"""Raw-current quality filtering shared by the paper's MI and dSNR analyses."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import Data_pipeline


def array_digest(array):
    """Hash dtype, shape, and ordered array contents in bounded chunks."""
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    for start in range(0, len(array), 256):
        digest.update(np.ascontiguousarray(array[start:start + 256]).tobytes())
    return digest.hexdigest()


def _linear_raw_statistics(stabilization):
    """Fit raw-current drift and robust residual noise using finite readings."""
    finite = np.isfinite(stabilization)
    count = finite.sum(axis=1)
    time = np.arange(stabilization.shape[1], dtype=np.float64)[None, :, None]
    safe_count = np.maximum(count, 1)
    mean_t = (finite * time).sum(axis=1) / safe_count
    mean_y = np.where(finite, stabilization, 0.0).sum(axis=1) / safe_count
    centered_t = time - mean_t[:, None, :]
    centered_y = np.where(finite, stabilization - mean_y[:, None, :], 0.0)
    denominator = np.where(finite, centered_t ** 2, 0.0).sum(axis=1)
    slope = np.divide(
        (centered_t * centered_y).sum(axis=1), denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    intercept = mean_y - slope * mean_t
    residual = stabilization - (slope[:, None, :] * time + intercept[:, None, :])
    # Entirely missing channels are already invalid; zeros avoid nanmedian warnings.
    residual = np.where(finite, residual, np.nan)
    residual = np.where((count == 0)[:, None, :], 0.0, residual)
    median = np.nanmedian(residual, axis=1)
    noise = 1.4826 * np.nanmedian(np.abs(residual - median[:, None, :]), axis=1)
    return count, slope, noise

def _recompute_response(raw, norm_index, background_reference, time_indices):
    """Normalize current and subtract the fitted baseline for QC-passing samples."""
    reference = raw[:, [norm_index], :]
    # QC ensures a finite, nonzero reference; retain the measured denominator.
    with np.errstate(over="raise", divide="raise", invalid="ignore"):
        response = raw / reference - 1.0
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    ref_index = int(background_reference.get("ref_t_idx", 14))
    method = background_reference.get("mode", "linear_fit")
    time = np.arange(raw.shape[1], dtype=np.float32)[None, :, None]
    if method == "linear_fit":
        stabilization_t = np.arange(ref_index + 1, dtype=np.float32)[None, :, None]
        stabilization_y = response[:, :ref_index + 1, :]
        t_mean = stabilization_t.mean(axis=1, keepdims=True)
        y_mean = stabilization_y.mean(axis=1, keepdims=True)
        t_centered = stabilization_t - t_mean
        slope = ((stabilization_y - y_mean) * t_centered).sum(axis=1, keepdims=True) / (t_centered ** 2).sum(axis=1, keepdims=True)
        intercept = y_mean - slope * t_mean
    elif method == "first_point":
        slope = -response[:, [0], :] / float(ref_index)
        intercept = -slope * float(ref_index)
    elif method == "median_slope":
        indices = np.arange(ref_index, dtype=np.float32)[None, :, None]
        slope = np.median(-response[:, :ref_index, :] / (float(ref_index) - indices), axis=1, keepdims=True)
        intercept = -slope * float(ref_index)
    else:
        raise ValueError(f"Unsupported saved background-correction method: {method!r}")
    corrected = (response - (slope * time + intercept)).astype(np.float32, copy=False)
    selected = corrected[:, time_indices, :]
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite response remains after raw-current quality checks.")
    return selected


def prepare_metric_data(
    bundle, *, data_root, average_mode=0, rh_values=(0, 50),
    min_reference_current=1e-9, min_baseline_snr=5.0,
    max_extrapolated_drift_fraction=1.0, show_progress=True,
):
    """Rebuild floor-free responses for the full common raw-QC-passing cohort.

    Start with the model bundle's physical sample identities, including both
    training/validation and held-out chips. Evaluate all loaded humidities,
    then select RH 0/50 for MI and dSNR. No random sampling is performed.
    Model inputs and SHAP observations are never modified.
    """
    for name, value in (("min_reference_current", min_reference_current),
                        ("min_baseline_snr", min_baseline_snr),
                        ("max_extrapolated_drift_fraction", max_extrapolated_drift_fraction)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    state = bundle["preprocess_state"]
    cut = state["cut_window"]
    norm_index = int(state["normalization_time_index"])
    background_index = int(state["background_reference_time_index"])
    original_times = np.arange(int(cut["t_start"]), min(int(cut["t_end"]), int(state["raw_time_length"])))
    if norm_index < 2 or background_index < 1 or not len(original_times):
        raise ValueError("Raw QC requires a valid baseline and analysis time window")
    relative_vg = np.arange(int(cut["v_start"]), int(cut["v_end"]))
    if len(relative_vg) != len(bundle["vg_axis"]) or len(original_times) != len(bundle["time_axis"]):
        raise ValueError("Analysis axes must match the saved preprocessing window")
    metadata = pd.concat([bundle[name]["metadata"] for name in ("training", "heldout_test")], ignore_index=True).copy()
    if metadata.duplicated(["split", "filename", "device_index"]).any():
        raise ValueError("Duplicate sample identities in the analysis cohort")
    metadata["source_row_index"] = np.arange(len(metadata), dtype=np.int64)
    n_sample, n_vg = len(metadata), len(relative_vg)
    responses = np.empty((n_sample, len(original_times), n_vg), dtype=np.float32)
    reason_names = ("low_reference", "nonfinite_baseline", "low_snr", "excessive_drift", "nonfinite_analysis")
    failures = {name: np.zeros((n_sample, n_vg), dtype=bool) for name in reason_names}
    valid_channel = np.ones((n_sample, n_vg), dtype=bool)
    min_snr, max_drift = np.empty(n_sample), np.empty(n_sample)
    effective_minimum = max(float(min_reference_current), 1e-9)
    extrapolation = max(0, int(original_times.max()) - norm_index)
    last_required = max(int(original_times.max()), norm_index, background_index)
    background = {"ref_t_idx": background_index, "mode": state["background_method"]}
    files = {}
    for split in ("training", "heldout_test"):
        for row in Data_pipeline.discover_files(split, data_root=data_root).itertuples():
            files[(split, row.filename)] = Path(row.path)
    groups = metadata.groupby(["split", "filename"], sort=False)
    for identity, frame in tqdm(groups, total=len(groups), desc="Raw-current QC", disable=not show_progress):
        rows = frame.index.to_numpy(dtype=np.int64)
        devices = frame["device_index"].to_numpy(dtype=np.int64)
        raw_file = Data_pipeline._average_one_file(np.load(files[identity], mmap_mode="r"), average_mode)
        starts = frame["vth_index"].to_numpy(dtype=np.int64) - int(state["minimum_index"])
        indices = starts[:, None] + relative_vg[None, :]
        raw = np.take_along_axis(raw_file[devices, :last_required + 1, :], indices[:, None, :], axis=2)
        baseline = np.asarray(raw[:, :norm_index + 1, :], dtype=np.float64)
        reference = np.abs(baseline[:, norm_index, :])
        count, slope, noise = _linear_raw_statistics(baseline)
        low_reference = (~np.isfinite(reference)) | (reference < effective_minimum) | (reference <= 0)
        finite_baseline = (count / float(norm_index + 1) >= 1.0) & (count >= 3)
        snr = np.divide(reference, noise, out=np.full_like(reference, np.inf), where=noise > 0)
        snr[low_reference] = 0
        drift = np.divide(np.abs(slope) * extrapolation, reference,
                          out=np.full_like(reference, np.inf), where=reference > 0)
        block = {"low_reference": low_reference, "nonfinite_baseline": ~finite_baseline,
                 "low_snr": snr < min_baseline_snr,
                 "excessive_drift": drift > max_extrapolated_drift_fraction,
                 "nonfinite_analysis": ~np.isfinite(raw[:, original_times, :]).all(axis=1)}
        for name, failed in block.items():
            failures[name][rows] = failed
            valid_channel[rows] &= ~failed
        min_snr[rows], max_drift[rows] = np.min(snr, axis=1), np.max(drift, axis=1)
        passing = valid_channel[rows].all(axis=1)
        if passing.any():
            responses[rows[passing]] = _recompute_response(raw[passing], norm_index, background, original_times)
    valid_sample = valid_channel.all(axis=1)
    valid_response_digest = array_digest(responses[valid_sample])
    requested = sorted(set(float(value) for value in rh_values))
    if not requested or not np.isfinite(requested).all():
        raise ValueError("Select at least one finite analysis RH value")
    rh_mask = pd.to_numeric(metadata["rh_percent"]).isin(requested).to_numpy()
    selected = rh_mask & valid_sample
    selected_metadata = metadata.loc[selected].reset_index(drop=True).copy()
    expected_groups = metadata.loc[rh_mask].groupby(["gas", "rh_percent"]).size()
    counts = selected_metadata.groupby(["gas", "rh_percent"]).size().reindex(expected_groups.index, fill_value=0)
    if not selected.any() or (counts < 6).any():
        raise ValueError("Raw QC must retain at least six samples in every gas/RH group")
    x = responses[selected]
    y = selected_metadata["gas"].map(bundle["gas_label_map"]).to_numpy(dtype=np.int64)
    sample_qc = metadata.copy()
    sample_qc["quality_pass"] = valid_sample
    sample_qc["selected_for_metrics"] = selected
    sample_qc["num_valid_channels"] = valid_channel.sum(axis=1)
    sample_qc["num_failed_channels"] = n_vg - valid_channel.sum(axis=1)
    sample_qc["min_baseline_snr"] = min_snr
    sample_qc["max_extrapolated_drift_fraction"] = max_drift
    channel_qc = pd.DataFrame({"vg_index": np.arange(n_vg), "source_vg_index": relative_vg,
                              "vg_volts": bundle["vg_axis"], "num_samples": n_sample,
                              "num_valid_samples": valid_channel.sum(axis=0),
                              "num_excluded_samples": (~valid_channel).sum(axis=0)})
    for name, failed in failures.items():
        sample_qc[name + "_count"] = failed.sum(axis=1)
        channel_qc[name + "_count"] = failed.sum(axis=0)
    group_summary = sample_qc.groupby(["split", "rh_percent", "gas"], sort=False).agg(
        num_samples_before=("quality_pass", "size"), num_samples_valid=("quality_pass", "sum"),
        metric_samples=("selected_for_metrics", "sum")).reset_index()
    group_summary["num_samples_excluded"] = group_summary["num_samples_before"] - group_summary["num_samples_valid"]
    summary = {"quality_version": "raw-baseline-common-cohort-v1", "cohort_policy": "all_selected_vg_channels",
               "min_reference_current": float(min_reference_current), "effective_min_reference_current": effective_minimum,
               "min_baseline_snr": float(min_baseline_snr), "max_extrapolated_drift_fraction": float(max_extrapolated_drift_fraction),
               "min_finite_fraction": 1.0, "reference_time_index": norm_index,
               "analysis_original_time_indices": original_times.tolist(),
               "num_samples_before": n_sample, "num_samples_valid": int(valid_sample.sum()),
               "num_samples_excluded": int((~valid_sample).sum()), "num_invalid_channels": int((~valid_channel).sum()),
               "metric_rh_values": requested, "metric_num_samples_before": int(rh_mask.sum()),
               "metric_num_samples_valid": int(selected.sum()), "metric_num_samples_excluded": int((rh_mask & ~valid_sample).sum()),
               "valid_response_sha256": valid_response_digest, "metric_response_sha256": array_digest(x),
               "source_rows_sha256": hashlib.sha256(selected_metadata["source_row_index"].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
               "class_labels_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
    return {"x": x, "y": y, "metadata": selected_metadata, "quality_summary": summary,
            "sample_qc": sample_qc, "channel_qc": channel_qc, "group_summary": group_summary}
