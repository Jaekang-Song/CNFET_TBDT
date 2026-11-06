"""dSNR, mutual-information, and SHAP analyses for the paper dataset."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

from .Data_pipeline import format_rh_label, normalize_rh_value


DEFAULT_JOINT_TIME_INDICES = np.array([0, 1, 2, 3, 5, 8, 12, 18, 25, 35], dtype=np.int64)


def _balanced_indices(
    frame: pd.DataFrame,
    target_size: int,
    *,
    label_column: str = "gas",
    random_state: int = 0,
) -> np.ndarray:
    """Choose an approximately class-balanced subset from one metadata frame."""
    if target_size <= 0 or frame.empty:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    groups = list(frame.groupby(label_column, sort=True).groups.values())
    per_group = max(1, int(np.ceil(target_size / max(1, len(groups)))))
    chosen = []
    for group_indices in groups:
        group_indices = np.asarray(list(group_indices), dtype=np.int64)
        count = min(per_group, len(group_indices))
        chosen.extend(rng.choice(group_indices, size=count, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) > target_size:
        chosen = rng.choice(chosen, size=target_size, replace=False)
    return np.sort(chosen)


def sample_metric_data(
    bundle: Mapping,
    *,
    max_samples_per_gas: int = 500,
    random_state: int = 0,
) -> dict:
    """Sample class-balanced observations for exploratory analyses.

    Published MI/dSNR use Metric_preprocessing.prepare_metric_data instead."""
    rng = np.random.default_rng(int(random_state))
    sample_parts, metadata_parts = [], []
    for split_name in ("training", "heldout_test"):
        split = bundle[split_name]
        frame = split["metadata"].copy()
        frame["source_index"] = np.arange(len(frame), dtype=np.int64)
        frame["source_split"] = split_name
        metadata_parts.append(frame)
    metadata = pd.concat(metadata_parts, ignore_index=True)

    chosen_rows = []
    for _, gas_frame in metadata.groupby("gas", sort=True):
        candidate = gas_frame.index.to_numpy(dtype=np.int64)
        count = min(int(max_samples_per_gas), len(candidate))
        chosen_rows.extend(rng.choice(candidate, size=count, replace=False).tolist())
    selected = metadata.loc[np.sort(np.asarray(chosen_rows, dtype=np.int64))].reset_index(drop=True)

    for split_name in ("training", "heldout_test"):
        split_rows = selected[selected["source_split"] == split_name]
        if split_rows.empty:
            continue
        indices = split_rows["source_index"].to_numpy(dtype=np.int64)
        sample_parts.append(bundle[split_name]["x"][indices])
    x = np.concatenate(sample_parts, axis=0)

    # The concatenation order above follows split order, so rebuild metadata to match it.
    selected = pd.concat(
        [selected[selected["source_split"] == name] for name in ("training", "heldout_test")],
        ignore_index=True,
    )
    y = selected["gas"].map(bundle["gas_label_map"]).to_numpy(dtype=np.int64)
    return {"x": x, "y": y, "metadata": selected}



def full_metric_data(bundle: Mapping, *, data_root, average_mode=0, show_progress=True) -> dict:
    """Load both physical folders after device screening, including training RH 30.

    Published MI/dSNR use Metric_preprocessing.prepare_metric_data for raw-current QC."""
    from . import Data_pipeline

    parts, frames = [], []
    for split_name in ("training", "heldout_test"):
        split = bundle[split_name]
        parts.append(split["x"])
        frames.append(split["metadata"].copy())
        all_files = Data_pipeline.discover_files(split_name, rh_values=None, data_root=data_root)
        present = set(split["files"]["filename"])
        missing = all_files[~all_files["filename"].isin(present)]
        if missing.empty:
            continue
        raw, metadata = Data_pipeline.load_averaged_split(
            missing, average_mode=average_mode, show_progress=show_progress,
        )
        values, metadata = Data_pipeline.apply_preprocess_state(
            raw, metadata, bundle["preprocess_state"],
        )
        del raw
        metadata["model_split"] = "analysis_only"
        parts.append(values)
        frames.append(metadata)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata.duplicated(["split", "filename", "device_index"]).any():
        raise ValueError("An observation appears more than once in full-data dSNR.")
    labels = metadata["gas"].map(bundle["gas_label_map"])
    if labels.isna().any():
        raise ValueError("Full-data dSNR contains an unknown gas class.")
    return {"x": np.concatenate(parts, axis=0),
            "y": labels.to_numpy(dtype=np.int64), "metadata": metadata}


def _geometric_mean(values: Sequence[float], epsilon: float = 1e-12) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(np.maximum(values, epsilon)))))


def compute_directional_dsnr(
    x: np.ndarray,
    labels: np.ndarray,
    vg_axis: np.ndarray,
) -> dict:
    """Compute pairwise centroid-axis dSNR at each gate-voltage position."""
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    vg_axis = np.asarray(vg_axis, dtype=np.float64)
    if x.ndim != 3 or len(x) != len(labels) or x.shape[2] != len(vg_axis):
        raise ValueError("x must be (sample, time, Vg) and match labels/vg_axis")
    class_order = np.unique(labels)
    if len(class_order) < 2:
        raise ValueError("dSNR requires at least two gas classes")
    class_indices = {value: np.flatnonzero(labels == value) for value in class_order}

    pairwise_rows, summary_rows = [], []
    for vg_index, vg_value in enumerate(vg_axis):
        centroids, residuals, centroid_energies = {}, {}, []
        for class_value in class_order:
            values = np.asarray(x[class_indices[class_value], :, vg_index], dtype=np.float64)
            centroid = values.mean(axis=0)
            centroids[class_value] = centroid
            residuals[class_value] = values - centroid
            centroid_energies.append(float(np.dot(centroid, centroid)))

        components = []
        for class_a, class_b in combinations(class_order, 2):
            delta = centroids[class_a] - centroids[class_b]
            separation = float(np.linalg.norm(delta))
            if separation > 0.0:
                direction = delta / separation
                variance_a = float(np.mean(np.square(residuals[class_a] @ direction)))
                variance_b = float(np.mean(np.square(residuals[class_b] @ direction)))
            else:
                variance_a = variance_b = 0.0
            pooled_variance = max(0.0, 0.5 * (variance_a + variance_b))
            components.append((class_a, class_b, separation, variance_a, variance_b, pooled_variance))

        positive_variances = np.asarray(
            [item[5] for item in components if item[5] > 0.0],
            dtype=np.float64,
        )
        if positive_variances.size:
            epsilon_reference = float(np.median(positive_variances))
        else:
            epsilon_reference = max(float(np.median(centroid_energies)), np.finfo(float).tiny)
        denominator_epsilon = max(
            np.finfo(float).eps * epsilon_reference,
            np.finfo(float).tiny,
        )

        pair_scores, between_scores, within_scores = [], [], []
        for class_a, class_b, separation, variance_a, variance_b, pooled_variance in components:
            within = float(np.sqrt(pooled_variance))
            score = float(separation / np.sqrt(max(pooled_variance, denominator_epsilon)))
            pair_scores.append(score)
            between_scores.append(separation)
            within_scores.append(within)
            pairwise_rows.append(
                {
                    "vg_index": vg_index,
                    "vg_volts": float(vg_value),
                    "class_a": int(class_a),
                    "class_b": int(class_b),
                    "centroid_separation": separation,
                    "class_a_directional_variance": max(0.0, variance_a),
                    "class_b_directional_variance": max(0.0, variance_b),
                    "pooled_directional_variance": pooled_variance,
                    "dSNR": score,
                    "denominator_epsilon": denominator_epsilon,
                }
            )
        pair_scores_array = np.asarray(pair_scores, dtype=np.float64)
        summary_rows.append(
            {
                "vg_index": vg_index,
                "vg_volts": float(vg_value),
                "B": _geometric_mean(between_scores),
                "W": _geometric_mean(within_scores),
                "dSNR": _geometric_mean(pair_scores),
                "pairwise_median": float(np.median(pair_scores_array)),
                "pairwise_q25": float(np.percentile(pair_scores_array, 25.0)),
                "pairwise_q75": float(np.percentile(pair_scores_array, 75.0)),
                "gas_pair_count": len(pair_scores),
                "denominator_epsilon": denominator_epsilon,
            }
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "pairwise": pd.DataFrame(pairwise_rows),
        "definition": "centroid-axis projected residual variance",
    }


def _digamma_integer(value: int) -> float:
    gamma = 0.5772156649015329
    if value <= 1:
        return -gamma
    return -gamma + float(np.sum(1.0 / np.arange(1, value)))


def _kl_entropy_chebyshev(x: np.ndarray, neighbors: int, rng: np.random.RandomState) -> float:
    x = np.asarray(x, dtype=np.float64)
    sample_count, dimensions = x.shape
    if sample_count <= 1:
        return 0.0
    k = min(int(neighbors), sample_count - 1)
    jittered = x + 1e-12 * rng.standard_normal(x.shape)
    search = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(jittered)
    distances, _ = search.kneighbors(jittered)
    radius = distances[:, -1] + 1e-300
    return (
        _digamma_integer(sample_count)
        - _digamma_integer(k)
        + dimensions * float(np.mean(np.log(radius)))
        + dimensions * np.log(2.0)
    )


def joint_mi_per_voltage_bits(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    neighbors: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate I(selected-time vector at each Vg; gas) in bits."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.RandomState(int(random_state))
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / float(len(labels))
    output = np.empty(x.shape[2], dtype=np.float64)
    for vg_index in range(x.shape[2]):
        values = x[:, :, vg_index]
        entropy = _kl_entropy_chebyshev(values, neighbors, rng)
        conditional = 0.0
        for class_value, count, probability in zip(classes, counts, probabilities):
            class_values = values[labels == class_value]
            class_neighbors = min(int(neighbors), max(1, int(count) - 1))
            conditional += probability * _kl_entropy_chebyshev(class_values, class_neighbors, rng)
        output[vg_index] = max(0.0, entropy - conditional) / np.log(2.0)
    return output


def compute_mutual_information(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    joint_time_indices: Iterable[int] = DEFAULT_JOINT_TIME_INDICES,
    neighbors: int = 5,
    random_state: int = 0,
    joint_vg_indices: Iterable[int] | None = None,
) -> dict:
    """Compute the scalar MI map and the voltage/time joint MI profiles."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), copy=False)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    time_indices = np.unique(np.asarray(list(joint_time_indices), dtype=np.int64))
    if np.any(time_indices < 0):
        raise ValueError("joint_time_indices must be non-negative")
    # The task configs share one upper cut bound, while NumPy slicing yields
    # 42, 29, and 32 active time points for chemical, solvent, and bacteria.
    time_indices = time_indices[time_indices < x.shape[1]]
    if time_indices.size == 0:
        raise ValueError("No joint_time_indices fall inside the active time window")
    pointwise_nats = mutual_info_classif(
        x.reshape(len(x), -1),
        labels,
        discrete_features=False,
        n_neighbors=int(neighbors),
        random_state=int(random_state),
    )
    pointwise_bits = pointwise_nats.reshape(x.shape[1], x.shape[2]) / np.log(2.0)
    joint_bits = joint_mi_per_voltage_bits(
        x[:, time_indices, :],
        labels,
        neighbors=neighbors,
        random_state=random_state,
    )
    if joint_vg_indices is None:
        joint_vg_indices = np.rint(np.linspace(0, x.shape[2] - 1, 6)).astype(np.int64)
    vg_indices = np.unique(np.asarray(list(joint_vg_indices), dtype=np.int64))
    if not len(vg_indices) or np.any(vg_indices < 0) or np.any(vg_indices >= x.shape[2]):
        raise ValueError("joint_vg_indices must lie inside the analysis Vg window")
    joint_time_bits = joint_mi_per_voltage_bits(
        np.transpose(x[:, :, vg_indices], (0, 2, 1)), labels,
        neighbors=neighbors, random_state=random_state,
    )
    return {
        "pointwise_bits": pointwise_bits,
        "joint_per_voltage_bits": joint_bits,
        "joint_per_time_bits": joint_time_bits,
        "joint_vg_indices": vg_indices,
        "joint_time_indices": time_indices,
        "neighbors": int(neighbors),
        "sample_count": len(x),
    }


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output[0] if isinstance(output, tuple) else output


def _resolve_device(value: str | torch.device) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sample_shap_inputs(
    bundle: Mapping, *, background_size: int, eval_size: int,
    background_rh_values, evaluation_rh_values, random_state: int,
) -> dict:
    """Select observations by a seeded permutation of the eligible pools."""
    rng = np.random.default_rng(int(random_state))
    background = bundle["training"]
    evaluation = bundle["heldout_test"]
    background_frame = background["metadata"].reset_index(drop=True)
    eval_frame = evaluation["metadata"].reset_index(drop=True)
    background_mask = (
        background_frame["split"].eq("training")
        & background_frame["model_split"].eq("train")
        & background_frame["rh_percent"].isin(background_rh_values)
    )
    eval_mask = (
        eval_frame["split"].eq("heldout_test")
        & eval_frame["model_split"].eq("heldout_test")
        & eval_frame["rh_percent"].isin(evaluation_rh_values)
    )
    background_ids = rng.permutation(np.flatnonzero(background_mask))[:int(background_size)]
    eval_ids = rng.permutation(np.flatnonzero(eval_mask))[:int(eval_size)]
    if not len(background_ids) or not len(eval_ids):
        raise ValueError("Empty SHAP background or evaluation cohort.")
    if set(background_frame.iloc[background_ids]["rh_percent"]) != set(background_rh_values):
        raise ValueError("SHAP background does not contain every requested humidity.")
    if set(eval_frame.iloc[eval_ids]["rh_percent"]) != set(evaluation_rh_values):
        raise ValueError("SHAP evaluation does not contain every requested humidity.")
    return {
        "background_x": background["x"][background_ids],
        "background_metadata": background_frame.iloc[background_ids].reset_index(drop=True),
        "eval_x": evaluation["x"][eval_ids], "eval_y": evaluation["y"][eval_ids],
        "eval_metadata": eval_frame.iloc[eval_ids].reset_index(drop=True),
    }


def _true_class_values(values, labels: np.ndarray, input_shape: tuple[int, int, int]) -> np.ndarray:
    if isinstance(values, list):
        array = np.stack([np.asarray(item) for item in values], axis=-1)
    else:
        array = np.asarray(values)
    sample_count, time_count, vg_count = input_shape
    if array.shape[:3] == (sample_count, time_count, vg_count):
        class_last = array
    elif array.ndim == 4 and array.shape[0] == sample_count and array.shape[2:] == (time_count, vg_count):
        class_last = np.moveaxis(array, 1, -1)
    else:
        raise ValueError(f"Unsupported SHAP output shape: {array.shape}")
    return class_last[np.arange(sample_count), :, :, labels]


def _bin_shap_curve(axis, values, size):
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("SHAP bin size must be a positive integer.")
    axis, values = np.asarray(axis), np.asarray(values)
    if axis.ndim != 1 or axis.shape != values.shape or not len(axis):
        raise ValueError("SHAP coordinates and values must be aligned and nonempty.")
    return pd.DataFrame([
        {"coordinate": float(axis[start:start + size].mean()),
         "mean_abs_shap": float(values[start:start + size].mean()),
         "points_in_bin": len(values[start:start + size])}
        for start in range(0, len(values), size)
    ])


def shap_profiles_from_class_maps(
    class_maps, labels, metadata, vg_axis, time_axis, *,
    vg_bin_size=8, time_bin_size=2,
):
    """Average absolute SHAP within each gas, then weight gases equally."""
    class_maps = np.asarray(class_maps, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if (class_maps.shape != (len(classes), len(time_axis), len(vg_axis))
            or not np.array_equal(classes, np.arange(len(classes)))
            or not np.isfinite(class_maps).all() or np.any(class_maps < 0)):
        raise ValueError("Expected finite nonnegative maps for every true gas class.")
    if len(metadata) != len(labels) or not metadata["split"].eq("heldout_test").all():
        raise ValueError("SHAP aggregation requires held-out observations only.")
    humidities = metadata["rh_percent"].unique()
    if len(humidities) != 1:
        raise ValueError("Aggregate one held-out humidity at a time.")
    importance_map = class_maps.mean(axis=0)
    voltage = _bin_shap_curve(vg_axis, importance_map.mean(axis=0), vg_bin_size)
    exposure = np.asarray(time_axis) >= 0
    temporal = _bin_shap_curve(
        np.asarray(time_axis)[exposure], importance_map.mean(axis=1)[exposure], time_bin_size,
    )
    voltage = voltage.rename(columns={"coordinate": "vg_volts"})
    temporal = temporal.rename(columns={"coordinate": "time_minutes"})
    for frame in (voltage, temporal):
        maximum = float(frame["mean_abs_shap"].max())
        frame["normalized_importance"] = frame["mean_abs_shap"] / maximum if maximum > 0 else 0.0
        frame["split"] = "heldout_test"
        frame["rh_percent"] = humidities[0]
        frame["sample_count"] = len(labels)
    return {
        "gas_mean_abs_maps": class_maps, "mean_abs_map": importance_map,
        "voltage_profiles": voltage, "time_profiles": temporal,
        "settings": {
            "aggregation": "mean_samples_abs_within_gas_then_equal_gas_mean",
            "class_counts": counts.tolist(), "eval_size": len(labels),
            "evaluation_rh": [normalize_rh_value(humidities[0])],
            "vg_bin_size": int(vg_bin_size), "time_bin_size": int(time_bin_size),
            "voltage_time_points": len(time_axis), "time_display": "time >= 0",
            "plot_normalization": "each curve divided by its maximum after averaging",
        },
    }


def compute_shap(
    model: torch.nn.Module, bundle: Mapping, *,
    background_size: int = 8192, eval_size: int = 8192,
    background_rh_values=(0, 50), evaluation_rh_values=(30,),
    batch_size: int = 256, nsamples: int = 200,
    vg_bin_size: int = 8, time_bin_size: int = 2,
    device: str | torch.device = "auto", random_state: int = 0,
    reference_dir=None,
) -> dict:
    """Compute true-class GradientSHAP with the specified samples and bin sizes."""
    import shap
    from .Shap_reference import model_fingerprint, select_reference_inputs

    if batch_size <= 0 or nsamples <= 0:
        raise ValueError("SHAP batch size and integration sample count must be positive.")
    if reference_dir is None:
        sampled = _sample_shap_inputs(
            bundle, background_size=background_size, eval_size=eval_size,
            background_rh_values=background_rh_values,
            evaluation_rh_values=evaluation_rh_values, random_state=random_state,
        )
    else:
        sampled = select_reference_inputs(bundle, reference_dir)
        reference = sampled["manifest"]
        if (background_size != reference["background_size"]
                or min(eval_size, reference["eval_size"]) != reference["eval_size"]
                or set(background_rh_values) != set(reference["background_rh"])
                or set(evaluation_rh_values) != set(reference["evaluation_rh"])):
            raise ValueError("Requested SHAP cohorts differ from the archived sample selection.")
    device = _resolve_device(device)
    wrapped = _LogitsOnly(model).to(device).eval()
    background = torch.as_tensor(sampled["background_x"], dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(wrapped, background, local_smoothing=0)
    values_parts = []
    eval_x, eval_y = sampled["eval_x"], sampled["eval_y"]
    for start in range(0, len(eval_x), int(batch_size)):
        stop = min(start + int(batch_size), len(eval_x))
        batch = torch.as_tensor(eval_x[start:stop], dtype=torch.float32, device=device)
        raw = explainer.shap_values(batch, nsamples=int(nsamples),
                                    rseed=int(random_state) + start)
        values_parts.append(_true_class_values(raw, eval_y[start:stop], tuple(batch.shape)))
        print(f"SHAP evaluated {stop}/{len(eval_x)} samples")
    values = np.concatenate(values_parts, axis=0)
    classes = np.arange(bundle["num_classes"])
    if not np.array_equal(np.unique(eval_y), classes) or not np.isfinite(values).all():
        raise ValueError("SHAP must contain finite true-class values for every gas.")
    class_maps = np.stack([
        np.abs(values[eval_y == gas]).mean(axis=0, dtype=np.float64) for gas in classes
    ])
    result = shap_profiles_from_class_maps(
        class_maps, eval_y, sampled["eval_metadata"], bundle["vg_axis"], bundle["time_axis"],
        vg_bin_size=vg_bin_size, time_bin_size=time_bin_size,
    )
    result.update(
        true_class_values=values, eval_y=eval_y, eval_metadata=sampled["eval_metadata"],
        background_metadata=sampled["background_metadata"],
        background_size=len(background), method="gradient", source="recomputed",
        device=str(device),
    )
    result["settings"].update(
        background_size=len(background), background_rh=list(background_rh_values),
        nsamples=int(nsamples), batch_size=int(batch_size), random_seed=int(random_state),
        model_state_sha256=model_fingerprint(model), source="recomputed",
    )
    return result


def plot_dsnr(result: Mapping, *, figsize=(6.0, 3.5)):
    frame = result["summary"]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(frame["vg_volts"], frame["dSNR"], color="#9A6B32", linewidth=1.8)
    ax.set(xlabel="$V_G$ (V)", ylabel="Directional dSNR")
    return fig, ax


def plot_mutual_information(result: Mapping, time_axis: np.ndarray, vg_axis: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    image = axes[0].imshow(
        result["pointwise_bits"],
        origin="lower",
        aspect="auto",
        extent=[vg_axis[0], vg_axis[-1], time_axis[0], time_axis[-1]],
        cmap="viridis",
    )
    fig.colorbar(image, ax=axes[0], label="MI (bits)")
    axes[0].set(xlabel="$V_G$ (V)", ylabel="Time (min)", title="Pointwise MI")
    axes[1].plot(vg_axis, result["joint_per_voltage_bits"], color="#C65F5F", linewidth=1.8)
    axes[1].set(xlabel="$V_G$ (V)", ylabel="MI (bits)", title="Joint-time MI")
    return fig, axes


def plot_shap_profiles(result: Mapping, *, figsize=(9.0, 3.5), yscale="log"):
    domains = [("voltage_profiles", "vg_volts", "$V_G$ (V)")]
    if "time_profiles" in result:
        domains.append(("time_profiles", "time_minutes", "Time (min)"))
    fig, axes = plt.subplots(1, len(domains), figsize=figsize, constrained_layout=True, squeeze=False)
    flat_axes = axes.ravel()
    for ax, (key, coordinate, xlabel) in zip(flat_axes, domains):
        for (split_name, rh), frame in result[key].groupby(["split", "rh_percent"], sort=False):
            values = frame["mean_abs_shap"]
            maximum = float(values.max())
            normalized = values / maximum if maximum > 0 else values * 0
            keep = normalized > 0 if yscale == "log" else np.ones(len(frame), bool)
            ax.plot(frame.loc[keep, coordinate], normalized[keep],
                    label=f"{split_name}, {format_rh_label(rh)}",
                    marker="o", markersize=3, linewidth=1.5)
        ax.set(xlabel=xlabel, ylabel="Normalized mean |SHAP|", yscale=yscale)
        ax.legend(frameon=False)
    return fig, flat_axes[0] if len(domains) == 1 else flat_axes


def save_analysis_results(
    output_dir: str | Path,
    *,
    dsnr_result: Mapping | None = None,
    mi_result: Mapping | None = None,
    shap_result: Mapping | None = None,
) -> Path:
    """Save compact numerical artifacts needed to reproduce plotted curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dsnr_result is not None:
        dsnr_result["summary"].to_csv(output_dir / "dsnr_by_vg.csv", index=False)
        dsnr_result["pairwise"].to_csv(output_dir / "dsnr_pairwise_by_vg.csv", index=False)
        if "sample_metadata" in dsnr_result:
            dsnr_result["sample_metadata"].to_csv(output_dir / "dsnr_sample_metadata.csv", index=False)
    if mi_result is not None:
        np.savez_compressed(
            output_dir / "mutual_information.npz",
            pointwise_bits=mi_result["pointwise_bits"],
            joint_per_voltage_bits=mi_result["joint_per_voltage_bits"],
            joint_time_indices=mi_result["joint_time_indices"],
            joint_per_time_bits=mi_result["joint_per_time_bits"],
            joint_vg_indices=mi_result["joint_vg_indices"],
        )
        for key in ("time_coordinates", "vg_coordinates"):
            mi_result[key].to_csv(output_dir / f"mi_{key}.csv", index=False)
        mi_result["sample_metadata"].to_csv(output_dir / "mi_sample_metadata.csv", index=False)
        for key in ("sample_qc", "channel_qc", "group_summary"):
            mi_result[key].to_csv(output_dir / f"metric_{key}.csv", index=False)
        (output_dir / "metric_quality_summary.json").write_text(
            json.dumps(mi_result["quality_summary"], indent=2) + "\n", encoding="utf-8",
        )
    if shap_result is not None:
        shap_result["voltage_profiles"].to_csv(output_dir / "shap_RH30_vg.csv", index=False)
        shap_result["time_profiles"].to_csv(output_dir / "shap_RH30_time.csv", index=False)
        np.savez_compressed(
            output_dir / "shap_mean_abs_maps.npz",
            per_gas=shap_result["gas_mean_abs_maps"], mean=shap_result["mean_abs_map"],
        )
        if shap_result.get("true_class_values") is not None:
            np.savez_compressed(
                output_dir / "shap_heldout_true_class_values.npz",
                values=shap_result["true_class_values"], labels=shap_result["eval_y"],
            )
        shap_result["background_metadata"].to_csv(output_dir / "shap_background_metadata.csv", index=False)
        shap_result["eval_metadata"].to_csv(output_dir / "shap_heldout_sample_metadata.csv", index=False)
        (output_dir / "shap_settings.json").write_text(
            json.dumps(shap_result["settings"], indent=2) + "\n", encoding="utf-8",
        )
    return output_dir


__all__ = [
    "DEFAULT_JOINT_TIME_INDICES",
    "full_metric_data",
    "shap_profiles_from_class_maps",
    "compute_directional_dsnr",
    "compute_mutual_information",
    "compute_shap",
    "joint_mi_per_voltage_bits",
    "plot_dsnr",
    "plot_mutual_information",
    "plot_shap_profiles",
    "sample_metric_data",
    "save_analysis_results",
]
