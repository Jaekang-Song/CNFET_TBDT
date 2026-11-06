"""Train independent single-Vg CNNs and read the preserved humidity results.

Vg indices always refer to the aligned, cropped input, never the raw array.
The existing CNN1D_TBDT receives (sample, time, 1). Candidate selection uses
validation accuracy; held-out predictions are reporting artifacts only.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

import ML_models
from . import Data_pipeline, Run_analysis, Training_utils
from .Checkpoint_compat import (
    fingerprint_config, model_config_from_snapshot, training_config_from_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "results/single_vg"
_GRID_KEYS = ("hidden_size", "learning_rate", "weight_decay", "dropout_p")


def _safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _json(value):
    return json.dumps(_safe(value), sort_keys=True, allow_nan=False)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def expand_parameter_sets(config):
    """Expand the ordered grid, with the last axis (dropout) varying fastest."""
    grid = config["PARAMETER_GRID"]
    if tuple(grid) != _GRID_KEYS:
        raise ValueError(f"PARAMETER_GRID must use this order: {_GRID_KEYS}")
    if any(not isinstance(grid[key], (list, tuple)) or not grid[key] for key in grid):
        raise ValueError("Each parameter axis must be a nonempty list or tuple")
    return [
        {"candidate_index": index, **dict(zip(grid, values))}
        for index, values in enumerate(itertools.product(*(grid[key] for key in grid)))
    ]


def _indices(values, size, name):
    items = list(values)
    if not items or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in items):
        raise ValueError(f"{name} must contain integer indices")
    indices = np.asarray(items, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f"{name} contains duplicate or out-of-range indices")
    return indices


def _hash_array(digest, values):
    array = np.asarray(values)
    digest.update(_json({"shape": array.shape, "dtype": array.dtype.str}).encode())
    # Buffer in blocks so fingerprinting large datasets does not duplicate them.
    for block in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                           order="C", buffersize=1024 * 1024):
        digest.update(block.tobytes())


def _fingerprint(config, bundle, train_indices, val_indices):
    digest = hashlib.sha256(_json(fingerprint_config(config)).encode())
    for split in ("training", "heldout_test"):
        for key in ("x", "y"):
            _hash_array(digest, bundle[split][key])
        digest.update(_json(bundle[split]["metadata"].to_dict("list")).encode())
    for values in (train_indices, val_indices, bundle["vg_axis"]):
        _hash_array(digest, values)
    digest.update(_json(bundle["gas_label_map"]).encode())
    return digest.hexdigest()


def _select_best(records):
    best = {}
    for row in sorted(records, key=lambda item: (item["candidate_index"], item["seed"])):
        key = row["vg_index"], row["seed"]
        if key not in best or row["validation_accuracy"] > best[key]["validation_accuracy"]:
            best[key] = row
    return sorted(best.values(), key=lambda item: (item["vg_index"], item["seed"]))


def _tables(records, candidates, configured_vg):
    ordered = [dict(sorted(row.items())) for row in
               sorted(records, key=lambda item: (item["vg_index"], item["candidate_index"], item["seed"]))]
    best = _select_best(ordered)
    summary = []
    for row in best:
        count = sum(item["vg_index"] == row["vg_index"] for item in ordered)
        summary.append({
            **row,
            "vg_local_index": configured_vg.index(row["vg_index"]),
            "completed_runs": count,
            "expected_runs": len(candidates),
            "is_complete": count == len(candidates),
        })
    return pd.DataFrame(ordered), pd.DataFrame(best), pd.DataFrame(summary)


def _cached_record(marker, output_root, fingerprint):
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload["fingerprint"] != fingerprint:
        raise ValueError(f"Completed case belongs to different inputs: {marker}")
    record = payload["record"]
    required = ("checkpoint_path", "config_snapshot_path", "epoch_history_path",
                "validation_predictions_path", "heldout_test_predictions_path")
    return record if all((output_root / record[key]).is_file() for key in required) else None


@contextlib.contextmanager
def _sweep_lock(directory):
    # OS locks are released when an interrupted kernel/process exits.
    with (directory / ".sweep.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"Another sweep is using {directory}") from error
        try:
            yield
        finally:
            handle.seek(0)
            release()


def run_single_vg_sweep(config, bundle, output_root=None, vg_indices=None,
                        candidate_indices=None, resume=True, device=None):
    """Run requested cases, retaining immutable splits and durable completion.

    Resume permits adding requested Vg/candidate subsets to the same experiment.
    Changing the config, input values, labels, metadata, or split indices requires
    a different output root. Interrupted attempts are retained and retried in a
    new artifact directory; only a completed marker makes a case reusable.
    """
    directory = (Path(output_root) if output_root is not None
                 else PROJECT_ROOT / config["OUTPUT"]["models_dir"]).resolve()
    # Guard before creating directories or lock files, including relocated archives.
    if directory == ARCHIVE_ROOT.resolve() or ARCHIVE_ROOT.resolve() in directory.parents:
        raise ValueError("Preserved result archives cannot be used as sweep output")
    if any((ancestor / "archive_manifest.json").is_file()
           for ancestor in (directory, *directory.parents)):
        raise ValueError("Preserved result archives cannot be used as sweep output")
    directory.mkdir(parents=True, exist_ok=True)
    with _sweep_lock(directory):
        return _run_single_vg_sweep(config, bundle, directory, vg_indices, candidate_indices, resume, device)


def _run_single_vg_sweep(config, bundle, output_root, vg_indices, candidate_indices, resume, device):
    config = deepcopy(config)
    config["MODEL"] = model_config_from_snapshot(config["MODEL"])
    config["TRAINING"] = training_config_from_snapshot(config["TRAINING"])
    if (config["MODEL"].get("model_name") != "CNN1D_TBDT" or
            config["MODEL"].get("dim_v") != 1 or config["MODEL"].get("vg_groups") != 1):
        raise ValueError("Single-Vg training requires CNN1D_TBDT with dim_v=1 and vg_groups=1")
    if float(config["TRAINING"].get("vg_input_dropout_p", 0.0)) != 0.0:
        raise ValueError("Vg input dropout must be disabled for single-Vg training")
    candidates = expand_parameter_sets(config)
    training, heldout = bundle["training"], bundle["heldout_test"]
    x_train, x_heldout = np.asarray(training["x"]), np.asarray(heldout["x"])
    if x_train.ndim != 3 or x_heldout.ndim != 3 or x_train.shape[1:] != x_heldout.shape[1:]:
        raise ValueError("Both input splits must have the same (sample, time, Vg) shape")
    for split in (training, heldout):
        if not len(split["y"]) or len(split["x"]) != len(split["y"]) or len(split["y"]) != len(split["metadata"]):
            raise ValueError("Each nonempty split must have aligned inputs, labels, and metadata")
    label_map = dict(bundle["gas_label_map"])
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError("gas_label_map indices must be contiguous from zero")
    train_indices = _indices(bundle["train_indices"], len(x_train), "train_indices")
    val_indices = _indices(bundle["val_indices"], len(x_train), "val_indices")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training and validation indices overlap")
    configured_vg = _indices(config["VG_INDICES"], x_train.shape[2], "VG_INDICES").tolist()
    selected_vg = _indices(configured_vg if vg_indices is None else vg_indices,
                           x_train.shape[2], "vg_indices").tolist()
    if not set(selected_vg).issubset(configured_vg):
        raise ValueError("vg_indices must be a subset of CONFIG['VG_INDICES']")
    selected_candidates = _indices(range(len(candidates)) if candidate_indices is None else candidate_indices,
                                   len(candidates), "candidate_indices").tolist()
    if len(bundle["vg_axis"]) != x_train.shape[2]:
        raise ValueError("vg_axis must align with the cropped Vg input")
    fingerprint = _fingerprint(config, bundle, train_indices, val_indices)
    output_root = Path(output_root) if output_root is not None else PROJECT_ROOT / config["OUTPUT"]["models_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "sweep_manifest.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError("Output already contains a sweep; choose a new output_root or resume=True")
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint mismatch: config, data, labels, or split membership changed")
    else:
        if list(output_root.glob("runs/v*/c*/s*/completed.json")):
            raise ValueError("Completed cases exist without a sweep manifest")
        _atomic_text(manifest, _json({"schema_version": 1, "fingerprint": fingerprint, "config": config}))
    _atomic_text(output_root / "parameter_sets.csv", pd.DataFrame(candidates).to_csv(index=False))
    seed = int(config["TRAINING"].get("random_seed", 0))
    records = {}
    for vg in configured_vg:
        for candidate in candidates:
            index = candidate["candidate_index"]
            marker = output_root / "runs" / f"v{vg:03d}" / f"c{index:03d}" / f"s{seed}" / "completed.json"
            cached = _cached_record(marker, output_root, fingerprint)
            if cached is not None:
                records[vg, index] = cached

    def persist():
        frames = _tables(list(records.values()), candidates, configured_vg)
        for filename, frame in zip(("completed_runs.csv", "best_by_vg_seed.csv", "single_vg_summary.csv"), frames):
            _atomic_text(output_root / filename, frame.to_csv(index=False))
        return frames

    persist()
    for vg in selected_vg:
        train_view = x_train[:, :, [vg]]
        test_view = x_heldout[:, :, [vg]]
        full_training = Data_pipeline.SensorDataset(train_view, training["y"], training["metadata"])
        dataset_train = Subset(full_training, train_indices.tolist())
        dataset_val = Subset(full_training, val_indices.tolist())
        dataset_heldout = Data_pipeline.SensorDataset(test_view, heldout["y"], heldout["metadata"])
        for index in selected_candidates:
            if (vg, index) in records:
                continue
            candidate = candidates[index]
            case_config = deepcopy(config)
            case_config["MODEL"].update({key: candidate[key] for key in ("hidden_size", "dropout_p")})
            case_config["TRAINING"].update({key: candidate[key] for key in ("learning_rate", "weight_decay")})
            case_config["CASE_INPUT_SELECTION"] = {
                "selected_vg_indices": [vg],
                "selected_vg_axis_volts": [float(bundle["vg_axis"][vg])],
                "selected_time_indices": list(range(x_train.shape[1])),
                "dim_v": 1,
                "time_point_count": x_train.shape[1],
            }
            case_dir = output_root / "runs" / f"v{vg:03d}" / f"c{index:03d}" / f"s{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            attempt = "attempt_" + uuid.uuid4().hex[:12]
            log_path = case_dir / f"{attempt}.log"
            print(f"Single Vg {vg}: candidate {index + 1}/{len(candidates)}, seed {seed}")
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                result = Training_utils.fit_model(
                    dataset_train, dataset_val, num_classes=len(label_map),
                    model_config=case_config["MODEL"], training_config=case_config["TRAINING"], device=device,
                )
            y_true, y_pred = Run_analysis.predict_dataset(
                result["model"], dataset_heldout, device=result["device"],
                batch_size=int(case_config["TRAINING"].get("batch_size", 128)),
            )
            artifact_dir = Training_utils.save_run_artifacts(
                result, config=case_config, gas_label_map=label_map,
                output_root=case_dir, run_name=attempt,
            )
            log_path.replace(artifact_dir / "training_log.txt")
            np.save(artifact_dir / "heldout_test_true_labels.npy", y_true)
            np.save(artifact_dir / "heldout_test_predictions.npy", y_pred)
            predictions = heldout["metadata"].reset_index(drop=True).copy()
            predictions["y_true"], predictions["y_pred"] = y_true, y_pred
            predictions.to_csv(artifact_dir / "heldout_test_prediction_table.csv", index=False)
            row = {
                **candidate, "vg_index": vg, "vg_value": float(bundle["vg_axis"][vg]), "seed": seed,
                "selected_epoch": int(result["best_epoch"]),
                "selection_metric": "validation_accuracy",
                "validation_accuracy": float(result["validation"]["accuracy"]),
                "heldout_test_accuracy": 100.0 * float(np.mean(y_true == y_pred)),
            }
            rh = heldout["metadata"]["rh_percent"].to_numpy()
            for value in (0, 30, 50):
                mask = rh == value
                row[f"heldout_test_accuracy_rh{value}"] = 100.0 * float(np.mean(y_true[mask] == y_pred[mask])) if mask.any() else None
            for key, filename in {
                "checkpoint_path": "best_checkpoint.pt", "config_snapshot_path": "config_snapshot.json",
                "epoch_history_path": "training_history.csv", "validation_predictions_path": "validation_predictions.npy",
                "heldout_test_predictions_path": "heldout_test_predictions.npy",
            }.items():
                row[key] = (artifact_dir / filename).relative_to(output_root).as_posix()
            _atomic_text(case_dir / "completed.json", _json({"fingerprint": fingerprint, "record": row}))
            records[vg, index] = row
            persist()
    completed, best, summary = persist()
    return {"output_root": output_root, "completed_runs": completed,
            "best_by_vg_seed": best, "summary": summary, "fingerprint": fingerprint}


def load_archived_model(vg_index, archive_root=ARCHIVE_ROOT, device="cpu"):
    """Load a selected archived checkpoint with strict parameter matching.

    Snapshots store effective model fields in MODEL. Optional singleton SWEEP
    axes are also supported. Code-to-formula translation changes
    label names only; the checkpoint's integer class indices remain authoritative.
    """
    index = int(_indices([vg_index], 96, "vg_index")[0])
    directory = Path(archive_root) / "selected_models" / f"v{index:03d}"
    snapshot = json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
    config = snapshot["config"]
    checkpoint = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model_config = dict(config["MODEL"])
    sweep = config.get("SWEEP", {})
    for parameter, axis in (("hidden_size", "hidden_sizes"), ("dropout_p", "dropout_p"), ("dropout2d_p", "dropout2d_p")):
        if axis in sweep:
            if len(sweep[axis]) != 1:
                raise ValueError(f"Archived SWEEP.{axis} must resolve to one selected value")
            model_config[parameter] = sweep[axis][0]
    if "pool_options" in sweep:
        if len(sweep["pool_options"]) != 1:
            raise ValueError("Archived pooling must resolve to one selected option")
        mode, tau = sweep["pool_options"][0]
        model_config["pool_mode"] = mode
        if tau is not None:
            model_config["lse_tau"] = tau
    original_labels = {str(key): int(value) for key, value in checkpoint["gas_label_map"].items()}
    gas_label_map = {Data_pipeline.CHEMICAL_FORMULA_BY_CODE.get(key, key): value for key, value in original_labels.items()}
    if len(gas_label_map) != len(original_labels) or sorted(gas_label_map.values()) != list(range(len(gas_label_map))):
        raise ValueError("Archived gas labels do not form a unique contiguous class map")
    model_config = model_config_from_snapshot(model_config)
    model = ML_models.build_model(num_classes=len(gas_label_map), **model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model, {"run_dir": directory, "config": config, "snapshot": snapshot,
                   "model_config": model_config, "gas_label_map": gas_label_map,
                   "original_gas_label_map": original_labels,
                   "case_input_selection": config.get("CASE_INPUT_SELECTION", {}),
                   "checkpoint": checkpoint}


def load_single_vg_results(root=ARCHIVE_ROOT):
    """Read saved summary percentages without requiring the sensor dataset."""
    root = Path(root)
    frame = pd.read_csv(root if root.suffix.lower() == ".csv" else root / "single_vg_summary.csv")
    aliases = {
        "validation_accuracy": "best_test_mean", "heldout_test_accuracy": "best_heldout_mean",
        **{f"heldout_test_accuracy_rh{rh}": f"heldout_acc_rh{rh}_mean" for rh in (0, 30, 50)},
    }
    for destination, source in aliases.items():
        if destination not in frame and source in frame:
            frame[destination] = frame[source]
    required = {"vg_index", "vg_value", *aliases}
    if required - set(frame):
        raise ValueError(f"Summary is missing columns: {sorted(required - set(frame))}")
    return frame.sort_values("vg_index", kind="stable").reset_index(drop=True)


def plot_single_vg_results(results=ARCHIVE_ROOT, metric="rh30"):
    """Plot saved validation, overall, or RH 0/30/50 held-out accuracy."""
    import matplotlib.pyplot as plt

    columns = {"overall": "heldout_test_accuracy", "validation": "validation_accuracy",
               **{f"rh{rh}": f"heldout_test_accuracy_rh{rh}" for rh in (0, 30, 50)}}
    if metric not in columns:
        raise ValueError(f"metric must be one of {tuple(columns)}")
    frame = results if isinstance(results, pd.DataFrame) else load_single_vg_results(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["vg_value"], frame[columns[metric]], marker="o")
    ax.set_xlabel("Gate voltage (V)")
    ax.set_ylabel("Validation accuracy (%)" if metric == "validation" else "Held-out accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall held-out" if metric == "overall" else "Validation" if metric == "validation" else f"RH {metric[2:]}%")
    if "is_complete" in frame and not frame["is_complete"].eq(True).all():
        ax.set_title(ax.get_title() + " (partial candidate sweep)")
    fig.tight_layout()
    return fig, ax


__all__ = ["expand_parameter_sets", "run_single_vg_sweep", "load_archived_model",
           "load_single_vg_results", "plot_single_vg_results"]

