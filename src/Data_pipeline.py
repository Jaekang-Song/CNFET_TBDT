"""Local data loading and preprocessing for the paper notebooks.

The pipeline classifies gas identity and preserves concentration labels,
humidity, and the physical training/heldout-test folder boundary as metadata.
Every operation is self-contained in this repository.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "TBDT_Dataset"
DATA_ROOT = DATASET_ROOT / "0_chemical_gas"
SPLITS = ("training", "heldout_test")
SENSORS_PER_FILE = 256
AMBIENT_DATASETS = frozenset({"1_solvent", "2_bacteria"})
UNQUANTIFIED_DATASETS = frozenset({"1_solvent", "2_bacteria"})

# Filename codes are retained separately, while model labels use chemical formulas.
CHEMICAL_FORMULA_BY_CODE = {
    "A": "C3H6O",
    "B": "C6H6",
    "CD": "CO2",
    "E": "C2H5OH",
    "H": "H2",
    "HS": "H2S",
    "M": "CH3OH",
    "ND": "NO2",
    "NH": "NH3",
    "P": "C3H7OH",
    "Pr": "C3H8",
    "T": "C7H8",
}
_FILE_RE = re.compile(
    r"^(?:avg\d+_)?"
    r"(?P<experiment_set>[^_]+)_"
    r"(?P<chip>[^_]+)_"
    r"(?P<gas>[^_]+)_"
    r"(?P<concentration>[^_]+)_"
    r"(?P<rh_percent>[Aa]|[-+]?\d+(?:\.\d+)?)_"
    r"(?P<repeat_index>\d+)$"
)


def resolve_data_root(data_dir: str | Path) -> Path:
    """Resolve current configs and legacy run snapshots to the local dataset."""
    data_path = Path(data_dir)
    if data_path.is_absolute():
        return data_path
    if data_path.parts and data_path.parts[0].lower() == "data":
        data_path = Path("TBDT_Dataset", *data_path.parts[1:])
    return PROJECT_ROOT / data_path


def normalize_rh_value(value: float | str) -> float | str:
    """Keep numeric RH percentages and represent ambient humidity as 'A'."""
    if isinstance(value, str) and value.strip().upper() == "A":
        return "A"
    rh_value = float(value)
    return int(rh_value) if rh_value.is_integer() else rh_value


def format_rh_label(value: float | str) -> str:
    """Format numeric RH and ambient conditions for figure labels."""
    value = normalize_rh_value(value)
    return "Ambient" if value == "A" else f"RH {value:g}%"


def parse_sensor_filename(path: str | Path) -> dict:
    """Parse sample identifiers, retaining concentration as a filename label."""
    path = Path(path)
    match = _FILE_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Unexpected sensor filename: {path.name}")
    values = match.groupdict()
    rh_value = normalize_rh_value(values["rh_percent"])
    gas_code = values["gas"]
    return {
        "experiment_set": values["experiment_set"],
        "chip": values["chip"],
        "gas_code": gas_code,
        "gas": CHEMICAL_FORMULA_BY_CODE.get(gas_code, gas_code),
        "concentration": (
            "NQ" if values["concentration"].upper() == "NQ" else values["concentration"]
        ),
        "rh_percent": rh_value,
        "repeat_index": int(values["repeat_index"]),
        "filename": path.name,
    }


def _normalize_rh_values(rh_values: Iterable[float | str] | float | str | None):
    if rh_values is None:
        return None
    if isinstance(rh_values, str) and rh_values.strip().lower() == "all":
        return None
    if np.isscalar(rh_values):
        return {normalize_rh_value(rh_values)}
    return {normalize_rh_value(value) for value in rh_values}


def discover_files(
    split: str,
    rh_values: Iterable[float | str] | float | str | None = None,
    data_root: str | Path = DATA_ROOT,
) -> pd.DataFrame:
    """Select supplied files by physical split and RH, retaining concentration labels."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing data split directory: {split_dir}")

    allowed_rh = _normalize_rh_values(rh_values)
    dataset_name = Path(data_root).name
    if dataset_name in AMBIENT_DATASETS and allowed_rh is not None:
        # Legacy solvent/bacteria configs used 0 as the ambient condition code.
        allowed_rh = {"A" if value == 0 else value for value in allowed_rh}
    rows = []
    for path in natsorted(split_dir.glob("*.npy"), key=lambda item: item.name):
        row = parse_sensor_filename(path)
        if dataset_name != "0_chemical_gas":
            row["gas"] = row["gas_code"]
        if dataset_name in UNQUANTIFIED_DATASETS:
            # Older archives used numeric placeholders for unquantified samples.
            row["concentration"] = "NQ"
        if dataset_name in AMBIENT_DATASETS and row["rh_percent"] == 0:
            # Also support older dataset archives whose filenames still use 0.
            row["rh_percent"] = "A"
        if allowed_rh is not None and row["rh_percent"] not in allowed_rh:
            continue
        row["split"] = split
        row["path"] = path
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No .npy files matched split={split!r}, rh_values={rh_values!r}"
        )
    return pd.DataFrame(rows)


def summarize_files(data_root: str | Path = DATA_ROOT) -> pd.DataFrame:
    """Summarize available files by folder, humidity, and gas."""
    frames = [discover_files(split, data_root=data_root) for split in SPLITS]
    files = pd.concat(frames, ignore_index=True)
    return (
        files.groupby(["split", "rh_percent", "gas"], sort=True)
        .size()
        .rename("file_count")
        .reset_index()
    )


def _average_one_file(raw: np.ndarray, average_mode: int) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[0] != SENSORS_PER_FILE:
        raise ValueError(
            "Each data file must have shape (256, raw_vg, raw_time); "
            f"received {tuple(raw.shape)}"
        )
    n_avg = 2 ** int(average_mode)
    if n_avg <= 0 or 32 % n_avg != 0:
        raise ValueError("average_mode must produce a factor that divides 32")
    _, raw_vg, raw_time = raw.shape
    averaged = raw.reshape(32 // n_avg, n_avg, 8, raw_vg, raw_time)
    averaged = np.nanmean(averaged, axis=1, dtype=np.float32)
    averaged = averaged.reshape(SENSORS_PER_FILE // n_avg, raw_vg, raw_time)
    return np.transpose(averaged, (0, 2, 1)).astype(np.float32, copy=False)


def load_averaged_split(
    file_frame: pd.DataFrame,
    average_mode: int = 0,
    show_progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load files one at a time and return samples as (sample, time, Vg)."""
    if file_frame.empty:
        raise ValueError("file_frame is empty")
    n_avg = 2 ** int(average_mode)
    samples_per_file = SENSORS_PER_FILE // n_avg
    first = np.load(file_frame.iloc[0]["path"], mmap_mode="r")
    first_averaged = _average_one_file(first, average_mode)
    output = np.empty(
        (len(file_frame) * samples_per_file, *first_averaged.shape[1:]),
        dtype=np.float32,
    )
    metadata_parts = []

    rows = file_frame.reset_index(drop=True)
    iterator = rows.iterrows()
    if show_progress:
        split_key = str(rows.iloc[0]["split"])
        split_name = "held out test" if split_key == "heldout_test" else split_key
        iterator = tqdm(
            iterator,
            total=len(rows),
            desc=split_name,
            bar_format="{desc}: {percentage:3.0f}%|{bar}|",
            dynamic_ncols=True,
            leave=True,
        )

    for file_index, row in iterator:
        raw = np.load(row["path"], mmap_mode="r")
        averaged = _average_one_file(raw, average_mode)
        start = file_index * samples_per_file
        stop = start + samples_per_file
        output[start:stop] = averaged
        metadata_parts.append(
            pd.DataFrame(
                {
                    "experiment_set": row["experiment_set"],
                    "chip": row["chip"],
                    "gas_code": row["gas_code"],
                    "gas": row["gas"],
                    "concentration": row["concentration"],
                    "rh_percent": row["rh_percent"],
                    "repeat_index": row["repeat_index"],
                    "filename": row["filename"],
                    "split": row["split"],
                    "file_index": file_index,
                    "device_index": np.arange(samples_per_file, dtype=np.int32),
                }
            )
        )
    return output, pd.concat(metadata_parts, ignore_index=True)


def faulty_device_mask(
    data: np.ndarray,
    on_time_index: int = 0,
    on_vg_index: int = 2,
    off_vg_index: int = -1,
    open_current_max: float = 2e-6,
    short_current_min: float = 1e-6,
    on_off_ratio_min: float = 40.0,
) -> np.ndarray:
    """Identify open, shorted, or low-ratio devices."""
    current_on = data[:, on_time_index, on_vg_index]
    current_off = data[:, on_time_index, off_vg_index]
    safe_off = np.maximum(np.abs(current_off), 1e-9)
    return (
        (current_on < open_current_max)
        | (current_off > short_current_min)
        | ((np.abs(current_on) / safe_off) < on_off_ratio_min)
    )


def estimate_vth_indices(
    data: np.ndarray,
    reference_time_index: int = 14,
    search_margin: int = 10,
    minimum_index: int = 63,
    maximum_index: int = 87,
) -> np.ndarray:
    """Estimate a clipped threshold index independently for every device."""
    gradient = np.gradient(data[:, reference_time_index, :], axis=1)
    indices = np.argmin(gradient[:, search_margin:-search_margin], axis=1)
    indices = indices + search_margin
    return np.clip(indices, minimum_index, maximum_index).astype(np.int32)


def fit_preprocess_state(
    training_data: np.ndarray,
    cut_window: Mapping[str, int],
) -> dict:
    """Fit and record preprocessing choices using the training folder only."""
    vth = estimate_vth_indices(training_data)
    raw_vg_length = int(training_data.shape[2])
    minimum_index, maximum_index = 63, 87
    aligned_length = raw_vg_length - (maximum_index - minimum_index)
    aligned_vg = np.arange(aligned_length, dtype=np.float32) * 0.03
    aligned_vg -= 0.5 * (aligned_length - 1) * 0.03
    return {
        "fit_split": "training",
        "heldout_test_used_for_fit": False,
        "reference_time_index": 14,
        "search_margin": 10,
        "minimum_index": minimum_index,
        "maximum_index": maximum_index,
        "raw_time_length": int(training_data.shape[1]),
        "raw_vg_length": raw_vg_length,
        "aligned_vg": aligned_vg,
        "fit_device_count": int(training_data.shape[0]),
        "fit_vth_min": int(vth.min()),
        "fit_vth_max": int(vth.max()),
        "fit_vth_median": float(np.median(vth)),
        "normalization_time_index": 14,
        "background_reference_time_index": 14,
        "background_method": "linear_fit",
        "cut_window": {key: int(value) for key, value in cut_window.items()},
    }


def _linear_background(response: np.ndarray, reference_time_index: int) -> tuple[np.ndarray, np.ndarray]:
    fit_time = np.arange(reference_time_index + 1, dtype=np.float32)
    centered_time = fit_time - fit_time.mean()
    fit_values = response[:, : reference_time_index + 1, :]
    centered_values = fit_values - fit_values.mean(axis=1, keepdims=True)
    denominator = float(np.sum(centered_time * centered_time))
    slope = np.sum(centered_values * centered_time[None, :, None], axis=1) / denominator
    intercept = fit_values.mean(axis=1) - slope * fit_time.mean()
    return slope, intercept


def apply_preprocess_state(
    data: np.ndarray,
    metadata: pd.DataFrame,
    state: Mapping,
    chunk_size: int = 512,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the established preprocessing recipe and return the configured view."""
    if int(data.shape[1]) != int(state["raw_time_length"]):
        raise ValueError("Time dimension does not match the fitted preprocessing state")
    if int(data.shape[2]) != int(state["raw_vg_length"]):
        raise ValueError("Vg dimension does not match the fitted preprocessing state")

    bad = faulty_device_mask(data)
    vth = estimate_vth_indices(
        data,
        reference_time_index=int(state["reference_time_index"]),
        search_margin=int(state["search_margin"]),
        minimum_index=int(state["minimum_index"]),
        maximum_index=int(state["maximum_index"]),
    )

    cut = state["cut_window"]
    t_start, requested_t_end = int(cut["t_start"]), int(cut["t_end"])
    v_start, v_end = int(cut["v_start"]), int(cut["v_end"])
    # NumPy slicing truncates at the available time length when it is shorter
    # than the configured upper bound (chemical=42, solvent=29, bacteria=32).
    t_end = min(requested_t_end, int(data.shape[1]))
    aligned_vg_length = int(data.shape[2]) - (
        int(state["maximum_index"]) - int(state["minimum_index"])
    )
    if not 0 <= t_start < t_end:
        raise ValueError(
            f"Invalid time cut [{t_start}:{requested_t_end}] for length {data.shape[1]}"
        )
    if not 0 <= v_start < v_end <= aligned_vg_length:
        raise ValueError(
            f"Invalid Vg cut [{v_start}:{v_end}] for aligned length {aligned_vg_length}"
        )
    norm_time = int(state["normalization_time_index"])
    background_time = int(state["background_reference_time_index"])
    output = np.empty((len(data), t_end - t_start, v_end - v_start), dtype=np.float32)
    relative_vg = np.arange(v_start, v_end, dtype=np.int64)

    for chunk_start in range(0, len(data), int(chunk_size)):
        chunk_end = min(chunk_start + int(chunk_size), len(data))
        chunk = data[chunk_start:chunk_end]
        starts = vth[chunk_start:chunk_end] - int(state["minimum_index"])
        gather_indices = starts[:, None] + relative_vg[None, :]
        selected = np.take_along_axis(chunk, gather_indices[:, None, :], axis=2)
        normalizer = selected[:, [norm_time], :]
        normalizer = np.where(np.abs(normalizer) < 1e-9, 1e-9, normalizer)
        response = np.nan_to_num(
            selected / normalizer - 1.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        slope, intercept = _linear_background(response, background_time)
        active_time = np.arange(t_start, t_end, dtype=np.float32)
        baseline = slope[:, None, :] * active_time[None, :, None] + intercept[:, None, :]
        corrected = response[:, t_start:t_end, :] - baseline
        output[chunk_start:chunk_end] = corrected.astype(np.float32, copy=False)

    kept_metadata = metadata.loc[~bad].reset_index(drop=True).copy()
    kept_metadata["vth_index"] = vth[~bad]
    kept_metadata["faulty_device"] = False
    kept_metadata["sample_index"] = np.arange(len(kept_metadata), dtype=np.int64)
    kept_metadata.attrs["removed_faulty_devices"] = int(bad.sum())
    return output[~bad], kept_metadata


def _encode_gases(
    frames: Sequence[pd.DataFrame],
) -> tuple[dict[str, int], list[str]]:
    gas_codes = sorted(
        {str(code) for frame in frames for code in frame["gas_code"].unique()}
    )
    labels_by_code = {}
    for code in gas_codes:
        labels = {
            str(label)
            for frame in frames
            for label in frame.loc[frame["gas_code"] == code, "gas"].unique()
        }
        if len(labels) != 1:
            raise ValueError(f"Gas code {code!r} has inconsistent labels: {sorted(labels)}")
        labels_by_code[code] = labels.pop()

    gas_order = [labels_by_code[code] for code in gas_codes]
    if len(set(gas_order)) != len(gas_order):
        raise ValueError("Multiple filename gas codes map to the same class label")
    mapping = {gas: index for index, gas in enumerate(gas_order)}
    return mapping, gas_order


def split_training_indices(
    labels: np.ndarray,
    val_size: float = 0.5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels), dtype=np.int64)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=float(val_size),
        random_state=int(random_state),
    )
    return train_indices, val_indices


class SensorDataset(Dataset):
    """Torch dataset backed by preprocessed local arrays."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame,
        return_metadata: bool = False,
    ):
        if len(data) != len(labels) or len(data) != len(metadata):
            raise ValueError("data, labels, and metadata must have equal lengths")
        self.data = data
        self.labels = np.asarray(labels, dtype=np.int64)
        self.metadata = metadata.reset_index(drop=True)
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        x = torch.as_tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        if not self.return_metadata:
            return x, y
        return x, y, self.metadata.iloc[int(index)].to_dict()


def prepare_data_bundle(
    *,
    train_rh_values: Iterable[float | str] | float | str | None = (0,),
    heldout_test_rh_values: Iterable[float | str] | float | str | None = (0,),
    average_mode: int = 0,
    cut_window: Mapping[str, int] | None = None,
    val_size: float = 0.5,
    random_state: int = 0,
    data_root: str | Path = DATA_ROOT,
    show_progress: bool = True,
) -> dict:
    """Load both physical splits and build training, validation, and test views."""
    if cut_window is None:
        cut_window = {"t_start": 14, "t_end": 56, "v_start": 11, "v_end": 107}

    training_files = discover_files("training", train_rh_values, data_root)
    heldout_files = discover_files("heldout_test", heldout_test_rh_values, data_root)
    training_raw, training_metadata = load_averaged_split(
        training_files,
        average_mode=average_mode,
        show_progress=show_progress,
    )
    state = fit_preprocess_state(training_raw, cut_window)
    training_x, training_metadata = apply_preprocess_state(
        training_raw,
        training_metadata,
        state,
    )
    del training_raw

    heldout_raw, heldout_metadata = load_averaged_split(
        heldout_files,
        average_mode=average_mode,
        show_progress=show_progress,
    )
    heldout_x, heldout_metadata = apply_preprocess_state(
        heldout_raw,
        heldout_metadata,
        state,
    )
    del heldout_raw

    gas_label_map, gas_order = _encode_gases((training_metadata, heldout_metadata))
    training_y = training_metadata["gas"].map(gas_label_map).to_numpy(dtype=np.int64)
    heldout_y = heldout_metadata["gas"].map(gas_label_map).to_numpy(dtype=np.int64)
    train_indices, val_indices = split_training_indices(
        training_y,
        val_size=val_size,
        random_state=random_state,
    )
    training_metadata["model_split"] = "train"
    training_metadata.loc[val_indices, "model_split"] = "validation"
    heldout_metadata["model_split"] = "heldout_test"

    training_dataset = SensorDataset(training_x, training_y, training_metadata)
    heldout_dataset = SensorDataset(heldout_x, heldout_y, heldout_metadata)
    aligned_vg = np.asarray(state["aligned_vg"], dtype=np.float32)
    time_axis_full = np.arange(int(state["raw_time_length"]), dtype=np.float32) * 0.5 - 8.0
    cut = state["cut_window"]

    return {
        "training": {
            "x": training_x,
            "y": training_y,
            "metadata": training_metadata,
            "files": training_files.drop(columns=["path"]),
        },
        "heldout_test": {
            "x": heldout_x,
            "y": heldout_y,
            "metadata": heldout_metadata,
            "files": heldout_files.drop(columns=["path"]),
        },
        "dataset_train": Subset(training_dataset, train_indices.tolist()),
        "dataset_val": Subset(training_dataset, val_indices.tolist()),
        "dataset_heldout_test": heldout_dataset,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "gas_label_map": gas_label_map,
        "gas_order": gas_order,
        "num_classes": len(gas_order),
        "preprocess_state": state,
        "time_axis": time_axis_full[int(cut["t_start"]) : int(cut["t_end"])],
        "vg_axis": aligned_vg[int(cut["v_start"]) : int(cut["v_end"])],
    }


def sample_summary(bundle: Mapping) -> pd.DataFrame:
    """Count usable samples by folder, model partition, humidity, and gas."""
    frames = [
        bundle["training"]["metadata"],
        bundle["heldout_test"]["metadata"],
    ]
    metadata = pd.concat(frames, ignore_index=True)
    return (
        metadata.groupby(["split", "model_split", "rh_percent", "gas"], sort=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )


__all__ = [
    "AMBIENT_DATASETS",
    "CHEMICAL_FORMULA_BY_CODE",
    "DATA_ROOT",
    "DATASET_ROOT",
    "PROJECT_ROOT",
    "SPLITS",
    "UNQUANTIFIED_DATASETS",
    "SensorDataset",
    "apply_preprocess_state",
    "discover_files",
    "estimate_vth_indices",
    "faulty_device_mask",
    "fit_preprocess_state",
    "format_rh_label",
    "load_averaged_split",
    "normalize_rh_value",
    "parse_sensor_filename",
    "prepare_data_bundle",
    "resolve_data_root",
    "sample_summary",
    "split_training_indices",
    "summarize_files",
]
