"""Load paper runs and create validation/test confusion matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import ML_models
from configs import load_config
from .Checkpoint_compat import model_config_from_snapshot
from .Data_pipeline import normalize_rh_value, format_rh_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = load_config()["OUTPUT"]
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / _DEFAULT_OUTPUT["models_dir"] / _DEFAULT_OUTPUT["runs_dirname"]
)

FIGURE2_CHEMICAL_ORDER = (
    "H2",
    "NO2",
    "NH3",
    "C3H6O",
    "C3H7OH",
    "CH3OH",
    "CO2",
    "C6H6",
    "H2S",
    "C3H8",
)
SOLVENT_GAS_ORDER = ("M", "E", "P", "T", "A")
_FIGURE2_REFERENCE_CLASS_COUNT = len(SOLVENT_GAS_ORDER)
_FIGURE2_REFERENCE_SIZE = (6.0, 6.0)
_FIGURE2_LEGACY_CODE_ORDER = ("H", "ND", "NH", "A", "P", "M", "CD", "B", "HS", "Pr")
_LEGACY_CODE_TO_FORMULA = dict(zip(_FIGURE2_LEGACY_CODE_ORDER, FIGURE2_CHEMICAL_ORDER))
_CHEMICAL_MATHTEXT_LABELS = {
    "H2": r"$\mathrm{H_2}$",
    "NO2": r"$\mathrm{NO_2}$",
    "NH3": r"$\mathrm{NH_3}$",
    "C3H6O": r"$\mathrm{C_3H_6O}$",
    "C3H7OH": r"$\mathrm{C_3H_7OH}$",
    "CH3OH": r"$\mathrm{CH_3OH}$",
    "CO2": r"$\mathrm{CO_2}$",
    "C6H6": r"$\mathrm{C_6H_6}$",
    "H2S": r"$\mathrm{H_2S}$",
    "C3H8": r"$\mathrm{C_3H_8}$",
}


def list_saved_runs(runs_root: str | Path = DEFAULT_RUNS_ROOT) -> list[str]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    runs = [path for path in root.iterdir() if (path / "best_checkpoint.pt").is_file()]
    return [path.name for path in sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)]


def resolve_run_dir(
    run_name: str | None = "pretrained",
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    root = Path(runs_root)
    if run_name is None or str(run_name).lower() == "latest":
        available = list_saved_runs(root)
        if not available:
            raise FileNotFoundError(f"No saved runs found under {root}")
        run_name = available[0]
    run_dir = root / str(run_name)
    if not (run_dir / "best_checkpoint.pt").is_file():
        raise FileNotFoundError(f"Run checkpoint not found: {run_dir}")
    return run_dir


def load_run_metadata(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    snapshot_path = run_dir / "config_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    gas_label_map = json.loads((run_dir / "gas_label_map.json").read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir,
        "snapshot": snapshot,
        "config": snapshot["config"],
        "gas_label_map": {str(key): int(value) for key, value in gas_label_map.items()},
    }


def load_model(
    run_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict]:
    run_dir = Path(run_dir)
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = load_run_metadata(run_dir)
    model_config = model_config_from_snapshot(
        checkpoint.get("model_config", metadata["config"]["MODEL"])
    )
    model_name = model_config.pop("model_name")
    model = ML_models.build_model(
        model_name=model_name,
        num_classes=len(metadata["gas_label_map"]),
        **model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    metadata["checkpoint"] = checkpoint
    return model, metadata


def _logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset,
    *,
    device: str | torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    true_parts, pred_parts = [], []
    for x, y in loader:
        logits = _logits(model(x.to(device)))
        true_parts.append(y.numpy())
        pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def load_saved_predictions(run_dir: str | Path) -> dict:
    """Load validation predictions and any separately saved heldout predictions."""
    run_dir = Path(run_dir)
    result = {}
    for split_name in ("validation", "heldout_test"):
        true_path = run_dir / f"{split_name}_true_labels.npy"
        pred_path = run_dir / f"{split_name}_predictions.npy"
        if split_name == "heldout_test" and not true_path.is_file() and not pred_path.is_file():
            continue
        if not true_path.is_file() or not pred_path.is_file():
            raise FileNotFoundError(f"Saved predictions are incomplete for {split_name}: {run_dir}")
        result[split_name] = {
            "y_true": np.load(true_path),
            "y_pred": np.load(pred_path),
        }
    return result


def confusion_matrix_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gas_label_map: Mapping[str, int],
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    gas_order = [name for name, _ in sorted(gas_label_map.items(), key=lambda item: item[1])]
    labels = np.arange(len(gas_order), dtype=np.int64)
    matrix = confusion_matrix(
        np.asarray(y_true),
        np.asarray(y_pred),
        labels=labels,
        normalize="true" if normalize else None,
    )
    if normalize:
        matrix = matrix * 100.0
    return pd.DataFrame(matrix, index=gas_order, columns=gas_order)


def _configure_figure2_style() -> None:
    """Set fonts and line widths for the confusion-matrix figure."""
    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "savefig.dpi": 1200,
            "figure.dpi": 1200,
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.default": "regular",
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#303030",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.bottom": False,
            "xtick.top": False,
            "ytick.left": True,
            "ytick.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "legend.frameon": False,
        }
    )


def _figure2_display_order(gas_label_map: Mapping[str, int]) -> list[str]:
    label_order = [
        name for name, _ in sorted(gas_label_map.items(), key=lambda item: item[1])
    ]
    available = set(label_order)
    if available == set(FIGURE2_CHEMICAL_ORDER):
        return list(FIGURE2_CHEMICAL_ORDER)
    if available == set(_FIGURE2_LEGACY_CODE_ORDER):
        return list(_FIGURE2_LEGACY_CODE_ORDER)
    if available == set(SOLVENT_GAS_ORDER):
        return list(SOLVENT_GAS_ORDER)
    return label_order


def _chemical_tick_label(label: str) -> str:
    formula = _LEGACY_CODE_TO_FORMULA.get(str(label), str(label))
    return _CHEMICAL_MATHTEXT_LABELS.get(formula, str(label))


def _figure2_tick_labels(display_order: list[str]) -> list[str]:
    available = set(display_order)
    chemical_labels = set(FIGURE2_CHEMICAL_ORDER)
    legacy_chemical_labels = set(_FIGURE2_LEGACY_CODE_ORDER)
    if available in (chemical_labels, legacy_chemical_labels):
        return [_chemical_tick_label(label) for label in display_order]
    return [str(label) for label in display_order]


def _figure2_size(
    class_count: int,
    figure_size: tuple[float, float] | None,
) -> tuple[float, float]:
    """Keep approximately the same heatmap-cell size across datasets."""
    if figure_size is not None:
        return tuple(float(value) for value in figure_size)
    scale = max(1.0, float(class_count) / _FIGURE2_REFERENCE_CLASS_COUNT)
    return tuple(value * scale for value in _FIGURE2_REFERENCE_SIZE)


def plot_figure2_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gas_label_map: Mapping[str, int],
    *,
    figure_size: tuple[float, float] | None = None,
):
    """Plot a row-normalized matrix with size scaled to its class count."""
    _configure_figure2_style()
    percent_frame = confusion_matrix_frame(
        y_true,
        y_pred,
        gas_label_map,
        normalize=True,
    )
    display_order = _figure2_display_order(gas_label_map)
    percent_frame = percent_frame.loc[display_order, display_order]
    matrix = percent_frame.to_numpy(dtype=np.float32) / 100.0

    resolved_figure_size = _figure2_size(len(display_order), figure_size)
    fig, ax = plt.subplots(figsize=resolved_figure_size, constrained_layout=True)
    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        alpha=1.0,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Accuracy (%)", fontsize=24)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    colorbar.ax.tick_params(labelsize=20)

    display_labels = _figure2_tick_labels(display_order)
    positions = np.arange(len(display_order))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        display_labels,
        rotation=45,
        rotation_mode="anchor",
        ha="right",
        va="baseline",
        fontsize=18,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(display_labels, fontsize=18)
    ax.set_xlabel("Predicted label", fontsize=24)
    ax.set_ylabel("True label", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.0, length=4.0)
    ax.tick_params(axis="x", which="major", bottom=True, top=False, pad=9, width=0.5)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{100.0 * value:.1f}",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "#1A1A1A",
                fontsize=12,
            )

    ax.set_xlim(-0.5, len(display_order) - 0.5)
    ax.set_ylim(len(display_order) - 0.5, -0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax, percent_frame


def plot_confusion_matrices(
    predictions: Mapping,
    gas_label_map: Mapping[str, int],
    *,
    normalize: bool = True,
    figsize: tuple[float, float] = (12.0, 5.0),
    cmap: str = "Blues",
):
    """Plot validation and heldout-test matrices side by side."""
    panels = (("validation", "Validation"), ("heldout_test", "Heldout Test"))
    frames = {
        key: confusion_matrix_frame(
            predictions[key]["y_true"],
            predictions[key]["y_pred"],
            gas_label_map,
            normalize=normalize,
        )
        for key, _ in panels
    }
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        sns.heatmap(
            frames[key],
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=100 if normalize else None,
            annot=True,
            fmt=".1f" if normalize else "d",
            square=True,
            cbar=ax is axes[-1],
            cbar_kws={"label": "Row-normalized (%)" if normalize else "Count"},
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    return fig, axes, frames


def evaluate_heldout(
    model, bundle, *, device, batch_size=128,
):
    """Print held-out accuracies and plot all physical held-out observations.

    Multiple humidity conditions produce one line per RH and one pooled line.
    A single condition produces one pooled line. Only physical held-out samples
    enter these results and the confusion matrix, including every RH condition.
    """
    y_true, y_pred = predict_dataset(model, bundle["dataset_heldout_test"],
                                    device=device, batch_size=batch_size)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metadata = bundle["heldout_test"]["metadata"].reset_index(drop=True)
    if not (len(y_true) == len(y_pred) == len(metadata)) or not len(metadata):
        raise ValueError("Prediction and held-out metadata lengths must match and be nonempty")
    if not metadata["split"].eq("heldout_test").all():
        raise ValueError("Evaluation must contain only heldout_test samples")
    correct = y_true == y_pred
    rh_values = metadata["rh_percent"].map(normalize_rh_value)
    conditions = sorted(rh_values.unique(), key=lambda value: (isinstance(value, str), value))
    rows = []
    def report(mask, label):
        total, right = int(mask.sum()), int(correct[mask].sum())
        accuracy = 100.0 * right / total
        rows.append({"Evaluation": label, "Samples": total, "Correct": right, "Accuracy (%)": accuracy})
        print(f"{label}: {accuracy:.4f}% ({right}/{total})")
    if len(conditions) > 1:
        for rh in conditions:
            report(rh_values.eq(rh).to_numpy(), "Held-out chips, " + format_rh_label(rh))
    report(np.ones(len(y_true), dtype=bool), "All held-out samples")
    mask = np.ones(len(y_true), dtype=bool)
    scope = "All held-out samples"
    fig, ax, frame = plot_figure2_confusion_matrix(y_true[mask], y_pred[mask], bundle["gas_label_map"])
    ax.set_title(scope)
    return {"y_true": y_true, "y_pred": y_pred, "figure": fig, "axes": ax,
            "confusion_frame": frame, "confusion_mask": mask, "confusion_scope": scope,
            "accuracy_rows": rows}


__all__ = [
    "evaluate_heldout",
    "DEFAULT_RUNS_ROOT",
    "confusion_matrix_frame",
    "list_saved_runs",
    "load_model",
    "load_run_metadata",
    "load_saved_predictions",
    "plot_figure2_confusion_matrix",
    "plot_confusion_matrices",
    "predict_dataset",
    "resolve_run_dir",
]
