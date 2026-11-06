"""Training and artifact utilities for the paper's gas classifier."""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import ML_models
from .Checkpoint_compat import training_config_from_snapshot


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def _subset_labels(dataset) -> np.ndarray:
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "labels"):
        labels = np.asarray(dataset.dataset.labels, dtype=np.int64)
        return labels[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))], dtype=np.int64)


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _balanced_sampler(dataset, seed: int) -> WeightedRandomSampler:
    labels = _subset_labels(dataset)
    counts = np.bincount(labels)
    if np.any(counts == 0):
        counts = np.maximum(counts, 1)
    sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=_seeded_generator(seed),
    )


def build_loaders(
    dataset_train,
    dataset_val,
    config: Mapping,
    *,
    seed: int,
) -> dict:
    use_balanced = bool(config.get("use_class_balanced_sampler", True))
    num_workers = int(config.get("num_workers", 0))
    common = {
        "batch_size": int(config.get("batch_size", 32)),
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
        "worker_init_fn": partial(_seed_worker, base_seed=seed) if num_workers > 0 else None,
    }
    return {
        "train": DataLoader(
            dataset_train,
            sampler=_balanced_sampler(dataset_train, seed) if use_balanced else None,
            shuffle=not use_balanced,
            generator=_seeded_generator(seed),
            **common,
        ),
        "validation": DataLoader(
            dataset_val,
            shuffle=False,
            generator=_seeded_generator(seed),
            **common,
        ),
    }


def _drop_vg_groups(x: torch.Tensor, probability: float, groups: int) -> torch.Tensor:
    probability = float(probability)
    groups = int(groups)
    if probability <= 0.0:
        return x
    if probability > 1.0:
        raise ValueError("Input dropout probability must be in [0, 1]")
    if x.ndim != 3:
        raise ValueError("Input dropout expects [batch, time, Vg]")
    length = x.shape[2]
    if groups <= 0:
        raise ValueError("Input dropout group count must be positive")
    groups = min(groups, length)

    keep_groups = torch.rand(x.shape[0], groups, device=x.device) >= probability
    all_groups_dropped = ~keep_groups.any(dim=1)
    if torch.any(all_groups_dropped):
        rows = torch.nonzero(all_groups_dropped, as_tuple=False).reshape(-1)
        fallback_groups = torch.randint(
            low=0,
            high=groups,
            size=(rows.numel(),),
            device=x.device,
        )
        keep_groups[rows, fallback_groups] = True

    keep = x.new_zeros((x.shape[0], length))
    for group_index in range(groups):
        start = (group_index * length) // groups
        stop = ((group_index + 1) * length) // groups
        keep[:, start:stop] = keep_groups[:, group_index].unsqueeze(1).to(x.dtype)
    return x * keep.reshape(x.shape[0], 1, length)


def apply_input_dropout(x: torch.Tensor, config: Mapping) -> torch.Tensor:
    return _drop_vg_groups(
        x,
        probability=float(config.get("vg_input_dropout_p", 0.0)),
        groups=int(config.get("vg_input_dropout_num_groups", 1)),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    true_parts, pred_parts = [], []
    loss_total = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = _logits(model(x))
        loss_total += float(criterion(logits, y).item())
        true_parts.append(y.cpu().numpy())
        pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(true_parts) if true_parts else np.empty(0, dtype=np.int64)
    y_pred = np.concatenate(pred_parts) if pred_parts else np.empty(0, dtype=np.int64)
    return {
        "loss": loss_total / max(1, len(y_true)),
        "accuracy": 100.0 * float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _metric_improved(
    candidate: float,
    best: float | None,
    *,
    min_delta: float = 0.0,
) -> bool:
    candidate = float(candidate)
    if np.isnan(candidate):
        return False
    if best is None or np.isnan(float(best)):
        return True
    return candidate > float(best) + min_delta


def _baseline_reached(value: float, baseline: float | None) -> bool:
    if baseline is None or np.isnan(float(value)):
        return False
    return float(value) >= float(baseline)


def fit_model(
    dataset_train,
    dataset_val,
    *,
    num_classes: int,
    model_config: Mapping,
    training_config: Mapping,
    device: str | torch.device | None = None,
) -> dict:
    """Train and restore the first checkpoint with maximum validation accuracy."""
    config = training_config_from_snapshot(training_config)
    supported = {
        "batch_size", "num_epochs", "patience", "num_workers",
        "early_stop_min_delta", "early_stop_baseline", "learning_rate",
        "weight_decay", "vg_input_dropout_p", "vg_input_dropout_num_groups",
        "random_seed", "use_class_balanced_sampler",
    }
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))
    if not np.isfinite(early_stop_min_delta) or early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be finite and nonnegative")
    seed = int(config.get("random_seed", 0))
    seed_everything(seed)
    device = get_default_device() if device is None else torch.device(device)
    model_kwargs = dict(model_config)
    model_name = model_kwargs.pop("model_name")
    model = ML_models.build_model(
        model_name=model_name,
        num_classes=int(num_classes),
        **model_kwargs,
    ).to(device)
    loaders = build_loaders(
        dataset_train,
        dataset_val,
        config,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-3)),
        weight_decay=float(config.get("weight_decay", 3e-3)),
    )
    criterion = nn.CrossEntropyLoss()

    early_stop_baseline = config.get("early_stop_baseline")
    if early_stop_baseline is not None:
        early_stop_baseline = float(early_stop_baseline)
    patience = int(config.get("patience", 40))

    best_epoch = 0
    best_selection_value = float("-inf")
    best_early_stop_value = None
    epochs_without_improvement = 0
    early_stop_active = early_stop_baseline is None
    best_model_state = None
    history = []

    for epoch in range(1, int(config.get("num_epochs", 120)) + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = apply_input_dropout(x, config)
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model(x))
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(y)
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_count += len(y)

        validation = evaluate(model, loaders["validation"], device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "train_accuracy": 100.0 * train_correct / max(1, train_count),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)

        selection_value = validation["accuracy"]
        if _metric_improved(
            selection_value,
            best_selection_value,
        ):
            best_selection_value = float(selection_value)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch}/{int(config.get('num_epochs', 120))}] | "
            f"train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.2f}% | "
            f"validation loss={row['validation_loss']:.4f}, acc={row['validation_accuracy']:.2f}% | "
            f"best val_acc={float(best_selection_value):.2f} @ {best_epoch}"
        )

        early_stop_value = validation["accuracy"]
        baseline_just_activated = False
        if not early_stop_active and _baseline_reached(
            early_stop_value,
            early_stop_baseline,
        ):
            early_stop_active = True
            baseline_just_activated = True
            epochs_without_improvement = 0

        if _metric_improved(
            early_stop_value,
            best_early_stop_value,
            min_delta=early_stop_min_delta,
        ):
            best_early_stop_value = float(early_stop_value)
            if early_stop_active:
                epochs_without_improvement = 0
        elif early_stop_active and not baseline_just_activated:
            epochs_without_improvement += 1

        if early_stop_active and epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}: val_acc did not improve "
                f"for {patience} epochs. "
                f"Selected checkpoint epoch: {best_epoch}."
            )
            break

    if best_model_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_model_state)
    restored_validation = evaluate(model, loaders["validation"], device)
    return {
        "model": model,
        "selected_checkpoint_reason": "best_val_acc",
        "selection_metric": "val_acc",
        "selection_metric_value": float(best_selection_value),
        "early_stop_metric": "val_acc",
        "early_stop_best_metric_value": best_early_stop_value,
        "early_stop_baseline": early_stop_baseline,
        "early_stop_min_delta": early_stop_min_delta,
        "model_state_dict": copy.deepcopy(best_model_state),
        "best_epoch": int(best_epoch),
        "history": pd.DataFrame(history),
        "validation": restored_validation,
        "device": str(device),
    }


def _json_ready(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_run_artifacts(
    result: Mapping,
    *,
    config: Mapping,
    gas_label_map: Mapping[str, int],
    output_root: str | Path,
    run_name: str | None = None,
) -> Path:
    """Save the checkpoint and all inputs needed by Analysis and ML_test."""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S_gas_cnn")
    run_dir = Path(output_root) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = {
        "model_state_dict": result["model_state_dict"],
        "selected_checkpoint_reason": result["selected_checkpoint_reason"],
        "selection_metric": result["selection_metric"],
        "selection_metric_value": result["selection_metric_value"],
        "early_stop_metric": result["early_stop_metric"],
        "early_stop_best_metric_value": result["early_stop_best_metric_value"],
        "early_stop_baseline": result["early_stop_baseline"],
        "early_stop_min_delta": result["early_stop_min_delta"],
        "best_epoch": int(result["best_epoch"]),
        "gas_label_map": dict(gas_label_map),
        "model_config": dict(config["MODEL"]),
        "preprocess_config": dict(config["PREPROCESS"]),
        "selection_config": dict(config["SELECTION"]),
    }
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")
    (run_dir / "gas_label_map.json").write_text(
        json.dumps(dict(gas_label_map), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    snapshot = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": str(run_name),
        "config": _json_ready(config),
    }
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["history"].to_csv(run_dir / "training_history.csv", index=False)
    np.save(run_dir / "validation_true_labels.npy", result["validation"]["y_true"])
    np.save(run_dir / "validation_predictions.npy", result["validation"]["y_pred"])

    metrics = {
        "best_epoch": int(result["best_epoch"]),
        "selected_checkpoint_reason": result["selected_checkpoint_reason"],
        "selection_metric": result["selection_metric"],
        "selection_metric_value": float(result["selection_metric_value"]),
        "validation_accuracy": float(result["validation"]["accuracy"]),
        "validation_loss": float(result["validation"]["loss"]),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


__all__ = [
    "apply_input_dropout",
    "build_loaders",
    "evaluate",
    "fit_model",
    "get_default_device",
    "save_run_artifacts",
    "seed_everything",
]

