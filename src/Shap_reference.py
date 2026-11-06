"""Validated sample identities and saved full-input SHAP for the humid model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def array_fingerprint(values):
    values = np.asarray(values)
    digest = hashlib.sha256(str((values.shape, str(values.dtype))).encode())
    for start in range(0, len(values), 128):
        digest.update(np.ascontiguousarray(values[start:start + 128]).view(np.uint8))
    return digest.hexdigest()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def read_reference(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("method") != "pointwise_gradient_shap":
        raise ValueError("Expected the saved pointwise GradientSHAP reference.")
    required = {"pointwise.npz", "axes.npz", "background_metadata.csv",
                "eval_metadata.csv", "eval_labels.npy"}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete SHAP reference manifest.")
    for name in required:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("SHAP reference checksum mismatch: " + name)
    return manifest


def select_reference_inputs(bundle, directory):
    """Select the archived 8,192 background and 7,572 evaluation observations."""
    directory = Path(directory)
    manifest = read_reference(directory)
    if bundle["gas_label_map"] != manifest["gas_label_map"]:
        raise ValueError("SHAP reference gas labels do not match the dataset.")
    selected = {}
    for label, split, membership, rhs in (
        ("background", "training", "train", manifest["background_rh"]),
        ("eval", "heldout_test", "heldout_test", manifest["evaluation_rh"]),
    ):
        saved = pd.read_csv(directory / (label + "_metadata.csv"))
        frame = bundle[split]["metadata"].reset_index(drop=True)
        keys = list(zip(frame["filename"], frame["device_index"]))
        if len(set(keys)) != len(keys):
            raise ValueError("SHAP sample identities must be unique within each physical split.")
        lookup = dict(zip(keys, range(len(keys))))
        wanted = list(zip(saved["filename"], saved["device_index"]))
        if len(set(wanted)) != len(wanted) or any(key not in lookup for key in wanted):
            raise ValueError("The dataset does not contain the exact saved SHAP samples.")
        indices = np.asarray([lookup[key] for key in wanted], dtype=np.int64)
        metadata = frame.iloc[indices].reset_index(drop=True)
        if (not metadata["split"].eq(split).all()
                or not metadata["model_split"].eq(membership).all()
                or set(metadata["rh_percent"]) != set(rhs)):
            raise ValueError("SHAP sample RH or train/validation/test membership differs.")
        for column in ("filename", "device_index", "gas_code", "rh_percent", "gas"):
            if metadata[column].tolist() != saved[column].tolist():
                raise ValueError("SHAP sample identity differs: " + column)
        x = bundle[split]["x"][indices]
        expected = manifest["background_input_sha256" if label == "background"
                            else "evaluation_input_sha256"]
        if array_fingerprint(x) != expected:
            raise ValueError("Preprocessed SHAP inputs differ from the saved reference.")
        if len(indices) != manifest["background_size" if label == "background" else "eval_size"]:
            raise ValueError("Unexpected SHAP cohort size.")
        selected[label + "_x"] = x
        selected[label + "_metadata"] = metadata
        if label == "eval":
            eligible = frame["rh_percent"].isin(rhs).to_numpy()
            if set(indices.tolist()) != set(np.flatnonzero(eligible).tolist()):
                raise ValueError("SHAP evaluation must include every eligible held-out sample.")
            selected["eval_y"] = bundle[split]["y"][indices]
            if not np.array_equal(selected["eval_y"], np.load(directory / "eval_labels.npy")):
                raise ValueError("SHAP reference true-class labels differ.")
    selected["manifest"] = manifest
    return selected


def load_reference_shap(model, bundle, directory, *, vg_bin_size=8, time_bin_size=2):
    """Load verified saved attributions and apply the same aggregation as recomputation."""
    from .Analysis_tools import shap_profiles_from_class_maps

    directory = Path(directory)
    selected = select_reference_inputs(bundle, directory)
    manifest = selected["manifest"]
    if model_fingerprint(model) != manifest["model_state_sha256"]:
        raise ValueError("Saved SHAP belongs to a different model. Set RECOMPUTE_SHAP = True.")
    with np.load(directory / "axes.npz", allow_pickle=False) as axes:
        if (not np.array_equal(axes["vg"], bundle["vg_axis"])
                or not np.array_equal(axes["time"], bundle["time_axis"])):
            raise ValueError("SHAP reference axes differ from the current preprocessing.")
    with np.load(directory / "pointwise.npz", allow_pickle=False) as values:
        maps = values["per_gas_maps"]
        if not np.allclose(maps.mean(axis=0), values["mean_abs_map"], rtol=1e-12, atol=1e-12):
            raise ValueError("Saved SHAP map is not the equal-gas average.")
    result = shap_profiles_from_class_maps(
        maps, selected["eval_y"], selected["eval_metadata"],
        bundle["vg_axis"], bundle["time_axis"],
        vg_bin_size=vg_bin_size, time_bin_size=time_bin_size,
    )
    result.update(
        true_class_values=None, background_metadata=selected["background_metadata"],
        eval_metadata=selected["eval_metadata"], eval_y=selected["eval_y"],
        background_size=len(selected["background_x"]), method="gradient",
        source="saved_reference", reference_manifest=manifest,
    )
    result["settings"].update(
        background_size=manifest["background_size"], background_rh=manifest["background_rh"],
        nsamples=manifest["nsamples"], batch_size=manifest["eval_batch_size"],
        model_state_sha256=manifest["model_state_sha256"],
        source="saved_reference",
    )
    return result
