"""Read fixed choices in existing artifacts without exposing runtime selectors.

The published architecture always uses these choices. Old snapshots remain
readable, and existing single-Vg fingerprints keep their original meaning.
"""

from copy import deepcopy


_FIXED_MODEL_SETTINGS = {
    "pool_mode": "avg",
    "classifier_head": "cosine",
    "learnable_cosine_scale": False,
    "use_late_mix": True,
}
_FIXED_TRAINING_SETTINGS = {
    "selection_metric": "val_acc",
    "early_stop_metric": "val_acc",
}


def model_config_from_snapshot(settings):
    """Remove legacy selectors only when they describe the fixed architecture."""
    settings = dict(settings)
    for key, expected in _FIXED_MODEL_SETTINGS.items():
        if key not in settings:
            continue
        value = settings.pop(key)
        if key == "pool_mode" and value == "avg_only":
            value = "avg"
        if value != expected:
            raise ValueError(f"Saved {key}={value!r} is incompatible; expected {expected!r}")
    # This value never affects an average-pooling checkpoint.
    settings.pop("lse_tau", None)
    return settings


def training_config_from_snapshot(settings):
    """Accept old accuracy-only settings without restoring metric selection."""
    settings = dict(settings)
    for key, expected in _FIXED_TRAINING_SETTINGS.items():
        if key in settings and str(settings.pop(key)).lower() != expected:
            raise ValueError(f"{key} must be {expected!r}")
    return settings


def fingerprint_config(config):
    """Give compatible saved and current single-Vg configurations the same fingerprint."""
    config = deepcopy(config)
    config["MODEL"] = model_config_from_snapshot(config["MODEL"])
    config["MODEL"].update(_FIXED_MODEL_SETTINGS)
    config["TRAINING"] = training_config_from_snapshot(config["TRAINING"])
    config["TRAINING"].update(_FIXED_TRAINING_SETTINGS)
    return config
