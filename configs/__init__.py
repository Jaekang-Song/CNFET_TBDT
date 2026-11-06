"""Named configuration loader for the paper notebooks."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module


AVAILABLE_CONFIGS = (
    "Config_TBDT_Chem_dry_pretrained",
    "Config_TBDT_Chem_humid_pretrained",
    "Config_TBDT_Solvent_pretrained",
    "Config_TBDT_Bacteria_pretrained",
)

# Map saved configuration aliases to the four task modules below.
_LEGACY_CONFIG_NAMES = {
    "Config_TBDT_chemical_gas_all": "Config_TBDT_Chem_humid_pretrained",
    "Config_TBDT_chemical_gas_dry": "Config_TBDT_Chem_dry_pretrained",
    "Config_TBDT_solvent": "Config_TBDT_Solvent_pretrained",
    "Config_TBDT_bacteria": "Config_TBDT_Bacteria_pretrained",
}


def load_config(config_name: str = "Config_TBDT_Chem_dry_pretrained") -> dict:
    """Load one named config and return an independent mutable copy."""
    config_name = _LEGACY_CONFIG_NAMES.get(config_name, config_name)
    if config_name not in AVAILABLE_CONFIGS:
        choices = ", ".join(AVAILABLE_CONFIGS)
        raise ValueError(f"Unknown config {config_name!r}. Choose one of: {choices}")
    module = import_module(f".{config_name}", package=__name__)
    return deepcopy(module.CONFIG)


__all__ = ["AVAILABLE_CONFIGS", "load_config"]
