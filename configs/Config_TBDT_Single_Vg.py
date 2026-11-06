"""Single-Vg chemical-gas experiment with the archived 24-candidate grid.

The 24-candidate grid is shared by saved results and new sweeps.
New training uses a 90% validation-accuracy early-stopping baseline.
Each cropped Vg channel is trained independently, using RH 0% for training and
validation, and RH 0%, 30%, and 50% for held-out reporting.
"""

from copy import deepcopy

from .Config_TBDT_Chem_humid_pretrained import CONFIG as _FULL_VG_CONFIG


# Indices refer to the 96 Vg channels after the configured preprocessing crop.
VG_INDICES = list(range(0, 96, 4))

# Insertion order fixes candidate indices 0-23 for reproducible resume behavior.
PARAMETER_GRID = {
    "hidden_size": [24, 48, 88],
    "learning_rate": [0.001, 0.002],
    "weight_decay": [0.001, 0.003],
    "dropout_p": [0.15, 0.35],
}

CONFIG = deepcopy(_FULL_VG_CONFIG)
CONFIG["CONFIG_NAME"] = "Config_TBDT_Single_Vg"
# Single-Vg training uses RH 0%, with RH 0/30/50 held out for evaluation.
CONFIG["SELECTION"]["train_rh_values"] = [0]
CONFIG.pop("ANALYSIS", None)
CONFIG["MODEL"].update({"dim_v": 1, "vg_groups": 1, "gn_groups": 8})
CONFIG["TRAINING"].update({
    "vg_input_dropout_p": 0.0,
    "vg_input_dropout_num_groups": 1,
    "early_stop_baseline": 90,
})
CONFIG["OUTPUT"]["models_dir"] = "results/models/tbdt_single_vg"
CONFIG["VG_INDICES"] = list(VG_INDICES)
CONFIG["PARAMETER_GRID"] = deepcopy(PARAMETER_GRID)
