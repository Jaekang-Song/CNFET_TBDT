"""Microbial-headspace classification at ambient (uncontrolled) humidity.

Model and training settings accompany the included pretrained checkpoint.
New training uses a 90% validation-accuracy early-stopping baseline.
"""


CONFIG = {
    "CONFIG_NAME": "Config_TBDT_Bacteria_pretrained",
    "DATASET": {
        "name": "Config_TBDT_bacteria",
        "data_dir": "TBDT_Dataset/2_bacteria",
    },
    "PREPROCESS": {
        "average_mode": 0,
        "cut_window": {
            "t_start": 14,
            "t_end": 56,
            "v_start": 11,
            "v_end": 107,
        },
    },
    "SELECTION": {
        "train_rh_values": ["A"],
        "heldout_test_rh_values": ["A"],
        "train_val_split": {
            "val_size": 0.5,
            "random_state": 0,
        },
    },
    "MODEL": {
        "model_name": "CNN1D_TBDT",
        "dim_v": 96,
        "num_layers": 1,
        "vg_groups": 8,
        "gn_groups": 8,
        "kernel_size": 5,
        "cosine_scale": 16,
        "vg_group_dropout_p": 0,
        "hidden_size": 8,
        "dropout_p": 0.2,
        "dropout2d_p": 0.0,
    },
    "TRAINING": {
        "batch_size": 128,
        "num_epochs": 120,
        "patience": 100,
        "num_workers": 0,
        "early_stop_min_delta": 0.01,
        "early_stop_baseline": 90,
        "learning_rate": 0.002,
        "weight_decay": 0.002,
        "vg_input_dropout_p": 0.3,
        "vg_input_dropout_num_groups": 12,
        "random_seed": 0,
        "use_class_balanced_sampler": True,
    },
    "OUTPUT": {
        "models_dir": "results/models/tbdt_bacteria_pretrained",
        "runs_dirname": "runs",
    },
}

