"""Chemical-gas classification: train and evaluate at RH 0%.

Model and training settings accompany the included pretrained checkpoint.
New training uses a 90% validation-accuracy early-stopping baseline.
"""


CONFIG = {
    "CONFIG_NAME": "Config_TBDT_Chem_dry_pretrained",
    "DATASET": {
        "name": "Config_TBDT_chemical_gas",
        "data_dir": "TBDT_Dataset/0_chemical_gas",
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
        "train_rh_values": [0],
        "heldout_test_rh_values": [0],
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
        "hidden_size": 112,
        "dropout_p": 0.4,
        "dropout2d_p": 0.125,
    },
    "TRAINING": {
        "batch_size": 128,
        "num_epochs": 140,
        "patience": 100,
        "num_workers": 0,
        "early_stop_min_delta": 0.01,
        "early_stop_baseline": 90,
        "learning_rate": 0.002,
        "weight_decay": 0.003,
        "vg_input_dropout_p": 0.3,
        "vg_input_dropout_num_groups": 8,
        "random_seed": 0,
        "use_class_balanced_sampler": True,
    },
    "OUTPUT": {
        "models_dir": "results/models/tbdt_chem_dry_pretrained",
        "runs_dirname": "runs",
    },
}

