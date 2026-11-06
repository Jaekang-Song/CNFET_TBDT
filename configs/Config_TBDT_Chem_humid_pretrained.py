"""Chemical-gas classification: train at RH 0/50 and evaluate at RH 0/30/50.

Model and training settings accompany the included pretrained checkpoint.
New training uses a 90% validation-accuracy early-stopping baseline.
"""


CONFIG = {
    "CONFIG_NAME": "Config_TBDT_Chem_humid_pretrained",
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
        "train_rh_values": [0, 50],
        "heldout_test_rh_values": [0, 30, 50],
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
        "hidden_size": 88,
        "dropout_p": 0.35,
        "dropout2d_p": 0.075,
    },
    "TRAINING": {
        "batch_size": 128,
        "num_epochs": 200,
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
        "models_dir": "results/models/tbdt_chem_humid_pretrained",
        "runs_dirname": "runs",
    },
    "ANALYSIS": {
        "random_seed": 0,
        "metric_rh_values": [0, 50],
        "metric_quality": {
            "min_reference_current": 1e-9,
            "min_baseline_snr": 5.0,
            "max_extrapolated_drift_fraction": 1.0,
        },
        "mi_joint_time_indices": [0, 1, 2, 3, 5, 8, 12, 18, 25, 35],
        "mi_joint_vg_indices": [0, 19, 38, 57, 76, 95],
        "mi_neighbors": 5,
        "shap_background_size": 8192,
        "shap_eval_size": 8192,
        "shap_batch_size": 256,
        "shap_nsamples": 200,
        "shap_background_rh_values": [0, 50],
        "shap_evaluation_rh_values": [30],
        "shap_vg_bin_size": 8,
        "shap_time_bin_size": 2,
        "shap_yscale": "log",
        "shap_reference_dir": "results/analysis_reference/chemical_humid_shap",
        "shap_device": "auto",
    },
}
