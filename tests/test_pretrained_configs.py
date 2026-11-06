"""Regression checks for the four task configurations and checkpoint settings.

These checks use only the standard library and protect the published numerical
settings, including the 90% validation-accuracy early-stopping baseline.
"""

import importlib
import unittest
from pathlib import Path

from configs import AVAILABLE_CONFIGS, load_config


class PretrainedConfigTests(unittest.TestCase):
    def test_only_four_canonical_pretrained_modules_are_published(self):
        expected = {
            "Config_TBDT_Chem_dry_pretrained",
            "Config_TBDT_Chem_humid_pretrained",
            "Config_TBDT_Solvent_pretrained",
            "Config_TBDT_Bacteria_pretrained",
        }
        self.assertEqual(set(AVAILABLE_CONFIGS), expected)
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        self.assertEqual({path.stem for path in config_dir.glob("Config_TBDT_*_pretrained.py")}, expected)
        for name in AVAILABLE_CONFIGS:
            module = importlib.import_module(f"configs.{name}")
            self.assertEqual(module.CONFIG["CONFIG_NAME"], name)
            self.assertNotIn("PROVISIONAL", module.CONFIG)
            self.assertFalse(hasattr(module, "ACTIVE_RUN_SI"))
            self.assertFalse(hasattr(module, "ACTIVE_RUN"))
            self.assertNotIn("SWEEP", module.CONFIG)
            self.assertNotIn("MODEL_SWEEP", module.CONFIG)

    def test_published_pretrained_hyperparameters(self):
        # hidden, dropout, channel dropout, LR, decay, epochs, baseline, Vg groups
        expected = {
            "Chem_dry": (112, 0.4, 0.125, 2e-3, 3e-3, 140, 90, 8),
            "Chem_humid": (88, 0.35, 0.075, 2e-3, 3e-3, 200, 90, 8),
            "Solvent": (56, 0.25, 0.1, 3e-3, 1.3e-3, 120, 90, 12),
            "Bacteria": (8, 0.2, 0.0, 2e-3, 2e-3, 120, 90, 12),
        }
        for experiment, values in expected.items():
            with self.subTest(experiment=experiment):
                config = load_config(f"Config_TBDT_{experiment}_pretrained")
                model, training = config["MODEL"], config["TRAINING"]
                actual = (
                    model["hidden_size"], model["dropout_p"], model["dropout2d_p"],
                    training["learning_rate"], training["weight_decay"],
                    training["num_epochs"], training["early_stop_baseline"],
                    training["vg_input_dropout_num_groups"],
                )
                self.assertEqual(actual, values)

    def test_all_pretrained_configs_use_only_the_shared_paper_model(self):
        expected_model = {
            "model_name": "CNN1D_TBDT",
            "dim_v": 96,
            "num_layers": 1,
            "vg_groups": 8,
            "gn_groups": 8,
            "kernel_size": 5,
            "cosine_scale": 16.0,
            "vg_group_dropout_p": 0.0,
        }
        for name in AVAILABLE_CONFIGS:
            with self.subTest(config=name):
                model = load_config(name)["MODEL"]
                self.assertEqual(
                    set(model), set(expected_model) | {"hidden_size", "dropout_p", "dropout2d_p"}
                )
                for key, expected in expected_model.items():
                    self.assertEqual(model[key], expected, key)

    def test_common_effective_training_settings(self):
        expected = {
            "batch_size": 128,
            "patience": 100,
            "num_workers": 0,
            "early_stop_min_delta": 0.01,
            "vg_input_dropout_p": 0.3,
            "random_seed": 0,
            "use_class_balanced_sampler": True,
        }
        for name in AVAILABLE_CONFIGS:
            with self.subTest(config=name):
                training = load_config(name)["TRAINING"]
                self.assertEqual(
                    set(training),
                    set(expected) | {
                        "num_epochs", "early_stop_baseline", "learning_rate",
                        "weight_decay", "vg_input_dropout_num_groups",
                    },
                )
                for key, value in expected.items():
                    self.assertEqual(training[key], value, key)

    def test_local_dataset_paths_selections_and_preprocessing(self):
        expected = {
            "Chem_dry": ("Config_TBDT_chemical_gas", "0_chemical_gas", [0], [0]),
            "Chem_humid": ("Config_TBDT_chemical_gas", "0_chemical_gas", [0, 50], [0, 30, 50]),
            "Solvent": ("Config_TBDT_solvent", "1_solvent", ["A"], ["A"]),
            "Bacteria": ("Config_TBDT_bacteria", "2_bacteria", ["A"], ["A"]),
        }
        outputs = set()
        for experiment, (dataset_name, directory, train_rh, heldout_rh) in expected.items():
            with self.subTest(experiment=experiment):
                config = load_config(f"Config_TBDT_{experiment}_pretrained")
                self.assertEqual(config["DATASET"], {
                    "name": dataset_name, "data_dir": f"TBDT_Dataset/{directory}",
                })
                self.assertEqual(config["PREPROCESS"], {
                    "average_mode": 0,
                    "cut_window": {"t_start": 14, "t_end": 56, "v_start": 11, "v_end": 107},
                })
                selection = config["SELECTION"]
                self.assertEqual(selection, {
                    "train_rh_values": train_rh,
                    "heldout_test_rh_values": heldout_rh,
                    "train_val_split": {"val_size": 0.5, "random_state": 0},
                })
                expected_output = f"results/models/tbdt_{experiment.lower()}_pretrained"
                self.assertEqual(config["OUTPUT"], {
                    "models_dir": expected_output,
                    "runs_dirname": "runs",
                })
                outputs.add(expected_output)
        self.assertEqual(len(outputs), 4)

    def test_default_aliases_and_independent_copies(self):
        self.assertEqual(load_config()["CONFIG_NAME"], "Config_TBDT_Chem_dry_pretrained")
        aliases = {
            "Config_TBDT_chemical_gas_all": "Config_TBDT_Chem_humid_pretrained",
            "Config_TBDT_chemical_gas_dry": "Config_TBDT_Chem_dry_pretrained",
            "Config_TBDT_solvent": "Config_TBDT_Solvent_pretrained",
            "Config_TBDT_bacteria": "Config_TBDT_Bacteria_pretrained",
        }
        for old_name, canonical_name in aliases.items():
            with self.subTest(alias=old_name):
                self.assertEqual(load_config(old_name), load_config(canonical_name))
                modified = load_config(old_name)
                modified["SELECTION"]["train_rh_values"].append("changed")
                modified["MODEL"]["hidden_size"] = -1
                self.assertNotEqual(modified, load_config(canonical_name))
        with self.assertRaises(ValueError):
            load_config("Config_TBDT_unknown_model")

    def test_humid_analysis_matches_documented_protocol(self):
        expected = {'random_seed': 0,
         'metric_rh_values': [0, 50],
         'metric_quality': {'min_reference_current': 1e-9, 'min_baseline_snr': 5.0,
                            'max_extrapolated_drift_fraction': 1.0},
         'mi_joint_time_indices': [0, 1, 2, 3, 5, 8, 12, 18, 25, 35],
         'mi_joint_vg_indices': [0, 19, 38, 57, 76, 95],
         'mi_neighbors': 5,
         'shap_background_size': 8192,
         'shap_eval_size': 8192,
         'shap_batch_size': 256,
         'shap_nsamples': 200,
         'shap_background_rh_values': [0, 50],
         'shap_evaluation_rh_values': [30],
         'shap_vg_bin_size': 8,
         'shap_time_bin_size': 2,
         'shap_yscale': 'log',
         'shap_reference_dir': 'results/analysis_reference/chemical_humid_shap',
         'shap_device': 'auto'}
        self.assertEqual(load_config("Config_TBDT_Chem_humid_pretrained")["ANALYSIS"], expected)


if __name__ == "__main__":
    unittest.main()
