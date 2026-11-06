"""Check the single-Vg experiment's archived search space and isolation."""

from copy import deepcopy
import unittest

from configs import AVAILABLE_CONFIGS, load_config
from configs.Config_TBDT_Single_Vg import CONFIG
from src.Single_vg_training import expand_parameter_sets


class SingleVgConfigTests(unittest.TestCase):
    def test_archived_grid_preserves_all_24_candidate_indices(self):
        self.assertEqual(list(CONFIG["PARAMETER_GRID"]), [
            "hidden_size", "learning_rate", "weight_decay", "dropout_p",
        ])
        self.assertEqual(CONFIG["PARAMETER_GRID"], {
            "hidden_size": [24, 48, 88],
            "learning_rate": [0.001, 0.002],
            "weight_decay": [0.001, 0.003],
            "dropout_p": [0.15, 0.35],
        })
        candidates = expand_parameter_sets(CONFIG)
        self.assertEqual(len(candidates), 24)
        self.assertEqual([row["candidate_index"] for row in candidates], list(range(24)))
        self.assertEqual(candidates[0], {
            "candidate_index": 0, "hidden_size": 24, "learning_rate": 0.001,
            "weight_decay": 0.001, "dropout_p": 0.15,
        })
        self.assertEqual(candidates[-1], {
            "candidate_index": 23, "hidden_size": 88, "learning_rate": 0.002,
            "weight_decay": 0.003, "dropout_p": 0.35,
        })
        self.assertEqual(len({
            tuple(row[key] for key in CONFIG["PARAMETER_GRID"]) for row in candidates
        }), 24)

    def test_single_channel_model_and_published_training_settings(self):
        self.assertEqual(CONFIG["VG_INDICES"], list(range(0, 96, 4)))
        self.assertEqual(CONFIG["MODEL"]["model_name"], "CNN1D_TBDT")
        self.assertEqual(CONFIG["MODEL"]["dim_v"], 1)
        self.assertEqual(CONFIG["MODEL"]["vg_groups"], 1)
        self.assertEqual(CONFIG["MODEL"]["gn_groups"], 8)
        self.assertEqual(CONFIG["MODEL"]["dropout2d_p"], 0.075)
        self.assertNotIn("pool_mode", CONFIG["MODEL"])
        training = CONFIG["TRAINING"]
        self.assertEqual(training["vg_input_dropout_p"], 0.0)
        self.assertEqual(training["vg_input_dropout_num_groups"], 1)
        self.assertEqual(training["early_stop_baseline"], 90)
        self.assertEqual(training["num_epochs"], 200)
        self.assertEqual(training["patience"], 100)
        self.assertEqual(training["early_stop_min_delta"], 0.01)
        self.assertEqual(training["batch_size"], 128)
        self.assertEqual(training["random_seed"], 0)
        self.assertNotIn("selection_metric", training)
        self.assertNotIn("early_stop_metric", training)

    def test_separate_config_preserves_full_vg_picker_and_split(self):
        full_vg = load_config("Config_TBDT_Chem_humid_pretrained")
        self.assertNotIn("Config_TBDT_Single_Vg", AVAILABLE_CONFIGS)
        self.assertEqual(len(AVAILABLE_CONFIGS), 4)
        self.assertEqual(CONFIG["DATASET"], full_vg["DATASET"])
        self.assertEqual(CONFIG["PREPROCESS"], full_vg["PREPROCESS"])
        self.assertEqual(CONFIG["SELECTION"], {
            "train_rh_values": [0],
            "heldout_test_rh_values": [0, 30, 50],
            "train_val_split": {"val_size": 0.5, "random_state": 0},
        })
        self.assertEqual(CONFIG["OUTPUT"], {
            "models_dir": "results/models/tbdt_single_vg", "runs_dirname": "runs",
        })
        self.assertNotIn("ANALYSIS", CONFIG)
        self.assertNotIn("PROVISIONAL", CONFIG)
        self.assertNotIn("scheduler_mode", CONFIG["TRAINING"])
        self.assertEqual(full_vg["MODEL"]["dim_v"], 96)
        self.assertEqual(full_vg["TRAINING"]["vg_input_dropout_p"], 0.3)
        # Independent nested dictionaries prevent a single-Vg edit from
        # mutating the full-Vg config imported by another notebook.
        from configs.Config_TBDT_Chem_humid_pretrained import CONFIG as original

        self.assertIsNot(CONFIG["MODEL"], original["MODEL"])
        self.assertIsNot(CONFIG["TRAINING"], original["TRAINING"])
        self.assertIsNot(CONFIG["SELECTION"], original["SELECTION"])

    def test_parameter_expansion_preserves_the_supplied_configuration(self):
        config = deepcopy(CONFIG)
        before = deepcopy(config)
        expand_parameter_sets(config)
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()

