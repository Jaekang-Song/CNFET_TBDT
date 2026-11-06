"""Check that stopping tolerance does not change checkpoint selection."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from configs import load_config
from src import Training_utils


class PretrainedTrainingTests(unittest.TestCase):
    def test_stopping_tolerance_and_maximum_validation_checkpoint_are_independent(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
        dataset = TensorDataset(torch.ones(4, 2, 2), torch.tensor([0, 1, 0, 1]))
        config = load_config("Config_TBDT_Bacteria_pretrained")
        training = dict(config["TRAINING"], num_epochs=5, patience=2,
                        early_stop_baseline=50.0, early_stop_min_delta=1.0,
                        vg_input_dropout_p=0.0, batch_size=4)
        evaluations = [
            {"accuracy": accuracy, "loss": 1.0,
             "y_true": np.array([0, 1]), "y_pred": np.array([0, 1])}
            for accuracy in (60.0, 60.5, 60.75, 60.75)
        ]
        with patch.object(Training_utils.ML_models, "build_model", return_value=model), \
             patch.object(Training_utils, "evaluate", side_effect=evaluations), \
             contextlib.redirect_stdout(io.StringIO()):
            result = Training_utils.fit_model(
                dataset, dataset, num_classes=2, model_config=config["MODEL"],
                training_config=training, device="cpu",
            )
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["selection_metric_value"], 60.75)
        self.assertEqual(result["early_stop_best_metric_value"], 60.0)
        self.assertEqual(result["early_stop_min_delta"], 1.0)

    def test_published_baseline_gates_patience_at_90_percent(self):
        cases = (
            ([89.0, 89.0, 89.0, 90.0, 90.0, 90.0], 7, 6, 4),
            ([89.0] * 5, 5, 5, 1),
        )
        for accuracies, maximum_epochs, expected_epochs, best_epoch in cases:
            with self.subTest(accuracies=accuracies):
                config = load_config("Config_TBDT_Chem_humid_pretrained")
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
                dataset = TensorDataset(torch.ones(4, 2, 2), torch.tensor([0, 1, 0, 1]))
                training = dict(config["TRAINING"], num_epochs=maximum_epochs,
                                patience=2, vg_input_dropout_p=0.0, batch_size=4)
                evaluations = [
                    {"accuracy": accuracy, "loss": 1.0,
                     "y_true": np.array([0, 1]), "y_pred": np.array([0, 1])}
                    for accuracy in accuracies + [max(accuracies)]
                ]
                with patch.object(Training_utils.ML_models, "build_model", return_value=model), \
                     patch.object(Training_utils, "evaluate", side_effect=evaluations), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = Training_utils.fit_model(
                        dataset, dataset, num_classes=2, model_config=config["MODEL"],
                        training_config=training, device="cpu",
                    )
                self.assertEqual(result["early_stop_baseline"], 90.0)
                self.assertEqual(len(result["history"]), expected_epochs)
                self.assertEqual(result["best_epoch"], best_epoch)
                self.assertEqual(result["selection_metric_value"], max(accuracies))

    def test_unused_scheduler_is_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "scheduler_mode"):
            Training_utils.fit_model(
                None, None, num_classes=2, model_config={},
                training_config={"scheduler_mode": "plateau"}, device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
