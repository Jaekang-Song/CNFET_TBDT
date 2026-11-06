"""Exercise the pretrained model configs through the supported model interface."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from configs import AVAILABLE_CONFIGS, load_config
from ML_models import CNN1D_TBDT, TBDT_MODEL_NAME, build_model
from src import Data_pipeline, Run_analysis, Training_utils


_ANALYTES = {
    "0_chemical_gas": Run_analysis.FIGURE2_CHEMICAL_ORDER,
    "1_solvent": Run_analysis.SOLVENT_GAS_ORDER,
    "2_bacteria": ("CA", "EC", "PSA", "SA", "SS"),
}


def _input_shape(config):
    cut = config["PREPROCESS"]["cut_window"]
    return cut["t_end"] - cut["t_start"], cut["v_end"] - cut["v_start"]


def _analytes(config):
    return _ANALYTES[Path(config["DATASET"]["data_dir"]).name]


class PretrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._previous_threads)

    def test_pretrained_configs_train_and_reload(self):
        for config_name in AVAILABLE_CONFIGS:
            with self.subTest(config=config_name):
                config = load_config(config_name)
                settings = dict(config["MODEL"])
                self.assertEqual(settings["model_name"], TBDT_MODEL_NAME)
                time_length, vg_channels = _input_shape(config)
                self.assertEqual((time_length, vg_channels), (42, 96))
                self.assertEqual(settings["dim_v"], vg_channels)
                num_classes = len(_analytes(config))
                settings["num_classes"] = num_classes
                model = build_model(**settings)
                self.assertIsInstance(model, CNN1D_TBDT)
                inputs = torch.randn(3, time_length, vg_channels)
                logits, auxiliary = model(inputs)
                self.assertEqual(tuple(logits.shape), (3, num_classes))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertFalse(auxiliary["features"].requires_grad)
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

                model.eval()
                with torch.no_grad():
                    expected, features = model(inputs, return_features=True)
                    self.assertTrue(torch.equal(features, model.forward_features(inputs)))
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = build_model(**settings).eval()
                restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                with torch.no_grad():
                    actual, _ = restored(inputs)
                self.assertTrue(torch.equal(actual, expected))

    def test_pretrained_training_artifacts_and_prediction_reload(self):
        for config_name in AVAILABLE_CONFIGS:
            with self.subTest(config=config_name), tempfile.TemporaryDirectory() as temporary:
                config = load_config(config_name)
                # Keep the configured settings; one synthetic epoch checks the full interface.
                config["TRAINING"]["num_epochs"] = 1
                analytes = _analytes(config)
                num_classes = len(analytes)
                input_shape = _input_shape(config)
                rng = np.random.default_rng(0)
                training_labels = np.tile(np.arange(num_classes), 2)
                validation_labels = np.arange(num_classes)

                def make_dataset(labels):
                    values = rng.standard_normal((len(labels), *input_shape)).astype(np.float32)
                    metadata = pd.DataFrame({"gas": [analytes[index] for index in labels]})
                    return Data_pipeline.SensorDataset(values, labels, metadata)

                training = make_dataset(training_labels)
                validation = make_dataset(validation_labels)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = Training_utils.fit_model(
                        training,
                        validation,
                        num_classes=num_classes,
                        model_config=config["MODEL"],
                        training_config=config["TRAINING"],
                        device="cpu",
                    )
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(len(result["history"]), 1)
                self.assertTrue(np.isfinite(result["validation"]["loss"]))
                label_map = {name: index for index, name in enumerate(analytes)}
                run_dir = Training_utils.save_run_artifacts(
                    result,
                    config=config,
                    gas_label_map=label_map,
                    output_root=temporary,
                    run_name=config_name,
                )
                restored, metadata = Run_analysis.load_model(run_dir, device="cpu")
                self.assertEqual(metadata["gas_label_map"], label_map)
                self.assertEqual(metadata["config"], config)
                before = Run_analysis.predict_dataset(
                    result["model"], validation, device="cpu", batch_size=3
                )
                after = Run_analysis.predict_dataset(
                    restored, validation, device="cpu", batch_size=3
                )
                np.testing.assert_array_equal(after[0], validation_labels)
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                inputs = torch.from_numpy(validation.data)
                with torch.no_grad():
                    expected, _ = result["model"](inputs)
                    actual, _ = restored(inputs)
                self.assertEqual(tuple(actual.shape), (num_classes, num_classes))
                self.assertTrue(torch.equal(actual, expected))

    def test_removed_architectures_are_rejected(self):
        for name in (
            "CNN1DClassifier_VgGate_CutGN_PoolSweep",
            "CNN1DClassifier_GroupPreservingCosine_bacteria_v1",
            "CNN1DClassifier_GroupStemMixedBlock_bacteria_v2",
            "CNN2DClassifier_TimeVgResidual",
            "unknown",
        ):
            with self.subTest(model=name), self.assertRaisesRegex(ValueError, "only 'CNN1D_TBDT'"):
                build_model(model_name=name)


if __name__ == "__main__":
    unittest.main()
