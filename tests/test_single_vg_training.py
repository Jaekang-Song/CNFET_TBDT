"""Small real sweeps and checkpoint contracts for the single-Vg experiment."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from configs.Config_TBDT_Single_Vg import CONFIG
from ML_models import build_model
from src import Run_analysis, Single_vg_training, Training_utils


def _small_bundle():
    rng = np.random.default_rng(14)
    training_labels = np.tile([0, 1], 4)
    test_labels = np.tile([0, 1], 3)
    return {
        "training": {
            "x": rng.standard_normal((8, 42, 96)).astype(np.float32),
            "y": training_labels,
            "metadata": pd.DataFrame({"gas": np.tile(["C3H6O", "H2"], 4), "rh_percent": 0}),
        },
        "heldout_test": {
            "x": rng.standard_normal((6, 42, 96)).astype(np.float32),
            "y": test_labels,
            "metadata": pd.DataFrame({"gas": np.tile(["C3H6O", "H2"], 3), "rh_percent": [0, 0, 30, 30, 50, 50]}),
        },
        "train_indices": np.array([0, 1, 2, 3]),
        "val_indices": np.array([4, 5, 6, 7]),
        "gas_label_map": {"C3H6O": 0, "H2": 1},
        "vg_axis": np.linspace(-1.41, 1.44, 96),
    }


class SingleVgTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._previous_threads)

    def test_real_subset_sweep_resume_and_saved_prediction_contract(self):
        config = deepcopy(CONFIG)
        config["TRAINING"]["num_epochs"] = 1
        config["TRAINING"]["batch_size"] = 4
        bundle = _small_bundle()
        original = bundle["training"]["x"].copy()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(Training_utils, "fit_model", wraps=Training_utils.fit_model) as fit, \
                 contextlib.redirect_stdout(io.StringIO()):
                result = Single_vg_training.run_single_vg_sweep(
                    config, bundle, temporary, vg_indices=[0, 4], candidate_indices=[0, 1], device="cpu",
                )
            self.assertEqual(fit.call_count, 4)
            self.assertEqual(len(result["completed_runs"]), 4)
            self.assertEqual(len(result["best_by_vg_seed"]), 2)
            self.assertEqual(len(result["summary"]), 2)
            self.assertTrue((result["summary"]["completed_runs"] == 2).all())
            self.assertTrue((result["summary"]["expected_runs"] == 24).all())
            self.assertFalse(result["summary"]["is_complete"].any())
            np.testing.assert_array_equal(original, bundle["training"]["x"])
            for row in result["completed_runs"].to_dict("records"):
                model_dir = (Path(temporary) / row["checkpoint_path"]).parent
                model, metadata = Run_analysis.load_model(model_dir, device="cpu")
                self.assertEqual(metadata["config"]["CASE_INPUT_SELECTION"]["selected_vg_indices"], [row["vg_index"]])
                self.assertEqual(metadata["config"]["MODEL"]["dim_v"], 1)
                inputs = torch.from_numpy(bundle["heldout_test"]["x"][:, :, [row["vg_index"]]])
                with torch.no_grad():
                    logits, _ = model(inputs)
                self.assertEqual(tuple(logits.shape), (6, 2))
                predicted = np.load(model_dir / "heldout_test_predictions.npy")
                np.testing.assert_array_equal(logits.argmax(1).numpy(), predicted)
                self.assertEqual(row["selected_epoch"], 1)
                labels = bundle["heldout_test"]["y"]
                self.assertAlmostEqual(row["heldout_test_accuracy"], 100 * np.mean(labels == predicted))
                for rh in (0, 30, 50):
                    mask = bundle["heldout_test"]["metadata"]["rh_percent"].to_numpy() == rh
                    self.assertAlmostEqual(row[f"heldout_test_accuracy_rh{rh}"], 100 * np.mean(labels[mask] == predicted[mask]))
            for filename in ("parameter_sets.csv", "completed_runs.csv", "best_by_vg_seed.csv", "single_vg_summary.csv"):
                self.assertTrue((Path(temporary) / filename).is_file())
            with patch.object(Training_utils, "fit_model", side_effect=AssertionError("Completed cases must not retrain")):
                resumed = Single_vg_training.run_single_vg_sweep(
                    config, bundle, temporary, vg_indices=[0, 4], candidate_indices=[0, 1], device="cpu",
                )
            pd.testing.assert_frame_equal(result["summary"], resumed["summary"])
            changed = deepcopy(bundle)
            changed["train_indices"] = np.array([1, 0, 2, 3])
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Single_vg_training.run_single_vg_sweep(config, changed, temporary, vg_indices=[0], candidate_indices=[0])
            changed = deepcopy(bundle)
            changed["heldout_test"]["x"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Single_vg_training.run_single_vg_sweep(config, changed, temporary, vg_indices=[0], candidate_indices=[0])
            with self.assertRaises(FileExistsError):
                Single_vg_training.run_single_vg_sweep(config, bundle, temporary, resume=False)

    def test_validation_tie_ignores_heldout_and_request_order(self):
        rows = [
            {"vg_index": 0, "seed": 0, "candidate_index": 1, "validation_accuracy": 80.0, "heldout_test_accuracy": 100.0},
            {"vg_index": 0, "seed": 0, "candidate_index": 0, "validation_accuracy": 80.0, "heldout_test_accuracy": 0.0},
            {"vg_index": 4, "seed": 0, "candidate_index": 0, "validation_accuracy": 79.0, "heldout_test_accuracy": 100.0},
            {"vg_index": 4, "seed": 0, "candidate_index": 1, "validation_accuracy": 80.0, "heldout_test_accuracy": 0.0},
        ]
        best = Single_vg_training._select_best(rows)
        self.assertEqual([(row["vg_index"], row["candidate_index"]) for row in best], [(0, 0), (4, 1)])

    def test_preserved_archive_roots_and_descendants_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "original_archive"
            relocated = Path(temporary) / "relocated_archive"
            original.mkdir()
            relocated.mkdir()
            # The known archive remains protected even if its manifest is absent.
            (original / "single_vg_summary.csv").write_bytes(b"original archive bytes\n")
            (relocated / "archive_manifest.json").write_bytes(b'{"preserved": true}\n')
            (relocated / "single_vg_summary.csv").write_bytes(b"relocated archive bytes\n")

            def inventory():
                return {path.relative_to(temporary).as_posix(): path.read_bytes()
                        for path in Path(temporary).rglob("*") if path.is_file()}

            before = inventory()
            with patch.object(Single_vg_training, "ARCHIVE_ROOT", original), \
                 patch.object(Training_utils, "fit_model", side_effect=AssertionError("Must reject before training")):
                for output in (original, original / "new/subdirectory", relocated, relocated / "new/subdirectory"):
                    with self.subTest(output=output), self.assertRaisesRegex(ValueError, "Preserved result archives"):
                        # An unusable bundle proves the guard runs before data inspection.
                        Single_vg_training.run_single_vg_sweep(CONFIG, None, output_root=output)
            self.assertEqual(inventory(), before)
            self.assertFalse((original / "new").exists())
            self.assertFalse((relocated / "new").exists())
            self.assertFalse(list(Path(temporary).rglob(".sweep.lock")))

    def test_original_checkpoint_schema_preserves_class_indices_and_weights(self):
        settings = dict(CONFIG["MODEL"], hidden_size=24, dropout_p=0.15)
        original = build_model(num_classes=2, **settings).eval()
        fixed = {key: value for key, value in settings.items() if key not in ("hidden_size", "dropout_p", "dropout2d_p", "pool_mode")}
        fixed.update({
            "classifier_head": "cosine", "learnable_cosine_scale": False,
            "use_late_mix": True,
        })
        source_config = {
            "MODEL": fixed,
            "SWEEP": {"hidden_sizes": [24], "dropout_p": [0.15], "dropout2d_p": [0.075], "pool_options": [["avg", None]]},
            "CASE_INPUT_SELECTION": {"selected_vg_indices": [0], "dim_v": 1, "time_point_count": 42},
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "selected_models/v000"
            directory.mkdir(parents=True)
            torch.save({"model_state_dict": original.state_dict(), "gas_label_map": {"A": 0, "H": 1}}, directory / "best_checkpoint.pt")
            (directory / "config_snapshot.json").write_text(json.dumps({"config": source_config}), encoding="utf-8")
            checkpoint_bytes = (directory / "best_checkpoint.pt").read_bytes()
            restored, metadata = Single_vg_training.load_archived_model(0, archive_root=temporary)
            self.assertEqual(metadata["gas_label_map"], {"C3H6O": 0, "H2": 1})
            self.assertEqual(metadata["original_gas_label_map"], {"A": 0, "H": 1})
            self.assertEqual(metadata["case_input_selection"]["selected_vg_indices"], [0])
            inputs = torch.randn(3, 42, 1)
            with torch.no_grad():
                self.assertTrue(torch.equal(original(inputs)[0], restored(inputs)[0]))
            self.assertEqual(checkpoint_bytes, (directory / "best_checkpoint.pt").read_bytes())

    def test_saved_results_and_all_rh_plots_need_no_sensor_dataset(self):
        summary = pd.DataFrame({
            "vg_index": [4, 0], "vg_value": [-1.29, -1.41],
            "best_test_mean": [70.0, 65.0], "best_heldout_mean": [60.0, 55.0],
            "heldout_acc_rh0_mean": [62.0, 57.0], "heldout_acc_rh30_mean": [60.0, 55.0],
            "heldout_acc_rh50_mean": [58.0, 53.0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            summary.to_csv(Path(temporary) / "single_vg_summary.csv", index=False)
            loaded = Single_vg_training.load_single_vg_results(temporary)
            self.assertEqual(loaded["vg_index"].tolist(), [0, 4])
            self.assertEqual(loaded["validation_accuracy"].tolist(), [65.0, 70.0])
            for metric in ("overall", "rh0", "rh30", "rh50", "validation"):
                fig, ax = Single_vg_training.plot_single_vg_results(temporary, metric=metric)
                self.assertEqual(len(ax.lines), 1)
                plt.close(fig)
            partial = loaded.copy()
            partial["is_complete"] = [True, False]
            fig, ax = Single_vg_training.plot_single_vg_results(partial)
            self.assertIn("partial candidate sweep", ax.get_title())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

