"""Regression checks for ambient humidity and numeric chemical-gas RH."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import load_config
from src import Analysis_tools, Data_pipeline


class AmbientHumidityTests(unittest.TestCase):
    def test_filename_fields_and_numeric_rh(self):
        ambient = Data_pipeline.parse_sensor_filename("Set4_3_A_0_A_1.npy")
        self.assertEqual(ambient["gas_code"], "A")
        self.assertEqual(ambient["rh_percent"], "A")
        self.assertEqual(ambient["repeat_index"], 1)
        self.assertEqual(
            Data_pipeline.parse_sensor_filename("avg2_Set1_3_EC_100_a_2.npy")["rh_percent"],
            "A",
        )
        for token, expected in (("0", 0), ("30", 30), ("50", 50), ("30.5", 30.5)):
            with self.subTest(token=token):
                self.assertEqual(
                    Data_pipeline.parse_sensor_filename(f"Set1_3_A_0_{token}_1.npy")["rh_percent"],
                    expected,
                )

    def test_selectors_and_legacy_ambient_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dataset in ("1_solvent", "2_bacteria", "0_chemical_gas"):
                root = Path(temporary) / dataset
                for split in Data_pipeline.SPLITS:
                    folder = root / split
                    folder.mkdir(parents=True)
                    for name in ("Set1_1_A_0_0_1.npy", "Set2_1_M_0_A_1.npy"):
                        (folder / name).touch()
                if dataset in ("1_solvent", "2_bacteria"):
                    for selector in ("A", ["A"], [0], None, "all"):
                        with self.subTest(dataset=dataset, selector=selector):
                            files = Data_pipeline.discover_files("training", selector, root)
                            self.assertEqual(len(files), 2)
                            self.assertEqual(set(files["rh_percent"]), {"A"})
                            self.assertEqual(set(files["gas"]), {"A", "M"})
                    summary = Data_pipeline.summarize_files(root)
                    self.assertEqual(set(summary["rh_percent"]), {"A"})
                else:
                    files = Data_pipeline.discover_files("training", [0], root)
                    self.assertEqual(len(files), 1)
                    self.assertEqual(files.iloc[0]["rh_percent"], 0)
                    self.assertEqual(files.iloc[0]["gas"], "C3H6O")

    def test_configs_select_ambient(self):
        for name in ("Config_TBDT_solvent", "Config_TBDT_bacteria"):
            selection = load_config(name)["SELECTION"]
            self.assertEqual(selection["train_rh_values"], ["A"])
            self.assertEqual(selection["heldout_test_rh_values"], ["A"])

    def test_shap_sampling_and_labels_accept_ambient(self):
        for conditions in (["A"], [0, 30, 50], [0, "A"]):
            with self.subTest(conditions=conditions):
                frame = pd.DataFrame(
                    [
                        {"gas": gas, "rh_percent": rh, "split": "training", "model_split": "train"}
                        for rh in conditions for gas in ("M", "E") for _ in range(2)
                    ]
                )
                labels = frame["gas"].map({"M": 0, "E": 1}).to_numpy()
                data = np.zeros((len(frame), 4, 2), dtype=np.float32)
                heldout_frame = frame.assign(split="heldout_test", model_split="heldout_test")
                bundle = {
                    "training": {"x": data, "y": labels, "metadata": frame},
                    "heldout_test": {"x": data, "y": labels, "metadata": heldout_frame},
                }
                sampled = Analysis_tools._sample_shap_inputs(
                    bundle, background_size=len(frame), eval_size=len(frame),
                    background_rh_values=conditions, evaluation_rh_values=conditions,
                    random_state=0,
                )
                self.assertEqual(set(sampled["background_metadata"]["rh_percent"]), set(conditions))
                self.assertEqual(set(sampled["eval_metadata"]["rh_percent"]), set(conditions))
                profiles = pd.DataFrame(
                    [
                        {"split": "heldout_test", "rh_percent": rh, "vg_volts": vg,
                         "mean_abs_shap": 0.1}
                        for rh in conditions for vg in (0.0, 1.0)
                    ]
                )
                fig, ax = Analysis_tools.plot_shap_profiles({"voltage_profiles": profiles})
                labels = ax.get_legend_handles_labels()[1]
                if "A" in conditions:
                    self.assertIn("heldout_test, Ambient", labels)
                if 0 in conditions:
                    self.assertIn("heldout_test, RH 0%", labels)
                self.assertFalse(any("A%" in label for label in labels))
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
