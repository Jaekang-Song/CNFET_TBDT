"""Checks for NQ labels, legacy placeholders, and chemical-gas concentrations."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import Data_pipeline


class UnquantifiedConcentrationTests(unittest.TestCase):
    def test_parse_concentration_separately_from_analyte_and_humidity(self):
        for token in ("NQ", "nq"):
            parsed = Data_pipeline.parse_sensor_filename(f"Set4_3_A_{token}_A_1.npy")
            self.assertEqual(parsed["gas_code"], "A")
            self.assertEqual(parsed["concentration"], "NQ")
            self.assertEqual(parsed["rh_percent"], "A")
            self.assertEqual(parsed["repeat_index"], 1)
        for token in ("0", "1", "10", "100", "500", "2500"):
            parsed = Data_pipeline.parse_sensor_filename(f"Set1_6_NH_{token}_50_1.npy")
            self.assertEqual(parsed["concentration"], token)
            self.assertEqual(parsed["rh_percent"], 50)
            self.assertEqual(parsed["gas"], "NH3")
        parsed = Data_pipeline.parse_sensor_filename("avg2_Set1_3_EC_NQ_A_2.npy")
        self.assertEqual(parsed["concentration"], "NQ")

    def test_discovery_supports_legacy_and_nq_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dataset in ("1_solvent", "2_bacteria"):
                root = Path(temporary) / dataset
                for split in Data_pipeline.SPLITS:
                    folder = root / split
                    folder.mkdir(parents=True)
                    for index, token in enumerate(("0", "100", "NQ"), start=1):
                        (folder / f"Set{index}_3_A_{token}_A_1.npy").touch()
                    files = Data_pipeline.discover_files(split, ["A"], root)
                    self.assertEqual(len(files), 3)
                    self.assertEqual(set(files["concentration"]), {"NQ"})
                    self.assertEqual(set(files["rh_percent"]), {"A"})
                    self.assertEqual(set(files["gas"]), {"A"})

    def test_renaming_does_not_change_loaded_samples(self):
        raw = np.arange(256 * 4 * 3, dtype=np.float32).reshape(256, 4, 3)
        with tempfile.TemporaryDirectory() as temporary:
            for dataset, token in (("1_solvent", "0"), ("2_bacteria", "100")):
                loaded = []
                for version, concentration in (("legacy", token), ("current", "NQ")):
                    root = Path(temporary) / version / dataset
                    folder = root / "training"
                    folder.mkdir(parents=True)
                    np.save(folder / f"Set1_3_A_{concentration}_A_1.npy", raw)
                    files = Data_pipeline.discover_files("training", ["A"], root)
                    values, metadata = Data_pipeline.load_averaged_split(
                        files, average_mode=1, show_progress=False
                    )
                    self.assertEqual(set(metadata["concentration"]), {"NQ"})
                    self.assertEqual(set(metadata["rh_percent"]), {"A"})
                    loaded.append((values, metadata))
                np.testing.assert_array_equal(loaded[0][0], loaded[1][0])
                pd.testing.assert_frame_equal(
                    loaded[0][1].drop(columns="filename"),
                    loaded[1][1].drop(columns="filename"),
                )

    def test_chemical_gas_concentrations_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "0_chemical_gas"
            folder = root / "training"
            folder.mkdir(parents=True)
            for index, token in enumerate(("0", "100", "500"), start=1):
                np.save(
                    folder / f"Set{index}_6_A_{token}_50_1.npy",
                    np.ones((256, 4, 3), dtype=np.float32),
                )
            files = Data_pipeline.discover_files("training", [50], root)
            self.assertEqual(set(files["concentration"]), {"0", "100", "500"})
            _, metadata = Data_pipeline.load_averaged_split(files, show_progress=False)
            self.assertEqual(set(metadata["concentration"]), {"0", "100", "500"})
            self.assertEqual(set(metadata["gas"]), {"C3H6O"})
            self.assertEqual(set(metadata["rh_percent"]), {50})


if __name__ == "__main__":
    unittest.main()
