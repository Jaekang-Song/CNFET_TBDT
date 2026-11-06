"""Concentration labels remain metadata and never exclude supplied files."""

import tempfile
import unittest
from pathlib import Path

from src import Data_pipeline


class PretrainedSelectionTests(unittest.TestCase):
    def test_all_supplied_concentrations_are_retained_in_each_physical_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "0_chemical_gas"
            for split, values in (("training", (2, 10, 100)), ("heldout_test", (3, 20, 200))):
                folder = root / split
                folder.mkdir(parents=True)
                for gas, concentrations in (("ND", values), ("NH", (1, 10))):
                    for concentration in concentrations:
                        (folder / f"Set1_3_{gas}_{concentration}_0_1.npy").touch()
                selected = Data_pipeline.discover_files(split, [0], root)
                self.assertEqual(
                    set(selected.loc[selected["gas_code"] == "ND", "concentration"]),
                    set(map(str, values)),
                )
                self.assertEqual(len(selected), 5)
                self.assertEqual(set(selected["split"]), {split})

    def test_rh_selection_does_not_depend_on_concentration_at_other_humidities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "0_chemical_gas"
            folder = root / "training"
            folder.mkdir(parents=True)
            for name in ("Set1_3_ND_10_0_1.npy", "Set2_3_ND_100_50_1.npy"):
                (folder / name).touch()
            selected = Data_pipeline.discover_files("training", [0], root)
            self.assertEqual(selected["concentration"].tolist(), ["10"])
            self.assertEqual(selected["rh_percent"].tolist(), [0])
            self.assertEqual(len(Data_pipeline.discover_files("training", [0, 50], root)), 2)

    def test_unquantified_ambient_samples_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dataset in ("1_solvent", "2_bacteria"):
                root = Path(temporary) / dataset
                folder = root / "training"
                folder.mkdir(parents=True)
                (folder / "Set1_3_A_NQ_A_1.npy").touch()
                selected = Data_pipeline.discover_files("training", ["A"], root)
                self.assertEqual(selected["concentration"].tolist(), ["NQ"])
                self.assertEqual(selected["rh_percent"].tolist(), ["A"])


if __name__ == "__main__":
    unittest.main()
