import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import Metric_preprocessing as Q


class MetricQualityTests(unittest.TestCase):
    def prepare(self, broken):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        assert Path(tempfile.gettempdir()).resolve() in root.parents
        raw = np.full((30, 56, 8), 2e-9, dtype=np.float32)
        raw[:, 16:, :] *= 1.25
        if broken:
            raw[0, 14, 2] = 0.5e-9
            raw[1, :, 1] = 2e-9 + np.arange(56) * 0.2e-9
            raw[2, :14, 3] = 2e-9 + ((np.arange(14) % 2) * 2 - 1) * 0.6e-9
            raw[3, 20, 4] = np.nan
            raw[4, 4, 5] = np.nan
        bundle = {'gas_label_map': {'A': 0, 'B': 1},
                  'time_axis': np.arange(42) * .5 - 1, 'vg_axis': np.arange(8),
                  'preprocess_state': {'normalization_time_index': 14,
                                       'background_reference_time_index': 14,
                                       'background_method': 'linear_fit', 'minimum_index': 0,
                                       'raw_time_length': 56,
                                       'cut_window': {'t_start': 14, 't_end': 56, 'v_start': 0, 'v_end': 8}}}
        files = {}
        for split, rh in [('training', 0), ('heldout_test', 50)]:
            values = raw if split == 'training' else np.full((30, 56, 8), 2e-9, dtype=np.float32)
            path = root / (split + '.npy'); np.save(path, values)
            metadata = pd.DataFrame({'split': split, 'filename': path.name,
                                     'device_index': np.arange(30), 'vth_index': 0,
                                     'gas': ['A'] * 15 + ['B'] * 15, 'rh_percent': rh})
            bundle[split] = {'metadata': metadata, 'x': np.full((30, 42, 8), -999., dtype=np.float32)}
            files[split] = pd.DataFrame({'filename': [path.name], 'path': [str(path)]})
        original_load = np.load
        with patch.object(Q.np, 'load', side_effect=lambda path, **kwargs: original_load(path)), \
             patch.object(Q.Data_pipeline, 'discover_files', side_effect=lambda split, **kwargs: files[split]), \
             patch.object(Q.Data_pipeline, '_average_one_file', side_effect=lambda data, mode: data):
            result = Q.prepare_metric_data(bundle, data_root=root, show_progress=False)
        for split in ('training', 'heldout_test'):
            self.assertTrue((bundle[split]['x'] == -999).all())
            self.assertNotIn('quality_pass', bundle[split]['metadata'])
        return result

    def test_any_failed_channel_excludes_whole_observation_from_both_metrics(self):
        result = self.prepare(True)
        self.assertEqual(result['x'].shape, (55, 42, 8))
        np.testing.assert_array_equal(result['metadata']['source_row_index'], np.arange(5, 60))
        for row, reason in enumerate(['low_reference', 'excessive_drift', 'low_snr', 'nonfinite_analysis', 'nonfinite_baseline']):
            self.assertGreater(result['sample_qc'].loc[row, reason + '_count'], 0)
        self.assertEqual(result['quality_summary']['metric_num_samples_excluded'], 5)

    def test_passing_raw_current_is_renormalized_instead_of_reusing_model_inputs(self):
        result = self.prepare(False)
        self.assertEqual(result['x'].shape, (60, 42, 8))
        np.testing.assert_array_equal(result['x'][:30, :2], 0)
        np.testing.assert_allclose(result['x'][:30, 2:], .25, atol=1e-6)
        np.testing.assert_array_equal(result['x'][30:], 0)
        self.assertEqual(result['quality_summary']['metric_num_samples_excluded'], 0)


if __name__ == '__main__':
    unittest.main()
