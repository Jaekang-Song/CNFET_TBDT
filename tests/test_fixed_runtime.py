"""Protect fixed runtime choices and compatibility with saved artifacts."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from configs import AVAILABLE_CONFIGS, load_config
from configs.Config_TBDT_Single_Vg import CONFIG
from ML_models import CNN1D_TBDT, build_model
from src import Analysis_tools, Single_vg_training, Training_utils
from src.Checkpoint_compat import model_config_from_snapshot
from test_single_vg_training import _small_bundle


class FixedRuntimeTests(unittest.TestCase):
    def test_fixed_choices_are_not_constructor_or_config_options(self):
        removed = {"pool_mode", "lse_tau", "classifier_head",
                   "learnable_cosine_scale", "use_late_mix"}
        self.assertTrue(removed.isdisjoint(inspect.signature(CNN1D_TBDT).parameters))
        for name in AVAILABLE_CONFIGS:
            config = load_config(name)
            self.assertTrue(removed.isdisjoint(config["MODEL"]))
            self.assertNotIn("selection_metric", config["TRAINING"])
            self.assertNotIn("early_stop_metric", config["TRAINING"])
        self.assertNotIn("method", inspect.signature(Analysis_tools.compute_shap).parameters)
        self.assertNotIn("shap_method", load_config("Config_TBDT_Chem_humid_pretrained")["ANALYSIS"])

    def test_average_pooling_fixed_cosine_scale_and_late_mixing(self):
        model = build_model(dim_v=8, hidden_size=8, vg_groups=8, gn_groups=8,
                            num_classes=2, dropout_p=0.0, dropout2d_p=0.0).eval()
        inputs = torch.randn(3, 5, 8)
        backbone = model._forward_backbone(inputs)
        expected_features = backbone.mean(dim=-1)
        logits, features = model(inputs, return_features=True)
        self.assertTrue(torch.equal(features, expected_features))
        expected_logits = 16.0 * F.linear(F.normalize(features, dim=1),
                                          F.normalize(model.class_weight, dim=1))
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertIn("_fixed_cosine_scale", dict(model.named_buffers()))
        self.assertNotIn("_fixed_cosine_scale", dict(model.named_parameters()))
        logits.sum().backward()
        self.assertIsNotNone(model.mix.weight.grad)
        self.assertIsNotNone(model.mix_gn.weight.grad)

    def test_old_snapshots_cannot_silently_change_the_architecture(self):
        fixed = {"pool_mode": "avg", "classifier_head": "cosine",
                 "learnable_cosine_scale": False, "use_late_mix": True,
                 "hidden_size": 24, "lse_tau": 1.0}
        before = deepcopy(fixed)
        self.assertEqual(model_config_from_snapshot(fixed), {"hidden_size": 24})
        self.assertEqual(fixed, before)
        for key, value in (("pool_mode", "avgmax"), ("pool_mode", "lse"),
                           ("classifier_head", "linear"),
                           ("learnable_cosine_scale", True), ("use_late_mix", False)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                model_config_from_snapshot(dict(fixed, **{key: value}))

    def test_old_single_vg_fingerprint_matches_without_rewriting_inputs(self):
        legacy = json.loads((Path(__file__).parent / "fixtures" /
                             "fixed_settings_snapshot.json").read_text(encoding="utf-8"))
        bundle = _small_bundle()
        args = (bundle, bundle["train_indices"], bundle["val_indices"])
        # The original implementation hashed the config verbatim.
        with patch.object(Single_vg_training, "fingerprint_config", side_effect=lambda value: value):
            previous = Single_vg_training._fingerprint(legacy, *args)
        current = deepcopy(CONFIG)
        before = deepcopy(current)
        self.assertEqual(Single_vg_training._fingerprint(current, *args), previous)
        self.assertEqual(current, before)
        current["TRAINING"]["learning_rate"] *= 2
        self.assertNotEqual(Single_vg_training._fingerprint(current, *args), previous)

    def test_removed_training_features_are_rejected(self):
        for config in ({"early_stop_metric": "val_loss"},
                       {"selection_metric": "val_loss"},
                       {"time_input_dropout_p": 0.3},
                       {"time_input_dropout_num_groups": 8}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Training_utils.fit_model(None, None, num_classes=2,
                                         model_config={}, training_config=config)


if __name__ == "__main__":
    unittest.main()
