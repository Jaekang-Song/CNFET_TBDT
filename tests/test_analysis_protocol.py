"""Regression checks for the published full-data dSNR and SHAP aggregation."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from src import Analysis_tools as A


class AnalysisProtocolTests(unittest.TestCase):
    def test_mean_absolute_true_class_then_equal_gas_mean(self):
        labels = np.array([0, 1, 1, 1])
        x = np.array([2, 10, -10, 10], dtype=np.float32)[:, None, None]
        frame = pd.DataFrame({"split":["heldout_test"]*4, "model_split":["heldout_test"]*4,
                              "rh_percent":[30]*4})
        background_frame = pd.DataFrame({"split":["training"]*2, "model_split":["train"]*2,
                                         "rh_percent":[0,50]})
        bundle = {"training":{"x":np.zeros((2,1,1),np.float32), "metadata":background_frame},
                  "heldout_test":{"x":x, "y":labels, "metadata":frame},
                  "num_classes":2, "vg_axis":np.array([0]), "time_axis":np.array([0])}
        class FakeExplainer:
            def __init__(self,*args,**kwargs): pass
            def shap_values(self, batch, **kwargs):
                raw = np.repeat(batch.cpu().numpy()[...,None],2,axis=-1)
                return raw
        with patch.dict("sys.modules",shap=SimpleNamespace(GradientExplainer=FakeExplainer)):
            result = A.compute_shap(torch.nn.Identity(),bundle,background_size=2,
                                    eval_size=4,batch_size=4,device="cpu")
        self.assertEqual(float(result["mean_abs_map"][0,0]),6.0)
        self.assertEqual(result["settings"]["class_counts"],[1,3])

    def test_bins_are_posthoc_means_and_time_plot_excludes_stabilization(self):
        values=np.arange(42*96,dtype=float).reshape(42,96)
        labels=np.array([0,1])
        metadata=pd.DataFrame({"split":["heldout_test"]*2,"rh_percent":[30]*2})
        r=A.shap_profiles_from_class_maps(np.stack([values,values]),labels,metadata,
                                         np.arange(96),np.arange(42)*.5-1)
        self.assertEqual(len(r["voltage_profiles"]),12)
        self.assertEqual(len(r["time_profiles"]),20)
        np.testing.assert_allclose(r["voltage_profiles"]["mean_abs_shap"],
                                   values.mean(0).reshape(12,8).mean(1))
        np.testing.assert_allclose(r["time_profiles"]["mean_abs_shap"],
                                   values.mean(1)[2:].reshape(20,2).mean(1))
        self.assertTrue(r["time_profiles"]["time_minutes"].ge(0).all())

    def test_shap_background_excludes_validation_and_evaluation_uses_all_rh30(self):
        train=pd.DataFrame({"split":["training"]*4,"model_split":["train","validation","train","train"],
                            "rh_percent":[0,0,50,30]})
        test=pd.DataFrame({"split":["heldout_test"]*5,"model_split":["heldout_test"]*5,
                           "rh_percent":[30,0,30,50,30]})
        bundle={"training":{"x":np.arange(4)[:,None,None],"metadata":train},
                "heldout_test":{"x":np.arange(5)[:,None,None],"y":np.arange(5),"metadata":test}}
        r=A._sample_shap_inputs(bundle,background_size=8192,eval_size=8192,
                               background_rh_values=[0,50],evaluation_rh_values=[30],random_state=0)
        self.assertEqual(set(r["background_x"].ravel()),{0,2})
        self.assertEqual(set(r["eval_y"]),{0,2,4})

    def test_full_dsnr_keeps_every_existing_sample_and_adds_missing_rh30(self):
        def frame(split,n):
            return pd.DataFrame({"split":[split]*n,"filename":[split+".npy"]*n,
                                 "device_index":np.arange(n),"gas":["G"]*n,"rh_percent":[0]*n})
        bundle={s:{"x":np.zeros((n,2,2)),"metadata":frame(s,n),
                   "files":pd.DataFrame({"filename":[s+".npy"]})}
                for s,n in [("training",503),("heldout_test",509)]}
        bundle.update(gas_label_map={"G":0},preprocess_state={})
        extra=pd.DataFrame({"split":["training"]*3,"filename":["extra.npy"]*3,
                            "device_index":np.arange(3),"gas":["G"]*3,"rh_percent":[30]*3})
        def discover(split,**kwargs):
            return pd.DataFrame({"filename":[split+".npy"]+(["extra.npy"] if split=="training" else [])})
        with patch("src.Data_pipeline.discover_files",side_effect=discover), \
             patch("src.Data_pipeline.load_averaged_split",return_value=(np.ones((3,2,2)),extra)), \
             patch("src.Data_pipeline.apply_preprocess_state",side_effect=lambda x,m,s:(x,m)):
            result=A.full_metric_data(bundle,data_root="unused")
        self.assertEqual(len(result["y"]),1015)
        self.assertEqual(result["metadata"]["rh_percent"].eq(30).sum(),3)
        self.assertEqual(len(bundle["training"]["x"]),503)


if __name__=="__main__":unittest.main()
