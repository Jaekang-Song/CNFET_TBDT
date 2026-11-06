# Single-Vg training and archived results

This guide explains the 24-candidate single-Vg chemical-gas sweep, its saved results, and the procedure for running the experiment through this repository. Start with [Single_Vg_training.ipynb](../Single_Vg_training.ipynb). Its default `RUN_TRAINING = False` previews the saved results; a new full sweep requires setting that flag to `True`.

The runtime configuration is [Config_TBDT_Single_Vg.py](../configs/Config_TBDT_Single_Vg.py), separate from the four full-Vg pretrained configurations. [Single_vg_training.py](../src/Single_vg_training.py) uses the shared data pipeline, training utilities, and `CNN1D_TBDT` architecture.

## 1. Experimental inputs and fixed split

New training uses the supplied files and their physical `training/` and `heldout_test/` chip split. The loader selects RH conditions from the configuration; concentration labels remain metadata.

| Setting | Value |
| --- | --- |
| Dataset | `TBDT_Dataset/0_chemical_gas/` |
| Training and validation RH | 0% |
| Held-out RH | 0%, 30%, 50% |
| Validation fraction | 0.5 of the physical training samples |
| Split random state | 0 |
| Training seed | 0 |
| Time points per sample | All 42 processed time points, from -1.0 to 19.5 min in 0.5 min steps |
| Available processed Vg points | 96 after alignment and cropping |
| Selected Vg indices | `0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92` |
| Input shape for one candidate | `(batch, 42, 1)` |

Vg indices refer to positions in the aligned, cropped 96-point `vg_axis` returned by the data loader. They are distinct from indices into a raw sensor file. The archived selected points span approximately -1.41 to 1.35 V.

The existing `Data_pipeline.split_training_indices` uses a seeded random `train_test_split`, without stratification. The split is constructed once, and every Vg/candidate run uses the same training and validation membership. Class-balanced sampling operates within the resulting model-training partition.

## 2. The 24 parameter sets

The grid varies four settings:

| Parameter | Values | Role |
| --- | --- | --- |
| Hidden channels | 24, 48, 88 | Width of the shared CNN |
| Learning rate | 0.001, 0.002 | AdamW update size |
| Weight decay | 0.001, 0.003 | AdamW regularization |
| Dropout probability | 0.15, 0.35 | Dropout applied to pooled features |

The Cartesian product contains `3 × 2 × 2 × 2 = 24` candidates. Each Vg receives the complete grid independently. Candidate order is stable: dropout changes fastest, followed by weight decay, learning rate, and hidden channels. With zero-based axis positions `h, l, w, d`, the candidate index is `8*h + 4*l + 2*w + d`.

The table below follows `PARAMETER_GRID` in [Config_TBDT_Single_Vg.py](../configs/Config_TBDT_Single_Vg.py).

| Candidate index | Hidden channels | Learning rate | Weight decay | Dropout |
| --- | --- | --- | --- | --- |
| 0 | 24 | 0.001 | 0.001 | 0.15 |
| 1 | 24 | 0.001 | 0.001 | 0.35 |
| 2 | 24 | 0.001 | 0.003 | 0.15 |
| 3 | 24 | 0.001 | 0.003 | 0.35 |
| 4 | 24 | 0.002 | 0.001 | 0.15 |
| 5 | 24 | 0.002 | 0.001 | 0.35 |
| 6 | 24 | 0.002 | 0.003 | 0.15 |
| 7 | 24 | 0.002 | 0.003 | 0.35 |
| 8 | 48 | 0.001 | 0.001 | 0.15 |
| 9 | 48 | 0.001 | 0.001 | 0.35 |
| 10 | 48 | 0.001 | 0.003 | 0.15 |
| 11 | 48 | 0.001 | 0.003 | 0.35 |
| 12 | 48 | 0.002 | 0.001 | 0.15 |
| 13 | 48 | 0.002 | 0.001 | 0.35 |
| 14 | 48 | 0.002 | 0.003 | 0.15 |
| 15 | 48 | 0.002 | 0.003 | 0.35 |
| 16 | 88 | 0.001 | 0.001 | 0.15 |
| 17 | 88 | 0.001 | 0.001 | 0.35 |
| 18 | 88 | 0.001 | 0.003 | 0.15 |
| 19 | 88 | 0.001 | 0.003 | 0.35 |
| 20 | 88 | 0.002 | 0.001 | 0.15 |
| 21 | 88 | 0.002 | 0.001 | 0.35 |
| 22 | 88 | 0.002 | 0.003 | 0.15 |
| 23 | 88 | 0.002 | 0.003 | 0.35 |

All remaining model and training settings are fixed.

## 3. Checkpoint and candidate selection

For each Vg, all 24 candidates start independent training runs with seed 0 and the fixed split.

1. Select the earliest epoch attaining that candidate's maximum validation accuracy. Checkpoint selection uses a strict improvement in validation accuracy.
2. For new training, start the early-stopping patience counter once validation accuracy reaches 90%. An improvement greater than 0.01 percentage points resets the counter. Runs that never reach 90% continue to the maximum epoch count. Checkpoint selection still uses maximum validation accuracy.
3. Evaluate the restored checkpoint on held-out samples and record overall and per-RH accuracies.
4. Select the candidate with the highest validation accuracy for that Vg. An exact candidate tie retains the earliest candidate index.

Held-out accuracy is never used for checkpoint selection, candidate selection, or breaking validation ties. Overall held-out accuracy pools all held-out samples; the per-RH columns report each humidity condition separately.

A full sweep contains `24 Vg × 24 candidates × 1 seed = 576` training runs and retains 24 selected models. The archived sweep completed all 576 runs and retains its recorded training settings.

## 4. Inspect the saved results

Run the notebook with `RUN_TRAINING = False`, or load the archived summary directly:

```python
from src.Single_vg_training import (
    load_single_vg_results,
    plot_single_vg_results,
)

archive_root = "results/single_vg"
summary = load_single_vg_results(archive_root)
fig, ax = plot_single_vg_results(summary, metric="rh30")
```

The helper defaults to held-out RH 30%, the unseen humidity condition. The notebook displays per-RH and overall curves separately. Available helper metrics are `"rh30"`, `"rh0"`, `"rh50"`, `"overall"`, and `"validation"`. Use `"overall"` to plot pooled held-out accuracy across all three RH conditions.

Saved accuracies and predictions can be inspected without the sensor dataset or a new training run. Loading an archived selected checkpoint is also supported:

```python
from src.Single_vg_training import load_archived_model

model, metadata = load_archived_model(60, device="cpu")
```

Here `60` is the cropped Vg index. The loader reconstructs the selected candidate's effective model parameters and preserves class indices while exposing the paper's chemical-formula labels. Running inference again requires the sensor data and the matching one-Vg input view.

Saved curves use the archived predictions. CPU re-evaluation can differ slightly from the archived GPU results; the largest observed accuracy difference was 0.0221 percentage points. The default notebook displays the saved evaluations.

## 5. Launch or resume a new sweep

Complete the [environment and dataset setup](../README.md#get-started), open `Single_Vg_training.ipynb`, and inspect the displayed parameter table. Keep the default preview route until you are ready to run training, then set `RUN_TRAINING = True` and run the notebook's training cells.

New training uses the shared preprocessing and supplied dataset. It reuses the same train/validation indices for every run and writes to `results/models/tbdt_single_vg/`. The saved archive remains available separately.

A complete Python invocation from the repository root uses the same data-loading path as the notebook:

```python
from configs.Config_TBDT_Single_Vg import CONFIG
from src import Data_pipeline
from src.Single_vg_training import run_single_vg_sweep

selection = CONFIG["SELECTION"]
preprocess = CONFIG["PREPROCESS"]
bundle = Data_pipeline.prepare_data_bundle(
    data_root=Data_pipeline.resolve_data_root(CONFIG["DATASET"]["data_dir"]),
    train_rh_values=selection["train_rh_values"],
    heldout_test_rh_values=selection["heldout_test_rh_values"],
    average_mode=preprocess["average_mode"],
    cut_window=preprocess["cut_window"],
    val_size=selection["train_val_split"]["val_size"],
    random_state=selection["train_val_split"]["random_state"],
)
result = run_single_vg_sweep(
    CONFIG,
    bundle,
    resume=True,
)
```

The result contains `summary`, `completed_runs`, and `best_by_vg_seed` DataFrames, plus the output-root path. Generated files include:

When only part of the 24-candidate grid has finished, summary rows have `is_complete=False` and plots are labeled **partial candidate sweep**. Their winners are provisional until all candidates at that Vg finish.
With the archived single seed, the associated seed-standard-deviation columns are zero. All archived candidates use the same time window and train/validation split.


