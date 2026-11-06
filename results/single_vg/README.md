# Saved single-Vg results

The tables report 576 completed candidates: 24 gate-voltage channels × 24 parameter sets.
One checkpoint per Vg was selected by validation accuracy. All accuracies are percentages.

- `single_vg_summary.csv` and `best_by_vg_seed.csv`: the 24 selected results.
- `completed_runs.csv`: all candidate results and effective hyperparameters.
- `selected_models/vXXX/`: the selected models, configurations, predictions, and evaluation tables.
- `archive_manifest.json`: artifact checksums.

The integer class order is stored in each model's `gas_label_map.json`. Validation and held-out predictions are stored separately. The held-out metrics include RH 0%, 30%, and 50%, with separate columns for each humidity.

Open [Single_Vg_training.ipynb](../../Single_Vg_training.ipynb) with `RUN_TRAINING = False` to display these results without downloading the sensor dataset. See the [single-Vg guide](../../docs/single_vg_training.md) for checkpoint loading and new training.
