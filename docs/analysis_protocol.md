# Analysis preprocessing and MI coordinates

MI and directional dSNR use the same complete cohort at **RH 0% and 50%**,
combining training-folder observations (both model-training and validation)
and held-out chips. The published dataset has **44,869** observations after
the classifier's device screening; raw-current quality control (QC) excludes **557**, leaving
**44,312** observations for both metrics. No per-gas subsampling is applied.

## Raw-current quality checks

Checks use aligned current before response normalization. A sample is retained
only if all 96 selected Vg channels satisfy all checks:

- Absolute reference current at raw time index 14 is at least **1 nA**.
- All 15 baseline readings (raw indices 0–14) and all analysis readings are finite.
- Baseline SNR is at least **5**: absolute reference current divided by
  `1.4826 × MAD` of residuals from a linear baseline fit, where MAD is the
  median absolute deviation.
- Extrapolated drift is no more than **1 × absolute reference current**:
  `abs(baseline slope) × (55 − 14) / abs(reference current) ≤ 1`.

Passing responses are recomputed from raw current as `I/I_reference − 1`,
without replacing the denominator with a floor, then corrected using the
linear baseline fit. The same retained samples and all 42 × 96 coordinates
are used throughout the metric calculations. This analysis filtering does
not change the classifier's inputs, training, held-out accuracy, or SHAP samples.

## MI calculations

The scalar heatmap evaluates every one of the **42 × 96** coordinates using
`mutual_info_classif`, with 5 neighbors and random seed 0. Both joint curves
use the Chebyshev k-nearest-neighbor entropy estimator, k = 5, seed 0,
without standardization. The seeded jitter follows NumPy's MT19937 sequence.
All three calculations use the same 44,312 samples and class labels.

All indices below are **zero-based within the cropped analysis input**.

### MI versus Vg: ten time features at each of the 96 Vg coordinates

| Time index | Raw time index | Time (min) |
| --- | --- | --- |
| 0 | 14 | −1.0 |
| 1 | 15 | −0.5 |
| 2 | 16 | 0.0 |
| 3 | 17 | 0.5 |
| 5 | 19 | 1.5 |
| 8 | 22 | 3.0 |
| 12 | 26 | 5.0 |
| 18 | 32 | 8.0 |
| 25 | 39 | 11.5 |
| 35 | 49 | 16.5 |

### MI versus time: six Vg features at each of the 42 time coordinates

| Vg index | Aligned, uncropped Vg index | Vg (V) |
| --- | --- | --- |
| 0 | 11 | −1.41 |
| 19 | 30 | −0.84 |
| 38 | 49 | −0.27 |
| 57 | 68 | 0.30 |
| 76 | 87 | 0.87 |
| 95 | 106 | 1.44 |

The displayed Vg values are rounded to two decimals; calculation and exported
coordinate tables retain the original float32 coordinates. Vg alignment is
performed per device, so raw-file Vg indices also depend on its threshold shift.
The complete output axes are Vg −1.41 to 1.44 V in 0.03 V steps and time −1 to
19.5 min in 0.5 min steps.

Section 3 of `Analysis.ipynb` keeps its existing heatmap and Vg-profile figure.
Time-wise MI is saved in `mutual_information.npz` without an additional figure.
The output folder also contains `mi_time_coordinates.csv`, `mi_vg_coordinates.csv`,
`mi_sample_metadata.csv`, the raw-current QC tables, and `metric_quality_summary.json`.
