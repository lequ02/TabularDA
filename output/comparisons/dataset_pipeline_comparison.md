# Synthetic-data pipeline comparison

Verified source snapshot: October 10, 2026, 11:58 AM Chicago. This report contains 210 distinct completed run records from the selected configurations; selected report coverage does not establish completion of the original 816-run matrix.

Classification scores are percentages; News and Housing display paired, unscaled R² / D² absolute-error scores. Both are higher-is-better, have a maximum of 1, and can be negative. NMAEσ remains in the CSV. Adult and Census KDD show binary F1 / macro F1; Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. Within Adult and Census KDD, the two seeds share the same prepared train, development, and test split; they are model-seed repeats, not independent holdouts.

Corrected runs are shown for other datasets; News uses the completed fresh log-target rerun. Missing configurations remain marked with a dash; completed zero scores are displayed as 0.0%.
MNIST12/28 use only `mnist_head_fixed_20261009`; Housing uses only `housing_no_faker_20261009`. News uses only `news_log_v1`; Census KDD uses `census_kdd_weighted_macro_f1_20261005`; remaining datasets use `corrected_v2`. Source paths in the CSV preserve each namespace.
This synthetic report has 60/60 completed MNIST configuration records across both datasets and seeds, including the four real-only baselines. Separately, 212/212 full-matrix MNIST fits are verified, including NB/PCA-GMM configurations excluded from these reports. All planned MNIST fits are complete.
The News rerun has 74 completed downstream runs across seeds 42 and 43. These reports select RF/XGB/DNN and generated-target arms, with all 15 configurations available for each seed and training mode. Every News downstream model is freshly fitted; no pilot or historical raw-target News records are reused. Full-table generators are fitted on log targets; features-only generators are reused only with verified provenance, and all labelers are fitted afresh on real training log targets.
The News rerun trains MSE on log(shares), uses no BatchNorm or LayerNorm, and selects checkpoints by raw-scale real-development MSE. Predictions are transformed back with exp before final metrics are computed in shares. Log training, normalization removal, and full-table generator/labeler changes are combined changes; comparisons with historical runs do not isolate their individual effects.
Census KDD uses class-weighted BCEWithLogitsLoss (positive weight = actual training negatives / positives) and development macro-F1 checkpoint selection, with threshold 0.5. Both loss and checkpoint selection changed from the earlier evaluation; improvements cannot be attributed to weighting alone. Pending weighted configurations remain missing rather than using earlier unweighted results.
News and Housing NMAEσ = MAE / σ_y; lower is better. σ_y is the population standard deviation (ddof=0) of the same real held-out test targets used to compute MAE. Both News seeds use σ_y = 9485.506480005333 over 7,929 rows, verified against the prepared test-table hash. The paper reports News R², not NMAEσ, so its reference lines appear only in the News R² panels. No paper reference is supplied for California Housing.
D² absolute error = 1 − MAE / MAE of a constant test-median prediction. Its zero benchmark is the test median; R² uses the test mean. D² uses absolute errors and R² uses squared errors. The two scores share a plotting axis but measure different prediction errors. Paper lines in the combined regression panels refer to R² only; no D² references are supplied.
Housing now uses all 58 completed corrected rerun fits, including fresh real-only baselines. Its eight fresh generators retain Latitude and Longitude as learned numerical features and use no Faker transformers. Earlier coordinate-flawed generators and scores remain historical diagnostic evidence and do not supply this report. See the [Housing rerun](../../audit/HOUSING_FAKER_RERUN_2026_10_09.md).
MNIST now uses the repaired downstream evaluator: both forward methods apply their declared ten-class output layers. Every displayed rerun model is freshly trained. Pending configurations remain blank; earlier models that bypassed those layers never fill missing cells. The full matrix plans 212 fits, with 116 report configurations first and 96 NB/PCA-GMM fits last. NB/PCA-GMM remain excluded from these reports. The retained MNIST12 TVAE artifacts still carry the documented held-out feature-collision caveat; the downstream repair does not remove it. See [the MNIST rerun](../../audit/mnist_head_rerun_20261009/README.md).
News uses a symmetric-log axis with a linear region from −0.001 to 0.001 to spread scores clustered near zero while retaining negative values. Other regression pairs use a symmetric-log axis with a linear region from −0.1 to 0.1 only if their range extends below −1. Tick labels and reported R²/D² values remain unscaled; both seeds share the same axis scale.
News and Housing run records save `test_scores.nmae_sigma` alongside `r2` and `mae`, with `target_normalization` recording σ_y, split, ddof, row count, and target-table hash. These saved results supply the report; normalization is not recomputed during plotting.
D² is computed remotely from each run's saved held-out predictions without retraining and stored in a `.d2.json` sidecar. The sidecar saves the test median, median-baseline MAE, prediction MAE, row count, split and target-table/prediction/run-record hashes. Original records and NMAEσ are preserved. The builder reads and verifies sidecars; it does not recompute D² from test data.

$$\mathrm{NMAE}_\sigma = \frac{\mathrm{MAE}}{\sigma_y}.$$

In the figure, triangles mark original data; hollow markers mark generated-target benchmarks. Original-data triangles use teal for binary F1, orange for macro F1, green for accuracy/R², and blue for D². CTGAN and TVAE have distinct colors within each metric.
Macro F1 averages the F1 scores of both classes. Paper references for binary datasets are shown only under binary F1; no matching paper macro F1 reference is supplied.

## Adult

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 66.9% / 78.9% | 68.0% / 79.4% |
| CTGAN, generated target | 59.7% / 74.7% | 63.7% / 76.8% |
| TVAE, generated target | 59.1% / 74.1% | 54.4% / 71.7% |
| CTGAN full features + RF | 62.3% / 76.1% | 64.1% / 77.1% |
| CTGAN full features + XGB | 62.3% / 76.2% | 65.6% / 77.7% |
| CTGAN full features + DNN | 68.8% / 78.4% | 67.7% / 77.9% |
| CTGAN X-only features + RF | 64.8% / 77.5% | 66.5% / 78.3% |
| CTGAN X-only features + XGB | 66.2% / 77.9% | 66.7% / 78.3% |
| CTGAN X-only features + DNN | 68.6% / 78.4% | 68.7% / 78.7% |
| TVAE full features + RF | 65.9% / 78.1% | 62.7% / 76.3% |
| TVAE full features + XGB | 67.3% / 78.6% | 65.2% / 77.3% |
| TVAE full features + DNN | 68.2% / 78.2% | 68.0% / 78.5% |
| TVAE X-only features + RF | 65.7% / 77.8% | 66.0% / 78.1% |
| TVAE X-only features + XGB | 66.8% / 78.4% | 67.1% / 78.6% |
| TVAE X-only features + DNN | 68.7% / 78.3% | 68.4% / 78.5% |
| Paper CTGAN (reference) | 60.1% / — | 60.1% / — |
| Paper TVAE (reference) | 62.6% / — | 62.6% / — |
| Paper Real (reference) | 66.9% / — | 66.9% / — |

## Covertype

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 47.9% | 53.0% |
| CTGAN, generated target | 44.6% | 44.9% |
| TVAE, generated target | 43.3% | 43.6% |
| CTGAN full features + RF | 54.1% | 56.9% |
| CTGAN full features + XGB | 59.8% | 50.5% |
| CTGAN full features + DNN | 72.9% | 69.6% |
| CTGAN X-only features + RF | 55.6% | 52.7% |
| CTGAN X-only features + XGB | 56.2% | 44.8% |
| CTGAN X-only features + DNN | 68.9% | 69.2% |
| TVAE full features + RF | 45.1% | 50.5% |
| TVAE full features + XGB | 42.9% | 49.2% |
| TVAE full features + DNN | 66.1% | 63.8% |
| TVAE X-only features + RF | 47.4% | 45.0% |
| TVAE X-only features + XGB | 45.7% | 45.5% |
| TVAE X-only features + DNN | 62.2% | 65.7% |
| Paper CTGAN (reference) | 32.4% | 32.4% |
| Paper TVAE (reference) | 43.3% | 43.3% |
| Paper Real (reference) | 65.2% | 65.2% |

## MNIST12

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 95.7% | 95.6% |
| CTGAN, generated target | 50.9% | 57.1% |
| TVAE, generated target | 93.0% | 92.7% |
| CTGAN full features + RF | 90.0% | 90.1% |
| CTGAN full features + XGB | 91.2% | 91.5% |
| CTGAN full features + DNN | 91.9% | 92.2% |
| CTGAN X-only features + RF | 88.9% | 88.8% |
| CTGAN X-only features + XGB | 90.1% | 90.5% |
| CTGAN X-only features + DNN | 91.7% | 90.5% |
| TVAE full features + RF | 93.2% | 93.1% |
| TVAE full features + XGB | 94.2% | 94.1% |
| TVAE full features + DNN | 94.1% | 94.0% |
| TVAE X-only features + RF | 92.9% | 92.8% |
| TVAE X-only features + XGB | 94.2% | 94.2% |
| TVAE X-only features + DNN | 93.9% | 93.7% |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## MNIST28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 98.0% | 98.0% |
| CTGAN, generated target | 52.0% | 62.6% |
| TVAE, generated target | 94.1% | 94.3% |
| CTGAN full features + RF | 86.3% | 88.4% |
| CTGAN full features + XGB | 87.5% | 89.4% |
| CTGAN full features + DNN | 90.9% | 93.0% |
| CTGAN X-only features + RF | 88.4% | 87.5% |
| CTGAN X-only features + XGB | 89.1% | 88.6% |
| CTGAN X-only features + DNN | 92.0% | 91.9% |
| TVAE full features + RF | 94.6% | 94.6% |
| TVAE full features + XGB | 96.0% | 95.8% |
| TVAE full features + DNN | 95.8% | 95.9% |
| TVAE X-only features + RF | 94.4% | 94.7% |
| TVAE X-only features + XGB | 95.9% | 95.9% |
| TVAE X-only features + DNN | 95.8% | 96.1% |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## Census KDD

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 55.6% / 75.9% | 54.9% / 75.4% |
| CTGAN, generated target | 43.5% / 68.2% | 51.4% / 73.9% |
| TVAE, generated target | 47.0% / 71.1% | 47.0% / 71.3% |
| CTGAN full features + RF | 52.1% / 74.3% | 49.7% / 73.2% |
| CTGAN full features + XGB | 55.0% / 75.9% | 54.1% / 75.2% |
| CTGAN full features + DNN | 54.2% / 75.6% | 53.9% / 75.4% |
| CTGAN X-only features + RF | 53.0% / 74.8% | 52.5% / 74.5% |
| CTGAN X-only features + XGB | 53.6% / 75.1% | 54.9% / 75.8% |
| CTGAN X-only features + DNN | 54.2% / 75.7% | 54.6% / 75.7% |
| TVAE full features + RF | 49.1% / 72.4% | 50.2% / 73.2% |
| TVAE full features + XGB | 50.9% / 73.7% | 53.2% / 74.9% |
| TVAE full features + DNN | 53.3% / 75.1% | 52.7% / 74.8% |
| TVAE X-only features + RF | 48.4% / 72.0% | 48.7% / 72.4% |
| TVAE X-only features + XGB | 49.7% / 73.0% | 50.2% / 73.2% |
| TVAE X-only features + DNN | 52.6% / 74.6% | 53.1% / 74.9% |
| Paper CTGAN (reference) | 39.1% / — | 39.1% / — |
| Paper TVAE (reference) | 37.7% / — | 37.7% / — |
| Paper Real (reference) | 49.4% / — | 49.4% / — |

## News R² / D²

| Configuration | Seed 42 (R² / D² absolute error) | Seed 43 (R² / D² absolute error) |
|---|---:|---:|
| Original data only | 0.005 / -0.086 | -0.063 / -0.003 |
| CTGAN, generated target | -0.014 / 0.010 | -0.025 / 0.020 |
| TVAE, generated target | -0.029 / 0.004 | -0.026 / 0.005 |
| CTGAN full features + RF | -0.005 / -0.004 | -0.001 / -0.008 |
| CTGAN full features + XGB | -0.013 / 0.016 | -0.000 / -0.002 |
| CTGAN full features + DNN | 0.002 / 0.006 | 0.000 / 0.010 |
| CTGAN X-only features + RF | -0.000 / 0.001 | -0.001 / -0.002 |
| CTGAN X-only features + XGB | 0.000 / -0.013 | -0.000 / -0.008 |
| CTGAN X-only features + DNN | 0.002 / 0.006 | 0.002 / 0.008 |
| TVAE full features + RF | 0.002 / -0.007 | -0.001 / -0.005 |
| TVAE full features + XGB | 0.000 / -0.011 | -0.005 / -0.010 |
| TVAE full features + DNN | 0.013 / -0.029 | 0.003 / -0.027 |
| TVAE X-only features + RF | 0.003 / -0.005 | 0.002 / -0.001 |
| TVAE X-only features + XGB | 0.004 / -0.045 | 0.000 / -0.019 |
| TVAE X-only features + DNN | 0.011 / -0.030 | 0.009 / -0.011 |
| Paper CTGAN (reference) | -0.430 / — | -0.430 / — |
| Paper TVAE (reference) | -0.200 / — | -0.200 / — |
| Paper Real (reference) | 0.140 / — | 0.140 / — |

## Housing R² / D²

| Configuration | Seed 42 (R² / D² absolute error) | Seed 43 (R² / D² absolute error) |
|---|---:|---:|
| Original data only | 0.525 / 0.296 | 0.500 / 0.268 |
| CTGAN, generated target | 0.608 / 0.391 | 0.655 / 0.436 |
| TVAE, generated target | 0.700 / 0.490 | 0.729 / 0.521 |
| CTGAN full features + RF | 0.788 / 0.599 | 0.791 / 0.599 |
| CTGAN full features + XGB | 0.813 / 0.626 | 0.821 / 0.628 |
| CTGAN full features + DNN | 0.801 / 0.612 | 0.799 / 0.607 |
| CTGAN X-only features + RF | 0.785 / 0.577 | 0.795 / 0.595 |
| CTGAN X-only features + XGB | 0.822 / 0.623 | 0.817 / 0.620 |
| CTGAN X-only features + DNN | 0.807 / 0.610 | 0.801 / 0.605 |
| TVAE full features + RF | 0.785 / 0.577 | 0.784 / 0.584 |
| TVAE full features + XGB | 0.821 / 0.625 | 0.821 / 0.629 |
| TVAE full features + DNN | 0.698 / 0.584 | 0.794 / 0.604 |
| TVAE X-only features + RF | 0.773 / 0.567 | 0.785 / 0.580 |
| TVAE X-only features + XGB | 0.815 / 0.621 | 0.820 / 0.622 |
| TVAE X-only features + DNN | 0.729 / 0.609 | 0.786 / 0.603 |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |


Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes source run-record paths, D² sidecar paths, and the retained NMAEσ scores.
