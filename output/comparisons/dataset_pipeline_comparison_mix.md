# Mix pipeline comparison

Mix training concatenates all real training rows and 100,000 synthetic rows; the ratio varies by dataset. The original-data row is the same real-only baseline. Development and test partitions remain real held-out data.

Verified source snapshot: October 10, 2026, 11:58 AM Chicago. This report contains 210 distinct completed run records from the selected configurations; selected report coverage does not establish completion of the original 816-run matrix.

Classification scores are percentages; News and Housing display paired, unscaled R² / D² absolute-error scores. Both are higher-is-better, have a maximum of 1, and can be negative. NMAEσ remains in the CSV. Adult and Census KDD show binary F1 / macro F1; Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. Within Adult and Census KDD, the two seeds share the same prepared train, development, and test split; they are model-seed repeats, not independent holdouts.

Corrected runs are shown for other datasets; News uses the completed fresh log-target rerun. Missing configurations remain marked with a dash; completed zero scores are displayed as 0.0%.
MNIST12/28 use only `mnist_head_fixed_20261009`; Housing uses only `housing_no_faker_20261009`. News uses only `news_log_v1`; Census KDD uses `census_kdd_weighted_macro_f1_20261005`; remaining datasets use `corrected_v2`. Source paths in the CSV preserve each namespace.
This mix report has 60/60 completed MNIST configuration records across both datasets and seeds, including the four real-only baselines. Separately, 212/212 full-matrix MNIST fits are verified, including NB/PCA-GMM configurations excluded from these reports. All planned MNIST fits are complete.
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
| CTGAN, generated target | 62.7% / 76.5% | 67.6% / 78.9% |
| TVAE, generated target | 56.7% / 73.3% | 63.7% / 76.6% |
| CTGAN full features + RF | 60.3% / 75.2% | 68.8% / 79.5% |
| CTGAN full features + XGB | 60.1% / 75.1% | 66.4% / 78.0% |
| CTGAN full features + DNN | 68.6% / 78.2% | 67.7% / 77.8% |
| CTGAN X-only features + RF | 67.3% / 79.0% | 66.2% / 78.1% |
| CTGAN X-only features + XGB | 65.7% / 77.8% | 67.1% / 78.6% |
| CTGAN X-only features + DNN | 68.5% / 78.1% | 68.7% / 78.6% |
| TVAE full features + RF | 66.0% / 78.0% | 62.7% / 76.4% |
| TVAE full features + XGB | 67.3% / 78.5% | 65.4% / 77.7% |
| TVAE full features + DNN | 68.6% / 78.6% | 68.6% / 78.8% |
| TVAE X-only features + RF | 64.7% / 77.4% | 65.7% / 78.1% |
| TVAE X-only features + XGB | 66.5% / 78.4% | 67.2% / 78.8% |
| TVAE X-only features + DNN | 69.0% / 78.7% | 68.9% / 79.1% |
| Paper CTGAN (reference) | 60.1% / — | 60.1% / — |
| Paper TVAE (reference) | 62.6% / — | 62.6% / — |
| Paper Real (reference) | 66.9% / — | 66.9% / — |

## Covertype

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 47.9% | 53.0% |
| CTGAN, generated target | 49.9% | 49.4% |
| TVAE, generated target | 47.9% | 50.0% |
| CTGAN full features + RF | 56.2% | 54.1% |
| CTGAN full features + XGB | 62.5% | 58.8% |
| CTGAN full features + DNN | 78.3% | 75.2% |
| CTGAN X-only features + RF | 58.9% | 56.0% |
| CTGAN X-only features + XGB | 53.2% | 62.6% |
| CTGAN X-only features + DNN | 74.1% | 75.4% |
| TVAE full features + RF | 54.2% | 58.9% |
| TVAE full features + XGB | 60.9% | 58.3% |
| TVAE full features + DNN | 75.1% | 70.9% |
| TVAE X-only features + RF | 53.9% | 54.6% |
| TVAE X-only features + XGB | 54.6% | 56.6% |
| TVAE X-only features + DNN | 73.1% | 73.2% |
| Paper CTGAN (reference) | 32.4% | 32.4% |
| Paper TVAE (reference) | 43.3% | 43.3% |
| Paper Real (reference) | 65.2% | 65.2% |

## MNIST12

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 95.7% | 95.6% |
| CTGAN, generated target | 65.7% | 78.0% |
| TVAE, generated target | 94.3% | 94.4% |
| CTGAN full features + RF | 92.9% | 93.0% |
| CTGAN full features + XGB | 94.0% | 94.1% |
| CTGAN full features + DNN | 93.9% | 94.2% |
| CTGAN X-only features + RF | 92.9% | 93.0% |
| CTGAN X-only features + XGB | 93.9% | 94.3% |
| CTGAN X-only features + DNN | 93.9% | 93.7% |
| TVAE full features + RF | 94.0% | 93.9% |
| TVAE full features + XGB | 94.9% | 94.9% |
| TVAE full features + DNN | 94.7% | 94.8% |
| TVAE X-only features + RF | 94.1% | 93.8% |
| TVAE X-only features + XGB | 94.9% | 94.7% |
| TVAE X-only features + DNN | 94.7% | 94.6% |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## MNIST28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 98.0% | 98.0% |
| CTGAN, generated target | 90.6% | 94.9% |
| TVAE, generated target | 96.5% | 96.4% |
| CTGAN full features + RF | 95.5% | 95.8% |
| CTGAN full features + XGB | 96.0% | 96.8% |
| CTGAN full features + DNN | 95.9% | 96.8% |
| CTGAN X-only features + RF | 96.3% | 95.2% |
| CTGAN X-only features + XGB | 96.4% | 96.0% |
| CTGAN X-only features + DNN | 96.4% | 96.4% |
| TVAE full features + RF | 96.1% | 95.9% |
| TVAE full features + XGB | 96.9% | 96.8% |
| TVAE full features + DNN | 96.6% | 96.7% |
| TVAE X-only features + RF | 95.9% | 96.0% |
| TVAE X-only features + XGB | 96.9% | 96.9% |
| TVAE X-only features + DNN | 96.8% | 96.8% |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## Census KDD

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 55.6% / 75.9% | 54.9% / 75.4% |
| CTGAN, generated target | 36.2% / 62.2% | 53.1% / 74.6% |
| TVAE, generated target | 56.5% / 76.6% | 54.3% / 75.1% |
| CTGAN full features + RF | 57.1% / 77.0% | 56.2% / 76.7% |
| CTGAN full features + XGB | 57.2% / 77.1% | 56.4% / 76.8% |
| CTGAN full features + DNN | 56.4% / 76.7% | 57.2% / 77.2% |
| CTGAN X-only features + RF | 55.7% / 76.4% | 56.0% / 76.5% |
| CTGAN X-only features + XGB | 57.1% / 77.1% | 56.9% / 76.8% |
| CTGAN X-only features + DNN | 56.9% / 77.1% | 57.1% / 77.1% |
| TVAE full features + RF | 55.8% / 76.3% | 55.4% / 76.1% |
| TVAE full features + XGB | 57.9% / 77.4% | 57.7% / 77.4% |
| TVAE full features + DNN | 57.0% / 77.1% | 57.0% / 77.1% |
| TVAE X-only features + RF | 55.2% / 75.9% | 57.1% / 77.2% |
| TVAE X-only features + XGB | 57.4% / 77.3% | 58.0% / 77.6% |
| TVAE X-only features + DNN | 57.5% / 77.4% | 57.2% / 77.2% |
| Paper CTGAN (reference) | 39.1% / — | 39.1% / — |
| Paper TVAE (reference) | 37.7% / — | 37.7% / — |
| Paper Real (reference) | 49.4% / — | 49.4% / — |

## News R² / D²

| Configuration | Seed 42 (R² / D² absolute error) | Seed 43 (R² / D² absolute error) |
|---|---:|---:|
| Original data only | 0.005 / -0.086 | -0.063 / -0.003 |
| CTGAN, generated target | -0.008 / 0.020 | -0.026 / 0.028 |
| TVAE, generated target | -0.022 / 0.036 | -0.022 / 0.036 |
| CTGAN full features + RF | -0.003 / 0.009 | -0.002 / 0.010 |
| CTGAN full features + XGB | -0.004 / 0.013 | 0.000 / 0.005 |
| CTGAN full features + DNN | 0.001 / 0.017 | 0.002 / 0.016 |
| CTGAN X-only features + RF | -0.002 / 0.008 | -0.002 / 0.008 |
| CTGAN X-only features + XGB | 0.002 / -0.009 | 0.003 / 0.008 |
| CTGAN X-only features + DNN | 0.000 / 0.018 | -0.001 / 0.023 |
| TVAE full features + RF | -0.003 / 0.007 | 0.000 / 0.004 |
| TVAE full features + XGB | -0.002 / 0.015 | -0.000 / 0.007 |
| TVAE full features + DNN | 0.002 / 0.007 | -0.001 / 0.016 |
| TVAE X-only features + RF | 0.002 / -0.002 | -0.001 / 0.006 |
| TVAE X-only features + XGB | -0.000 / -0.002 | -0.001 / 0.019 |
| TVAE X-only features + DNN | 0.004 / -0.001 | -0.001 / 0.020 |
| Paper CTGAN (reference) | -0.430 / — | -0.430 / — |
| Paper TVAE (reference) | -0.200 / — | -0.200 / — |
| Paper Real (reference) | 0.140 / — | 0.140 / — |

## Housing R² / D²

| Configuration | Seed 42 (R² / D² absolute error) | Seed 43 (R² / D² absolute error) |
|---|---:|---:|
| Original data only | 0.525 / 0.296 | 0.500 / 0.268 |
| CTGAN, generated target | 0.609 / 0.398 | 0.662 / 0.439 |
| TVAE, generated target | 0.673 / 0.484 | 0.726 / 0.521 |
| CTGAN full features + RF | 0.782 / 0.590 | 0.791 / 0.592 |
| CTGAN full features + XGB | 0.803 / 0.616 | 0.823 / 0.630 |
| CTGAN full features + DNN | 0.801 / 0.612 | 0.774 / 0.607 |
| CTGAN X-only features + RF | 0.790 / 0.589 | 0.774 / 0.587 |
| CTGAN X-only features + XGB | 0.815 / 0.620 | 0.804 / 0.611 |
| CTGAN X-only features + DNN | 0.807 / 0.612 | 0.798 / 0.605 |
| TVAE full features + RF | 0.778 / 0.572 | 0.774 / 0.568 |
| TVAE full features + XGB | 0.749 / 0.612 | 0.775 / 0.618 |
| TVAE full features + DNN | 0.807 / 0.608 | 0.789 / 0.602 |
| TVAE X-only features + RF | 0.759 / 0.553 | 0.762 / 0.558 |
| TVAE X-only features + XGB | 0.801 / 0.600 | 0.799 / 0.612 |
| TVAE X-only features + DNN | 0.789 / 0.604 | 0.786 / 0.597 |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |


Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes source run-record paths, D² sidecar paths, and the retained NMAEσ scores.
