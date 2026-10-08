# Synthetic-data pipeline comparison

Verified source snapshot: October 07, 2026, 11:41 PM Chicago. This report contains 193 distinct completed run records from the selected configurations; it does not establish completion of the original 816-run matrix or the added Housing study.

Classification scores are percentages; News and Housing report unscaled R² (which can be negative) and NMAEσ. Adult, Credit, and Census KDD show binary F1 / macro F1; Covertype and Intrusion use macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. Within Adult, Credit, and Census KDD, the two seeds share the same prepared train, development, and test split; they are model-seed repeats, not independent holdouts.

Only corrected runs are shown. Missing configurations remain marked with a dash; completed zero scores are displayed as 0.0%.
MNIST28 and News seed 42 use the separate `corrected_v2_seed42_mnist28_news` namespace. Census KDD uses `census_kdd_weighted_macro_f1_20261005`; remaining datasets use `corrected_v2`. Source paths in the CSV preserve each namespace.
Census KDD uses class-weighted BCEWithLogitsLoss (positive weight = actual training negatives / positives) and development macro-F1 checkpoint selection, with threshold 0.5. Both loss and checkpoint selection changed from the earlier evaluation; improvements cannot be attributed to weighting alone. Pending weighted configurations remain missing rather than using earlier unweighted results.
News and Housing NMAEσ = MAE / σ_y; lower is better. σ_y is the population standard deviation (ddof=0) of the same real held-out test targets used to compute MAE. News seed 42 uses σ_y = 9485.506480005333 over 7,929 rows, verified against the prepared test-table hash. The paper reports News R², not NMAEσ, so its reference lines appear only in the News R² panels. No paper reference is supplied for California Housing.
When a large negative R² expands a paired plot beyond −1, that pair uses a symmetric-log axis with a linear region from −0.1 to 0.1. Tick labels and reported R² values remain unscaled; the axis keeps the extreme result visible while separating the other scores and paper references.
News and Housing run records save `test_scores.nmae_sigma` alongside `r2` and `mae`, with `target_normalization` recording σ_y, split, ddof, row count, and target-table hash. These saved results supply the report; normalization is not recomputed during plotting.

$$\mathrm{NMAE}_\sigma = \frac{\mathrm{MAE}}{\sigma_y}.$$

In the figure, triangles mark original data; hollow markers mark generated-target benchmarks. Original-data triangles use teal for binary F1, orange for macro F1, and green for accuracy/R². CTGAN and TVAE have distinct colors within each metric.
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
| Original data only | 95.3% | 95.5% |
| CTGAN, generated target | 51.4% | 55.8% |
| TVAE, generated target | 92.9% | 92.6% |
| CTGAN full features + RF | 89.5% | 89.6% |
| CTGAN full features + XGB | 90.8% | 90.6% |
| CTGAN full features + DNN | 91.7% | 91.7% |
| CTGAN X-only features + RF | 88.5% | 88.3% |
| CTGAN X-only features + XGB | 89.7% | 89.3% |
| CTGAN X-only features + DNN | 91.2% | 90.3% |
| TVAE full features + RF | 92.8% | 92.5% |
| TVAE full features + XGB | 93.9% | 93.8% |
| TVAE full features + DNN | 93.9% | 93.7% |
| TVAE X-only features + RF | 92.7% | 92.4% |
| TVAE X-only features + XGB | 93.8% | 93.7% |
| TVAE X-only features + DNN | 93.7% | 93.3% |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## MNIST28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 97.9% | 97.9% |
| CTGAN, generated target | 51.5% | — |
| TVAE, generated target | 94.3% | — |
| CTGAN full features + RF | 85.9% | — |
| CTGAN full features + XGB | 87.2% | — |
| CTGAN full features + DNN | 90.9% | — |
| CTGAN X-only features + RF | 88.7% | — |
| CTGAN X-only features + XGB | 89.3% | — |
| CTGAN X-only features + DNN | 92.4% | — |
| TVAE full features + RF | 94.8% | — |
| TVAE full features + XGB | 96.1% | — |
| TVAE full features + DNN | 95.9% | — |
| TVAE X-only features + RF | 95.0% | — |
| TVAE X-only features + XGB | 96.0% | — |
| TVAE X-only features + DNN | 96.0% | — |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## Credit

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN, generated target | 10.4% / 54.8% | 9.3% / 54.1% |
| TVAE, generated target | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN full features + RF | 73.7% / 86.8% | 77.8% / 88.9% |
| CTGAN full features + XGB | 80.0% / 90.0% | 84.2% / 92.1% |
| CTGAN full features + DNN | 51.6% / 75.8% | 38.1% / 69.0% |
| CTGAN X-only features + RF | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN X-only features + XGB | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN X-only features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE full features + RF | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE full features + XGB | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE full features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE X-only features + RF | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE X-only features + XGB | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE X-only features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| Paper CTGAN (reference) | 67.2% / — | 67.2% / — |
| Paper TVAE (reference) | 9.8% / — | 9.8% / — |
| Paper Real (reference) | 72.0% / — | 72.0% / — |

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

## Intrusion

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | — | 20.9% |
| CTGAN, generated target | — | — |
| TVAE, generated target | — | — |
| CTGAN full features + RF | — | — |
| CTGAN full features + XGB | — | — |
| CTGAN full features + DNN | — | — |
| CTGAN X-only features + RF | — | — |
| CTGAN X-only features + XGB | — | — |
| CTGAN X-only features + DNN | — | — |
| TVAE full features + RF | — | — |
| TVAE full features + XGB | — | — |
| TVAE full features + DNN | — | — |
| TVAE X-only features + RF | — | — |
| TVAE X-only features + XGB | — | — |
| TVAE X-only features + DNN | — | — |
| Paper CTGAN (reference) | 52.8% | 52.8% |
| Paper TVAE (reference) | 51.1% | 51.1% |
| Paper Real (reference) | 86.2% | 86.2% |

## News

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | -0.246 | -0.002 |
| CTGAN, generated target | 0.017 | -93.089 |
| TVAE, generated target | -0.076 | -0.028 |
| CTGAN full features + RF | -0.010 | -0.008 |
| CTGAN full features + XGB | -0.008 | -0.045 |
| CTGAN full features + DNN | 0.028 | 0.027 |
| CTGAN X-only features + RF | -0.024 | — |
| CTGAN X-only features + XGB | -0.025 | — |
| CTGAN X-only features + DNN | 0.029 | — |
| TVAE full features + RF | 0.002 | -0.005 |
| TVAE full features + XGB | -0.019 | -0.025 |
| TVAE full features + DNN | 0.031 | 0.030 |
| TVAE X-only features + RF | 0.006 | — |
| TVAE X-only features + XGB | -0.049 | — |
| TVAE X-only features + DNN | 0.024 | — |
| Paper CTGAN (reference) | -0.430 | -0.430 |
| Paper TVAE (reference) | -0.200 | -0.200 |
| Paper Real (reference) | 0.140 | 0.140 |

## Housing R²

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.525 | 0.500 |
| CTGAN, generated target | — | — |
| TVAE, generated target | — | — |
| CTGAN full features + RF | — | — |
| CTGAN full features + XGB | — | — |
| CTGAN full features + DNN | — | — |
| CTGAN X-only features + RF | — | — |
| CTGAN X-only features + XGB | — | — |
| CTGAN X-only features + DNN | — | — |
| TVAE full features + RF | — | — |
| TVAE full features + XGB | — | — |
| TVAE full features + DNN | — | — |
| TVAE X-only features + RF | — | — |
| TVAE X-only features + XGB | — | — |
| TVAE X-only features + DNN | — | — |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |

## News NMAEσ

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.320 | 0.317 |
| CTGAN, generated target | 0.275 | 0.407 |
| TVAE, generated target | 0.309 | 0.261 |
| CTGAN full features + RF | 0.294 | 0.304 |
| CTGAN full features + XGB | 0.282 | 0.288 |
| CTGAN full features + DNN | 0.324 | 0.298 |
| CTGAN X-only features + RF | 0.306 | — |
| CTGAN X-only features + XGB | 0.320 | — |
| CTGAN X-only features + DNN | 0.332 | — |
| TVAE full features + RF | 0.298 | 0.299 |
| TVAE full features + XGB | 0.306 | 0.337 |
| TVAE full features + DNN | 0.323 | 0.308 |
| TVAE X-only features + RF | 0.308 | — |
| TVAE X-only features + XGB | 0.308 | — |
| TVAE X-only features + DNN | 0.343 | — |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |

## Housing NMAEσ

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.540 | 0.561 |
| CTGAN, generated target | — | — |
| TVAE, generated target | — | — |
| CTGAN full features + RF | — | — |
| CTGAN full features + XGB | — | — |
| CTGAN full features + DNN | — | — |
| CTGAN X-only features + RF | — | — |
| CTGAN X-only features + XGB | — | — |
| CTGAN X-only features + DNN | — | — |
| TVAE full features + RF | — | — |
| TVAE full features + XGB | — | — |
| TVAE full features + DNN | — | — |
| TVAE X-only features + RF | — | — |
| TVAE X-only features + XGB | — | — |
| TVAE X-only features + DNN | — | — |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |


Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

Credit's test split contains 10 positive cases among 9,992 rows. Its binary F1 is sensitive to each positive prediction; 0.0% is a measured result, not a missing run.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes the source run-record path for each score.
