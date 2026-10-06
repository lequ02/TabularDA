# Mix pipeline comparison

Mix training concatenates all real training rows and 100,000 synthetic rows; the ratio varies by dataset. The original-data row is the same real-only baseline. Development and test partitions remain real held-out data.

This report contains 128 distinct completed run records from the selected comparison configurations; it does not establish completion of the full 816-run matrix.

Classification scores are percentages; News reports unscaled R² (which can be negative) and NMAEσ. Adult, Credit, and Census KDD show binary F1 / macro F1; Covertype and Intrusion use macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. Within Adult, Credit, and Census KDD, the two seeds share the same prepared train, development, and test split; they are model-seed repeats, not independent holdouts.

Only corrected runs are shown. Missing configurations remain marked with a dash; completed zero scores are displayed as 0.0%.
MNIST28 and News seed 42 use the separate `corrected_v2_seed42_mnist28_news` namespace. Source paths in the CSV preserve that namespace; all other displayed runs use `corrected_v2`.
News NMAEσ = MAE / σ_y; lower is better. σ_y is the population standard deviation (ddof=0) of the same real held-out test targets used to compute MAE. Seed 42 uses σ_y = 9485.506480005333 over 7,929 rows, verified against the prepared test-table hash. The paper reports News R², not NMAEσ, so its reference lines appear only in the R² panels.
News run records save `test_scores.nmae_sigma` alongside `r2` and `mae`, with `target_normalization` recording σ_y, split, ddof, row count, and target-table hash. These saved results supply the report; normalization is not recomputed during plotting.

$$\mathrm{NMAE}_\sigma = \frac{\mathrm{MAE}}{\sigma_y}.$$

In the figure, triangles mark original data; hollow markers mark generated-target benchmarks. Original-data triangles use teal for binary F1, orange for macro F1, and green for accuracy/R². CTGAN and TVAE have distinct colors within each metric.
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
| TVAE full features + DNN | 68.6% / 78.6% | 68.6% / 78.8% |
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
| TVAE full features + DNN | 75.1% | 70.9% |
| TVAE X-only features + DNN | 73.1% | 73.2% |
| Paper CTGAN (reference) | 32.4% | 32.4% |
| Paper TVAE (reference) | 43.3% | 43.3% |
| Paper Real (reference) | 65.2% | 65.2% |

## MNIST12

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 95.3% | — |
| CTGAN, generated target | 66.5% | 80.7% |
| TVAE, generated target | 94.1% | — |
| CTGAN full features + RF | 92.5% | 92.7% |
| CTGAN full features + XGB | 93.8% | 94.0% |
| CTGAN full features + DNN | 93.7% | — |
| CTGAN X-only features + RF | 92.4% | 92.4% |
| CTGAN X-only features + XGB | 93.5% | 93.6% |
| CTGAN X-only features + DNN | 93.9% | 93.8% |
| TVAE full features + DNN | 94.6% | — |
| TVAE X-only features + DNN | 94.6% | — |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## MNIST28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 97.9% | — |
| CTGAN, generated target | 91.0% | — |
| TVAE, generated target | 96.5% | — |
| CTGAN full features + RF | 95.3% | — |
| CTGAN full features + XGB | 95.7% | — |
| CTGAN full features + DNN | 96.0% | — |
| CTGAN X-only features + RF | 95.7% | — |
| CTGAN X-only features + XGB | 96.3% | — |
| CTGAN X-only features + DNN | 96.2% | — |
| TVAE full features + DNN | 96.6% | — |
| TVAE X-only features + DNN | 96.6% | — |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## Credit

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN, generated target | 17.3% / 58.4% | 20.6% / 60.1% |
| TVAE, generated target | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN full features + RF | 73.7% / 86.8% | 73.7% / 86.8% |
| CTGAN full features + XGB | 77.8% / 88.9% | 66.7% / 83.3% |
| CTGAN full features + DNN | 55.2% / 77.6% | 57.1% / 78.5% |
| CTGAN X-only features + RF | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN X-only features + XGB | 0.0% / 50.0% | 0.0% / 50.0% |
| CTGAN X-only features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE full features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| TVAE X-only features + DNN | 0.0% / 50.0% | 0.0% / 50.0% |
| Paper CTGAN (reference) | 67.2% / — | 67.2% / — |
| Paper TVAE (reference) | 9.8% / — | 9.8% / — |
| Paper Real (reference) | 72.0% / — | 72.0% / — |

## Census KDD

| Configuration | Seed 42 (binary / macro F1) | Seed 43 (binary / macro F1) |
|---|---:|---:|
| Original data only | 40.5% / 69.0% | 47.9% / 72.7% |
| CTGAN, generated target | 58.1% / 77.5% | 37.1% / 67.2% |
| TVAE, generated target | 0.0% / 48.4% | 25.0% / 61.1% |
| CTGAN full features + RF | 14.3% / 55.7% | 0.0% / 48.4% |
| CTGAN full features + XGB | 26.5% / 61.9% | 34.2% / 65.8% |
| CTGAN full features + DNN | 21.8% / 59.5% | 38.6% / 67.9% |
| CTGAN X-only features + RF | 10.7% / 53.9% | 0.0% / 48.4% |
| CTGAN X-only features + XGB | 22.7% / 59.9% | 8.9% / 52.9% |
| CTGAN X-only features + DNN | 28.8% / 63.0% | 2.7% / 49.8% |
| TVAE full features + DNN | 12.2% / 54.6% | 0.0% / 48.4% |
| TVAE X-only features + DNN | 0.0% / 48.4% | 0.0% / 48.4% |
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
| TVAE full features + DNN | — | — |
| TVAE X-only features + DNN | — | — |
| Paper CTGAN (reference) | 52.8% | 52.8% |
| Paper TVAE (reference) | 51.1% | 51.1% |
| Paper Real (reference) | 86.2% | 86.2% |

## News

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | -0.246 | — |
| CTGAN, generated target | 0.022 | — |
| TVAE, generated target | -0.033 | — |
| CTGAN full features + RF | 0.012 | — |
| CTGAN full features + XGB | 0.002 | — |
| CTGAN full features + DNN | 0.033 | — |
| CTGAN X-only features + RF | 0.000 | — |
| CTGAN X-only features + XGB | -0.021 | — |
| CTGAN X-only features + DNN | 0.026 | — |
| TVAE full features + DNN | 0.031 | — |
| TVAE X-only features + DNN | 0.029 | — |
| Paper CTGAN (reference) | -0.430 | -0.430 |
| Paper TVAE (reference) | -0.200 | -0.200 |
| Paper Real (reference) | 0.140 | 0.140 |

## News NMAEσ

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.320 | — |
| CTGAN, generated target | 0.278 | — |
| TVAE, generated target | 0.255 | — |
| CTGAN full features + RF | 0.297 | — |
| CTGAN full features + XGB | 0.316 | — |
| CTGAN full features + DNN | 0.308 | — |
| CTGAN X-only features + RF | 0.303 | — |
| CTGAN X-only features + XGB | 0.288 | — |
| CTGAN X-only features + DNN | 0.327 | — |
| TVAE full features + DNN | 0.307 | — |
| TVAE X-only features + DNN | 0.310 | — |
| Paper CTGAN (reference) | — | — |
| Paper TVAE (reference) | — | — |
| Paper Real (reference) | — | — |


Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

Credit's test split contains 10 positive cases among 9,992 rows. Its binary F1 is sensitive to each positive prediction; 0.0% is a measured result, not a missing run.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes the source run-record path for each score.
