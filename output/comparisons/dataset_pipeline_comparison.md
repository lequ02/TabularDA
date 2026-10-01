# Synthetic-data pipeline comparison

All scores are percentages. Adult, Credit, and Census KDD use binary F1; Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. Within Adult, Credit, and Census KDD, the two seeds share the same prepared train, development, and test split; they are model-seed repeats, not independent holdouts.

Credit seed 43 is partially complete. Missing configurations remain marked with a dash; completed zero scores are displayed as 0.0%.

## Pilot · seed 42

| Configuration | Adult | Covertype | MNIST28 |
|---|---:|---:|---:|
| Original data only | 66.9% | 47.9% | 97.9% |
| CTGAN, generated target | 59.7% | 44.6% | 51.5% |
| TVAE, generated target | — | — | — |
| CTGAN full features + RF | 62.3% | 54.1% | 85.9% |
| CTGAN full features + XGB | 62.3% | 59.8% | 87.2% |
| CTGAN X-only features + RF | 63.7% | 52.2% | 88.5% |
| CTGAN X-only features + XGB | 66.2% | 56.2% | 88.7% |
| CTGAN full features + DNN | 68.8% | 72.9% | 90.9% |
| CTGAN X-only features + DNN | 68.6% | 68.9% | 91.8% |
| TVAE full features + DNN | — | — | — |
| TVAE X-only features + DNN | — | — | — |
| Paper CTGAN (reference) | 60.1% | 32.4% | 37.1% |
| Paper TVAE (reference) | 62.6% | 43.3% | 79.4% |
| Paper Real (reference) | 66.9% | 65.2% | 91.6% |

## Corrected · seed 42

| Configuration | Adult | Covertype | MNIST12 | Credit | Census KDD |
|---|---:|---:|---:|---:|---:|
| Original data only | 66.9% | 47.9% | 95.3% | 0.0% | 40.5% |
| CTGAN, generated target | 59.7% | 44.6% | 51.4% | 10.4% | — |
| TVAE, generated target | 59.1% | 43.3% | 92.9% | 0.0% | — |
| CTGAN full features + RF | 62.3% | 54.1% | 89.5% | 73.7% | — |
| CTGAN full features + XGB | 62.3% | 59.8% | 90.8% | 80.0% | — |
| CTGAN X-only features + RF | 64.8% | 55.6% | 88.5% | 0.0% | — |
| CTGAN X-only features + XGB | 66.2% | 56.2% | 89.7% | 0.0% | — |
| CTGAN full features + DNN | 68.8% | 72.9% | 91.7% | 51.6% | — |
| CTGAN X-only features + DNN | 68.6% | 68.9% | 91.2% | 0.0% | — |
| TVAE full features + DNN | 68.2% | 66.1% | 93.9% | 0.0% | — |
| TVAE X-only features + DNN | 68.7% | 62.2% | 93.7% | 0.0% | — |
| Paper CTGAN (reference) | 60.1% | 32.4% | 39.4% | 67.2% | 39.1% |
| Paper TVAE (reference) | 62.6% | 43.3% | 79.3% | 9.8% | 37.7% |
| Paper Real (reference) | 66.9% | 65.2% | 88.6% | 72.0% | 49.4% |

## Corrected · seed 43

| Configuration | Adult | Credit | Census KDD |
|---|---:|---:|---:|
| Original data only | 68.0% | — | 47.9% |
| CTGAN, generated target | 63.7% | 9.3% | — |
| TVAE, generated target | 54.4% | 0.0% | — |
| CTGAN full features + RF | 64.1% | 77.8% | — |
| CTGAN full features + XGB | 65.6% | 84.2% | — |
| CTGAN X-only features + RF | 66.5% | 0.0% | — |
| CTGAN X-only features + XGB | 66.7% | 0.0% | — |
| CTGAN full features + DNN | 67.7% | 38.1% | — |
| CTGAN X-only features + DNN | 68.7% | 0.0% | — |
| TVAE full features + DNN | 68.0% | — | — |
| TVAE X-only features + DNN | 68.4% | 0.0% | — |
| Paper CTGAN (reference) | 60.1% | 67.2% | 39.1% |
| Paper TVAE (reference) | 62.6% | 9.8% | 37.7% |
| Paper Real (reference) | 66.9% | 72.0% | 49.4% |


Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

Credit's test split contains 10 positive cases among 9,992 rows. Its binary F1 is sensitive to each positive prediction; 0.0% is a measured result, not a missing run.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes the source run-record path for each score.
