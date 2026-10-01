# Synthetic-data pipeline comparison

All scores are percentages. Adult, Credit, and Census KDD use binary F1; Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. Seed 42 is used unless the column says seed 43. The two seeds share the same prepared train, development, and test split.

| Configuration | Adult · pilot, seed 42 | Adult · corrected, seed 42 | Adult · corrected, seed 43 | Covertype · pilot | Covertype · corrected | MNIST28 · pilot | MNIST12 · corrected | Credit · corrected, seed 42 | Census KDD · corrected, seed 42 | Census KDD · corrected, seed 43 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original data only | 66.9% | 66.9% | 68.0% | 47.9% | 47.9% | 97.9% | 95.3% | 0.0% | 40.5% | 47.9% |
| CTGAN, generated target | 59.7% | 59.7% | 63.7% | 44.6% | 44.6% | 51.5% | 51.4% | 10.4% | — | — |
| TVAE, generated target | — | 59.1% | 54.4% | — | 43.3% | — | 92.9% | 0.0% | — | — |
| CTGAN full features + RF | 62.3% | 62.3% | 64.1% | 54.1% | 54.1% | 85.9% | 89.5% | 73.7% | — | — |
| CTGAN full features + XGB | 62.3% | 62.3% | 65.6% | 59.8% | 59.8% | 87.2% | 90.8% | 80.0% | — | — |
| CTGAN X-only features + RF | 63.7% | 64.8% | 66.5% | 52.2% | 55.6% | 88.5% | 88.5% | 0.0% | — | — |
| CTGAN X-only features + XGB | 66.2% | 66.2% | 66.7% | 56.2% | 56.2% | 88.7% | 89.7% | 0.0% | — | — |
| CTGAN full features + DNN | 68.8% | 68.8% | 67.7% | 72.9% | 72.9% | 90.9% | 91.7% | 51.6% | — | — |
| CTGAN X-only features + DNN | 68.6% | 68.6% | 68.7% | 68.9% | 68.9% | 91.8% | 91.2% | 0.0% | — | — |
| TVAE full features + DNN | — | 68.2% | 68.0% | — | 66.1% | — | 93.9% | 0.0% | — | — |
| TVAE X-only features + DNN | — | 68.7% | 68.4% | — | 62.2% | — | 93.7% | 0.0% | — | — |
| Paper CTGAN (reference) | 60.1% | 60.1% | 60.1% | 32.4% | 32.4% | 37.1% | 39.4% | 67.2% | 39.1% | 39.1% |
| Paper TVAE (reference) | 62.6% | 62.6% | 62.6% | 43.3% | 43.3% | 79.4% | 79.3% | 9.8% | 37.7% | 37.7% |
| Paper Real (reference) | 66.9% | 66.9% | 66.9% | 65.2% | 65.2% | 91.6% | 88.6% | 72.0% | 49.4% | 49.4% |

Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "TGAN". Its scores use different splits and average multiple downstream classifiers.

Credit's test split contains 10 positive cases among 9,992 rows. Its binary F1 is sensitive to each positive prediction; 0.0% is a measured result, not a missing run.

The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. Prepared and synthetic CSVs are not present locally, so row-level leakage checks cannot be independently repeated here.

The CSV alongside this table includes the source run-record path for each score.
