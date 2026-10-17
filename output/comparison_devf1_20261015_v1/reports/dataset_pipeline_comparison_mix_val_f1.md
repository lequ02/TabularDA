# Mix training: development F1 / R² checkpoint selection

Status: 446/602 distinct source runs re-evaluated. Pending cells are blank; these tables are provisional.

Checkpoint rule: highest real-development binary F1 for Adult; highest macro F1 for Census KDD, Covertype, MNIST12 and MNIST28; highest raw-scale development R² for News and Housing. Ties select the earliest epoch. Selection is restricted to each original completed history; training budgets, splits, thresholds, synthetic tables and losses are preserved.

Every displayed test result was evaluated again on the server. MNIST panels continue to display test accuracy; binary and regression panels retain both displayed metrics. News and Housing select the same checkpoints under R² and squared error. Census retains its weighted loss. Mix concatenates all original training rows and 100,000 synthetic rows.

Sources: G:/SummerResearch_Retained_20261013_2221. Original server input files are checked against retained records and saved artifact hashes. Historical generated tables are reused; generators and labelers are not refitted.

Some older synthetic-table hashes come from the October 13 audit rather than training-time records. Later hashes alone cannot certify historical training-time bytes; exact history and checkpoint reproduction are additionally required for recovered states.

Caveats remain: Adult TabDDPM has an unresolved historical generator-source defect; MNIST12 TVAE retains the documented feature-collision caveat. These alternate evaluations do not clear either caveat. Known broken configurations excluded from the retained snapshot remain excluded.

Paper references are unchanged and use different splits and classifier averages. References in binary panels apply to binary F1; regression references apply to R². Original comparison files are preserved.

## adult

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 67.38% / 78.93% | 68.39% / 79.58% |
| CTGAN, generated target | 62.14% / 76.17% | 68.87% / 79.27% |
| CTGAN full features + RF | 64.70% / 77.75% | 68.84% / 79.45% |
| CTGAN full features + XGB | 67.08% / 78.95% | 68.59% / 79.66% |
| CTGAN full features + DNN | 69.62% / 79.30% | 69.13% / 79.14% |
| CTGAN X-only features + RF | 68.65% / 79.64% | 66.99% / 78.88% |
| CTGAN X-only features + XGB | 68.67% / 79.63% | 68.33% / 79.46% |
| CTGAN X-only features + DNN | 69.81% / 79.66% | 69.02% / 79.09% |
| TVAE, generated target | 59.66% / 74.88% | 63.66% / 76.59% |
| TVAE full features + RF | 66.63% / 78.67% | 65.38% / 77.89% |
| TVAE full features + XGB | 67.70% / 79.12% | 67.36% / 78.88% |
| TVAE full features + DNN | 69.33% / 79.41% | 68.37% / 78.93% |
| TVAE X-only features + RF | 66.38% / 78.49% | 67.36% / 79.14% |
| TVAE X-only features + XGB | 68.31% / 79.51% | 67.19% / 78.77% |
| TVAE X-only features + DNN | 69.92% / 79.71% | 68.99% / 79.29% |
| TabDDPM, generated target | 63.98% / 77.39% | 66.73% / 78.74% |
| TabDDPM full features + RF | 64.91% / 77.87% | 68.58% / 79.68% |
| TabDDPM full features + XGB | 68.23% / 79.52% | 69.03% / 79.91% |
| TabDDPM full features + DNN | 69.39% / 79.35% | 68.95% / 79.21% |
| TabDDPM X-only features + RF | 65.82% / 78.35% | 68.44% / 79.64% |
| TabDDPM X-only features + XGB | 67.78% / 79.25% | 68.12% / 79.35% |
| TabDDPM X-only features + DNN | 69.49% / 79.71% | 69.06% / 79.35% |
| Paper CTGAN (reference) | 60.1% / — | 60.1% / — |
| Paper TVAE (reference) | 62.6% / — | 62.6% / — |
| Paper Real (reference) | 66.9% / — | 66.9% / — |

## covertype

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 51.98% | 53.19% |
| CTGAN, generated target | 52.96% | 50.14% |
| CTGAN full features + RF | 57.99% | Pending |
| CTGAN full features + XGB | 66.10% | 58.83% |
| CTGAN full features + DNN | 78.25% | 75.18% |
| CTGAN X-only features + RF | 58.85% | 55.21% |
| CTGAN X-only features + XGB | Pending | 68.46% |
| CTGAN X-only features + DNN | 76.74% | 75.43% |
| TVAE, generated target | 48.99% | 49.83% |
| TVAE full features + RF | Pending | 58.71% |
| TVAE full features + XGB | Pending | 60.23% |
| TVAE full features + DNN | Pending | Pending |
| TVAE X-only features + RF | Pending | 57.10% |
| TVAE X-only features + XGB | Pending | 57.24% |
| TVAE X-only features + DNN | Pending | Pending |
| TabDDPM, generated target | 49.03% | 49.14% |
| TabDDPM full features + RF | 65.47% | Pending |
| TabDDPM full features + XGB | 66.27% | Pending |
| TabDDPM full features + DNN | Pending | 77.21% |
| TabDDPM X-only features + RF | 62.70% | 66.63% |
| TabDDPM X-only features + XGB | 68.21% | 68.05% |
| TabDDPM X-only features + DNN | 78.06% | 76.82% |
| Paper CTGAN (reference) | 32.4% | 32.4% |
| Paper TVAE (reference) | 43.3% | 43.3% |
| Paper Real (reference) | 65.2% | 65.2% |

## mnist12

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | Pending | Pending |
| CTGAN, generated target | Pending | Pending |
| CTGAN full features + RF | 92.89% | 93.00% |
| CTGAN full features + XGB | Pending | Pending |
| CTGAN full features + DNN | Pending | Pending |
| CTGAN X-only features + RF | Pending | 92.95% |
| CTGAN X-only features + XGB | Pending | Pending |
| CTGAN X-only features + DNN | Pending | Pending |
| TVAE, generated target | Pending | Pending |
| TVAE full features + RF | Pending | Pending |
| TVAE full features + XGB | Pending | Pending |
| TVAE full features + DNN | Pending | Pending |
| TVAE X-only features + RF | 94.11% | Pending |
| TVAE X-only features + XGB | Pending | Pending |
| TVAE X-only features + DNN | Pending | Pending |
| TabDDPM, generated target | Pending | Pending |
| TabDDPM full features + RF | 80.39% | Pending |
| TabDDPM full features + XGB | Pending | Pending |
| TabDDPM full features + DNN | Pending | Pending |
| TabDDPM X-only features + RF | Pending | 91.48% |
| TabDDPM X-only features + XGB | Pending | Pending |
| TabDDPM X-only features + DNN | Pending | Pending |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## mnist28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | Pending | Pending |
| CTGAN, generated target | 90.60% | 94.95% |
| CTGAN full features + RF | Pending | Pending |
| CTGAN full features + XGB | Pending | 96.83% |
| CTGAN full features + DNN | Pending | Pending |
| CTGAN X-only features + RF | 96.27% | Pending |
| CTGAN X-only features + XGB | Pending | Pending |
| CTGAN X-only features + DNN | Pending | Pending |
| TVAE, generated target | Pending | Pending |
| TVAE full features + RF | Pending | Pending |
| TVAE full features + XGB | Pending | 96.78% |
| TVAE full features + DNN | Pending | Pending |
| TVAE X-only features + RF | Pending | Pending |
| TVAE X-only features + XGB | Pending | Pending |
| TVAE X-only features + DNN | Pending | Pending |
| TabDDPM, generated target | 89.22% | Pending |
| TabDDPM full features + RF | 92.10% | 94.16% |
| TabDDPM full features + XGB | 93.07% | Pending |
| TabDDPM full features + DNN | Pending | 80.76% |
| TabDDPM X-only features + RF | Pending | Pending |
| TabDDPM X-only features + XGB | Pending | 34.45% |
| TabDDPM X-only features + DNN | Pending | Pending |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## census_kdd

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 55.56% / 75.86% | 54.87% / 75.41% |
| CTGAN, generated target | 36.25% / 62.20% | 53.14% / 74.64% |
| CTGAN full features + RF | 57.05% / 77.00% | 56.16% / 76.69% |
| CTGAN full features + XGB | 57.24% / 77.13% | 56.43% / 76.75% |
| CTGAN full features + DNN | 56.39% / 76.71% | 57.21% / 77.17% |
| CTGAN X-only features + RF | 55.65% / 76.40% | 55.99% / 76.53% |
| CTGAN X-only features + XGB | 57.09% / 77.13% | 56.86% / 76.83% |
| CTGAN X-only features + DNN | 56.87% / 77.07% | 57.12% / 77.13% |
| TVAE, generated target | 56.51% / 76.59% | 54.27% / 75.13% |
| TVAE full features + RF | 55.84% / 76.34% | 55.41% / 76.13% |
| TVAE full features + XGB | 57.92% / 77.44% | 57.67% / 77.36% |
| TVAE full features + DNN | 57.02% / 77.06% | 56.97% / 77.11% |
| TVAE X-only features + RF | 55.19% / 75.87% | 57.12% / 77.25% |
| TVAE X-only features + XGB | 57.45% / 77.26% | 58.03% / 77.58% |
| TVAE X-only features + DNN | 57.47% / 77.40% | 57.24% / 77.19% |
| TabDDPM, generated target | 43.38% / 67.97% | 47.02% / 70.35% |
| TabDDPM full features + RF | 52.20% / 74.68% | 57.96% / 77.58% |
| TabDDPM full features + XGB | 54.87% / 75.96% | 59.04% / 78.19% |
| TabDDPM full features + DNN | 55.17% / 76.13% | 56.41% / 76.89% |
| TabDDPM X-only features + RF | 55.92% / 76.55% | 54.58% / 75.89% |
| TabDDPM X-only features + XGB | 56.79% / 76.93% | 57.46% / 77.32% |
| TabDDPM X-only features + DNN | 55.81% / 76.56% | 57.30% / 77.37% |
| Paper CTGAN (reference) | 39.1% / — | 39.1% / — |
| Paper TVAE (reference) | 37.7% / — | 37.7% / — |
| Paper Real (reference) | 49.4% / — | 49.4% / — |

## news

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.0048 / -0.0857 | -0.0628 / -0.0033 |
| CTGAN, generated target | -0.0082 / 0.0199 | -0.0263 / 0.0282 |
| CTGAN full features + RF | -0.0028 / 0.0094 | -0.0018 / 0.0097 |
| CTGAN full features + XGB | -0.0044 / 0.0126 | 0.0003 / 0.0049 |
| CTGAN full features + DNN | 0.0008 / 0.0170 | 0.0017 / 0.0163 |
| CTGAN X-only features + RF | -0.0020 / 0.0081 | -0.0020 / 0.0080 |
| CTGAN X-only features + XGB | 0.0021 / -0.0091 | 0.0026 / 0.0083 |
| CTGAN X-only features + DNN | 0.0001 / 0.0182 | -0.0011 / 0.0225 |
| TVAE, generated target | -0.0221 / 0.0358 | -0.0216 / 0.0362 |
| TVAE full features + RF | -0.0031 / 0.0069 | 0.0001 / 0.0041 |
| TVAE full features + XGB | -0.0018 / 0.0150 | -0.0002 / 0.0073 |
| TVAE full features + DNN | 0.0023 / 0.0066 | -0.0012 / 0.0157 |
| TVAE X-only features + RF | 0.0020 / -0.0018 | -0.0008 / 0.0056 |
| TVAE X-only features + XGB | -0.0002 / -0.0015 | -0.0012 / 0.0186 |
| TVAE X-only features + DNN | 0.0037 / -0.0013 | -0.0008 / 0.0198 |
| TabDDPM, generated target | 0.0019 / -0.1291 | 0.0028 / -0.0183 |
| TabDDPM full features + RF | -0.0006 / -0.0041 | -0.0024 / 0.0106 |
| TabDDPM full features + XGB | 0.0027 / -0.0244 | -0.0008 / 0.0111 |
| TabDDPM full features + DNN | -0.0184 / -0.0298 | 0.0006 / 0.0130 |
| TabDDPM X-only features + RF | -0.0019 / 0.0027 | -0.0022 / 0.0147 |
| TabDDPM X-only features + XGB | -0.0012 / -0.0046 | -0.0011 / -0.0110 |
| TabDDPM X-only features + DNN | -0.0016 / -0.0358 | -0.0062 / -0.0337 |
| Paper CTGAN (reference) | -0.430 / — | -0.430 / — |
| Paper TVAE (reference) | -0.200 / — | -0.200 / — |
| Paper Real (reference) | 0.140 / — | 0.140 / — |

## california_housing

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.5246 / 0.2961 | 0.5003 / 0.2684 |
| CTGAN, generated target | 0.6087 / 0.3981 | 0.6625 / 0.4393 |
| CTGAN full features + RF | 0.7818 / 0.5897 | 0.7909 / 0.5923 |
| CTGAN full features + XGB | 0.8034 / 0.6155 | 0.8234 / 0.6298 |
| CTGAN full features + DNN | 0.8010 / 0.6116 | 0.7737 / 0.6073 |
| CTGAN X-only features + RF | 0.7898 / 0.5885 | 0.7743 / 0.5868 |
| CTGAN X-only features + XGB | 0.8153 / 0.6201 | 0.8043 / 0.6113 |
| CTGAN X-only features + DNN | 0.8072 / 0.6121 | 0.7985 / 0.6046 |
| TVAE, generated target | 0.6731 / 0.4843 | 0.7263 / 0.5214 |
| TVAE full features + RF | 0.7779 / 0.5722 | 0.7738 / 0.5684 |
| TVAE full features + XGB | 0.7493 / 0.6116 | 0.7751 / 0.6181 |
| TVAE full features + DNN | 0.8065 / 0.6080 | 0.7891 / 0.6024 |
| TVAE X-only features + RF | 0.7591 / 0.5525 | 0.7622 / 0.5577 |
| TVAE X-only features + XGB | 0.8011 / 0.5997 | 0.7993 / 0.6125 |
| TVAE X-only features + DNN | 0.7890 / 0.6039 | 0.7860 / 0.5974 |
| TabDDPM, generated target | 0.7998 / 0.5971 | 0.7745 / 0.5481 |
| TabDDPM full features + RF | 0.7951 / 0.5976 | 0.7834 / 0.5657 |
| TabDDPM full features + XGB | 0.8288 / 0.6384 | 0.8039 / 0.5871 |
| TabDDPM full features + DNN | 0.8020 / 0.6064 | 0.7834 / 0.5715 |
| TabDDPM X-only features + RF | 0.7907 / 0.5930 | 0.7817 / 0.5628 |
| TabDDPM X-only features + XGB | 0.8203 / 0.6215 | 0.8114 / 0.5970 |
| TabDDPM X-only features + DNN | 0.7987 / 0.6020 | 0.7923 / 0.5843 |
| Paper CTGAN (reference) | Not reported | Not reported |
| Paper TVAE (reference) | Not reported | Not reported |
| Paper Real (reference) | Not reported | Not reported |

