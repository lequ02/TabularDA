# Synthetic training: development F1 / R² checkpoint selection

Status: 477/602 distinct source runs re-evaluated. Pending cells are blank; these tables are provisional.

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
| CTGAN, generated target | 62.13% / 75.78% | 64.66% / 77.24% |
| CTGAN full features + RF | 64.83% / 77.76% | 67.47% / 79.00% |
| CTGAN full features + XGB | 65.83% / 78.26% | 67.55% / 79.00% |
| CTGAN full features + DNN | 69.58% / 79.63% | 69.10% / 79.14% |
| CTGAN X-only features + RF | 66.46% / 78.55% | 66.86% / 78.81% |
| CTGAN X-only features + XGB | 67.83% / 79.21% | 66.93% / 78.69% |
| CTGAN X-only features + DNN | 69.53% / 79.67% | 68.99% / 79.33% |
| TVAE, generated target | 59.13% / 74.13% | 63.59% / 76.40% |
| TVAE full features + RF | 66.84% / 78.68% | 65.45% / 77.90% |
| TVAE full features + XGB | 67.79% / 79.13% | 67.44% / 78.81% |
| TVAE full features + DNN | 69.46% / 79.59% | 68.73% / 79.22% |
| TVAE X-only features + RF | 67.55% / 79.04% | 66.84% / 78.75% |
| TVAE X-only features + XGB | 68.22% / 79.44% | 67.39% / 78.91% |
| TVAE X-only features + DNN | 69.37% / 79.15% | 69.26% / 79.50% |
| TabDDPM, generated target | 65.64% / 78.22% | 66.81% / 78.77% |
| TabDDPM full features + RF | 66.34% / 78.56% | 68.30% / 79.56% |
| TabDDPM full features + XGB | 68.30% / 79.56% | 68.34% / 79.57% |
| TabDDPM full features + DNN | 69.35% / 79.35% | 69.02% / 79.27% |
| TabDDPM X-only features + RF | 66.24% / 78.54% | 66.89% / 78.70% |
| TabDDPM X-only features + XGB | 67.27% / 79.01% | 67.88% / 79.18% |
| TabDDPM X-only features + DNN | 69.44% / 79.54% | 68.79% / 79.17% |
| Paper CTGAN (reference) | 60.1% / — | 60.1% / — |
| Paper TVAE (reference) | 62.6% / — | 62.6% / — |
| Paper Real (reference) | 66.9% / — | 66.9% / — |

## covertype

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 51.98% | 53.19% |
| CTGAN, generated target | 46.83% | 46.12% |
| CTGAN full features + RF | 61.75% | 61.87% |
| CTGAN full features + XGB | 66.78% | 62.59% |
| CTGAN full features + DNN | 71.99% | 73.61% |
| CTGAN X-only features + RF | 63.74% | 60.76% |
| CTGAN X-only features + XGB | 64.39% | 65.74% |
| CTGAN X-only features + DNN | 71.48% | 71.04% |
| TVAE, generated target | 47.11% | 47.44% |
| TVAE full features + RF | 59.58% | 56.60% |
| TVAE full features + XGB | 59.76% | 57.77% |
| TVAE full features + DNN | 71.00% | 65.99% |
| TVAE X-only features + RF | 54.62% | 56.23% |
| TVAE X-only features + XGB | 51.93% | 58.69% |
| TVAE X-only features + DNN | 66.26% | 68.10% |
| TabDDPM, generated target | 51.61% | 53.05% |
| TabDDPM full features + RF | 66.01% | 69.52% |
| TabDDPM full features + XGB | 68.06% | 68.14% |
| TabDDPM full features + DNN | 76.91% | 74.45% |
| TabDDPM X-only features + RF | 65.52% | 66.25% |
| TabDDPM X-only features + XGB | 69.69% | 68.16% |
| TabDDPM X-only features + DNN | 76.24% | 75.53% |
| Paper CTGAN (reference) | 32.4% | 32.4% |
| Paper TVAE (reference) | 43.3% | 43.3% |
| Paper Real (reference) | 65.2% | 65.2% |

## mnist12

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | Pending | Pending |
| CTGAN, generated target | Pending | 56.94% |
| CTGAN full features + RF | Pending | Pending |
| CTGAN full features + XGB | 91.23% | 91.54% |
| CTGAN full features + DNN | 91.94% | Pending |
| CTGAN X-only features + RF | Pending | Pending |
| CTGAN X-only features + XGB | Pending | Pending |
| CTGAN X-only features + DNN | Pending | Pending |
| TVAE, generated target | Pending | Pending |
| TVAE full features + RF | Pending | 93.09% |
| TVAE full features + XGB | Pending | Pending |
| TVAE full features + DNN | 93.90% | Pending |
| TVAE X-only features + RF | Pending | Pending |
| TVAE X-only features + XGB | Pending | Pending |
| TVAE X-only features + DNN | Pending | Pending |
| TabDDPM, generated target | Pending | Pending |
| TabDDPM full features + RF | 66.10% | 67.20% |
| TabDDPM full features + XGB | Pending | Pending |
| TabDDPM full features + DNN | 19.35% | 24.65% |
| TabDDPM X-only features + RF | 76.73% | 73.32% |
| TabDDPM X-only features + XGB | Pending | Pending |
| TabDDPM X-only features + DNN | 32.73% | 31.83% |
| Paper CTGAN (reference) | 39.4% | 39.4% |
| Paper TVAE (reference) | 79.3% | 79.3% |
| Paper Real (reference) | 88.6% | 88.6% |

## mnist28

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | Pending | Pending |
| CTGAN, generated target | 51.96% | 63.11% |
| CTGAN full features + RF | 86.33% | Pending |
| CTGAN full features + XGB | Pending | Pending |
| CTGAN full features + DNN | Pending | Pending |
| CTGAN X-only features + RF | Pending | 87.53% |
| CTGAN X-only features + XGB | Pending | 88.55% |
| CTGAN X-only features + DNN | Pending | Pending |
| TVAE, generated target | Pending | Pending |
| TVAE full features + RF | Pending | Pending |
| TVAE full features + XGB | Pending | Pending |
| TVAE full features + DNN | Pending | 95.95% |
| TVAE X-only features + RF | Pending | Pending |
| TVAE X-only features + XGB | Pending | Pending |
| TVAE X-only features + DNN | Pending | Pending |
| TabDDPM, generated target | 52.75% | 33.58% |
| TabDDPM full features + RF | 59.85% | 56.09% |
| TabDDPM full features + XGB | Pending | 47.53% |
| TabDDPM full features + DNN | 14.48% | 17.04% |
| TabDDPM X-only features + RF | 54.49% | Pending |
| TabDDPM X-only features + XGB | 47.37% | Pending |
| TabDDPM X-only features + DNN | 16.19% | Pending |
| Paper CTGAN (reference) | 37.1% | 37.1% |
| Paper TVAE (reference) | 79.4% | 79.4% |
| Paper Real (reference) | 91.6% | 91.6% |

## census_kdd

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 55.56% / 75.86% | 54.87% / 75.41% |
| CTGAN, generated target | 43.46% / 68.15% | 51.37% / 73.95% |
| CTGAN full features + RF | 52.07% / 74.30% | 49.75% / 73.21% |
| CTGAN full features + XGB | 55.04% / 75.92% | 54.34% / 75.45% |
| CTGAN full features + DNN | 54.21% / 75.57% | 53.95% / 75.37% |
| CTGAN X-only features + RF | 52.97% / 74.82% | 52.54% / 74.50% |
| CTGAN X-only features + XGB | 53.59% / 75.10% | 54.90% / 75.80% |
| CTGAN X-only features + DNN | 54.25% / 75.73% | 54.56% / 75.74% |
| TVAE, generated target | 46.95% / 71.08% | 47.04% / 71.33% |
| TVAE full features + RF | 49.08% / 72.41% | 50.20% / 73.38% |
| TVAE full features + XGB | 50.91% / 73.71% | 53.17% / 74.95% |
| TVAE full features + DNN | 53.33% / 75.08% | 52.74% / 74.80% |
| TVAE X-only features + RF | 48.40% / 72.05% | 48.67% / 72.37% |
| TVAE X-only features + XGB | 49.69% / 73.03% | 50.15% / 73.23% |
| TVAE X-only features + DNN | 52.61% / 74.64% | 53.09% / 74.93% |
| TabDDPM, generated target | 48.31% / 72.35% | 45.31% / 69.55% |
| TabDDPM full features + RF | 47.84% / 72.52% | 54.02% / 75.50% |
| TabDDPM full features + XGB | 50.37% / 73.87% | 57.27% / 77.25% |
| TabDDPM full features + DNN | 52.28% / 74.78% | 55.24% / 76.27% |
| TabDDPM X-only features + RF | 54.14% / 75.53% | 51.71% / 74.32% |
| TabDDPM X-only features + XGB | 56.66% / 76.88% | 54.58% / 75.83% |
| TabDDPM X-only features + DNN | 54.77% / 75.99% | 54.19% / 75.73% |
| Paper CTGAN (reference) | 39.1% / — | 39.1% / — |
| Paper TVAE (reference) | 37.7% / — | 37.7% / — |
| Paper Real (reference) | 49.4% / — | 49.4% / — |

## news

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.0048 / -0.0857 | -0.0628 / -0.0033 |
| CTGAN, generated target | -0.0141 / 0.0104 | -0.0249 / 0.0196 |
| CTGAN full features + RF | -0.0050 / -0.0038 | -0.0008 / -0.0081 |
| CTGAN full features + XGB | -0.0132 / 0.0158 | -0.0001 / -0.0015 |
| CTGAN full features + DNN | 0.0018 / 0.0058 | 0.0001 / 0.0102 |
| CTGAN X-only features + RF | -0.0001 / 0.0007 | -0.0006 / -0.0018 |
| CTGAN X-only features + XGB | 0.0004 / -0.0131 | -0.0005 / -0.0076 |
| CTGAN X-only features + DNN | 0.0016 / 0.0064 | 0.0019 / 0.0083 |
| TVAE, generated target | -0.0287 / 0.0045 | -0.0262 / 0.0053 |
| TVAE full features + RF | 0.0016 / -0.0069 | -0.0011 / -0.0055 |
| TVAE full features + XGB | 0.0001 / -0.0110 | -0.0046 / -0.0103 |
| TVAE full features + DNN | 0.0127 / -0.0295 | 0.0034 / -0.0269 |
| TVAE X-only features + RF | 0.0034 / -0.0045 | 0.0019 / -0.0007 |
| TVAE X-only features + XGB | 0.0037 / -0.0447 | 0.0003 / -0.0191 |
| TVAE X-only features + DNN | 0.0113 / -0.0302 | 0.0090 / -0.0113 |
| TabDDPM, generated target | -0.0509 / -0.0532 | -0.0043 / 0.0007 |
| TabDDPM full features + RF | 0.0026 / -0.0216 | -0.0010 / 0.0032 |
| TabDDPM full features + XGB | -0.0029 / -0.0422 | 0.0027 / -0.0014 |
| TabDDPM full features + DNN | -0.0160 / -0.0255 | 0.0024 / -0.0017 |
| TabDDPM X-only features + RF | 0.0012 / -0.0105 | -0.0021 / 0.0060 |
| TabDDPM X-only features + XGB | 0.0015 / -0.0079 | 0.0065 / -0.0437 |
| TabDDPM X-only features + DNN | -0.0077 / -0.0514 | -0.0010 / -0.0510 |
| Paper CTGAN (reference) | -0.430 / — | -0.430 / — |
| Paper TVAE (reference) | -0.200 / — | -0.200 / — |
| Paper Real (reference) | 0.140 / — | 0.140 / — |

## california_housing

| Configuration | Seed 42 | Seed 43 |
|---|---:|---:|
| Original data only | 0.5246 / 0.2961 | 0.5003 / 0.2684 |
| CTGAN, generated target | 0.6076 / 0.3909 | 0.6554 / 0.4364 |
| CTGAN full features + RF | 0.7880 / 0.5990 | 0.7910 / 0.5988 |
| CTGAN full features + XGB | 0.8131 / 0.6257 | 0.8207 / 0.6284 |
| CTGAN full features + DNN | 0.8014 / 0.6122 | 0.7991 / 0.6065 |
| CTGAN X-only features + RF | 0.7847 / 0.5768 | 0.7953 / 0.5945 |
| CTGAN X-only features + XGB | 0.8216 / 0.6227 | 0.8173 / 0.6197 |
| CTGAN X-only features + DNN | 0.8069 / 0.6100 | 0.8009 / 0.6049 |
| TVAE, generated target | 0.7000 / 0.4900 | 0.7290 / 0.5209 |
| TVAE full features + RF | 0.7848 / 0.5774 | 0.7844 / 0.5841 |
| TVAE full features + XGB | 0.8209 / 0.6245 | 0.8212 / 0.6289 |
| TVAE full features + DNN | 0.6976 / 0.5837 | 0.7943 / 0.6041 |
| TVAE X-only features + RF | 0.7725 / 0.5668 | 0.7855 / 0.5795 |
| TVAE X-only features + XGB | 0.8151 / 0.6215 | 0.8195 / 0.6219 |
| TVAE X-only features + DNN | 0.7294 / 0.6087 | 0.7860 / 0.6034 |
| TabDDPM, generated target | 0.8020 / 0.6045 | 0.7805 / 0.5691 |
| TabDDPM full features + RF | 0.7977 / 0.6054 | 0.7893 / 0.5725 |
| TabDDPM full features + XGB | 0.8302 / 0.6395 | 0.8151 / 0.5964 |
| TabDDPM full features + DNN | 0.8013 / 0.6079 | 0.7923 / 0.5809 |
| TabDDPM X-only features + RF | 0.7952 / 0.6029 | 0.7861 / 0.5699 |
| TabDDPM X-only features + XGB | 0.8209 / 0.6266 | 0.8243 / 0.6138 |
| TabDDPM X-only features + DNN | 0.8025 / 0.6087 | 0.7961 / 0.5942 |
| Paper CTGAN (reference) | Not reported | Not reported |
| Paper TVAE (reference) | Not reported | Not reported |
| Paper Real (reference) | Not reported | Not reported |

