# Synthetic-data comparison — workbook-reported results

**Local source:** `D:/SummerResearch/final_results.xlsx`, sheet `final_results!A1:O29`. All “Our” and alternative-method scores below come exclusively from this workbook. The requested `D:/SummerResearch/final/_results.xlsx` does not exist; this is the previously confirmed file.

**Selection:** use the workbook’s `_end` columns consistently. The separate `_max` table below is included for reference; maxima may come from different epochs and are not validation-selected scores. Values are copied as reported, with no leakage adjustment or substitution from individual experiment logs.

## Paper-compatible synthetic table

Columns use the paper’s metrics: binary F1 for Adult/Census/Census-KDD/Credit; macro-F1 for Covertype/Intrusion; accuracy for MNIST; R² for News. Higher is better. F1/accuracy use a 0–1 scale.

| Method | Adult (F1) | Census (Adult alias)* (F1) | Census-KDD (F1) | Credit (F1) | Covertype (Macro-F1) | Intrusion (Macro-F1) | MNIST12 (Accuracy) | MNIST28 (Accuracy) | News (R²) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paper Identity | 0.669 | 0.669 | 0.494 | 0.720 | 0.652 | 0.862 | 0.886 | 0.916 | 0.140 |
| Paper CTGAN | 0.601 | 0.601 | 0.391 | 0.672 | 0.324 | 0.528 | 0.394 | 0.371 | -0.430 |
| Paper TVAE | 0.626 | 0.626 | 0.377 | 0.098 | 0.433 | 0.511 | 0.793 | 0.794 | -0.200 |
| Our Original | — | — | — | 0.8571 | 0.6871 | — | 0.9579 | 0.9776 | — |
| Our CTGAN | 0.6572 | — | — | 0.0639 | 0.7227 | — | 0.7370 | 0.5205 | — |
| Categorical | 0.6261 | 0.6223 | — | 0.4286 | 0.6550 | — | 0.8490 | 0.8193 | — |
| Gaussian | 0.4452 | 0.4660 | — | 0.5781 | 0.5940 | — | 0.6484 | 0.3591 | — |
| PCA-GMM | — | — | — | 0.4848 | 0.4388 | — | — | — | — |
| Our TVAE | — | — | — | — | — | — | — | — | — |

*Local Census is an Adult alias, so the paper Adult reference is repeated for that column. The paper’s `census` corresponds to `census_kdd`, shown separately. Paper CTGAN is the supplied screenshot’s `TGAN(1)` row; Identity is its real-data reference.

**Missing means missing:** “—” denotes an absent workbook dataset, method, or metric. The workbook has no Census-KDD, Intrusion, News, TVAE, RF, or XGBoost rows; no MNIST PCA-GMM rows; and some binary-F1 cells are empty, including Adult/Census Original and Census CTGAN. No scores from the downloaded logs were used to fill these cells.

## Consistent local comparison: macro-F1 at end

This supplementary table makes every available original-data row comparable with the workbook’s synthetic rows. Do not compare its binary-dataset macro-F1 directly with the paper’s binary F1.

| Dataset | Our Original | Our CTGAN | Categorical | Gaussian | PCA-GMM | Our TVAE |
| --- | --- | --- | --- | --- | --- | --- |
| Adult | 0.7924 | 0.7773 | 0.7301 | 0.5585 | 0.7378 | — |
| Census (Adult alias)* | 0.7977 | 0.7738 | 0.7264 | 0.6631 | 0.7585 | — |
| Census-KDD | — | — | — | — | — | — |
| Credit | 0.9284 | 0.4241 | 0.7140 | 0.7887 | 0.7414 | — |
| Covertype | 0.6871 | 0.7227 | 0.6550 | 0.5940 | 0.4388 | — |
| Intrusion | — | — | — | — | — | — |
| MNIST12 | 0.9576 | 0.7252 | 0.8463 | 0.6164 | — | — |
| MNIST28 | 0.9776 | 0.4682 | 0.8174 | 0.3282 | — | — |
| News | — | — | — | — | — | — |

## What the workbook reports

- **Adult and local Census:** CTGAN has the highest synthetic macro-F1; Original is higher still.

- **Covertype:** CTGAN macro-F1 is **0.7227**, above Original **0.6871** and all other listed synthetic methods.

- **Credit:** Gaussian has the highest synthetic fraud F1 (**0.5781**) and macro-F1 (**0.7887**); Original remains higher (**0.8571 / 0.9284**).

- **MNIST12/28:** categorical accuracy (**0.8490 / 0.8193**) exceeds our CTGAN (**0.7370 / 0.5205**) and the published TVAE averages (**0.793 / 0.794**) numerically. Original remains higher (**0.9579 / 0.9776**). This does not establish a controlled win against TVAE: no local TVAE result is in the workbook.

**Interpretation:** these are workbook-reported rankings, not validated synthetic-only performance claims. The audit established that some workbook rows average mixed real-plus-synthetic and synthetic-only runs. Paper scores also average classifiers under a different protocol. Neither issue is corrected by copying the workbook.

## Appendix: workbook maximum scores, paper-compatible metrics

| Method | Adult | Census (Adult alias)* | Census-KDD | Credit | Covertype | Intrusion | MNIST12 | MNIST28 | News |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Our Original | — | — | — | 0.9375 | 0.6926 | — | 0.9580 | 0.9785 | — |
| Our CTGAN | 0.6789 | — | — | 0.1170 | 0.7285 | — | 0.7656 | 0.5615 | — |
| Categorical | 0.6319 | 0.6274 | — | 0.4444 | 0.6609 | — | 0.8499 | 0.8216 | — |
| Gaussian | 0.4452 | 0.4779 | — | 0.6786 | 0.5962 | — | 0.6792 | 0.3624 | — |
| PCA-GMM | — | — | — | 0.5098 | 0.4702 | — | — | — | — |
| Our TVAE | — | — | — | — | — | — | — | — | — |

Paper source: user-provided Table 6 image and [CTGAN paper](https://arxiv.org/pdf/1907.00503). [Detailed audit](RESULTS_REVIEW.md). The workbook was read without modification.
