# Paper comparison and TVAE Covertype substitution

All final joint CTGAN/TVAE baselines are compared using the paper's metric: binary F1 for Adult/Census-KDD/Credit, macro-F1 for Covertype/Intrusion, accuracy for MNIST, and R-squared for News. April MNIST micro-F1 equals accuracy in single-label classification. Adult, Census and the misrouted Census_Kdd entries are averaged within each method and counted once. No independent final Census-KDD result is available.

[Paper Table 6 and Section 5.2](https://arxiv.org/pdf/1907.00503). The paper averages downstream classifiers; the local record uses DNNs and reports test maxima/final epochs. Matching a metric does not match the full protocol. No combined paper/local average is calculated across unlike metrics.

## Final baseline comparison

| dataset | method | metric | paper_score | endpoint | value | difference_from_paper |
| --- | --- | --- | --- | --- | --- | --- |
| Adult | ctgan | Binary F1 | 0.6010 | end | 0.6581 | 0.0571 |
| Adult | ctgan | Binary F1 | 0.6010 | max | 0.6657 | 0.0647 |
| Adult | tvae | Binary F1 | 0.6260 | end | 0.5667 | -0.0593 |
| Adult | tvae | Binary F1 | 0.6260 | max | 0.6234 | -0.0026 |
| Census_KDD | ctgan | Binary F1 | 0.3910 | - | - | - |
| Census_KDD | tvae | Binary F1 | 0.3770 | - | - | - |
| Covertype | ctgan | Macro-F1 | 0.3240 | end | 0.4748 | 0.1508 |
| Covertype | ctgan | Macro-F1 | 0.3240 | max | 0.4835 | 0.1595 |
| Covertype | tvae | Macro-F1 | 0.4330 | end | 0.2672 | -0.1658 |
| Covertype | tvae | Macro-F1 | 0.4330 | max | 0.2972 | -0.1358 |
| Credit | ctgan | Binary F1 | 0.6720 | - | - | - |
| Credit | tvae | Binary F1 | 0.0980 | - | - | - |
| Intrusion | ctgan | Macro-F1 | 0.5280 | - | - | - |
| Intrusion | tvae | Macro-F1 | 0.5110 | - | - | - |
| MNIST12 | ctgan | Accuracy | 0.3940 | end | 0.5326 | 0.1386 |
| MNIST12 | ctgan | Accuracy | 0.3940 | max | 0.5544 | 0.1604 |
| MNIST12 | tvae | Accuracy | 0.7930 | end | 0.9219 | 0.1289 |
| MNIST12 | tvae | Accuracy | 0.7930 | max | 0.9252 | 0.1322 |
| MNIST28 | ctgan | Accuracy | 0.3710 | end | 0.4077 | 0.0367 |
| MNIST28 | ctgan | Accuracy | 0.3710 | max | 0.5341 | 0.1631 |
| MNIST28 | tvae | Accuracy | 0.7940 | end | 0.9291 | 0.1351 |
| MNIST28 | tvae | Accuracy | 0.7940 | max | 0.9361 | 0.1421 |
| News | ctgan | R-squared | -0.4300 | - | - | - |
| News | tvae | R-squared | -0.2000 | - | - | - |

Missing rows are not zero. Credit is present only in the older workbook. Intrusion and News have no final baseline in these records. Adult binary F1 and MNIST accuracy must not be confused with the macro-F1 scores used in the synthesis-family analysis.

## What if TVAE Covertype were 0.433?

Only the TVAE Covertype baseline is replaced: 0.2972 (maximum) or 0.2672 (final epoch) becomes the paper's 0.433. The paper does not provide separate maximum/final-epoch values; applying the same number to both is an explicit assumption. All decoupled configurations, all other datasets, and CTGAN remain unchanged. Original recorded tables are not overwritten.

| family | scenario | metric | joint_mean | proposed_mean | gain | relative_gain | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All_methods | Recorded | macro_max | 0.6361 | 0.6948 | 0.0587 | 9.22% | 0.1717 |
| All_methods | Recorded | macro_end | 0.6177 | 0.6831 | 0.0654 | 10.58% | 0.1882 |
| All_methods | Paper TVAE Covertype: hypothetical | macro_max | 0.6531 | 0.6948 | 0.0417 | 6.39% | 0.2109 |
| All_methods | Paper TVAE Covertype: hypothetical | macro_end | 0.6385 | 0.6831 | 0.0447 | 7.00% | 0.2624 |
| No_Gaussian | Recorded | macro_max | 0.6361 | 0.7537 | 0.1176 | 18.49% | 0.0707 |
| No_Gaussian | Recorded | macro_end | 0.6177 | 0.7429 | 0.1251 | 20.26% | 0.0822 |
| No_Gaussian | Paper TVAE Covertype: hypothetical | macro_max | 0.6531 | 0.7537 | 0.1006 | 15.41% | 0.0940 |
| No_Gaussian | Paper TVAE Covertype: hypothetical | macro_end | 0.6385 | 0.7429 | 0.1044 | 16.35% | 0.1177 |
| RF_XGB | Recorded | macro_max | 0.6361 | 0.8013 | 0.1652 | 25.97% | 0.0467 |
| RF_XGB | Recorded | macro_end | 0.6177 | 0.7880 | 0.1703 | 27.56% | 0.0654 |
| RF_XGB | Paper TVAE Covertype: hypothetical | macro_max | 0.6531 | 0.8013 | 0.1482 | 22.70% | 0.0520 |
| RF_XGB | Paper TVAE Covertype: hypothetical | macro_end | 0.6385 | 0.7880 | 0.1496 | 23.42% | 0.0776 |

P values here are two-sided paired t calculations across four task differences. In the substituted scenario they are mechanical, hypothetical outputs of a hybrid table, not valid evidence from a new experiment or a leakage correction. Primary planned definitions and original broad-family results remain separate.

## Same-feature-generator RF/XGBoost comparisons

| scenario | analysis | datasets | joint_mean | rf_xgb_mean | gain | relative_gain | lower_95 | upper_95 | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recorded | CTGAN features | 4.0000 | 0.5462 | 0.7900 | 0.2438 | 0.4463 | -0.0823 | 0.5698 | 0.0977 | 0.1953 |
| Recorded | TVAE features | 4.0000 | 0.7260 | 0.8126 | 0.0866 | 0.1193 | -0.1254 | 0.2987 | 0.2844 | 0.2844 |
| Paper TVAE Covertype: hypothetical | CTGAN features | 4.0000 | 0.5462 | 0.7900 | 0.2438 | 0.4463 | -0.0823 | 0.5698 | 0.0977 | 0.1953 |
| Paper TVAE Covertype: hypothetical | TVAE features | 4.0000 | 0.7599 | 0.8126 | 0.0527 | 0.0693 | -0.0516 | 0.1570 | 0.2062 | 0.2062 |

Holm adjustment applies to the two generator comparisons within each scenario. CTGAN comparisons are unchanged by the TVAE substitution. Excluding Covertype leaves the recorded and hypothetical scenarios identical. Target-inclusion effects within decoupled methods are also unchanged.

## All versions and named aliases

The accompanying all_baseline_metrics_and_versions.csv retains every available CTGAN/TVAE metric in March 23, April 2, April 29 and the older final_results.xlsx extraction. all_paper_metric_comparisons.csv includes only matching paper metrics, with source, alias and provenance labels. Nonmatching macro/weighted/micro/loss values are retained as evidence but not compared to another metric.

Historical repetition check: 80 matched March/April 2 baseline metric cells; 80 agree with April 29 within 1e-12. These repeated records do not count as independent trials.

The older workbook has no TVAE rows and includes a mixed/synthetic aggregation defect found in the audit. Its CTGAN values are historical evidence only and are not used in the hypothetical analysis.

### Older-workbook comparisons: audit-flagged historical evidence

| dataset_label | method | metric | value | paper_score | difference_from_paper |
| --- | --- | --- | --- | --- | --- |
| mnist12 | ctgan | test_accuracy_max | 0.7656 | 0.3940 | 0.3716 |
| mnist28 | ctgan | test_accuracy_max | 0.5615 | 0.3710 | 0.1905 |
| mnist12 | ctgan | test_accuracy_end | 0.7370 | 0.3940 | 0.3430 |
| mnist28 | ctgan | test_accuracy_end | 0.5205 | 0.3710 | 0.1495 |
| covertype | ctgan | test_f1_macro_max | 0.7285 | 0.3240 | 0.4045 |
| covertype | ctgan | test_f1_macro_end | 0.7227 | 0.3240 | 0.3987 |
| adult | ctgan | test_f1_binary_max | 0.6789 | 0.6010 | 0.0779 |
| credit | ctgan | test_f1_binary_max | 0.1170 | 0.6720 | -0.5550 |
| adult | ctgan | test_f1_binary_end | 0.6572 | 0.6010 | 0.0562 |
| credit | ctgan | test_f1_binary_end | 0.0639 | 0.6720 | -0.6081 |

Mixed-model substitutions are also saved in substitution_mixed_model_sensitivity.csv, with singularity flags; they do not replace the four-task sensitivity results. Known inconsistent Covertype test versions, test-fitted preprocessing, test-max selection, incomplete split/checkpoint provenance, and related MNIST tasks remain unresolved.

## Figures

Figures 22-25 are in ../figures as PNG and vector PDF. They are also included in ../legacy_artifacts_report.html. Input snapshots and hashes are in inputs/paper_comparison_manifest.json at the analysis-folder root.
