# Final-results credibility and CTGAN comparison

The observed advantage is real in the available logs for some datasets and survives a conservative **fixed-model duplicate-test-row removal** calculation. That is not the same as proving the advantage survives retraining a fully corrected pipeline. The workbook should not be used unchanged as evidence of synthetic-only performance.

## Sources and scope

- Read `D:/SummerResearch/final_results.xlsx`, sheet `final_results!A1:O29`, without modifying it. It has 28 result rows and no formulas.
- Read the final-results CSV and aggregation notebook in the user-provided Drive folder, then switched to the authoritative local download at `G:/summer_research/download2`.
- Inventoried the local ZIP archives without extracting their entire contents. Read 85 experiment CSV logs and streamed eight raw train/test pairs covering six datasets and both available MNIST split versions.
- The downloaded `final_results.csv` matches all 294 populated numeric workbook cells to its nine-decimal rounding. Thus the Drive/local CSV is the source of the same summary, while the folders also contain newer experiments that the workbook does not summarize.
- No original workbook, datasets, logs, checkpoints, or repository implementation files were changed. Remote notebooks were inspected as text, not executed. No weights or row-level prediction files were found in the downloaded archive manifests.

Source archive locations include:

- `G:/summer_research/download2/final_results-20260918T023824Z-1-001.zip`
- `G:/summer_research/download2/credit-20260918T023918Z-1-001.zip`
- `G:/summer_research/download2/mnist12-20260918T023822Z-1-001.zip`
- `G:/summer_research/download2/mnist28-20260918T023819Z-1-001.zip`
- `G:/summer_research/download2/data/data-20260918T024419Z-1-*.zip`

## 1. The workbook combines different experimental conditions

In `final_results/best_result_from_csv.ipynb`, code cell 5 checks generic method substrings (`ctgan`, `gaussian`, `categorical`, `pca_gmm`) under the `synthetic` category before checking `mix`. A filename such as `DNN_Credit_train_mix_augment_ctgan_...` therefore becomes `Synthetic/ctgan`. Cell 6 groups by these mistaken labels and averages numeric columns. It does not retain run identity, dataset hashes, epoch identity, seeds, or run counts in the final summary.

This is not just a possible bug: all 12 metrics for each Credit row reconcile exactly to the downloaded mixed and synthetic-only runs. The older MNIST12 categorical, CTGAN, and Gaussian rows likewise reconcile to averages of mixed and synthetic-only runs.

Examples using end-of-run macro-F1:

| Dataset/method | Synthetic-only | Mixed real + synthetic | Workbook |
|---|---:|---:|---:|
| Credit, categorical | 0.4995746384 | 0.9284462094 | 0.714010424 |
| Credit, CTGAN | 0.2928844815 | 0.5552423774 | 0.424063429 |
| MNIST12 old, categorical | 0.7956842734 | 0.8968414184 | 0.846262846 |
| MNIST12 old, CTGAN | 0.5354064929 | 0.9149026323 | 0.725154563 |

The workbook's MNIST28 categorical/CTGAN/Gaussian end scores instead match the individual older synthetic-only logs, not an average with the currently available mixed logs. Other rows cannot all be reconstructed from the downloaded logs. Do not assume a single consistent aggregation or complete run provenance across the workbook.

Earlier in this review, some workbook accuracy and binary/macro-F1 combinations appeared incompatible with one confusion matrix. Averaging different runs explains why that identity need not hold. Those combinations alone are therefore **not evidence of fabricated metrics**. The demonstrated issue is combining different experimental conditions under one label.

## 2. Data-version correction

The user's concern about old/new data is valid. The new local download has a 60,000/10,000 MNIST12 split, while the workspace's MNIST12 files were 56,000/14,000. The newer experiment logs use accuracy increments consistent with 10,000 test records; the older ones use 14,000. This is compatible evidence, not a cryptographic run-to-data binding.

These are **within-version** overlaps in the new download, using exact feature-row matching without the target. Newlines are normalized; no numeric rounding or fuzzy matching is applied. Matching features with different labels are included conservatively.

| Dataset/version | Training rows | Test rows | Test features also in training | Full row including label matches |
|---|---:|---:|---:|---:|
| Adult | 38,095 | 9,526 | 18 | 17 |
| Census | 38,095 | 9,526 | 18 | 17 |
| Covertype | 464,809 | 116,203 | 0 | 0 |
| Credit | 274,807 | 10,000 | 50 | 50 |
| MNIST12 new | 60,000 | 10,000 | 832 | 828 |
| MNIST12 old | 56,000 | 14,000 | 1,120 | 1,114 |
| MNIST28 new | 60,000 | 10,000 | 16 | 16 |
| MNIST28 old | 56,000 | 14,000 | 16 | 16 |

Cross-version comparison can indeed overstate the overlap relevant to an actual run: the old MNIST28 test set matches 4,075 rows in the new training set, but only 16 in its own old training set. The analogous old MNIST12 test/new train comparison gives 4,891 matching rows. These cross-version counts were not used in the bounds below.

Within-version overlap still exists. Natural duplicate images and lossy 12x12 binarization/downsampling can account for matches without duplicated source IDs. Of the 832 matching newer MNIST12 test features, 718 have label 1. The appropriate interpretation depends on whether the research claims unseen source records, unseen images after preprocessing, or draws from a distribution that permits repeated feature vectors. Source IDs are needed to distinguish these cases reliably.

Adult and Census remain identical to one another after newline normalization in the supplied data. They must not count as two independent benchmark datasets. Census KDD is a separate dataset in the download and is not the `census` row in this workbook. Intrusion is not a row in this workbook, so the earlier Intrusion overlap finding is not evidence against any particular workbook row.

## 3. Reconstructed comparison from individual logs

The following values use **synthetic-only runs** and the first epoch with minimum recorded **dev loss**, rather than independently maximizing each test metric. MNIST uses the explicitly labeled newer run folders. These are retrospective log selections, not new training runs or fully leakage-corrected metrics. Remaining preprocessing and validation-design issues are not repaired by this selection.

| Dataset | CTGAN macro-F1 | Categorical macro-F1 | Gaussian macro-F1 | PCA-GMM macro-F1 |
|---|---:|---:|---:|---:|
| Adult | 0.772769 | 0.724105 | 0.579008 | 0.713762 |
| Census (duplicate Adult data) | 0.767381 | 0.720733 | 0.635142 | 0.743520 |
| Credit | 0.303090 | 0.499575 | 0.637405 | 0.545715 |
| MNIST12 new | 0.472743 | 0.792293 | 0.302901 | 0.786185 |
| MNIST28 new | 0.410331 | 0.805903 | 0.474746 | 0.789244 |

The newer Covertype folder lacks a matched CTGAN log in the supplied archives. The workbook favors CTGAN on Covertype macro-F1, but its aggregate should not be compared directly with a newer single-run result. A new-version head-to-head Covertype conclusion is unavailable.

Additional newer baselines: TVAE macro-F1 is 0.922230 on MNIST12 and 0.931166 on MNIST28 under this same selection rule. Thus outperforming this CTGAN implementation does not establish outperforming all available synthesis baselines. RF/XGBoost labelers also outperform CTGAN in these logs, but these are single available configurations, not controlled replicated estimates.

## 4. Would the advantage survive removing overlap?

### Fixed models: conservative accuracy bounds

Suppose two fitted models were evaluated on the same N test rows, and k matching-feature rows are removed from both. If their original accuracy difference is d, their difference on the retained rows lies within:

`[(d - k/N)/(1 - k/N), (d + k/N)/(1 - k/N)]`, clipped to [-1, 1].

Reason: each removed row can contribute at most +1 or -1 to the difference in correct counts. This does not assume the models classified matching rows correctly. It is a bound, not an invented corrected score.

For newer categorical-versus-CTGAN runs selected by minimum dev loss:

| Dataset | Categorical accuracy | CTGAN accuracy | Removed feature matches | Minimum remaining categorical lead |
|---|---:|---:|---:|---:|
| MNIST12 | 79.31% | 53.35% | 832 / 10,000 | **19.24 percentage points** |
| MNIST28 | 80.92% | 42.97% | 16 / 10,000 | **37.85 percentage points** |

PCA-GMM also retains an accuracy lead under this calculation: at least 18.51 points on newer MNIST12 and 36.11 points on newer MNIST28. On Adult, even the upper accuracy bound for categorical, Gaussian, and PCA-GMM stays below CTGAN. Census PCA-GMM retains a small accuracy lead but trails CTGAN on macro-F1; these are different claims.

### Credit: bounds on macro-F1

The real Credit test contains 9,983 negative and 17 positive records. The 50 overlaps contain 49 negatives and one positive. The logged synthetic-only accuracy and binary-F1 values permit recovery of integer confusion counts; the resulting macro-F1 values independently match the logs. Enumerating every feasible way to remove those class counts gives:

| Method | Selected epoch | TP / FN / FP / TN before removal | Macro-F1 after removing overlaps: possible range |
|---|---:|---|---:|
| CTGAN | 65 | 17 / 0 / 5,702 / 4,281 | 0.301534–0.303975 |
| Gaussian | 10 | 4 / 13 / 8 / 9,975 | **0.610583–0.699698** |
| Categorical | 66 | 0 / 17 / 0 / 9,983 | 0.499598 |
| PCA-GMM | 111 | 5 / 12 / 82 / 9,901 | **0.536842–0.591484** |

The Gaussian and PCA-GMM lower bounds exceed CTGAN's upper bound. This supports a surviving **fixed-model** Credit macro-F1 advantage if the logs and supplied same-version test set correspond as indicated. The categorical model predicts no fraud cases and should not be described as a successful fraud detector merely because accuracy is high.

### What these bounds cannot establish

- They do not recompute individual predictions or provide an exact adjusted MNIST macro-F1.
- They hold the fitted model, preprocessing, and selected epoch fixed. Refitting a test-fitted scaler, retraining generators/classifiers on corrected splits, reserving dev before synthesis, or changing test-driven tuning can affect all predictions.
- They do not resolve near-duplicate/entity leakage, unknown checkpoint provenance, or changes in the data distribution caused by deleting duplicates.
- The archives do not include row-level predictions, saved model weights, or split-hash manifests for the reviewed runs. Therefore a fully corrected rerun cannot be reconstructed solely from these downloaded results.

## 5. Baseline quality and interpretation

In the supplied Credit synthetic CSVs (100,000 rows each):

| Generator/labeler | Negative labels | Fraud labels | Fraud proportion |
|---|---:|---:|---:|
| CTGAN | 67,012 | 32,988 | 32.988% |
| Categorical | 100,000 | 0 | 0% |
| Gaussian | 99,865 | 135 | 0.135% |
| Real test | 9,983 | 17 | 0.170% |

This severe CTGAN label-distribution mismatch is consistent with its 5,702 false positives. The cause still needs verification (model configuration, data preparation, target metadata, or a poorly fitted generator). It should not automatically be attributed to the CTGAN algorithm generally. Beating this run does not demonstrate beating a properly configured and tuned CTGAN baseline.

The categorical Credit result is a particularly clear summary problem: the workbook's 0.714010 macro-F1 averages a synthetic-only classifier that finds no fraud with a mixed-data classifier that benefits from real training records.

## Conclusion and required next evidence

**Yes, there is substantial evidence that the newer categorical/PCA-GMM MNIST runs and Gaussian/PCA-GMM Credit runs beat the available CTGAN runs even after removing exact-overlap test rows from the evaluation of fixed models. No, this does not establish a universal or fully leakage-free superiority claim.** Adult/Census macro-F1 favors CTGAN, a comparable newer Covertype baseline is missing, and the original summary combines conditions incorrectly.

Before publishing a corrected result, rebuild the summary with explicit run IDs and train mode, reserve immutable train/dev/test partitions before fitting any generator/labeler, fit preprocessing on training only, validate the CTGAN target distribution, and evaluate fixed validation-selected checkpoints on the same test IDs. Save row-level predictions and repeat across prespecified seeds. The corrected report must keep historical workbook aggregates, reconstructed individual-run scores, and duplicate-removal bounds separate.

Reproducibility evidence is in this directory's `archive_manifest.json`, `local_log_manifest.json`, `local_overlap.json`, `reconstructed.json`, and `sensitivity.json`. The original workbook and downloaded archives remain untouched.
