# Census KDD: completed-run before/after comparison

Snapshot: October 5, 2026, 7:40 p.m. Chicago time. All 32 completed configurations are seed 42; seed 43 has not started. The matrix remains incomplete.

Before: unweighted binary cross-entropy and minimum real-development loss checkpoint selection. After: training-count-weighted BCEWithLogitsLoss and maximum real-development macro-F1 checkpoint selection. The reported metric below is **test binary F1**, on the same held-out rows and saved synthetic inputs. The threshold remains strictly greater than 0.5; architecture, batch order, splits, and budgets are unchanged.

**30 of 32 configurations improved; two declined.** All eight completed configurations with zero prior binary F1 now have positive F1. Mean binary F1 across these completed configurations rose from 0.219 to 0.418; this is descriptive across different arms, not independent replication.

Real-only: **0.405 → 0.556** (precision 0.797 → 0.450; recall 0.271 → 0.725).

| Construction | Synthetic-only F1 | Mixed-data F1 |
|---|---:|---:|
| CTGAN X-only · DNN | 0.000 → 0.542 | 0.288 → 0.569 |
| CTGAN full · DNN | 0.000 → 0.542 | 0.218 → 0.564 |
| CTGAN full · Generated targets | 0.543 → 0.435 | 0.581 → 0.362 |
| CTGAN X-only · Gaussian NB | 0.273 → 0.282 | 0.274 → 0.279 |
| CTGAN X-only · Categorical NB | 0.265 → 0.274 | 0.266 → 0.273 |
| CTGAN X-only · PCA/GMM | 0.296 → 0.324 | 0.298 → 0.313 |
| CTGAN X-only · RF | 0.000 → 0.530 | 0.107 → 0.557 |
| CTGAN X-only · XGB | 0.000 → 0.536 | 0.227 → 0.571 |
| CTGAN full · Gaussian NB | 0.270 → 0.278 | 0.272 → 0.274 |
| CTGAN full · Categorical NB | 0.291 → 0.302 | 0.292 → 0.293 |
| CTGAN full · PCA/GMM | 0.304 → 0.313 | 0.302 → 0.314 |
| CTGAN full · RF | 0.000 → 0.521 | 0.143 → 0.571 |
| CTGAN full · XGB | 0.000 → 0.550 | 0.265 → 0.572 |
| TVAE full · Generated targets | 0.000 → 0.470 | 0.000 → 0.565 |
| TVAE X-only · Gaussian NB | 0.271 → 0.280 | 0.271 → 0.290 |
| TVAE X-only · Categorical NB | 0.272 → 0.275 | — |

“—” means that configuration was not completed at this snapshot. “Full” means a full-table generator; for labeler rows its generated targets were replaced. “X-only” means a separate features-only generator. Mixed training uses all real training rows plus 100,000 synthetic rows.

The strongest recovery is in RF/XGB/DNN relabeled arms. NB/PCA-GMM arms improve only slightly. CTGAN generated-target baselines worsen: synthetic-only F1 0.543 → 0.435, mixed F1 0.581 → 0.362. Their recall increases but precision falls (synthetic 0.473 → 0.298; mixed 0.515 → 0.225), consistent with more false positives. The procedure therefore fixes collapse without improving every construction.

These results compare the combined objective/selection change; they do not establish the separate causal effect of class weighting. One seed and a partially completed matrix do not establish a general advantage. Weighted losses are not compared.

Sources: matched *.run.json records in remote output/corrected_v2/census_kdd/acc and output/census_kdd_weighted_macro_f1_20261005/census_kdd/acc. For every pair, dataset/seed/method/training mode, split manifest, and synthetic input path were checked; completed records had saved predictions and checkpoints. Only completed records were used; no test-epoch maxima were selected.
