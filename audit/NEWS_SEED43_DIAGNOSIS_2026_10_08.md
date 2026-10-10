# News seed-43 failure diagnosis

Inspected October 8, 2026, approximately 12:04–12:11 a.m. Chicago. All diagnostics ran read-only on the research server using its dedicated environment, with CPU inference and two computation threads. Active experiment workers were preserved. No models were retrained, checkpoints edited, predictions clipped, or reported results replaced.

## Confirmed failure

The extreme negative R² values arise from catastrophic overpredictions by the downstream `DNN_News` checkpoints. Recomputing MAE, MSE, and R² from the saved predictions reproduces the stored metrics within floating-point tolerance. CPU checkpoint replay reproduces the failing predictions, establishing that they are actual model behavior.

| Seed-43 configuration | Test R² | Worst article source ID | Actual shares | Predicted shares | Worst row's share of total squared error |
|---|---:|---:|---:|---:|---:|
| CTGAN full features + DNN, mix | −5216.564 | 17955 | 4,500 | 60,849,504 | 99.46% |
| CTGAN generated target, synthetic | −93.089 | 23062 | 1,900 | 7,967,745.5 | 94.53% |
| TVAE generated target, mix | −1.499 | 28369 | 6,800 | 961,054.7 | 51.07% |

The CTGAN full-feature DNN labeler did not generate targets of 60 million: its synthetic targets range from approximately −1,630 to 13,983. Its real-development R² was 0.01659 and it passed the configured gate. The explosive values are produced by the separate downstream network.

TVAE generated-target synthetic-only is much less affected: R² = −0.02771, NMAEσ = 0.26116, versus the seed-43 original baseline's NMAEσ = 0.31710. CTGAN full-feature DNN synthetic-only has R² = 0.02678, and TVAE full-feature DNN mix has R² = 0.02961. These distinguish individual failing configurations from the entire generator or labeler family.

## Confirmed mechanism: collapsed BatchNorm variances

`src/modeling/models_folder/model_news.py` applies ReLU before BatchNorm in four hidden blocks. In the failed checkpoints, some hidden units have almost-zero stored variance, causing enormous evaluation gains when a real input activates them.

For CTGAN full-feature DNN mix, 61 third-layer and eight fourth-layer BatchNorm units have running variance below 1e−5. Maximum evaluation gains are approximately 389 and 1,551 in those layers. On article 17955, the largest third-layer normalized activation is 1,974.8; the fourth-layer activation reaches 1,786,129 before the final 60.8-million prediction.

A targeted fourth-layer unit (unit 35) has running variance 3.43e−13. Its normalized output is constant at approximately 4.893 across **all 3,172 development rows** and the last 2,048 synthetic rows checked. The failing test article drives that same unit to 1,786,129. This directly explains why development checkpoint selection missed the failure: the relevant hidden activation did not occur in development.

Corresponding probes establish the same mechanism for CTGAN generated-target synthetic (third-layer unit 196: constant on development/synthetic probe, test activation 49,370) and TVAE generated-target mix (third-layer unit 74: constant on development/synthetic probe, test activation 15,497).

For seed-42 CTGAN full-feature DNN mix, the third/fourth layers have no units below that variance threshold. On the same article 17955, its prediction is approximately 13,079 rather than 60.8 million. The two seeds share identical prepared splits and table hashes; this is variation in generator/model fitting, rather than a different test set.

## Contributing pipeline behavior and limits of attribution

The training loader uses `shuffle=False`. Mix construction concatenates real rows first and synthetic rows afterward, so each epoch ends with 100,000 synthetic rows. BatchNorm's moving statistics consequently emphasize the final synthetic batches. This is a plausible contributor to variance collapse and poor transfer to real inputs; proving the independent effect of shuffling requires a controlled rerun.

Synthetic features also underrepresent some real tails. For example, the failing CTGAN generated-target article has `kw_max_min = 98,700`, whereas its generated feature table reaches only 18,424.18. However, the worst CTGAN+DNN mix article has no individual raw feature beyond approximately five real-training standard deviations. The explanation is a hidden-network activation failure, rather than simply one astronomically large raw input.

The selected checkpoints had unremarkable development R²: 0.02437 (CTGAN+DNN mix), 0.01009 (CTGAN generated-target synthetic), and −0.02271 (TVAE generated-target mix). News is also difficult more generally: both labeler and downstream development R² are near zero, and the real targets have a heavy tail. That general difficulty does not explain the million-scale overpredictions; the checkpoint probes do.

Any repair should be evaluated under a documented, consistent News protocol and a separate output namespace. The current failed results remain evidence. The read-only diagnosis has not established the performance of any proposed repair.

## Evidence

- `audit/news_seed43_prediction_diagnosis_2026_10_08.json`: saved metrics, recomputed prediction checks, error concentration, selected-epoch metrics, and cross-seed comparisons.
- `audit/news_seed43_input_diagnosis_2026_10_08.json`: verified prepared-table hashes, split equality, feature ranges, synthetic targets, and DNN labeler quality reports.
- `audit/news_seed43_checkpoint_diagnosis_2026_10_08.json`: checkpoint hashes, replayed predictions, per-layer activations, BatchNorm variances/gains, and development/synthetic activation probes.

## Target distributions and output arithmetic

Verified October 8, 2026, at 12:13 a.m. Chicago. Values below are target shares, rounded to the nearest share. Each synthetic table has 100,000 rows. Full denotes features from a full-table generator; RF/XGB/DNN denotes replacement targets. The real test has 7,929 rows and real train has 28,543 rows.

| Training or held-out table | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Real test | 3,388 | 1,400 | 1 | 441,000 |
| Real train | 3,399 | 1,400 | 4 | 843,300 |
| CTGAN full, generated targets | 2,955 | 1,577 | 4 | 65,604 |
| CTGAN full, RF targets | 6,849 | 4,776 | 908 | 197,900 |
| CTGAN full, XGB targets | 7,698 | 3,489 | -5,792 | 304,737 |
| CTGAN full, DNN targets | 2,906 | 2,607 | -1,630 | 13,983 |
| TVAE full, generated targets | 1,388 | 1,153 | 4 | 68,504 |
| TVAE full, RF targets | 4,632 | 3,300 | 838 | 118,740 |
| TVAE full, XGB targets | 4,471 | 2,493 | -9,684 | 231,756 |
| TVAE full, DNN targets | 2,934 | 2,540 | -427 | 18,225 |
| TVAE features-only, DNN targets | 2,956 | 2,560 | -1,540 | 19,818 |
| CTGAN full DNN mix: real + synthetic, 128,543 rows | 3,016 | 2,423 | -1,630 | 843,300 |

The mean and maximum alone do not describe the tail: the real test's 99th percentile is 34,932, versus 5,529 for TVAE generated targets and 7,936 for CTGAN full DNN targets. The negative relabeled targets are actual unconstrained regressor outputs, not negative observed shares. Tail compression is a separate utility concern from the hidden activation explosion.

For source ID 17955, fourth-layer unit 35 receives activation 1,188.0304. Its running mean is approximately zero, running variance is 3.4279e-13, epsilon is 1e-5, learned scale is 4.754274, and offset is 4.892887. Thus BatchNorm computes approximately `(1188 - 0) / sqrt(3.4279e-13 + 1e-5) * 4.754274 + 4.892887 = 1,786,130`. Its final-layer weight of 4.584878 contributes 8,189,186.5 shares. The remaining units contribute approximately 52,660,328 shares, plus a small output bias. CPU float32 replay is 60,849,524 versus the saved GPU prediction of 60,849,504; the small discrepancy is numerical rounding. This single unit illustrates the amplification but does not account for the entire prediction.

The final affine output has no bound, so training-target maxima do not impose a prediction maximum. The actual mix training maximum was 843,300, despite the synthetic DNN-target maximum being 13,983.

Exact statistics, table paths, real-table hashes, checkpoint hash, replay arithmetic, and contemporaneous active processes are recorded in `audit/news_seed43_target_distributions_2026_10_08.json`.

## Proposed News repair protocol (not executed or validated)

1. Replace hidden BatchNorm with LayerNorm, using `Linear -> LayerNorm -> ReLU -> Dropout`. LayerNorm uses each row's current hidden-feature statistics in training and evaluation, removing the specific collapsed running-variance mechanism demonstrated here. Preserve hidden widths and dropout. This is a mechanism-based proposal, not a demonstrated performance improvement or a general guarantee against extreme predictions.
2. Shuffle training rows each epoch with the experiment seed. Keep development/test ordering and source-ID alignment deterministic. Current mix training ends each epoch with the synthetic block; shuffling addresses that distribution ordering, although its independent benefit has not been measured.
3. Consider a separately documented variant that standardizes downstream targets using only real-training mean 3,399.202466 and population standard deviation 12,450.692423. Apply the same transform to real and synthetic training targets and development targets; invert predictions before raw-scale evaluation and persistence. This improves optimization conditioning but does not independently remove the BatchNorm failure. It does not repair synthetic tail coverage.
4. Evaluate a consistent revised News protocol across both seeds, the real-only baseline, and all compared arms, with existing verified synthetic tables and unchanged splits, sample counts, epochs, batch size, learning rate, patience, and development-only checkpoint selection. Use a new output namespace and retain the historical failed runs. Record that this protocol was motivated by inspection of existing test failures; do not choose among repairs using final test scores.

The current model class is also used for California Housing. Implementation must isolate the News protocol and coordinate with active queues before deployment so existing jobs and Housing behavior are preserved. No experiment code was changed or new training launched for this proposal. Post-hoc clipping would change reported predictions without repairing the diagnosed network behavior.
