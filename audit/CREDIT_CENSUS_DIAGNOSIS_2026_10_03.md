# Credit and Census KDD: diagnosis of the plotted synthetic-only results

October 4 follow-up: [saved-checkpoint output-layer inspection](CENSUS_DNN_OUTPUT_LAYER_DIAGNOSIS_2026_10_04.md) shows that eight of the nine failing Census arms have exclusively negative output weights and a negative bias after a nonnegative ReLU representation. Their probabilities are globally bounded below 0.5. This identifies a specific learned output-layer mechanism for the selected models' zero F1, beyond the history-based findings below.

Checked existing remote artifacts on October 3, 2026, about 2:52–3:00 p.m. Chicago time. No experiments, model fits, threshold searches, or changes to research settings were performed. All 30 completed synthetic-only tables represented in the supplied figure were read and their target counts matched their saved run records. Saved predictions and epoch histories were inspected for these tables and the four real-only baselines. Census KDD seed 43 has no completed synthetic-only records for the plotted methods.

Machine-readable evidence: [credit_census_diagnosis_2026_10_03.json](credit_census_diagnosis_2026_10_03.json). Read-only remote diagnostic source: [diagnose_credit_census_2026_10_03.py](diagnose_credit_census_2026_10_03.py).

## Credit

Real training has 429 positives among 247,400 rows (0.1734%). Real development has 53 among 27,415; real test has 10 among 9,992. A representative 100,000-row sample preserving the training prevalence would contain about 173 positives.

| Construction | Positive labels / 100,000, seed 42 | Seed 43 |
|---|---:|---:|
| CTGAN full, generated target | 32,793 | 32,792 |
| TVAE full, generated target | 4 | 1 |
| CTGAN full + RF | 21,991 | 20,979 |
| CTGAN full + XGB | 20,825 | 19,435 |
| CTGAN X-only + RF | 2 | 2 |
| CTGAN X-only + XGB | 1 | 6 |
| CTGAN full + DNN | 26,783 | 27,150 |
| CTGAN X-only + DNN | 225 | 345 |
| TVAE full + DNN | 20 | 91 |
| TVAE X-only + DNN | 197 | 236 |

The mechanisms differ:

- **CTGAN generated targets: inflated positive prevalence and many false positives.** Seed 42 finds all 10 real frauds but flags 172 negatives; seed 43 finds all 10 but flags 196 negatives. This gives binary F1 10.42% and 9.26%. There is no positive-class omission here.
- **A concrete sampling cause for that inflation:** the installed `ctgan.data_sampler.DataSampler.__init__` transforms category counts with `log(count + 1)` when `log_frequency=True`. Its `sample_original_condvec` uses those stored probabilities, and `CTGAN.sample` calls it. The saved full-table generator parameters have `log_frequency=True`; the source marks `Class` categorical. For Credit's training counts, `log(430) / (log(246972) + log(430)) = 0.3281124`, closely matching the generated fraud share. The function's docstring says original frequency, but the inspected implementation uses the log-transformed probabilities. This is an observation about the installed implementation, not a claim that all CTGAN versions behave this way.
- **TVAE generated targets and CTGAN X-only RF/XGB: severe minority underrepresentation.** These arms have only 1–6 positives per 100,000 and predict no real-test positives. TVAE seed 42 has PR-AUC 0.00595 and maximum probability 0.00314. Its failure is more than a threshold mismatch. For X-only RF/XGB, counts describe the regions assigned positive by those labelers; they do not by themselves prove every oracle-positive feature region is absent.
- **Other zero-F1 arms: positives exist, but all probabilities stay below 0.5.** The real-only Credit models also predict all negatives, with maximum probabilities 0.391 and 0.376. TVAE full + DNN seed 42 has 20 synthetic positives, PR-AUC 0.823, and maximum probability 0.1325. CTGAN X-only + DNN has 225/345 positives and maximum probabilities 0.258/0.336. Thus zero binary F1 does not establish zero training positives or zero ranking information.
- **Full CTGAN relabeling helps:** RF/XGB reduce false positives to 1–2 while recovering 7–8 of the 10 frauds. Their binary F1 is 73.68–84.21%, as in the figure.

## Census KDD

Real training has 9,931 positives among 162,962 rows (6.094%). Development has 1,147 among 18,076 (6.345%); test has 579 among 9,523 (6.080%).

| Construction, seed 42 | Positive labels / 100,000 |
|---|---:|
| CTGAN full, generated target | 12,323 |
| TVAE full, generated target | 1,458 |
| CTGAN full + RF | 2,228 |
| CTGAN full + XGB | 3,190 |
| CTGAN X-only + RF | 2,529 |
| CTGAN X-only + XGB | 3,470 |
| CTGAN full + DNN | 3,471 |
| CTGAN X-only + DNN | 4,145 |
| TVAE full + DNN | 1,636 |
| TVAE X-only + DNN | 1,496 |

**Every failing plotted Census arm contains positive training labels.** All nine zero-F1 arms select epoch 1 by real-development loss and predict zero positives on test; their maximum test probabilities range from 0.4087 to 0.4689. All miss the same 579 positives. Their 93.92% accuracy and 48.43% macro F1 are the expected all-negative scores on this test set.

The development histories establish why those checkpoints were selected. For CTGAN full + RF, epoch 1 has development loss 0.1844 and binary F1 0; epoch 32 has training binary F1 0.8355 and development binary F1 0.4386, but development loss 0.4447. Other failing arms also learn positives later while development loss worsens. Selection therefore returns the initial conservative checkpoint; these runs did not stop training after one epoch. Later development F1 is diagnostic evidence, not a replacement result, and no later test scores were selected.

The pipeline uses unweighted binary cross-entropy, a fixed probability threshold of 0.5, and development-loss checkpoint selection. Underrepresented positives and distribution shift are consistent with conservative probabilities followed by poor transfer/overconfidence as training progresses. The observed histories support that account; causal contributions of the loss, architecture, and synthetic features have not been isolated by controlled reruns. Hard relabeling also changes a probabilistic target into a labeler's decision, potentially reducing positive prevalence even when features are retained.

There is direct evidence of feature-distribution distortion. For seed 42, real `CAPGAIN` is exactly zero in 96.37% of training rows, versus 39.36% in CTGAN full and 68.43% in CTGAN X-only. Its standard deviation is 4,646.68 in real training, versus 1,633.86 in CTGAN full and 839.84 in TVAE full. Real `DIVVAL` has standard deviation 1,971.46, versus 326.26 and 432.19 respectively. These show distorted point masses and compressed tails, but their separate effect on classifier performance has not been measured.

CTGAN with generated targets is an exception: 12,323 positives, 370 true positives, 413 false positives, and binary F1 54.33%. It does not exhibit the all-negative prediction behavior.

## What can be called collapse?

For seed 42, all four examined Credit feature sources have 100,000 distinct feature fingerprints out of 100,000 rows. Census CTGAN full/X-only also have 100,000; TVAE full/X-only have 99,998. These checks rule out widespread exact-row repetition, not concentration around a few modes or poor rare-mode coverage.

The supported diagnosis is **severe minority-label underrepresentation in some Credit arms; substantial class-prior inflation in full CTGAN Credit; and minority underrepresentation, feature-distribution distortion, and conservative selected checkpoints in Census KDD**. Neither dataset supports a blanket claim that the synthetic tables contain no positive class. A formal claim of feature mode collapse would require additional coverage or class-conditional diagnostics.

Any investigation of development-chosen thresholds, class weighting, or alternative checkpoint criteria should be a separately documented sensitivity analysis. Preserve the existing fixed-threshold, development-loss results and never choose thresholds or epochs from test performance.
