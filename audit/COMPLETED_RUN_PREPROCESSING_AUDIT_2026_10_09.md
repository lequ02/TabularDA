# Completed-run preprocessing audit

Audited October 9, 2026, Chicago time. The frozen inventory started at 10:02 a.m. and contains **871 completed non-Housing real-data runs**, across every namespace under the remote repository's `output/`. Housing's old eight generators were inspected only as a positive control. No fitting, sampling, optimizer steps, experiment changes, or job interruptions occurred. Small CPU inference and gradient probes used copies of saved models in memory.

**Faker was confined to Housing. Other confirmed implementation defects remain in completed MNIST, CategoricalNB, and Adult PCA/GMM results.** Bad training outcomes and default sampling behavior are distinguished below from those defects. Counts refer to completed artifacts, not planned matrices; multiple protocols and pilots are kept separate.

| Dataset | Completed runs inspected | Artifact-level finding |
|---|---:|---|
| Adult | 121 | 16 CategoricalNB runs erase binary indicators; eight full-table PCA/GMM runs send categorical indicators through numerical PCA. |
| Census KDD | 185 | 12 CategoricalNB runs erase binary indicators, including four weighted seed-42 runs reusing affected labelers. Unweighted checkpoint collapse is a separate verified training outcome. |
| Covertype | 121 | Eight seed-42 CategoricalNB runs erase binary indicators. |
| Credit | 109 | No Faker or erased binary-bin/PCA defect detected. CTGAN full-table sampling severely inflates fraud prevalence. |
| MNIST12 | 94 | Every downstream model bypasses its ten-class output layer; eight seed-42 runs also use affected CategoricalNB labels. |
| MNIST28 | 97 | Every downstream model bypasses its ten-class output layer. Includes 15 pilot runs. |
| News | 143 | Three historical raw-target runs have previously replayed catastrophic BatchNorm amplification. All 74 completed `news_log_v1` runs have no BatchNorm state. |
| Intrusion | 1 | Real-only baseline; no completed synthetic run references a generator. |

## 1. No Faker replacement outside Housing

Inspected all **78 distinct fitted generator pickles**, including 70 non-Housing generators. Actual saved metadata and parameters match their provenance after normalizing JSON tuple/list representation. Every non-Housing input column remains in the fitted CTGAN/TVAE transformer. No non-Housing transformer is `AnonymizedFaker`, another PII generator, or a transformer with zero output columns. All 78 generator hashes remained unchanged during inspection.

The non-Housing RDT columns use either categorical passthrough or `FloatFormatter`. The latter's actual settings clip generated numerical values to real-training bounds and learn rounding, consistent with the saved SDV defaults. For example, Credit `Amount` is rounded to cents, while `V1` has no learned decimal rounding. These are ordinary generation transforms; they neither replace observed columns with external fake data nor erase columns before fitting.

All **70 non-Housing generator fit-table hashes, row counts, and column orders reproduce** from saved real training inputs, including the explicitly log-transformed targets of fresh News full-table fits. This is evidence about fit provenance; poor generator fidelity can still occur.

## 2. MNIST: the declared class-output layer is unused

The active remote `DNN_MNIST12.forward()` and `DNN_MNIST28.forward()` return their last hidden tensor before calling `self.output`. Their declared output layers are 256→10 and 128→10, respectively; actual forward output widths are **256 and 128**. Cross-entropy and argmax operate on those widths instead of ten class logits.

Every one of the **191 completed MNIST checkpoints** was loaded and probed with its actual image preprocessing: natural numeric pixel order and unchanged binary values, without the tabular StandardScaler. The first 128 held-out predictions match the saved file for every checkpoint. The output-layer hook is never called, and both output weight and bias receive no gradient. Within each dataset/seed, all methods retain identical output-head parameters. These checks establish actual completed-run use, rather than only a source-code risk.

The saved full prediction files currently contain no labels outside 0–9. That does not repair the objective: training still normalizes cross-entropy over 246 or 118 additional hidden channels. Good accuracy does not establish use of the intended ten-class architecture. Merely attaching the untrained head to old checkpoints cannot repair them.

Repair requires calling the output layer and rerunning all affected downstream arms, including their real-only baselines, under a separate namespace. The inspected generators and synthetic tables can be reused subject to their own provenance and existing overlap caveats. The DNN **labelers** are separate small MLPs and do call their output layers; this finding concerns downstream evaluation models.

## 3. CategoricalNB: fitted bins erase observed binary features

Twenty saved CategoricalNB artifacts contain quantile edges `[0.0]` for some real features that take both 0 and 1. `digitize(0,[0]) == digitize(1,[0]) == 1`. Fitted category counts confirm only one occupied category for **all 2,372 affected artifact/column pairs**, so the information was already lost during training, not just misdescribed in saved bin metadata.

For example, Census KDD seed-42 CTGAN X-only `ACLSWKR_ Federal government` has counts `[0, 162962]`: every training row receives the same code regardless of whether it belongs to that category. The artifacts lose 318 binary indicators per affected Census labeler, 80 per Adult labeler, 41 per Covertype labeler, and 74 per MNIST12 labeler.

| Affected artifacts | Labelers | Completed downstream runs |
|---|---:|---:|
| Adult seeds 42/43, both generators, full/X-only | 8 | 16 |
| Census KDD seeds 42/43, both generators, X-only | 4 | 8 original-protocol + 4 weighted seed-42 |
| Covertype seed 42, both generators, full/X-only | 4 | 8 |
| MNIST12 seed 42, both generators, full/X-only | 4 | 8 |
| Total | 20 | **44** |

Current `naive_bayes.py` explicitly preserves binary indicators with edge 0.5. Existing trained artifacts still have the old defect. Repair requires refitting these labelers, producing new targets on verified existing features, and rerunning their downstream arms. Changing only inference edges is insufficient because the fitted category counts have already lost the signal.

## 4. Adult full-table PCA/GMM: categorical features enter numerical PCA

Four Adult full-table labelers—CTGAN/TVAE, seeds 42/43—have **107 numerical PCA input columns instead of the six original numerical features**. The extra 101 columns are one-hot categorical indicators. Their saved PCA retains 61 components, and their GMMs have zero categorical input columns. The intended Bernoulli treatment of categorical indicators is therefore absent in these artifacts.

This affects **eight completed downstream runs**, synthetic and mix for each labeler. Adult X-only PCA/GMM artifacts do not have this defect. Source fixes cannot retroactively refit the saved PCA and GMM. Refit the affected labelers and rerun their downstream arms; verified full-table generator samples can be reused.

All 54 inspected saved PCA/GMM labelers reproduce their saved target prefix, and their first-row prediction is invariant between singleton and 128-row batches under the current implementation. The older batch-dependent likelihood bug was not reproduced in these completed real-data artifacts. Binary-indicator PCA pollution is a different, still-present artifact problem.

## 5. CTGAN's generation conditions use log frequencies

The installed `DataSampler.sample_original_condvec` says original frequency in its docstring but uses `_discrete_column_category_prob`, which was built with `log(count+1)` when `log_frequency=True`. The fitted generator artifacts retain that default.

Credit has only one categorical full-table column, its target. Thus its generation condition directly chooses fraud with probability

`log(429+1) / (log(246971+1) + log(429+1)) = 0.3281124268`.

Real training prevalence is **429 / 247400 = 0.1734%**. Saved seed-42/43 full-table samples contain **32,793 / 32,792 fraud labels per 100,000**, agreeing closely with the 32.8% conditional probability. This is a concrete library sampling mechanism, not incorrect decoding, negative observed labels, or an un-normalization error.

It affects interpretation of the generated-target comparison: relabelers can benefit partly by correcting a deliberately altered sampling marginal. With multiple categorical columns, target-conditioned probabilities alone do not determine the unconditional generated target frequency, so the Credit calculation must not be generalized numerically to every dataset. Changing this setting requires an explicit, separately recorded experimental protocol; this audit preserved existing settings and results.

## 6. News and Census: verified training failures

Historical raw-target News seed-43 CTGAN full DNN mix, CTGAN generated-target synthetic, and TVAE generated-target mix have saved test R² approximately **−5216.56, −93.09, and −1.50**. Earlier frozen-checkpoint probes directly showed near-zero BatchNorm running variances amplifying held-out hidden activations. One saved CTGAN+DNN prediction is 60.85 million shares for an article with 4,500 actual shares. See [the checkpoint diagnosis](NEWS_SEED43_DIAGNOSIS_2026_10_08.md). Fixed real-first/synthetic-last batch order is a contributor; its independent effect was not established by a controlled retraining experiment.

The present scan finds near-zero BatchNorm variances in 19 historical News checkpoints, but that alone does not prove catastrophic behavior in all 19. All **74 completed `news_log_v1` checkpoints contain no BatchNorm running state**, their metrics/normalization replay, and their test R² spans −0.06283 to 0.01265. They do not show those historical extreme scores. Raw-target News DNN/XGB negative predictions were previously traced to their actual unconstrained output arithmetic, with correct feature and target scaling; see [the negative-label diagnosis](NEWS_NEGATIVE_LABEL_DIAGNOSIS_2026_10_08.md).

Thirty-eight completed unweighted Census checkpoints have exclusively nonpositive output weights and negative bias after a nonnegative final hidden representation. Their fixed 0.5 positive decision is mathematically impossible. This verifies a selected-model failure, not label encoding or Faker corruption; training imbalance and checkpoint selection are discussed in [the Census diagnosis](CENSUS_DNN_OUTPUT_LAYER_DIAGNOSIS_2026_10_04.md). Weighted protocols remain separate.

## 7. Checks that passed and limits

- All 120 non-Housing prepared CSV hashes match every referencing manifest; all 20 distinct manifests match embedded records and have disjoint source IDs.
- All 871 completed records remain identical to the inventory. Their held-out source IDs, true targets, finite predictions, and checked accuracy/macro-F1 or MAE/R² metrics agree. All 143 News saved NMAE definitions/denominators agree with real test targets.
- All 348 distinct saved labelers reproduce their saved targets on the first 128 synthetic rows. Saved DNN feature normalization, regression target mean/scale, and class mappings agree with real training data. All checked labeler hashes remain unchanged. Reproducing labels does not validate the method when its preprocessing is defective.
- The RF/XGB artifacts have `random_state=None`, a reproducibility/provenance limitation. This audit did not establish that their realized seeds are incorrect or that ambient RNG caused a particular score failure.
- Existing MNIST12 TVAE synthetic/holdout feature-overlap caveats remain in [the leakage review](LEAKAGE_REVIEW_2026_09_30.md). Verified training fit hashes do not eliminate naturally matching generated holdout patterns or certify an exclusively unseen-feature task.

Also inventoried 54 earlier Gaussian-only simulated rows, 378 newer seven-dataset simulated rows, and 406 historical simulated rows. Their generator paths use plain `ctgan.CTGAN/TVAE` with explicit discrete columns, without SDV metadata detection/Faker. The newer benchmark uses separate labelers and sklearn MLP downstream models, so the real-data MNIST head defect does not apply. The historical simulated results have previously documented CategoricalNB, RNG, and cache-provenance problems; see [the simulated audit](simulated_benchmark_audit.md). This turn reviewed their inventories/source paths, not a new full replay of all 838 simulated scores. Historical workbooks without run-level lineage are not newly certified.

## Evidence and reuse decisions

- [Inventory](completed_preprocessing_inventory_2026_10_09.json): every completed structured record, namespace, referenced generator and prepared path.
- [Fitted transformers](completed_fitted_transformers_2026_10_09.json): actual retained columns, parameters, library source, hashes, sampling probabilities.
- [Record validation](completed_record_validation_2026_10_09.json): per-run split/metric/checkpoint checks and correctly preprocessed MNIST probes.
- [Saved labelers](completed_saved_labelers_2026_10_09.json): normalization, bin/PCA schema and saved-target replay checks.
- [Finding confirmations](completed_findings_confirmation_2026_10_09.json): exact fitted category counts, PCA widths, MNIST hooks/gradients and target prevalence.
- [Affected-run table](completed_run_preprocessing_findings_2026_10_09.csv): all 871 non-Housing runs with precise defect flags; flags can overlap.
- [Simulated coverage](completed_simulated_coverage_2026_10_09.json): completed table inventories, configurations, and inspected source paths.

Prioritize repairing the MNIST downstream outputs and their baselines, then the affected CategoricalNB and Adult PCA/GMM labelers/downstream arms. These require neither universal generator refits nor rewriting historical scores. Housing's generator refits are a separate ongoing job, preserved during this audit. This audit made no experimental repairs.
