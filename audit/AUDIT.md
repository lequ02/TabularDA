# Code and data audit

Audit date: 2026-09-17. Scope: current working tree, including existing uncommitted changes.

**Verdict: this codebase is not ready to support a claim of bug-free, leakage-free experimental results.** There are reproducible correctness failures and evaluation contamination. This is not evidence of intentional misconduct. An audit cannot establish that no undiscovered bugs exist.

Production code, datasets, saved models, and existing user changes were preserved. Only this `audit/` directory was added. Fixes and experiment regeneration have not been performed.

## Highest-priority findings

### 1. P1 — Shared splitter puts the same records in train and test

Location: `src/commons/create_train_test.py:64–66`, `:86–88`.

When a category exists only in training, the function splits its matching rows but never removes the original matching rows from training. It appends `temp_train` to those originals and appends `temp_test` to test. This both duplicates training records and places test records in training. The opposite branch appends `temp_train` twice.

Reproduction: 40 distinct input IDs become 44 output rows, with ID 13 in both train and test and three duplicate training IDs. The opposite branch yields 43 rows and three duplicate training IDs. These are unique artificial IDs, so the leakage is proven independently of naturally duplicated datasets.

Fix: partition immutable source IDs once; avoid redistributing categories using the test set. Fit an encoder on training and handle unknown categories. Assert disjoint IDs, preserved row counts, and no introduced duplicates. Rebuild all affected splits and downstream synthetic data/models.

### 2. P1 — Test sets contain exact training-row matches

Streaming SHA-256 comparison of the current raw `{dataset}_train.csv` and `{dataset}_test.csv` files found:

| Dataset | Train rows | Test rows | Test rows exactly matching training |
|---|---:|---:|---:|
| Adult | 38,095 | 9,526 | 17 |
| Census | 38,095 | 9,526 | 17 |
| Census KDD | 181,050 | 9,551 | 300 |
| Covertype | 464,809 | 116,203 | 0 |
| Credit | 274,807 | 10,000 | 50 |
| Intrusion | 4,888,431 | 10,000 | 8,082 |
| MNIST12 | 56,000 | 14,000 | 1,114 |
| MNIST28 | 60,000 | 10,000 | 16 |
| News | 31,715 | 7,929 | 0 |

These are exact serialized row matches including the label, excluding headers/newlines. Natural duplicates and lossy image preprocessing can produce identical rows without the same original record ID. These counts therefore do not attribute all overlap to finding 1. Conversely, zero exact matches do not rule out near duplicates, shared entities, or other leakage. Intrusion's 80.82% overlap is especially material if scores are interpreted as performance on unseen records.

Fix: retain source identity, define the intended generalization unit, and group duplicates/related entities before splitting. Record overlap checks with each experiment. The one-hot, synthetic, archived, and NPZ datasets were not exhaustively cross-compared.

### 3. P1 — Adult and the local Census dataset are identical

Locations: `src/datasets.py:53`, `src/modeling_thuy/constants.py:34`; files `data/adult/adult_{train,test}.csv` and `data/census/census_{train,test}.csv`.

Both train files have the same SHA-256; both test files have the same SHA-256. Thus the local `census` results are not independent evidence on a second dataset. A separate `census_kdd` directory and loader exist, while several modeling paths use `census`. The included SDGym README describes Census as KDD Census.

Fix: establish the intended dataset identity, route the correct loader/files, and record dataset hashes and source identifiers in results. Relabel existing results where appropriate; rerun comparisons that intended to use KDD Census.

### 4. P1 — Test performance selects model checkpoints

Locations: `src/train.py:188`, `src/modeling_huy/train.py:193`, `src/modeling_sabina/train.py:240–255`.

These loops use test loss or test F1 for early stopping and checkpoint selection, then report performance on that same test set. This tunes the model to the reported holdout.

Fix: train/validation/test separation; select checkpoints on validation only and evaluate the restored checkpoint on test once. `modeling_thuy` does use a dev set for early stopping, so this specific defect should not be generalized to that loop. However, its dev set is split after augmentation: the generator/labeler has already seen original rows that may enter dev. For an independent real-data validation estimate, reserve dev before fitting either generator or labeler.

### 5. P1 — Preprocessing is fitted independently on validation/test

Locations: `src/data_loader.py:163–166`, `src/modeling_thuy/data_loader.py:146–148`, and the `_standardize` methods in Huy and Sabina loaders.

Every call creates a new scaler and calls `fit_transform`, including evaluation calls. Evaluation therefore uses holdout-wide statistics and a feature coordinate system different from training. This can change scores in either direction.

Reproduction: training values `[0,2]` and test values `[100,102]` both become `[-1,1]`; with the training scaler, test should become `[99,101]`.

Fix: fit preprocessing once on the training partition, persist it with the model, and call only `transform` on dev/test/inference.

### 6. P1 — Huy evaluation discards the true label

Location: `src/modeling_huy/data_loader.py:45–55`.

`train_columns` contains features only. Test alignment drops all other columns, including the target; then `y = test_df.iloc[:, -1]` selects the last remaining feature. Evaluation measures predictions against a feature that is also in X.

Reproduction: expected labels `[0,1]` become `[10,20]`, the final feature's values.

Fix: extract the named target before feature alignment and exclude it explicitly from X. Recompute all affected metrics.

### 7. P1 — GMM posterior calculation depends on other prediction rows

Locations: `src/synthesize_data/GaussMix_nb.py:231`, `:308`.

For each class/component separately, the code subtracts the maximum log likelihood across prediction samples. Those class-specific constants alter relative class likelihoods. For a singleton prediction, every numeric likelihood becomes one, eliminating the numeric evidence entirely. Multiplying unsmoothed categorical probabilities can also produce zero likelihoods; the regression normalization has no zero-denominator guard.

Reproduction: two clearly separated classes predict `[0,1]` together but `[0,0]` individually. The second singleton gets probabilities `[0.5,0.5]`.

Fix: form joint log probabilities including priors and categorical contributions, then normalize across classes/components per sample with log-sum-exp. Add single-row versus batch invariance tests and zero-likelihood tests. Regenerate labels produced by this method.

## Additional confirmed correctness problems

| Priority | Location | Finding and remedy |
|---|---|---|
| P1 | `src/data_loader.py:22` | Constructor gets both datasets from `train_option`; `test_option` does not select the test data. A run labeled train-synthetic/test-original actually tests synthetic data. Load the requested test source explicitly and record actual provenance. |
| P1 | `src/modeling_thuy/data_loader.py:280,334,341`; `classification_train.py:113,151–159` | MNIST12 has 144 features but the loader requires 784. MNIST28 returns `[N,1,28,28]`, while the selected DNN receives input size 1 and expects dense vectors. Reproduced MNIST12 rejection and MNIST28 matrix-multiplication failure. Match loader shape, numeric pixel order, model input size, and summary input. Lexicographically sorting pixel names also scrambles spatial order for CNN use. |
| P2 | `src/modeling_thuy/classification_train.py:292`; `regression_train.py:229` | Reload-and-evaluate of the best checkpoint is commented out; final reported scores come from the last epoch, potentially after patience has expired. Restore the selected checkpoint before the final report. |
| P2 | `src/modeling_thuy/data_loader.py:218` | Synthetic sampling condition compares `df1_sample` against the size of df2 instead of comparing `df2_sample`. Requested 100 total / 20 synthetic produced 120 total / 40 synthetic in a probe. Validate counts and sample each source from its own requested count. |
| P2 | `src/modeling_thuy/data_loader.py:156`; analogous loaders | Manual floor-division batching silently discards remainder rows. Five rows with batch size four become four; three rows crash with an empty ConcatDataset. Use a TensorDataset over all rows with DataLoader and deliberate `drop_last` policy. |
| P2 | `src/synthesize_data/create_synthetic_data/CreateSyntheticData.py:191` | Always stratifies by target, including regression. Unique continuous labels raise a minimum-class-size ValueError. Honor `is_classification`; do not stratify on raw continuous labels. |
| P2 | `src/commons/handle_missing_values.py:21` | `SimpleImputer(strategy)` raises TypeError in the installed environment. Use `strategy=strategy`; fit on training features only, reuse on holdouts, and do not impute target labels as ordinary features. |
| P2 | `src/synthesize_data/ensemble.py:69–96` | `eval_metrics` is defined only when verbose=True but returned unconditionally. Quiet ensemble generation can write its CSV and then raise UnboundLocalError. Compute metrics independently of logging. |
| P2 | `src/synthesize_data/naive_bayes.py:47–63` | CategoricalNB receives min-max-scaled continuous values instead of meaningful integer categories. Fractional values collapse under integer conversion, losing most numeric distinctions. Define training-fitted bins/category encodings or use a model appropriate to continuous features. |
| P2 | `src/trainer.py:40` | Adam is reconstructed every call; the outer loop calls once per epoch. Optimizer moment history resets each epoch. Construct it once and preserve state. |

Several invalid-option branches construct `ValueError(...)` without `raise`, allowing bad configuration to continue. The classification entrypoint also exposes datasets such as Credit/Covertype that its model-selection branches do not implement.

## Security, privacy, and reproducibility

- **Unsafe model deserialization:** `src/synthesize_data/synthesizer.py:424` and `bayes_net.py:245` use `pickle.load`. Loading an untrusted artifact can execute code. The audit did not deserialize any saved model. Establish artifact trust/provenance; for state dictionaries, explicitly use restricted loading supported by the pinned framework version. Synthetic data generation by itself is not a privacy guarantee.
- **Raw data in logs:** splitting, GMM fitting, and loader debug output print real records/features. Notebook output is also retained. Remove or redact row-level logging before using confidential data or sharing logs/notebooks. No confidential-data classification was available to determine actual disclosure.
- **Secret/network review:** a heuristic scan of 282 source/notebook/config/document text files found no matches for private-key blocks, GitHub tokens, AWS access-key IDs, or long literal API-key/password assignments. No obvious data-upload/exfiltration path was found in the reviewed Python code. This is a limited negative finding, not a guarantee; Git history, binaries, archives, every notebook output, and every dependency were not audited for secrets or malware.
- **Dependency setup is incomplete:** root requirements omit direct synthesis dependencies such as SDV, XGBoost, and pgmpy, while the embedded SDGym requirements pin a separate 2019-era stack. The active Python 3.11 environment lacks SDV. Most root requirements are unpinned. Use separate, documented, locked environments if legacy benchmark code must remain.
- **Environment conflicts:** `python -m pip check` reports eight conflicts in the shared active environment. Many concern unrelated installed tools; they are not all defects in this repository. Full dependency vulnerability/advisory verification was not performed.
- **No reliable automated suite established:** files named `test.py` include top-level data-loading/experiment scripts. Broad pytest collection could trigger downloads or experiments. Instead, safe isolated probes were run. Add deterministic regression tests covering the proven failures before accepting corrected results.
- **Provenance gaps:** outputs are not consistently bound to source IDs, split hashes, generator checkpoint hashes, package versions, configuration, and seeds. Existing checkpoints cannot be certified as having used the current split merely because filenames match. Preserve old results as historical artifacts and regenerate a traceable run.
- **Reporting caveat:** `src/modeling_thuy/post_processing.ipynb` consumes `macro_max`/`micro_max` columns from external CSVs unavailable in the reviewed paths. Their upstream selection procedure was not established. Confirm these are not best-over-test-epoch/model scores before treating them as unbiased results.

## Checks performed and limitations

- Parsed 196 Python files; one legacy Python 2 example fails Python 3 syntax parsing (`GLRM/examples/pca_test.py:38`).
- Inventoried 32 notebooks / 361 code cells and inspected targeted data-processing/reporting cells. Notebooks were not executed end to end.
- Ran Ruff's focused fatal/syntax/undefined-name checks: 19 diagnostics across source, notebooks, and vendored examples. Examples include undefined `BayesianNetwork` in benchmark evaluation, undefined `s` in the Adult benchmark maker, and undefined `DNN_Covertype` in Sabina training. Some notebook diagnostics can depend on execution state; do not equate every diagnostic with a confirmed executed failure.
- Ran deterministic runtime probes and streamed exact-row checks over nine saved dataset pairs. Evidence is in `evidence.jsonl` (initial probes plus data checks) and `probes.jsonl` (expanded runtime probes).
- No full model training, saved-checkpoint loading, GPU validation, external dataset downloading, paper/poster verification, or complete privacy attack evaluation was performed. Vendor code received targeted/static review, not line-by-line certification.

Reproduce the read-only checks from the repository root:

```powershell
python -B audit/reproduce.py --data
python -m ruff check src SDGym-research GLRM pyglrm --select E9,F63,F7,F82 --output-format concise --no-cache
python -m pip check
```

The probe program reports observed defects and exits normally when those observations are collected; it is not a green/passing regression suite. It needs pandas, NumPy, scikit-learn, and PyTorch. It only fits a tiny artificial GMM, never production models.

## Recommended repair order

1. Correct dataset identity and split logic; establish immutable train/dev/test IDs and duplicate policy.
2. Fit all preprocessing, generators, and labelers using training only. Correct target extraction, GMM probabilities, and requested evaluation sources.
3. Repair model shapes, batching, mixture counts, regression splitting, and best-checkpoint reporting; add focused regression tests.
4. Pin environments, bind outputs to provenance, regenerate synthetic datasets, retrain, and evaluate on the untouched test set.
5. Recompute tables/figures from those runs; separately assess disclosure/memorization if synthetic data will be released.
