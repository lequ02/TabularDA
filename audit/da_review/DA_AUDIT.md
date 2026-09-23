# Audit of `G:/DA` against the earlier `D:/SummerResearch` findings

Audit date: 2026-09-22. Scope: the files currently present in `G:/DA`, compared with the prior [code audit](../AUDIT.md). This is an assessment of the code and saved CSVs, not a certification of every historical run. No `G:/DA` files, checkpoints, or notebooks were modified or executed. I did not load pickle or PyTorch checkpoint files.

**Verdict:** Most important earlier defects remain. Several changes fix the old MNIST input-shape mismatch and add model branches, but they introduce or expose other critical failures. In particular, the saved Covertype test set in this copy has 80.57% exact training-row overlap. Do not treat this copy as a clean replacement or its existing result CSV as a leakage-free benchmark.

## Evidence and version comparison

- SHA-256 of the 85 Python files under each `src/` tree: 60 unchanged, 25 changed, one added in `G:/DA`. Unchanged source substantiates the status of many previously proved code defects; changed files were inspected separately.
- Read-only isolated probes are in [`probe_da.py`](probe_da.py). They do not train models or write to `G:/DA`. All 86 `src` Python files parsed successfully. Focused Ruff checks found six undefined-name diagnostics: four in notebook cells (whose execution context is uncertain), one in `modeling_sabina/train.py:149` (`DNN_Covertype`), and one in legacy `train_old.py:68` (`load_state_dict`).
- Exact raw CSV comparisons used SHA-256 of serialized feature rows (all fields but the last target column) and full rows, with line endings stripped. Headers matched within each pair. Exact features can be naturally duplicated, so they do not by themselves prove the same original source ID. The Covertype result was also independently checked on the one-hot CSVs used for training, with columns aligned by name.
- Historical checkpoints and logs lack a verified link to a specific code commit and split hash. The code/data defects documented below apply to the current copy and to any results produced with the affected paths and files; they cannot automatically be attributed to every saved run.

## Highest-priority findings

| Finding | Status in `G:/DA` | Direct evidence |
|---|---|---|
| Shared split function leaks IDs and duplicates rows | **Persists; one additional regression** | `src/commons/create_train_test.py:29,59-61,80-83`. Forty distinct artificial IDs become 44 rows with one ID in both splits; the opposite branch produces 43 rows and duplicates three training IDs. The new `test_size_frac = test_size / len(df)` treats fractional sizes such as `0.2` as row counts divided by dataset size, unlike the prior version. |
| Exact train/test overlap in saved CSVs | **Persists, with a much worse Covertype split** | Table below. `G:/DA` has a 10,000-row Covertype test file, versus 116,203 in the earlier copy. Its training file has the same SHA-256 as the prior copy. The new test set has 8,057 full-row matches in training; its remaining 1,943 rows match the prior copy's test file. The one-hot files also have 8,057 exact matches after aligning columns. |
| Adult and Census are identical | **Persists** | Both raw train files have identical SHA-256; both raw test files have identical SHA-256. They are not two independent datasets. |
| Census KDD is routed incorrectly in Thuy classifier | **New critical configuration error** | `src/modeling_thuy/constants.py:45`: `census_kdd` calls `create_path_dict('census', 'income')`. Thus the selected KDD classifier reads Adult/Census train, test, and synthetic files, despite a separate `data/census_kdd/` directory and added KDD model. An isolated constants probe confirmed identical `census` and `census_kdd` paths. The separate root loader's KDD branch does not repair this Thuy path. |
| Test data chooses checkpoints | **Persists in root, Huy, Sabina** | `src/train.py:188-190`, `src/modeling_huy/train.py:193-195`, and `src/modeling_sabina/train.py:240-256` still use test loss/F1 for checkpointing. Thuy selects by dev but computes and writes test metrics every epoch (`classification_train.py:238-258`) and reports the final epoch rather than reloading the selected checkpoint (`:270-274`). Its dev split is formed after synthetic data has been generated (`data_loader.py:25-79`). |
| Evaluation data gets its own scaler | **Persists** | `src/modeling_thuy/data_loader.py:146-148`, root `src/data_loader.py:175-176`, and unchanged Huy/Sabina loaders fit a new `StandardScaler` on each call. Artificial train `[0,2]` and test `[100,102]` both became `[-1,1]`. This uses test-wide statistics and different feature coordinates. |
| Huy test labels become a feature | **Persists** | `src/modeling_huy/data_loader.py:45-55` is byte-for-byte unchanged. Its column alignment removes the target before taking the last remaining column as `y`; the earlier isolated probe established this failure. |
| GMM classification depends on prediction batch | **Persists** | `src/synthesize_data/GaussMix_nb.py:181-199` still subtracts the maximum log likelihood across samples separately for each class, changing relative class likelihoods. The changed regression branch does not alter this classification path. Its zero-sum guard avoids division by zero but leaves all-zero probabilities. |
| Root loader ignores requested test source | **Persists** | `src/data_loader.py:22` still initializes both sources from `train_option`; the addition of a KDD loader branch does not change this. |
| MNIST model/data mismatch | **Changed, not fully fixed** | The separate 784-feature/4-D MNIST loader was removed, and the active loader now supplies flat 144/784-feature tensors. This fixes the old input-shape mismatch. But `model_mnist12.py:100-109` and `model_mnist28.py:192-201` define a 10-class `self.output` layer and return the preceding hidden layer without calling it. A two-row forward probe returned `[2,256]` and `[2,128]`, not `[2,10]`; class prediction may include values outside 0–9. |
| Multiclass classification entrypoint crashes on binary F1 | **New** | `src/modeling_thuy/classification_main.py:36` adds `binary` F1 for every dataset; `classification_train.py:346-358` evaluates each requested average. Scikit-learn raises `ValueError: Target is multiclass but average='binary'` on MNIST, Covertype, and Intrusion. |
| PCA-GMM condition raises for a pandas Index | **New regression** | `src/synthesize_data/pca_gmm.py:61` changed a working `len(self.numerical_cols) > 0` test to `self.numerical_cols != []`. The normal caller in `synthesizer.py:88,207` passes a pandas `Index` from `.columns.difference(...)`; comparing a nonempty Index to `[]` raises `ValueError: Lengths must match to compare` in the isolated probe. This blocks those PCA-GMM generation paths. |

### Exact overlap in the saved raw CSV pairs

| Dataset | Train rows | Test rows | Test feature rows in train | Full test rows in train |
|---|---:|---:|---:|---:|
| Adult | 38,095 | 9,526 | 18 | 17 |
| Census (same Adult data) | 38,095 | 9,526 | 18 | 17 |
| Census KDD | 181,050 | 9,551 | 301 | 300 |
| Credit | 274,807 | 10,000 | 50 | 50 |
| MNIST12 (new 60k/10k split) | 60,000 | 10,000 | 832 | 828 |
| MNIST28 | 60,000 | 10,000 | 16 | 16 |
| **Covertype** | **464,809** | **10,000** | **8,057** | **8,057** |
| News | 31,715 | 7,929 | 0 | 0 |

There is no `G:/DA/data/intrusion/` directory with a saved raw split, so the previous Intrusion overlap count of 8,082/10,000 cannot be checked against this copy. Zero exact overlap for News does not rule out other kinds of leakage. The older MNIST12 files under `old_data_mnist12/` are a different split and should not be cross-compared to the current test for leakage estimates.

## Other earlier findings

| Earlier finding | Status and evidence |
|---|---|
| Thuy mixed sampling takes the wrong synthetic count | **Persists.** `src/modeling_thuy/data_loader.py:213-222` is unchanged in the relevant section: the second conditional tests `df1_sample` instead of `df2_sample`. |
| Thuy batching drops remainder and fails below batch size | **Persists.** `data_loader.py:150-173` retains floor division. A five-row artificial input produced four rows; a three-row input with batch size four raised `AssertionError` from an empty `ConcatDataset`. |
| Regression splitter stratifies continuous targets | **Persists.** `src/synthesize_data/create_synthetic_data/CreateSyntheticData.py:131-133` still supplies `stratify=data[target]` even for `news.py:13` with `is_classification=False`. |
| Mean imputation raises | **Persists.** `src/commons/handle_missing_values.py:21` is unchanged and passes `strategy` positionally to `SimpleImputer`; the earlier runtime probe produced `TypeError`. |
| Quiet ensemble returns undefined metrics | **Persists.** `src/synthesize_data/ensemble.py:69-96` is unchanged; `eval_metrics` is assigned under `if self.verbose` but always returned. |
| CategoricalNB applied to scaled continuous values | **Persists.** `src/synthesize_data/naive_bayes.py:42-63` is unchanged. The min-max values are used as CategoricalNB category codes rather than fitted bins. |
| Root Adam optimizer resets per epoch | **Persists.** `src/trainer.py:40` is unchanged and constructs Adam on each `train()` call. |
| Invalid configuration branches do not raise | **Persists.** For example `src/modeling_thuy/data_loader.py:75,198,206` still constructs `ValueError(...)` without raising it. |
| Thuy best checkpoint is not used for final report | **Persists.** `classification_train.py:270` and unchanged regression trainer leave reload/evaluate commented out. |
| Unsafe pickle deserialization | **Persists.** `src/synthesize_data/synthesizer.py:280` and unchanged `bayes_net.py:245` use `pickle.load`; this is a code-execution risk for untrusted artifacts. No artifacts were deserialized in this audit. |
| Real rows/features printed to logs | **Persists.** `commons/create_train_test.py:48`, `modeling_huy/data_loader.py:36`, and other debug statements remain. This is a disclosure route if logs/notebooks are shared; it does not establish that confidential data was actually exposed. |
| Incomplete/unpinned environment and weak provenance | **Persists.** Root `requirements.txt` is byte-for-byte unchanged. No manifest binds result CSVs/checkpoints to current code commit, split hashes, and source IDs. |

The G copy's `src/modeling_thuy/output/best_result_from_csv.ipynb` retains additional reporting defects. Its cell 12 classifies filenames by generic method keywords without first separating `train_mix` from `train_synthetic`; cell 13 averages the resulting groups and uses **maximum test F1 over epochs**. Cell 15 writes `final_results.csv`. The notebook also has a later filename parser (cell 42) that recognizes train mode, but that does not repair cell 15's CSV. These results should be rebuilt from run-level records, with explicit training mode and validation-selected epochs. This CSV is a different artifact from `D:/SummerResearch/final_results.xlsx`; their numbers should not be assumed identical.

## Practical implication and repair order

The current `G:/DA` code/data cannot support a claim that its synthetic methods beat CTGAN on leakage-free tests. The Covertype split alone invalidates an unseen-row interpretation for 80.57% of that test file. Separately, model shape/output, metrics, GMM, PCA-GMM, and dataset routing errors can prevent execution or misstate what was evaluated. These findings do not prove intentional manipulation or quantify the corrected ranking.

Repair and regenerate in this order: (1) freeze source IDs and rebuild disjoint train/dev/test splits, especially Covertype; (2) point `census_kdd` to the real KDD files and remove duplicate Adult/Census benchmarking; (3) fit encoders/scalers/generators on training only and select checkpoints only on dev; (4) fix MNIST outputs, dataset-specific metrics, PCA-GMM, and GMM probabilities; (5) rebuild results from individual run IDs and saved test predictions on the same immutable test IDs. Repeat with recorded seeds and validate the CTGAN label distribution as discussed in the [results review](../results_review/RESULTS_REVIEW.md).

Audit limits: source and saved CSV inspection, focused artificial probes, and notebook text review. No full training, checkpoint loading, complete dependency security review, notebook execution, or exhaustive review of vendored libraries or Git history was performed. An audit cannot prove absence of undiscovered bugs or privacy leakage.
