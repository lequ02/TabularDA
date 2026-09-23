# Corrected experiment run list (frozen before new test scores)

This list derives from `FIX_PLAN.md`, `src/synthesize_data/main.py`, `final_results.xlsx`, and the later `April29.csv` reconstruction. Historical `census` is an Adult alias and is excluded as an independent dataset. Historical `compare_sdv_*` files that relabeled samples from a full-table generator are not valid X-only comparisons.

## Common settings

- Datasets: Adult, genuine Census KDD, Credit, Covertype, Intrusion, MNIST12, MNIST28, News.
- Split version: `corrected_v2`. Make one source-image split for MNIST; use the 60,000/10,000 MNIST28 source version and derive both 12×12 and 28×28 inputs from those IDs. Keep the historical 56,000/14,000 MNIST12 version identified as `legacy_56k_14k`; do not pool its scores.
- Reserve real train/dev/test before any fit. All methods within a dataset and seed share the same real dev/test IDs. Group identical processed features where the claim is performance on unseen features. Record duplicate policy and both raw and model-input overlap counts.
- Seeds: 42, 43, 44. Synthetic sample count: 100,000 for every synthetic-only arm. Each CTGAN and TVAE fit uses 500 epochs and batch size 500 on the configured GPU; record the effective parameters, package versions, and fit data hash. Every classifier arm uses batch size 128, learning rate 0.001, and a budget of 100 epochs with patience 30. Mixed arms use all reserved real training rows plus the same 100,000 synthetic rows. Choose the checkpoint by minimum real-dev loss, then evaluate the restored checkpoint on real test once.
- New artifacts go under `data/corrected_v2`, `sdv trained model/corrected_v2`, and `output/corrected_v2`; no historical CSV/checkpoint/workbook is a corrected result.

## Raw dataset sources

The synthesis step reads the original source again for each generator run, then
creates train/dev/test files under `data/corrected_v2/<dataset>/seed_<seed>/`.
These corrected files do not exist until synthesis has run.

| Corrected dataset | Source used by `src/datasets.py` |
| --- | --- |
| Adult | UCI repository ID 2 via `fetch_ucirepo` |
| Census KDD | UCI repository ID 117 via `fetch_ucirepo` |
| Credit | Local `data/credit/creditcard.csv` |
| Covertype | UCI repository ID 31 via `fetch_ucirepo` |
| Intrusion | Local `data/intrusion/kddcup.data.corrected.csv` |
| MNIST12 and MNIST28 | OpenML `mnist_784`, version 1; MNIST12 is derived from the same source images |
| News | UCI repository ID 332 via `fetch_ucirepo` |

The two local source CSVs are ignored by Git and must be copied to those paths
on the GPU host. The UCI and OpenML loaders need access to their respective
sources (or an existing local cache). Historical `data/<dataset>/` splits are
not inputs to corrected runs.

## Arms for each classification dataset

For Adult, Census KDD, Credit, Covertype, Intrusion, MNIST12, and MNIST28, run each applicable arm below as **synthetic-only** and **mixed** training, plus one real-only arm. A mixed arm uses the same 100,000 synthetic rows and the same reserved real training rows; it gets its own result row. The generator's full-table and X-only models are fitted independently on real train.

| Generator | Target method | Historical family |
| --- | --- | --- |
| none | real target | Original |
| CTGAN full table | generated target | CTGAN |
| TVAE full table | generated target | TVAE |
| CTGAN X-only | GaussianNB, CategoricalNB, PCA-GMM, RF, XGBoost | `gaussian`, `categorical`, `pca_gmm`, `rf`, `xgb` |
| TVAE X-only | GaussianNB, CategoricalNB, PCA-GMM, RF, XGBoost | `sdv_tvae_*` |

The original workbook reports a subset of these dataset/method pairs; `April29.csv` adds TVAE and ensemble families. A missing or unsupported pair remains an explicit failed/skipped run, never a substituted method. The historic full-table `compare_sdv_*` arms are identified separately as invalid relabeling comparisons; they must not be merged with the corrected X-only rows.

## News regression

Run real-only, full-table CTGAN, full-table TVAE, CTGAN X-only plus PCA-GMM regression/RF regression/XGBoost regression, and TVAE X-only plus the same regression labelers. Include synthetic-only and mixed training for each synthetic arm. Use MSE for dev selection; report MSE, MAE, and R² on test. GaussianNB, CategoricalNB, and Bayesian-network classification labelers on the continuous `shares` target are invalid historical configurations, not News regression results.

## Metrics and record gate

Report binary F1 and macro-F1 for Adult/Census KDD/Credit; macro-F1 for Covertype/Intrusion; accuracy and macro-F1 for MNIST; MSE/MAE/R² for News. Credit additionally requires fraud-class precision, recall, and PR-AUC. Save per-seed row-level test predictions and one run record containing source/split hashes, source IDs, label counts, columns, exact and near-duplicate checks, code and package versions, generator metadata/parameters, classifier settings, selected dev epoch, classification synthetic label counts (or a target range for News), and output paths. Tables and figures may use only completed records and must retain real-only, synthetic-only, and mixed rows separately.

No corrected scores are entered here. Run the tiny end-to-end checks, SDV API check, full GPU matrix, and result builder on suitable compute before treating any new comparison as complete.

On a GPU host with the pinned requirements installed, `python scripts/run_corrected_matrix.py --dataset adult --seed 42` runs one isolated dataset/seed and writes logs under `output/corrected_v2/logs`. Run `python scripts/run_corrected_matrix.py` for the complete matrix. The script writes `failures.json`; the result builder refuses to publish tables until every planned run record exists. Neither command has been executed on the local laptop.
