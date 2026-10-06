# Corrected experiment run list

The seed-42 method list was frozen before new test scores. Seed 43 was added on 2026-09-28 to repeat every configuration; no methods were added or removed. This list derives from `FIX_PLAN.md`, `src/synthesize_data/main.py`, `final_results.xlsx`, and the later `April29.csv` reconstruction, with the expanded arms requested for the corrected run. Historical `census` is an Adult alias and is excluded as an independent dataset. Relabeling full-table features is a separate experiment from fitting a generator on X only.

## Common settings

- Datasets: Adult, genuine Census KDD, Credit, Covertype, Intrusion, MNIST12, MNIST28, News, California Housing.
- Split version: `corrected_v2`. Make one source-image split for MNIST; use the 60,000/10,000 MNIST28 source version and derive both 12×12 and 28×28 inputs from those IDs. Keep the historical 56,000/14,000 MNIST12 version identified as `legacy_56k_14k`; do not pool its scores.
- Reserve real train/dev/test before any fit. All methods within a dataset and seed share the same real dev/test IDs. Group identical processed features where the claim is performance on unseen features. Record duplicate policy and both raw and model-input overlap counts.

- Seeds: 42 and 43. Synthetic sample count: 100,000 for every synthetic arm. Each CTGAN and TVAE fit uses 500 epochs and batch size 500 on the configured GPU; record the effective parameters, package versions, and fit data hash. Every downstream model arm uses batch size 128, learning rate 0.001, and a budget of 100 epochs with patience 30. Mixed arms use all reserved real training rows plus the same 100,000 synthetic rows. Choose the checkpoint by minimum real-dev loss, then evaluate the restored checkpoint on real test once.
- New artifacts go under `data/corrected_v2`, `sdv trained model/corrected_v2`, and `output/corrected_v2`; no historical CSV/checkpoint/workbook is a corrected result.

October 1 repair: if a classification label is absent from the initial training partition, reserve its first source feature group for training before any fit, retaining whole groups and disjoint source IDs. This corrects Intrusion's unevaluated split, which previously put both `spy.` rows in test. Census KDD's DNN labeler uses unweighted cross-entropy to match its original majority-accuracy gate; the gate is unchanged. PCA/GMM selects numerical source columns before one-hot encoding in both feature-source pipelines. Completed results produced by the previous full-feature PCA selection are retained in an archive before rerunning that arm.

October 4 extension: California Housing is an additional regression dataset. It uses scikit-learn's `fetch_california_housing(as_frame=True)`, all eight numerical features, and the untransformed `MedHouseVal` target in units of $100,000. It uses the shared feature-group split with a requested 20% test set and approximately 10% of the remainder for development; actual counts and source IDs come from its manifest. Its downstream architecture is the same existing `DNN_News` used for News, with the same training settings and regression labelers. This extends the original 816-run matrix by 74 planned runs; it does not imply any California Housing results exist or change the completed eight-dataset results.

## Raw dataset sources

The synthesis step reads the original source again for each generator run, then
creates train/dev/test files under `data/corrected_v2/<dataset>/seed_<seed>/`.
These corrected files do not exist until synthesis has run.

| Corrected dataset | Source used by `src/datasets.py` |
| --- | --- |
| Adult | UCI repository ID 2 via `fetch_ucirepo` |
| Census KDD | UCI repository ID 117 via `fetch_ucirepo` |
| Credit | OpenML dataset 1597 via `fetch_openml` |
| Covertype | UCI repository ID 31 via `fetch_ucirepo` |
| Intrusion | Full KDD Cup 1999 via `fetch_kddcup99(percent10=False)` |
| MNIST12 and MNIST28 | OpenML `mnist_784`, version 1; MNIST12 is derived from the same source images |
| News | UCI repository ID 332 via `fetch_ucirepo` |
| California Housing | scikit-learn `fetch_california_housing(as_frame=True)` |

The UCI, OpenML, and scikit-learn loaders need access to their respective
sources (or an existing local cache). Historical `data/<dataset>/` splits are
not inputs to corrected runs.

## Experiment configuration table

Each synthetic row below is evaluated with **synthetic-only** and **mixed** downstream training. Mixed training combines the same 100,000 synthetic rows with the reserved real training rows. The full-table and X-only generators are fitted independently on real train. Full-table relabeling uses the saved full-table sample's features, drops its generated target, and predicts a new target from a model fitted on real train. It therefore stays separate from both the full-table baseline and the X-only arms.

| Feature source | Target source | Classification methods (7 datasets) | Regression methods (News and California Housing) | Run keys |
| --- | --- | --- | --- | --- |
| Real train | Real target | Real-only baseline | Real-only baseline | `real` |
| Full-table CTGAN | Generated target | CTGAN baseline | CTGAN baseline | `ctgan` |
| Full-table TVAE | Generated target | TVAE baseline | TVAE baseline | `tvae` |
| X-only CTGAN | Predictor trained on real train | GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, DNN | PCA-GMM, RF, XGB, DNN | `gaussian`, `categorical`, `pca_gmm`, `rf`, `xgb`, `dnn` |
| X-only TVAE | Predictor trained on real train | GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, DNN | PCA-GMM, RF, XGB, DNN | `tvae_<method>` |
| Full-table CTGAN features, target dropped | Predictor trained on real train | GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, DNN | PCA-GMM, RF, XGB, DNN | `compare_<method>` |
| Full-table TVAE features, target dropped | Predictor trained on real train | GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, DNN | PCA-GMM, RF, XGB, DNN | `tvae_compare_<method>` |

| Dataset | Task | Generator invocations | Synthetic tables | Downstream runs including real-only |
| --- | --- | ---: | ---: | ---: |
| Adult | Classification | 2 | 26 | 53 |
| Census KDD | Classification | 2 | 26 | 53 |
| Credit | Classification | 2 | 26 | 53 |
| Covertype | Classification | 2 | 26 | 53 |
| Intrusion | Classification | 2 | 26 | 53 |
| MNIST12 | Classification | 2 | 26 | 53 |
| MNIST28 | Classification | 2 | 26 | 53 |
| News | Regression | 2 | 18 | 37 |
| California Housing | Regression | 2 | 18 | 37 |
| **Total, per seed** | | **18** | **218** | **445** |

The extended two-seed target is 36 generator invocations, 436 synthetic tables, and 890 downstream runs. The original eight-dataset target remains 816 downstream runs. Both seeds use the same reserved real split; the seed changes generator and model randomness.

GaussianNB and CategoricalNB are classification-only, so they are excluded from the continuous targets of News and California Housing. Both regression datasets use MSE for downstream dev selection and report MSE, MAE, and R² on test. DNN labeling selects its own checkpoint using the reserved real dev split and fails if dev loss does not plateau or if it does not beat the task-specific trivial baseline: majority-class accuracy and macro-F1 for classification, macro-F1 plus fraud recall and PR-AUC for Credit, and mean-predictor R² for regression. The DNN dev report is saved beside each DNN-labeled CSV; regression DNN `dev_loss` is MSE after target standardization, while `dev_r2` uses the original target units. Passing these gates can be confirmed only after the GPU run.

## Metrics and record gate

Report binary F1 and macro-F1 for Adult/Census KDD/Credit; macro-F1 for Covertype/Intrusion; accuracy and macro-F1 for MNIST; MSE/MAE/R² for News and California Housing. Credit additionally requires fraud-class precision, recall, and PR-AUC. Save per-seed row-level test predictions and one run record containing source/split hashes, source IDs, label counts, columns, exact and near-duplicate checks, code and package versions, generator metadata/parameters, classifier settings, selected dev epoch, classification synthetic label counts (or a target range for regression), and output paths. Tables and figures may use only completed records and must retain real-only, synthetic-only, and mixed rows separately.

Report individual scores for both seeds and their between-seed variation.

## Saved artifacts

All paths below are created on the GPU host during the run. Each downstream run record lists the exact prepared split, generator, predictor (when applicable), synthetic table, and downstream checkpoint paths and fails if a required model file is missing.

| Artifact | Saved location for dataset `D`, seed 42 |
| --- | --- |
| Prepared real train/dev/test | `data/corrected_v2/D/seed_42/D_seed42_real_{train,dev,test}_{raw,onehot}.csv`; `split_manifest.json` records their hashes and source IDs |
| Full-table CTGAN/TVAE and X-only CTGAN/TVAE models | `sdv trained model/corrected_v2/D/seed_42/D_seed42_{ctgan,tvae}_{full,xonly}.pkl`, with matching `.provenance.json` |
| Synthesized tables | `data/corrected_v2/D/seed_42/D_seed42_GENERATOR_{full,xonly}_TARGET_100k.csv`, with matching `.quality.json` |
| Target predictors | Beside each predictor-labeled synthetic CSV: matching `.predictor.pkl` for NB, PCA-GMM, RF, or XGB, or `.predictor.pt` for DNN; the DNN also writes `.dnn.json` dev metrics |
| Downstream evaluation model | `output/corrected_v2/D/weight/*.weights.pth`, selected using real-dev loss |
| Evaluation record and real-test predictions | `output/corrected_v2/D/acc/*.run.json` and `*.predictions.csv` |
| Checked results | `output/corrected_v2/results/per_run.csv` and `summary.csv` |

The full run and the CTGAN pilot use the same filenames. The pilot uses separate `pilot_ctgan_v1` directories. See [PILOT_RUN_SPEC.md](PILOT_RUN_SPEC.md).

No corrected scores are entered here. Run the tiny end-to-end checks, SDV API check, full GPU matrix, and result builder on suitable compute before treating any new comparison as complete.

On a GPU host with the pinned requirements installed, `python scripts/run_corrected_matrix.py --dataset adult --seed 42` runs one isolated dataset/seed and writes logs under `output/corrected_v2/logs`. Run `python scripts/run_corrected_matrix.py` for the complete matrix. The script writes `failures.json`; the result builder refuses to publish tables until every planned run record exists. Neither command has been executed on the local laptop.
