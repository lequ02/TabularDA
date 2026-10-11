# TabDDPM: four-worker pilot and full experiment plan

Revised October 10, 2026. Scope confirmed by the user: **TabDDPM only, seeds 42 and 43**. The seed-5 CTGAN/TVAE text came from another chat and does not apply. Planning inventory: **3:23 p.m. Chicago**. Subsequently implemented, validated, and launched at **10:53 p.m. Chicago**; see the [launch handoff](TAB_DDPM_LAUNCH_2026_10_10.md) for executed changes, exact commands, and status.

## Use 500 epochs; no development-data search is required

The proposed 20,000-versus-30,000-step search is removed. It was an optional baseline-strengthening step, not a requirement for this pilot. Freeze **500 epochs, batch 500, CUDA, and 100,000 synthetic rows per source** for both pilot and full run.

CTGAN/TVAE currently use `epochs=500`. TabDDPM's current `steps=20000` counts optimizer updates, each on a batch. Thus **500 steps is not 500 epochs**. With our DataLoader:

`updates = 500 * len(train_loader) = 500 * ceil(actual_training_rows / 500)`

Run 500 complete passes, including the final short batch, and derive the existing linear learning-rate schedule from that total. Save epochs and actual updates. Development data still selects downstream checkpoints and enforces existing labeler gates; it need not tune the generator budget. [Official batch-update loop](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py).

| Dataset | Historical verified training rows | Updates at 500 epochs/batch 500 |
| --- | ---: | ---: |
| Housing | 14,861 | 15,000 |
| News | 28,543 | 29,000 |
| Adult | 34,280 | 34,500 |
| MNIST12 / MNIST28, each | 54,094 | 54,500 |
| Census KDD | 162,962 | 163,000 |
| Covertype | 513,911 | 514,000 |

Compute actual updates from the verified fit table, not this historical table. Equal epoch counts do not establish equal compute or time across architectures; CTGAN also uses its own random/conditional sampling convention.

Keep **1,000 diffusion timesteps**, MLP widths 256/256, time embedding 128, dropout 0, quantile numerical transforms, native multinomial categorical diffusion, cosine schedule, AdamW learning rate 0.001 with linear decay, weight decay 0.00001, final weights, and ancestral sampling. Diffusion timesteps govern the noise/reverse process, not the number of training passes. No budget or architecture change follows interim test results in this namespace.

## Full design and pilot scope

The full study covers **Adult, Census KDD, Covertype, MNIST12, MNIST28, News, and California Housing**, seeds **42/43**. Credit and Intrusion remain excluded. Fit independent unconditional joint and X-only generators per dataset/seed:

| Construction | Fit data | Synthetic X | Synthetic Y |
| --- | --- | --- | --- |
| Joint baseline | Real training `(X, Y)` | Saved joint sample | Jointly generated target |
| Joint relabeling | Same joint model/sample | Exactly baseline X and row order | Real-trained labeler prediction |
| Our X-only method | Real training X only | Separate saved X-only sample | Real-trained labeler prediction |

Evaluate each table with synthetic-only training and mix training. Mix uses all reserved real training rows plus all 100,000 synthetic rows. Keep batch 128, learning rate 0.001, maximum 100 downstream epochs, patience 30, and each dataset's current corrected development-selection rule.

**Pilot: Adult, Housing, and MNIST12; both seeds; RF/XGB/DNN; both training modes; generated-target and real-only baselines.** Adult tests mixed categorical/numerical classification; Housing tests numerical regression; MNIST12 tests image-derived inputs and the repaired ten-class evaluator while MNIST28 runs in parallel.

The pilot contains **12 full-budget generator fits and 90 primary downstream records**. Each dataset/seed has fifteen records: one real-only plus seven synthetic constructions evaluated twice (`1 + 2*(1 + 2*3)`). Its completed fits, samples, and evaluations belong to the full study and are reused directly. This is not a reduced-epoch or reduced-row smoke run.

**First look:** Adult and Housing seed 42, thirty primary records including two real-only baselines. **Pilot review:** all ninety predefined records, plus any already completed slow-dataset results. Keep the fixed full queues running in the background during analysis. No automatic score-based stopping or dataset selection is introduced. If the user chooses to stop an unpromising exploratory study, record the remaining full matrix as incomplete.

The full inventory retains Gaussian NB, Categorical NB, and PCA/GMM supplementary classification arms, and PCA/GMM supplementary regression arms. Do not drop them to reduce runtime. Run generated-target and RF/XGB/DNN downstream comparisons before supplementary evaluations when their tables are available. Retain the current simple generation-stage boundary: both fits and all labeler tables are created before downstream evaluation starts. No new fine-grained scheduler is needed.

Judge the pilot by correctness, paired utility effects, consistency across both seeds/tasks, harm in mix training, and observed resource/time cost. Show every RF/XGB/DNN result and their equal-weight summary; do not choose the best test labeler. Positive effects across more than one task are useful evidence, but do not invent a significance gate from models sharing the same test set. Treat macro F1, MNIST accuracy, R², D², and NMAEσ as distinct outcomes.

## Four workers and their queues

Use **four independent Python processes** in persistent remote tmux sessions, each running one task at a time. They have separate random states and disjoint dataset ownership, while sharing one GPU.

| Worker/session | Queue, in order | Generator fits | Full downstream records | Reason |
| --- | --- | ---: | ---: | --- |
| `tabddpm-1` | MNIST28 42 → MNIST28 43 | 4 | 54 | Start the widest inputs and costly sampling/downstream work immediately. |
| `tabddpm-2` | Covertype 42 → Covertype 43 | 4 | 54 | Start the largest training table early; 500 epochs require many more updates. |
| `tabddpm-3` | Housing 42 → Housing 43 → News 42 → News 43 → Census KDD 42 → Census KDD 43 | 12 | 130 | Get regression pilot results first, then exercise News log targets; defer the larger Census workload. |
| `tabddpm-4` | Adult 42 → Adult 43 → MNIST12 42 → MNIST12 43 | 8 | 108 | Get mixed-type pilot results, then moderate-width image results. Avoid simultaneous initial preparation of both MNIST resolutions. |
| **Total** | Seven datasets, two seeds | **28** | **346** | Includes fourteen fresh real-only baselines. |

This starts two likely slow streams alongside two faster streams. Ordering reduces coincident preparation but does not guarantee complementary CPU/GPU phases. Four workers are not four GPUs, and previous CTGAN/TVAE concurrency does not prove diffusion speedup. Measure combined throughput, memory, and sampling time before promising a finish date.

Full inventory: **166 synthetic tables, 332 synthetic/mix evaluations, and 14 real-only evaluations**. RF/XGB/DNN reporting uses **210** records; **136** are supplementary NB/PCA-GMM records. The three pilot datasets have 146 full-inventory records, of which ninety are the primary pilot.

Use fresh originals in the new namespace so strict current-source/artifact verification applies uniformly. Existing corrected CTGAN/TVAE records are comparison references; do not refit those generators here. Pair only matching split IDs and valid current protocols.

## Minimal Python changes

General main-file consolidation is underway in another chat. Reconcile its final reviewed source instead of building a second runner or importing audit/cache runtimes into production. The latest read-only inspection still found earlier source in remote main. Staged files are not evidence of completed deployment/validation.

| Existing file | TabDDPM-specific change |
| --- | --- |
| `src/synthesize_data/tabddpm.py` | Add explicit 500-epoch training; derive/log/save actual update count and decay schedule. Preserve explicitly requested step-based smoke checks with conflicting budget arguments rejected. Call the maintained prepared-input verifier. Support News joint fitting on `log(shares)`, inverse `exp` once at output, and log-target arguments to existing labelers/quality/provenance. Persist source paths/hashes and refuse existing or mismatched artifacts. Preserve X-only target independence, schemas, numerical MNIST pixels/rounding, and reload. |
| `src/synthesize_data/main.py` | Pass the explicit epoch budget. Permit News log targets for TabDDPM after that path is implemented/verified; remove the current rejection only then. Reuse current dataset dispatch. |
| `scripts/run_corrected_matrix.py` | Extend existing `run_experiment` with generator selection and epoch budget. Pass both to generation, retain corrected Census/News flags, and prioritize primary TabDDPM comparisons ahead of supplements. Keep current remote/CUDA guard, `check=True`, namespace verification, and strict classifier resume. Preserve unrelated historical/default interfaces. |
| `scripts/build_corrected_results.py` | Reuse the explicit dataset/seed selection already added by the concurrent consolidation. Require the exact 346-record TabDDPM matrix. Further editing is needed only if current validation cannot consume new provenance fields. Aggregate once after all workers finish. |
| `audit/tabddpm_smoke.py` / existing repair checks | Add only missing epoch/update, News log/exp, feature-source hash, and public-dispatch checks. Execute remotely; reuse existing schema, seed-independence, reload, and inventory checks. |

No new `run_tabddpm.py`, per-dataset launch files, scheduler, general predictor cache, or dependency bundle. Existing unconditional TabDDPM methods are already registered. Production failures raise their real errors: no try/except, retries, device/model substitutions, error swallowing, or missing-metric substitutions.

Incorporate/verify the existing corrections through their maintained main files:

| Existing main files | Correction/deployment dependency |
| --- | --- |
| `CreateSyntheticData.py`, `news.py`, `synthesizer.py` | Prepare once and verify hashes/source IDs; shared saved features; save completed fit before sampling; train-only preparation. Preserve existing float32 sequential transformation and no-Faker protections. |
| `classification_main.py`, `classification_train.py`, `model_census.py`, `run_record.py` | Weighted Census logits/BCEWithLogitsLoss, actual training negatives/positives weight including mix targets, sigmoid probabilities, threshold strictly above 0.5, development macro-F1 selection, and saved protocol/counts. Reuse the local consolidation rather than pilot subclasses/runtime activation patches. |
| `commons/log_target.py`, `model_news.py`, `regression_main.py`, `regression_train.py`, `pca_gmm.py`, `ensemble.py`, `dnn_labeler.py`, `run_record.py` | Existing News log-target labeler/downstream implementation, no-BatchNorm News model, raw-share evaluation/selection, and saved normalization. Housing retains raw targets, existing model/dropout, and all eight features. Reconcile with verified successful runtimes and deploy missing/different source. |
| `model_mnist12.py`, `model_mnist28.py`, existing preparation/labelers | Verify repaired ten-logit heads/gradients, natural pixel order, numerical-column PCA routing before one-hot encoding, corrected categorical-NB bins/indicators, fixed grouped split/rare-class handling, and train-fitted encodings/scaling. Preserve these fixes; do not rewrite architectures. |
| `model_multiclass.py` | Include this existing direct dependency if the deployed Covertype trainer imports it. Its architecture is unchanged. |

`commons/log_target.py` and `models_folder/model_multiclass.py` already exist locally but were missing in inspected remote main. They are required dependencies, not speculative new modules. Transfer only changed files and direct dependencies, preserving remote originals and checking hashes. Do not clean unrelated modules/notebooks/artifacts.

## Simple invocation and execution

Extend the current public function, which presently hardcodes CTGAN/TVAE, to accept this proposed call:

```python
run_experiment(seed=42, datasets=["adult"], generators=("tabddpm",),
               generator_epochs=500, namespace="tabddpm_500ep_20261010_v1")
```

Use a short in-session loop per worker, not new launcher files. For worker 4:

```python
for dataset in ["adult", "mnist12"]:
    for seed in (42, 43):
        run_experiment(seed=seed, datasets=[dataset], generators=("tabddpm",),
                       generator_epochs=500, namespace="tabddpm_500ep_20261010_v1")
```

Workers 1–3 use `["mnist28"]`, `["covertype"]`, and `["california_housing", "news", "census_kdd"]`. This fixes dataset-first, seed-second order. Set namespace before importing modules that bind it, through the maintained runner's existing checks. The new generator/epoch function arguments are not currently supported and must be implemented before these calls work.

1. Finish the concurrent consolidation, inspect scoped diffs, and freeze main source. Recheck live processes, tmux, queues, and latest logs before transfer/launch. Do not interrupt or duplicate another chat's work.
2. Validate remotely with `/home/thuy/miniconda3/envs/env`. Latest inventory: Python 3.10.21, PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.15.3, SDV 1.18.0, CTGAN 0.10.2, XGBoost 2.1.4. Recheck versions/CUDA at execution. Reuse existing TabDDPM and correction checks; verify 500 epochs is not 500 updates. Tiny fixtures stay outside production namespaces.
3. Check four-process resource use with production dimensions/batches before expensive fits. Start with two native-library threads per process (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, PyTorch). These are library limits, not CPU affinity. At 3:23 p.m., no experiment workers/default tmux server/GPU compute processes were active; RAM available was approximately 28.4 GiB, GPU memory free 10,924 MiB, disk free 416 GiB. Recheck allocation because other work can start later; inspect active swap I/O rather than treating swap occupancy alone as current pressure.
4. Use `tabddpm_500ep_20261010_v1` under data, generator models, and output. Copy prepared inputs only after checking manifests/source IDs/hashes against corrected references; otherwise use the maintained preparer and verify the resulting split. Split random state remains 42, independent of experiment seed. MNIST resolutions share source-image IDs. Fit transforms/generators/labelers on train only; downstream scaling remains real-train fitted.
5. Launch exactly four persistent workers with the static queues. Record session, command, dataset/seed, namespace, budgets, current stage, source/config hashes, and log with Chicago timestamps. Measure early combined throughput/RAM/GPU use; retain the agreed budgets and actual errors.
6. A failure stops that worker and preserves traceback/artifacts; other disjoint workers can continue. Save completed generators before sampling. No automatic generator overwrite/restart. Classifier-only resume skips only compatible completed records after hash/protocol verification. Interrupted generation/sample/label work requires explicit stage recovery that preserves completed fits. Full optimizer-state training resume is not required for this minimal initial change.
7. Review first-look/pilot results while fixed queues proceed. Keep all labelers and remaining datasets; record harm as well as gains. New settings motivated by test results belong to separately declared follow-ups.
8. Verify records, selected-development weights, source IDs, sample counts, same-X identities, and generator/predictor provenance. Retain News/Housing saved MAE/R²/NMAEσ and test-target normalization (`ddof=0`, rows, target-table hash). Use the existing remote D² sidecar builder for comparison figures. Missing normalization is an error. Aggregate the full 346-record set once; interim views use only completed records and show incompleteness.

Planning evidence: `.cache/tabddpm_plan_20261010/remote_inventory_pilot.json`. The previous proposal is preserved in `plan_before_pilot_revision.md` in that ignored folder. No production experiment tests were executed by this planning turn.

## Conditional TabDDPM follow-up

The initial pilot/full run uses the already implemented unconditional joint/X-only constructions, keeping additional implementation small. Its claim concerns **unconditional joint TabDDPM versus supervised target synthesis**, extending the mechanism study to diffusion.

Native classification TabDDPM conditions on Y and samples conditions from real training class frequencies; our adapter sets `is_y_cond=False`. Do not call this joint baseline a reproduction of native conditional TabDDPM. If the pilot supports a stronger competitiveness study, add native conditional generation and same-X conditional relabeling in a separate follow-up. [Official configuration](https://github.com/yandex-research/tab-ddpm/blob/main/CONFIG_DESCRIPTION.md), [sampler](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/sample.py).

That optional five-classification-dataset/two-seed follow-up requires ten generator fits, eighty main RF/XGB/DNN evaluations, or 140 evaluations including supplementary labelers. Finish/stop the active run first or use a separately frozen source runtime; do not modify source used by current jobs. Gains/significance are not guaranteed, seeds/labelers do not create independent tasks, and MNIST12/MNIST28 share source images. Preserve existing split/overlap/leakage caveats.
