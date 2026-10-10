# Full News log-target rerun: launched

The user requested a complete new News experiment on October 8, 2026, and explicitly excluded **all three pilot runs**, including the real-only baseline, from reuse. This note supersedes the earlier plan to extend the pilot namespace.

The new entry point is `scripts/run_news_log_experiment.py`. Its default action is a read-only preflight; training requires an explicit stage. The user explicitly authorized concurrent execution, and the full rerun launched on October 8, 2026, at **6:18 p.m. Chicago**. Production code runs from an isolated source snapshot under `.cache/news_log_runtime_20261008/`; the canonical source files used by existing workers were preserved. Validation copies remain separately under `.cache/news_log_code_validation_20261008/` on the server.

## Live launch

- Persistent session: `news-log-v1`; output namespace: `news_log_v1`.
- Worker command: `taskset -c 6,7 /home/thuy/miniconda3/envs/env/bin/python -u /home/thuy/Research/minh_data_synth/TabularDA/.cache/news_log_runtime_20261008/scripts/run_news_log_experiment.py --namespace news_log_v1 --stage all`.
- Worker log: `/home/thuy/Research/minh_data_synth/TabularDA/output/news_log_v1/worker.log`; status: `worker_status.json` in the same directory. Downstream logs remain under `logs/seed_<seed>/`.
- CPU affinity is two cores; OMP, OpenBLAS, and MKL each use two threads. CUDA uses GPU 0 alongside the existing MNIST28 job. Production epochs, batches, rows, architectures, and quality gates are unchanged.
- The isolated runtime links `data`, `output`, and `sdv trained model` to their canonical directories. It writes only the new News namespace. Labeler caches live under `.cache/news_log_runtime_20261008/.cache/news_log_v1/`. Preserve this runtime for provenance and resumption; run resumes from this exact snapshot.
- All ten transferred files passed byte-level hash checks; replaced snapshot originals are preserved in the runtime's `.deployment_originals/`. The snapshot includes the already deployed metadata/Faker fix. All four reused News features-only checkpoints were additionally checked to contain no Faker transformers.
- At initial verification both real split manifests were prepared, the namespace lock was held, the worker was running, and MNIST28 continued at epoch 170/500. No new downstream records were complete. Launch is not matrix completion.
- At 6:21 p.m. Chicago, the News worker remained active in CPU preprocessing for the first seed-42 CTGAN fit; no generator epoch progress or News GPU allocation had appeared yet. MNIST28 had advanced to epoch 171/500. No errors appeared in the News log.
- Exact command, environment, source hashes, preflight, queue/process/resource snapshots, and timestamps: `audit/news_log_launch_2026_10_08.json`. Current initial health evidence: `audit/news_log_launch_health_2026_10_08.json`. Remote launch evidence is also saved as `output/news_log_v1/launch.json`.

## Frozen experiment

- News only, seeds 42 and 43, using the existing split manifests and unchanged raw/one-hot real partitions. Source namespaces are `corrected_v2_seed42_mnist28_news` for seed 42 and `corrected_v2` for seed 43. Feature columns remain unchanged; a consistent column order is used for the new shared labelers.
- Fit four new full-table generators: CTGAN and TVAE for each seed, trained on `(X, log(shares))`. Keep 500 epochs, batch size 500, CUDA, and 100,000 generated rows. Inverse-transform generated targets with `exp()` before saving tables in shares.
- Reuse all four existing features-only generator checkpoints and their saved RF-arm synthetic features. The old RF targets are discarded. Check production parameters, fitting-table provenance, schema, counts, finite features, split hashes, and file hashes before reuse. No features-only generation or fitting is needed.
- Fit PCA/GMM, RF, XGB, and DNN once per seed on the real training data with log targets: eight labeler fits. Apply each fitted labeler to the four fixed feature sources. Save each table's predictor artifact and, for DNN, development report. PCA/GMM retains the original numerical-column list, 99% PCA variance, ten mixture components, and its existing GMM settings. RF/XGB constructor settings remain unchanged; DNN keeps its existing architecture, optimizer, epoch budget, early stopping, and raw-share R² quality gate.
- DNN labeler standardization applies to log targets. Undo that standardization before `exp()`. Its checkpoint selection still uses development MSE in the standardized target representation; its reported development R² and quality gate remain in shares.
- Run 74 fresh downstream fits: 37 per seed, including a new original baseline and synthetic-only/mix evaluation of all 18 constructions. No pilot or historical downstream weights/results are imported.
- Downstream uses the existing corrected loader and training loop, learning rate 0.001, batch size 128, at most 100 epochs, patience 30, and unchanged loader order/mixing. The News log option removes BatchNorm, retains hidden widths 512/512/256/128 and dropout 0.4, and trains with MSE on log targets. Development checkpoint selection remains raw-share MSE. Test is evaluated after checkpoint selection. Save predictions, MAE, MSE, R², NMAEσ, and required held-out target-normalization metadata in shares.
- The new default namespace is `news_log_v1`, used consistently for prepared data, generator checkpoints, and outputs. Pilot namespaces are rejected. Existing experiment defaults remain raw targets with their original normalization behavior.

## Reuse and failure behavior

Source splits and features-only checkpoints are copied only when byte-for-byte verification succeeds. A different destination artifact is an error. Source changes after the first prepared preflight are an error.

Newly completed full-table samples, labeler caches, and downstream records can be reused inside this namespace. Labeler caches record the real fitting inputs, feature-source hashes, relevant code hashes, combined predictions, and predictor hashes. Each labeler fits once and predicts all four feature groups, preventing redundant fits and making fitted-model comparisons consistent across groups. Changed inputs or differing saved tables are errors. Incomplete labeler artifacts require inspection rather than silent retraining or overwriting.

A nonblocking OS lock at `output/<namespace>/.runner.lock` now prevents two live runners from mutating the same namespace. A second launch exits before preparation or training. The lock is released automatically when the worker exits; the remaining lock file is not a stale running-job marker. This protects the namespace, not GPU/CPU scheduling across different experiments. Verified separately with no-training stubs at 6:08 p.m. Chicago; see `audit/news_log_launch_conflicts_2026_10_08.json`. The earlier 6:00 p.m. validation snapshot precedes this small runner-only addition.

Nonpositive real targets and nonfinite or numerically invalid inverse-log predictions fail explicitly. There is no target clipping, bias correction, automatic retry, substitute model, gate weakening, or selective checkpoint replacement. Full generator samples still use the existing SDV production defaults.

All generated inputs are verified before downstream training begins. Aggregation uses the existing `scripts/build_corrected_results.py --matrix news-log` mode, requiring exactly the 74 expected completed records. Partial runs must not be described as a completed matrix.

## Remote entry points

Use the deployed snapshot at `/home/thuy/Research/minh_data_synth/TabularDA/.cache/news_log_runtime_20261008` with `/home/thuy/miniconda3/envs/env/bin/python`. The commands below are available for later inspected resumption; do not start another worker while `news-log-v1` is active:

```text
python scripts/run_news_log_experiment.py --namespace news_log_v1
python -u scripts/run_news_log_experiment.py --namespace news_log_v1 --stage generators
python -u scripts/run_news_log_experiment.py --namespace news_log_v1 --stage downstream
```

`--stage all` executes generation/relabeling followed by downstream fitting. Long execution must use a persistent remote session and record its command/session/logs at launch. Reinvocation reuses verified completed work; it does not reuse pilots. The runner refuses execution from the laptop. Per-downstream logs go under `output/news_log_v1/logs/seed_<seed>/`; generator/relabeler output should be captured in the persistent worker log.

## Focused validation

Verified October 8, 2026, at 6:00 p.m. Chicago and repeated before launch at 6:16 p.m. Evidence is in `audit/news_log_code_validation_2026_10_08.json`; the current JSON contains the later source hashes and successful checks. The earlier verification notes below describe those checks. The production fits are now running.

- Compiled all changed/new Python sources and checked CLI help and the local-execution guard. Existing default model retains four BatchNorm layers; the log option has no BatchNorm or LayerNorm.
- Verified exact expected coverage of 74 configurations and 37 per seed, using the existing method mapping and result builder.
- Checked log/inverse round trips, negative log predictions converting to positive shares, invalid input rejection, and inverse overflow/underflow failures.
- Checked the actual downstream validation function computes loss and metrics in shares after inverse transformation.
- Used fixed test estimators to verify RF/XGB/PCA target conversion and raw-share metric reporting without fitting production models.
- Ran one isolated 20-row CPU DNN fixture to verify saved log-target normalization, predictor metadata, positive inverse predictions, and raw-share development R². The fixture bypassed convergence enforcement because it did not converge within its 500-epoch budget; this bypass is confined to the validation call. The production runner uses the unchanged enforced gate. The fixture is not an experimental result.
- Used three-row fake generators and a fixed labeler to exercise generation, artifact preparation, table splitting, resume, and changed-input rejection. Two generator fits and one labeler fit supplied all four fixture source groups; rerunning added no fits. These temporary fixture artifacts were removed.
- Actual remote preflight verified the six real partitions per seed, four production X-only checkpoints, and four saved feature tables. Environment: SDV 1.18.0, CTGAN 0.10.2, RDT 1.13.2, PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, XGBoost 2.1.4; CUDA is available.
- Active remote jobs and resources were inspected before isolated validation. Active queues and production source files were preserved. No full generator fit, News relabeling job, downstream experiment, or comparison-report refresh was started.

The production fits are incomplete; in particular, the enforced DNN labeler gate may reject a log-target fit with insufficient raw-share development R². Such a failure must be reported rather than hidden or repaired by tuning a failing arm.
