# Three-run News log-target pilot

Launched October 8, 2026, at 12:45:26 a.m. Chicago on `thuy@10.24.10.133`. The user selected log first and limited the pilot to three runs. Yeo-Johnson training is deferred; its transform round-trip was checked during preflight but no Yeo-Johnson experiment was launched.

## Verified final results

All three training runs finished successfully by October 8, 2026, 1:19:23 a.m. Chicago. Verified against final predictions and full CPU checkpoint replay at 4:57 p.m. Chicago. Each record has 7,929 held-out predictions with matching source IDs/targets and saved NMAE normalization. Selected epochs match the minimum raw-scale development MSE in the saved history; no test-based checkpoint selection was used.

| News seed-43 training table | Historical raw + BatchNorm R² | Log + no normalization R² | Historical MAE | Pilot MAE | MAE reduction | Selected epoch / epochs run |
|---|---:|---:|---:|---:|---:|---:|
| Original real-only | -0.002101 | -0.002940 | 3,007.84 | 2,433.42 | 19.10% | 14 / 45 |
| CTGAN generated-target synthetic-only | -93.089477 | -0.031517 | 3,858.22 | 2,446.42 | 36.59% | 31 / 62 |
| TVAE generated-target synthetic-only | -0.027713 | -0.036714 | 2,477.25 | 2,396.91 | 3.24% | 17 / 48 |

Pilot NMAEσ: original 0.256541, CTGAN 0.257911, TVAE 0.252692. All use the same saved test sigma 9,485.506480 shares.

The pilot improves MAE in all three cases but does not establish a general R² improvement: original is slightly worse, TVAE is worse, and CTGAN's improvement chiefly reflects removing its historical catastrophic failure. None has positive R². Log training plus normalization removal are combined changes; the effect of log alone is not isolated, and the failed CTGAN+DNN mix arm was not part of this pilot.

Final prediction ranges in shares: original 528.63–46,854.12; CTGAN 614.01–109,248.18; TVAE 822.44–15,831.12. There are no negative or million-scale final predictions in these three runs. Their prediction means are 1,954.29 / 1,899.58 / 1,370.73 versus a real-test target mean of 3,388.06. These support concerns about upper-tail underprediction; they do not by themselves establish its cause.

Evidence: `audit/news_log_pilot_results_2026_10_08.json`. Local summaries/status are under `output/news_log_pilot_20261008_0048/`, including `pilot_comparison.csv` and `log/results/`. Existing comparison reports remain separate.

## Reporting repair

After training completed, the worker failed in aggregation because it invoked the full-matrix completeness check (887 missing configurations for a three-run pilot). Training records and selected weights were already saved successfully. On October 8 at 4:56 p.m. Chicago, the reporting step was repaired without rerunning training: the existing builder now has an explicit `news-transform-pilot` specification containing exactly these three configurations, and the pilot runner has a summarize-only action using it. Exact-completeness validation remains enforced. The old worker error log is preserved; final status is now `complete` with three completed records.

The two deployed reporting source files and previous status were backed up remotely under `.cache/news_pilot_reporting_repair_20261008/`, and contents were verified by SHA-256. Details are in `audit/news_log_pilot_reporting_repair_2026_10_08.json`. The training-source hash below refers to the original source used to train; the later reporting repair has a separate recorded hash.

## Scope and protocol

- Dataset News, seed 43. Three runs: original real-only, CTGAN full-table generated-target synthetic-only, and TVAE full-table generated-target synthetic-only. No relabelers, mixing, new generator fits, or reduced synthetic tables.
- Same prepared real train/dev/test: 28,543 / 3,172 / 7,929 rows. Each synthetic table has 100,000 rows. Source table hashes and production generator provenance were verified remotely: 500 epochs, batch size 500, CUDA, real-train fitting hash, and requested seed.
- Downstream hidden widths 512/512/256/128, ReLU, dropout 0.4; no BatchNorm or LayerNorm. Inputs use the existing loader's real-train-fitted StandardScaler. Existing training order is preserved (`shuffle=False`) to avoid another protocol change.
- Objective MSE on `log(shares)`; inverse `exp` before raw-scale development scoring and final held-out evaluation. No extra target standardization, clipping, bias correction, fallback, or retries. Non-finite outputs fail explicitly and appear in task logs/status.
- Adam, learning rate 0.001, batch size 128, at most 100 epochs, patience 30 with the existing runner's `stale > 30` stopping convention. Checkpoint selected by raw-scale real-development MSE with tolerance 1e-5. Test is evaluated once after selection; reports persist MAE, MSE, R², and NMAEσ plus the required test normalization metadata.
- Explicit CPU execution, two threads, to avoid adding GPU contention to active experiment queues. Existing GPU queues were inspected and preserved.
- This is exploratory. Historical raw-target baselines used BatchNorm and CUDA, so comparisons with them combine target, architecture, and device differences. Three runs do not isolate the effect of log alone or establish a seed-averaged improvement. No existing comparison results are replaced.

## Execution and evidence

Remote repository: `/home/thuy/Research/minh_data_synth/TabularDA`.

Session: `news_log_pilot_20261008_0048`.

Command:

```text
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/thuy/miniconda3/envs/env/bin/python -u scripts/run_news_target_transform_pilot.py worker --output /home/thuy/Research/minh_data_synth/TabularDA/output/news_log_pilot_20261008_0048 --transform log
```

Output namespace: `output/news_log_pilot_20261008_0048/`.

- `worker.log`: persistent worker progress.
- `pilot_status.json`: three planned tasks with running/completed/failed status and per-task log paths.
- `preflight.json`: source hashes, generator/checkpoint hashes, split counts, and transform round-trip checks.
- `logs/seed43_{real_original,ctgan_synthetic,tvae_synthetic}_log.log`: individual training logs.
- `log/news/acc/*.run.json`, `*.epochs.csv`, `*.predictions.csv`: completed records, development history, and final held-out predictions.
- `log/news/model/*.weights.pth`: selected downstream weights.
- `log/results/`: existing corrected-results builder aggregates completed records after the pilot finishes.
- `pilot_comparison.csv`: final pilot metrics, generated after the worker finishes.

Source: `scripts/run_news_target_transform_pilot.py`, transferred as a new file without replacing existing experiment code; local/remote SHA-256 `29369fe58c6ee9c29a6c16bf885394abc03ee0b303d086da7f37da25db69a8db`.

Local launch evidence: `audit/news_log_pilot_launch_2026_10_08.json`, including current processes/resources, existing queue log tails, exact command, and preflight evidence.

Verified approximately 12:45:52 a.m. Chicago: tmux session and worker/training processes were active; real-only epochs 1–4 had finite objective and development metrics. CTGAN and TVAE were pending. No final held-out results had completed at that inspection. Launch and epoch logs are not completion evidence; use verified `*.run.json` records and status when following up.
