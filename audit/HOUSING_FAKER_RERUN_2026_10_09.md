# Housing rerun after removing Faker

Completed **58/58 downstream fits** on October 9, 2026: seed 43 finished at **3:18 p.m. Chicago**, seed 42 at **3:30 p.m. Chicago**. Both states, exact expected records, prediction source IDs, development checkpoint selection, production settings, saved target normalization, and all eight fresh generator coordinate transformers were verified for the 9:42 p.m. report refresh. Both comparison reports now use this namespace exclusively, including its new real-only baselines. Historical coordinate-flawed scores remain preserved. See [rerun verification](comparison_rerun_verification_latest.json) and [report refresh](comparison_refresh_latest_reruns_2026_10_09.json). The dated notes below describe launch-time state.

Launched October 9, 2026, at **10:00:58 a.m. Chicago** on `thuy@10.24.10.133`. The user explicitly restricted this launch to **California Housing only**. No other dataset was launched or restarted.

Both seeds, 42 and 43, run in persistent sessions `housing-no-faker-42` and `housing-no-faker-43`. Each seed performs four fresh CTGAN/TVAE generator fits (full-table and features-only), RF/XGB/DNN relabeling of both feature sources, and 29 fresh downstream fits: a real-only baseline plus synthetic-only and mix evaluation of fourteen constructions. Total planned coverage is **8 generator fits and 58 downstream fits**. NB/PCA-GMM arms are excluded from this requested report scope.

Generator settings remain 500 epochs, batch size 500, CUDA, and 100,000 synthetic rows per table. Downstream settings remain batch size 128, learning rate 0.001, at most 100 epochs, patience 30, and development-loss checkpoint selection. Existing Housing architectures, target handling, labeler settings and quality gates, loader order, and full-real-plus-synthetic mixing are preserved.

The seven prepared files per seed (six real partition CSVs and the split manifest) were copied from `corrected_v2` and verified against their saved hashes. No generator, predictor, synthetic table, downstream weight, or result is reused. The fixed preprocessing was checked remotely on both seeds, both generators, and both feature sources: all columns are retained, coordinates remain unchanged, and no Faker transformers appear. Installed versions and production parameters passed validation.

The new namespace is `housing_no_faker_20261009` under the remote repository's `data/`, `sdv trained model/`, and `output/` directories. Source code is frozen in `/home/thuy/Research/minh_data_synth/TabularDA/.cache/housing_no_faker_runtime_20261009`; canonical source files and previous outputs were preserved. The existing verified task helper was copied into that snapshot with only its root, cache, and namespace paths adjusted. [The worker source](housing_faker_rerun_20261009/worker.py) sequences those existing tasks and the existing downstream modeling CLI. Failures surface and stop the affected seed; there are no automatic retries or parameter changes.

Each session runs:

```text
env CORRECTED_RUN_NAMESPACE=housing_no_faker_20261009 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 /home/thuy/miniconda3/envs/env/bin/python -u /home/thuy/Research/minh_data_synth/TabularDA/.cache/housing_no_faker_runtime_20261009/scripts/run_housing_faker_rerun.py --seed 42
```

Use `--seed 43` for the other session. Worker logs are `output/housing_no_faker_20261009/seed_<seed>.worker.log`; per-task logs are `logs/seed_<seed>/`; seed states are `seed_<seed>.status.json` in that output directory. These are fresh-run workers, not automatic resume entry points. Inspect any interrupted or failed task before deciding how to resume it; never launch a duplicate worker.

At **10:01:49 a.m. Chicago**, both real-only baselines were actively training at approximately epochs 18–19/100. Both sessions were live, with no failures. Zero downstream run records were complete at that snapshot. All 74 checked historical Housing JSON/checkpoint artifacts remained unchanged. The previous completion queue and News log-target worker had already finished; no existing experiment was interrupted. Launch is not completion, and the comparison figures still show the earlier Housing results with their caveat.

Launch commands, source hashes, split hashes, environment, eight preprocessing checks, and resource snapshots: [launch evidence](housing_faker_rerun_launch_2026_10_09.json). Initial process/log/status and preservation checks: [health evidence](housing_faker_rerun_health_2026_10_09.json). Preserve the runtime for provenance and future recovery.
