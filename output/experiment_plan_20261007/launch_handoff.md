# Remote experiment launch — October 7, 2026

All three persistent tmux sessions launched at **2:44 p.m. Chicago** on `thuy@10.24.10.133`. The latest verified snapshot is **3:23:17 p.m. Chicago**. Old sessions `mnist28-news-seed42` and `summer-research` were removed after confirming they contained only dead panes or stale monitors. Their pane histories were saved; completed artifacts were preserved.

The approved queue covers **169 initially missing downstream evaluations and 16 new generator fits**, plus preparation and RF/XGB/DNN relabeling. It includes unfinished CTGAN/TVAE runs and California Housing. Intrusion, NB, PCA/GMM, extra controls, extra seeds, new classification datasets and Tab-DDPM are excluded. The completed simulated benchmark is not rerun.

| Session | Default scope | Active task at the snapshot |
| --- | --- | --- |
| `research-slow` | MNIST28 seed 43 | CTGAN full-table fit/sample; training epoch **16/500** |
| `research-fast-1` | Census 42 → Census 43 → Housing 42 | Census 42 TVAE full-table DNN relabeling, synthetic downstream training |
| `research-fast-2` | MNIST12 43 → News 43 → Housing 43 | MNIST12 43 TVAE full-table DNN relabeling, synthetic downstream training |

These are default allocations. Workers claim ready tasks under a shared lock, allowing the faster workers to help each other and evaluate MNIST28 tables as they become ready. Only the slow worker claims MNIST28 generator fits. Every task has a single recorded claimant; running work is not duplicated.

**Verified progress:** 5/169 new downstream records completed, leaving 164 evaluations unfinished (including the two running evaluations). Three preparation checks and two missing MNIST12 relabeling tables also completed. Across all 242 queue tasks: 10 complete, 3 running, 229 pending, **0 failed**. No new generator fit has completed yet. The five completed evaluations are:

- `mnist12_seed43_real_original`
- `mnist12_seed43_tvae_full_generated_synthetic`
- `mnist12_seed43_tvae_full_generated_mix`
- `mnist12_seed43_tvae_full_dnn_mix`
- `census_kdd_seed42_tvae_full_dnn_mix`

Completion requires the runner's record validation to pass, including development selection, prediction source IDs, referenced weights and applicable provenance, weighted Census and News normalization checks. A live process or saved checkpoint alone is not counted as a completed evaluation. These records remain remote; this launch task did not refresh local comparison figures or statistical reports.

At the snapshot all three tmux panes were live. GPU utilization was **76%**, GPU memory **1,280/11,264 MiB**, and available RAM approximately **23 GiB**. These are instantaneous measurements, not peak memory guarantees or a completion-time estimate.

## Commands, logs and outputs

Remote repository: `/home/thuy/Research/minh_data_synth/TabularDA`.

Queue directory: `/home/thuy/Research/minh_data_synth/TabularDA/.cache/full_completion_20261007/`.

Each session runs the following command with its own name and log:

```sh
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg \
  /home/thuy/miniconda3/envs/env/bin/python -u \
  /home/thuy/Research/minh_data_synth/TabularDA/.cache/full_completion_20261007/queue_runner.py \
  worker --name research-slow \
  > /home/thuy/Research/minh_data_synth/TabularDA/.cache/full_completion_20261007/research-slow.log 2>&1
```

Use `research-fast-1` or `research-fast-2` for the corresponding worker command/log. Live queue state is `state.json`; task logs are `logs/<task-id>.log`. Existing-source features, completed generator models and synthetic tables are preserved. New X-only feature samples are retained under `features/`; unfinished artifacts replaced by the original trainer are first preserved under `preserved_partial/` with hash checks.

Output namespaces remain `census_kdd_weighted_macro_f1_20261005` for weighted Census and `corrected_v2` for the other newly scheduled results. Existing authoritative seed-42 MNIST28/News results remain in `corrected_v2_seed42_mnist28_news`. Task IDs include dataset, seed, generator, feature source, labeler and training mode.

The queue does **not** automatically retry failed tasks or alter scientific settings. An actual task failure is recorded with its log and causes that worker to exit; other workers can finish independent ready tasks. `completed.json` is written only after all 242 tasks complete. Inspect the failure and dependencies before any manual recovery; do not rerun queue initialization or launch duplicate workers.

## Deployment and evidence

Only three isolated files were deployed. Remote contents matched local SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `experiment_task.py` | `907fa17d582e003bd6936f566fe7ff80327d3c1562865bdf97c3c26c86735a58` |
| `queue_runner.py` | `cd3f28ccbc346e9bd8912b025f860d7df67c30506fe22b053a7c823dae1c328a` |
| `parallel_queues.json` | `ccd6dcbe4d4c6e65395e0694385c8e5376163d387222013f760815831401bed3` |

Remote syntax, environment and production-factory checks passed before launch. CUDA was available; installed versions were torch 2.5.1, SDV 1.18.0, CTGAN 0.10.2, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 and XGBoost 2.1.4. Generator settings remain 500 epochs, batch size 500 and 100,000 synthetic rows; downstream training settings remain unchanged.

Closing the last old tmux session initially raced the old tmux server's shutdown, causing `new-session` to fail before any worker started. The no-worker/pending-only state was verified; a fresh tmux server then launched all three workers successfully. No experiment failed or was retried during this correction.

Local evidence: `audit/experiment_plan_20261007/launch_status.json`, `launch_result.txt`, `launch_preflight_sources.json`, `inventory_full_schedule.json`, and the two deployed script sources. The remote `launch.json` records session commands and launch times; `old_tmux_capture.json` preserves the old panes. The planning JSON's `planned_not_launched` field describes its immutable pre-launch snapshot; the local manifest and this handoff record the actual launch.
