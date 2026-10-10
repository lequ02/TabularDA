# MNIST28 parallel fits — October 8, 2026

The user authorized using the two idle fast workers for the ready, independent MNIST28 seed-43 X-only generator fits. The queue routing was adjusted at **11:53:39 a.m. Chicago**, with no worker restart or change to the running full-table fit.

| Session | Task | Claim time, Chicago | Task log |
| --- | --- | --- | --- |
| `research-slow` | TVAE full-table, seed 43 | October 8, 9:07:36 a.m. (existing run) | `logs/mnist28_seed43_tvae_full_fit_sample.log` |
| `research-fast-1` | CTGAN X-only, seed 43 | October 8, 11:53:39 a.m. | `logs/mnist28_seed43_ctgan_xonly_fit_sample.log` |
| `research-fast-2` | TVAE X-only, seed 43 | October 8, 11:53:43 a.m. | `logs/mnist28_seed43_tvae_xonly_fit_sample.log` |

Remote queue directory: `/home/thuy/Research/minh_data_synth/TabularDA/.cache/full_completion_20261007/`. Existing worker command lines and worker logs remain as documented in [launch_handoff.md](launch_handoff.md). All three child fit processes were live and present in NVIDIA's compute-process list at **11:54 a.m.** GPU utilization was **100%**, memory use **1,768/11,264 MiB**, and available RAM approximately **20 GiB**. Full-table TVAE had completed 328/500 epochs. The two new fits had entered their 500-epoch training loops. These instantaneous resource readings do not establish future peak use or a completion-time estimate.

All initially scheduled non-MNIST28 evaluation blocks were verified complete before reassignment: Census 42 (9), Census 43 (29), MNIST12 43 (15), News 43 (29), Housing 42 (29), Housing 43 (29). MNIST28 had 9/29 evaluations complete. Across the queue after reassignment: **210 completed tasks, 3 running, 29 pending, no failed tasks**; completed downstream evaluations remain **149/169**. Generator launches are not counted as completed fits.

The two pending tasks' scheduling `kind` changed from `heavy_fit` to `fit`, removing the existing slow-worker restriction. Their default owners changed to the two fast workers. This is a routing change only: arguments, dependencies, seeds, training inputs, epochs, batch size, sample count, output paths and scientific settings were preserved. The changes occurred under the existing `state.lock`; the original state was saved first as `state.before_parallel_xonly_20261008.json`. Running and completed tasks were not modified. No production or deployed helper source was edited. Once fits and tables complete, the existing queue will dispatch ready RF/XGB/DNN relabeling and downstream evaluations.

Remote `parallel_xonly_routing_20261008.json` records before/after routing fields, unchanged argument hashes, resource headroom and the backup path. Local evidence is in `audit/experiment_plan_20261007/parallel_fit_preflight_20261008.json`, `parallel_fit_routing_20261008.json` and `parallel_fit_status_20261008.json`. Source hashes still match the original deployment: `experiment_task.py` SHA-256 `907fa17d582e003bd6936f566fe7ff80327d3c1562865bdf97c3c26c86735a58`; `queue_runner.py` SHA-256 `cd3f28ccbc346e9bd8912b025f860d7df67c30506fe22b053a7c823dae1c328a`.

The existing failure policy remains: failures are recorded and surfaced, without automatic retries, changed settings or fallback models. Consult the remote live `state.json` for later progress rather than this dated snapshot.

Follow-up at **11:56:42 a.m. Chicago** confirmed epoch progress in all three fits: CTGAN X-only **1/500**, TVAE X-only **5/500**, and full-table TVAE **333/500**. GPU utilization was **100%**, memory use **1,922/11,264 MiB**, available RAM approximately **20 GiB**, and no queue failures were recorded. Evidence: `audit/experiment_plan_20261007/parallel_fit_progress_20261008.json`.
