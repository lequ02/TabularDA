# MNIST downstream repair and parallel rerun plan

Completed October 10, 2026, **6:27:18 a.m. Chicago**: all three workers finished successfully, including supplementary NB/PCA-GMM training. At **11:58 a.m.**, all **212/212** saved downstream records passed the existing runner's artifact/provenance verification and were aggregated with the existing result builder. Both comparison reports now include all **60/60** selected MNIST records per training mode; NB/PCA-GMM stay outside those reports. See [worker completion evidence](../comparison_refresh_preflight_2026_10_10.json), [verified records and aggregation](../comparison_rerun_verification_latest.json), and [latest report checks](../comparison_refresh_2026_10_10.json). The launch and timing estimates below are historical.

Launched October 9, 2026, **12:59:53 p.m. Chicago**, with exactly three concurrent MNIST training workers: `mnist-head-1`, `mnist-head-2`, and `mnist-head-3`. The full 212-fit CTGAN/TVAE matrix is queued; NB/PCA-GMM runs last. The two existing Housing workers were preserved. Launching is not completion evidence.

Live status, exact commands, current jobs and resources are recorded in [launch evidence](../mnist_three_worker_launch_2026_10_09.json) and [latest health check](../mnist_three_worker_health_2026_10_09.json). Each worker's status JSON and logs live remotely in the output namespace below.

## Repair and scope

Only two production lines changed: `DNN_MNIST12.forward()` and `DNN_MNIST28.forward()` now return `self.output(x)`. Architectures, activations, dropout, BatchNorm, labelers, preprocessing, seeds, split manifests, mixing, and evaluation settings are preserved. Remote originals are backed up under `.cache/mnist_head_source_backup_20261009_123717/`. Remote train/eval probes return ten logits and confirm nonzero output weight/bias gradients; no optimizer steps were performed.

Retrain **212 downstream configurations**: two datasets × two seeds × 53 configurations. Each dataset/seed has one real-only baseline and 26 CTGAN/TVAE constructions evaluated with synthetic-only and mixed training. Every downstream model starts fresh; no old downstream weights or scores are reused.

| Phase | Fits | Included configurations |
|---|---:|---|
| Primary | 116 | Real-only; generated targets; full-table and features-only RF/XGB/DNN relabeling, both generators and seeds |
| Supplementary | 96 | Gaussian NB, Categorical NB, PCA/GMM; full-table and features-only, both generators and seeds |

**All primary fits must complete and verify before supplementary labeling/training starts.** The phased worker enforces this barrier. Current reports still exclude NB/PCA-GMM; running supplementary configurations does not add them to those reports.

Reuse 88 verified synthetic tables and all 16 generator checkpoints. Four MNIST12 seed-42 Categorical NB tables need replacement because their saved labelers erased binary indicators. MNIST28 seed 43 needs twelve missing NB/PCA-GMM tables. The secondary preparation step uses the existing corrected labeler functions, real training data, and verified saved generator features; it does not fit or sample a generator. NB alpha/binning and the dataset-specific PCA/GMM settings remain those in the corrected source. Pixels remain categorical for MNIST PCA/GMM (`numerical_columns_pca_gmm=[]`). Affected historical tables remain untouched. Existing MNIST12 TVAE holdout-feature collision caveats remain applicable.

## Runtime and preservation

Namespace: **`mnist_head_fixed_20261009`**. Frozen source: `/home/thuy/Research/minh_data_synth/TabularDA/.cache/mnist_head_fixed_runtime_20261009`. Staged plan/worker: `.cache/mnist_head_rerun_plan_20261009/{plan.json,run_plan.py}`. Output: `output/mnist_head_fixed_20261009/`.

Prepared partitions, retained tables/labelers and generator/provenance files are verified and linked into the new namespace. The four bad Categorical NB and twelve missing tables have no reused link; their new artifacts will be written only in the new namespace. Existing source/result namespaces are preserved. Inputs for MNIST28 seed 42 come from `corrected_v2_seed42_mnist28_news`; the other three blocks use `corrected_v2`. `source_snapshot.json` records the frozen source hash. Completed records are verified before being preserved on an explicit rerun of a worker; partial logs cause an error and require inspection, rather than being overwritten or automatically retried.

Downstream parameters: batch 128, learning rate 0.001, epoch budget 100, patience 30, real-development-loss checkpoint selection, seeds 42/43. Use the same held-out source IDs and original loader order. CPU thread limits control contention and do not change these research settings.

## Capacity and timing evidence

At the 12:32 p.m. inventory, the server had one GTX 1080 Ti (11,264 MiB), four physical CPU cores/eight logical CPUs, about 26 GiB available RAM, and 422 GiB disk space. Only the two Housing workers were doing experiment work. Eight GPU samples ranged 33–43%, averaging 37.75%, with about 640 MiB used.

Five-second fixed-batch GPU probes ran alongside Housing, with no optimizer steps. Two probes reached 96–97% GPU utilization after startup; four reached 100%, with approximately 1,514 MiB total GPU memory. Doubling the same mixed MNIST12/MNIST28 workload from two to four increased combined forward/backward throughput from 626 to 1,119 steps/s (79%). These probes omit CSV loading, Adam updates, validation and CPU metric computation. These earlier probes informed the resource assessment; the user subsequently selected exactly three workers. They do not guarantee full-run speedup or RAM peaks.

Historical task timings from 44 verified completed tasks:

| Dataset | Real-only | Synthetic-only | Mixed |
|---|---:|---:|---:|
| MNIST12 | 5.9 min | 10.5 min | 16.0 min |
| MNIST28 | 8.1 min | 15.0 min | 22.9 min |

The three primary queues contain 38/39/39 fits, with approximately 10.2 historical hours per queue. Each supplementary queue contains 32 fits, with approximately 8.6 historical hours. The resulting ideal elapsed estimate is **18.8 hours**, before label preparation, startup and contention. Corrected losses can change early stopping, so this is provisional and is not a completion forecast. Replace it with timings from completed fixed runs.

At launch, available RAM was approximately 27 GiB and GPU memory use was 638 MiB of 11,264 MiB. Housing had 15/29 seed-42 and 16/29 seed-43 completed downstream fits, and both workers were still training. Its progress and runtime may be affected by shared CPU/GPU use; no Housing job was interrupted or restarted.

## Persistent three-worker sequence

The commands recorded in the launch evidence have already been executed. **Do not launch duplicate workers.** The unlaunched four-lane routing and source snapshot were preserved remotely under `.cache/mnist_head_rerun_plan_20261009/before_three_workers/`; the new plan changes only routing. All 212 scientific configurations and input hashes are unchanged.

| Session | Primary fits | Supplementary fits | First training job |
|---|---:|---:|---|
| `mnist-head-1` | 38 | 32 | MNIST28, seed 42, real-only |
| `mnist-head-2` | 39 | 32 | MNIST28, seed 43, real-only |
| `mnist-head-3` | 39 | 32 | MNIST12, seed 42, real-only |

Each persistent [worker](run_worker.py) runs its primary lane through the existing [phased runner](run_plan.py), then waits for all three primary lanes to finish. Worker 1 verifies the 116 primary records and prepares the sixteen needed NB/PCA-GMM label tables; workers 2/3 wait. After successful label preparation, all three run their supplementary lanes. Worker 1 verifies all 212 completed records after those lanes finish. A failed child process raises its real error and records failure; waiting workers detect failed or dead peers. There are no automatic retries, configuration substitutions, or overwritten partial logs. Exclusive worker/lane locks prevent duplicate training.

Environment: `/home/thuy/miniconda3/envs/env/bin/python`, `CORRECTED_RUN_NAMESPACE=mnist_head_fixed_20261009`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MPLBACKEND=Agg`, `CUDA_VISIBLE_DEVICES=0`. Training still uses batch 128, learning rate 0.001, 100-epoch budget, patience 30, and the saved real development partition. Package versions and deployment hashes are saved in [three-worker deployment evidence](../mnist_three_worker_deployment_2026_10_09.json).

Remote worker logs: `output/mnist_head_fixed_20261009/worker-<lane>.log`; status: `worker-<lane>.status.json`; per-fit logs: `logs/<phase>/<run-id>.log`; checkpoints and records: `<dataset>/weight/` and `<dataset>/acc/`. Attach with `tmux attach -t mnist-head-1` (or 2/3). Dataset, seed, method/mode, source hashes and expected run IDs are in [plan.json](plan.json).

After all 212 records verify, aggregate using the existing builder:

```bash
/home/thuy/miniconda3/envs/env/bin/python \
  /home/thuy/Research/minh_data_synth/TabularDA/.cache/mnist_head_fixed_runtime_20261009/scripts/build_corrected_results.py \
  --runs /home/thuy/Research/minh_data_synth/TabularDA/output/mnist_head_fixed_20261009 \
  --out /home/thuy/Research/minh_data_synth/TabularDA/output/mnist_head_fixed_20261009/results \
  --generators ctgan tvae
```

Then refresh current RF/XGB/DNN report panels from the verified new MNIST namespace, preserving historical results and feature-collision caveats. Supplementary NB/PCA-GMM results remain outside the current report selection. Do not fill missing new coverage with old evaluator scores.

Earlier evidence: [preflight](../mnist_head_rerun_preflight_2026_10_09.json), [concurrency probes](../mnist_head_concurrency_probe_2026_10_09.json), [source deployment/backups](../mnist_head_fix_deployment_2026_10_09.json), [initial staging](../mnist_head_plan_staging_2026_10_09.json), and [prelaunch validation](../mnist_head_plan_final_validation_2026_10_09.json).
