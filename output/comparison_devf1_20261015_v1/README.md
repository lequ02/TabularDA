# Alternate checkpoint reports — October 15, 2026

The remote queue is re-evaluating all **602 distinct source runs** in the synthetic and mix comparison figures. Final plots are generated only after all planned evaluations pass. Consult [live receiver status](client_status.json) for the latest copied status; it updates every five minutes while this computer is running.

## Parallel execution update — October 17

At 1:49 p.m. Chicago, with 446 runs complete, dispatch moved to a parallel scheduler. The active Covertype worker continued uninterrupted; only its old dispatcher was stopped to prevent duplicate launches. The new scheduler retires that dispatcher after its existing worker exits. The three unrelated GPU experiments remain untouched.

- Active session: `comparison-devf1-parallel-20261017`.
- Command: `bash output/comparison_devf1_20261015_v1/launch_parallel.sh`.
- Queue log: `parallel_queue.log`; sampled resources: `parallel_telemetry.jsonl`; current workers: `active.json`.
- Three initial workers, growing to at most five if average GPU utilization remains below 95% and resource reservations fit.
- New workers use normal CPU priority, one Torch/BLAS CPU thread each, and all available CPU cores for scheduling. They retain the original batch size, training computation, seeds, checkpoint rule, and exact history/checkpoint checks.
- Starts are staggered by ten seconds. Dataset rotation interleaves MNIST12, MNIST28 and Covertype.
- RAM reservations use measured dataset/mode peak RSS plus 10% and 0.25 GiB. New workers retain a 6 GiB own-RSS guard and require 8 GiB of system RAM headroom; the original adopted worker retains its earlier 10 GiB system guard until it finishes.
- Each worker retains its 8% GPU allocator cap. Admission reserves 1.1 GiB per worker and at least 3 GiB free GPU memory. These are memory limits, not GPU compute-utilization limits.
- Failed jobs are surfaced and are not automatically retried. The scheduler stops launching additional jobs on a failure and preserves the other active workers' results.
- The completed-report builder and automatic local file receiver remain in use. See `parallel_handover.json` for process identities and source hashes.

The subsequent parallel-worker revision removes per-row CPU batch assembly. The DataLoader iterator and its RNG draws are unchanged; sequential tensor slices are made contiguous and the first batch is checked against the original collation. Original per-epoch history and checkpoint-tensor checks remain mandatory. Existing workers finish with their original implementation; upcoming workers use the new fetcher. The first accelerated worker reproduced its historical development metrics, and five concurrent workers reached a sampled 98% GPU utilization with ample RAM/GPU-memory headroom. The fetcher's source is retained locally in `fast_batch_fetch.py` and remotely in the output namespace.

## Selection

| Dataset | Checkpoint selected by | Test metrics retained in figures |
|---|---|---|
| Adult | Highest development binary F1 | Binary and macro F1 |
| Census KDD | Highest development macro F1 | Binary and macro F1 |
| Covertype | Highest development macro F1 | Macro F1 |
| MNIST12 / MNIST28 | Highest development macro F1 | Accuracy |
| News / California Housing | Highest development R² | R² and D² |

Selection uses each original completed history and chooses the earliest exact maximum. This recovers the comparison requested after the Adult audit; it does not extend training beyond the original stopping point. Losses, splits, seeds 42/43, input tables, learning rate, batch size, thresholds, and architectures are preserved. All source runs in the figures are included; the additional NB/PCA-GMM configurations outside these figures are outside this queue.

There are 308 changed checkpoints requiring historical training replay and 294 unchanged checkpoints requiring test evaluation only. Every News and Housing run selects the same epoch under maximum development R² as under its original loss selection. Census already used macro-F1 selection; two histories contain a slightly higher exact maximum than the original tolerance-based selector retained.

## Outputs

Once complete, the receiver copies these files into `reports/` here:

- `dataset_pipeline_comparison_val_f1.png`, `.csv`, `.md`
- `dataset_pipeline_comparison_mix_val_f1.png`, `.csv`, `.md`
- `checkpoint_selection.csv`, `status.json`, and `artifact_hashes.json`

Reports preserve the requested generator grouping and omit the duplicate TabDDPM real-only reference. Original figures and records are preserved. Each displayed score links to its alternative result record under `jobs/`. Recovered weights and predictions remain on the server, with their hashes and locations recorded in those result records.

## Provenance and caveats

Retained report membership, metadata, histories and downstream source come from `G:/SummerResearch_Retained_20261013_2221`. The server's original row tables are checked against saved hashes before use. Generators and labelers are reused. Every replayed training/development metric must match the original history within 1e-8; all original checkpoint tensors must reproduce exactly. Selected-checkpoint development scores are checked again before test data is loaded. Test row IDs and targets must match the original predictions, and reported metrics are checked independently from new saved predictions.

Some older synthetic tables have hashes from the October 13 saved-artifact audit rather than hashes saved at training time. Exact trajectory and checkpoint reproduction provide additional evidence for recovery; the later hashes alone cannot certify historical training-time bytes. Historical MNIST runtime symlinks are resolved to their physical table before checking that audit hash.

Adult TabDDPM retains its unresolved historical generator-source caveat. MNIST12 TVAE retains its feature-collision caveat. Alternative checkpoint testing does not clear these caveats. Defective configurations excluded from the retained snapshot remain excluded. Paper references and their differing evaluation protocols remain unchanged.

## Original serial execution (superseded by the parallel scheduler above)

- Server: `thuy@10.24.10.133`; repository `/home/thuy/Research/minh_data_synth/TabularDA`.
- Environment: `/home/thuy/miniconda3/envs/env`; original package versions checked per run.
- Namespace: `output/comparison_devf1_20261015_v1`; tmux session `comparison-devf1-20261015`.
- Command: `bash output/comparison_devf1_20261015_v1/launch.sh`; log `queue.log` in that namespace.
- One sequential worker, one CPU thread, CPU affinity 0, niceness 19, GPU allocator cap 8%. The worker exits if available RAM falls below 10 GiB, its own peak RSS exceeds 6 GiB, or GPU headroom falls below 2 GiB.
- Two other experiments, PIDs 2312829 and 2317009, were active at launch. Neither was interrupted or modified.
- The queue stops on errors. Completed verified jobs are preserved and may be resumed. Failed attempts are retained for diagnosis; no substitute model or altered experiment is used.
- The local file receiver performs status reads and artifact transfers only. It stops and writes `receiver_failure.txt` on a queue failure or connection error. No experiment runs locally.

Initial startup checks found and corrected a Python filename collision, the explicit weighted-objective mapping for older Census protocol metadata, and the audit-hash lookup for historical MNIST data symlinks. The failed attempts and their exact source are preserved remotely under `failed_attempts/`. Completed verified evaluations were preserved when resuming.

[Launch record](launch.json) · [Inventory counts](inventory_summary.json)
