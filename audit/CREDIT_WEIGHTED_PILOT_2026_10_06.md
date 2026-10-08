# Credit weighted macro-F1 pilot

Requested October 6, 2026: three pilots to investigate Credit's all-negative downstream predictions. Seed 42 is used, matching the three-arm Census pilot scope.

The three runs are real-only, synthetic-only CTGAN full-table features with DNN-predicted targets (`compare_dnn`), and synthetic-only CTGAN features-only generation with DNN-predicted targets (`dnn`). Existing corrected generator fits, labelers, 100,000-row synthetic tables, split manifests, and train-fitted scaling are reused. No data generation or labeler training is launched.

Procedure: `BCEWithLogitsLoss` with positive weight = each arm's actual training negatives / positives; real-development macro-F1 checkpoint selection; sigmoid probabilities and strict probability > 0.5 predictions; existing Credit `DNN_Adult` architecture `[128,64,32,16]` with existing BatchNorm/dropout; Adam at learning rate 0.001, batch size 128, at most 100 epochs, patience 30, and unchanged unshuffled batch order.

These pilots change both the objective and checkpoint selection. They do not isolate the causal effect of weighting. Minority precision, recall, binary F1, and confusion counts should be compared with matching preserved original records; merely predicting positives does not establish a useful fix. The held-out Credit test set contains only 10 fraud cases, so report counts along with scores.

Canonical remote repository: `/home/thuy/Research/minh_data_synth/TabularDA`.

- Output namespace: `output/credit_weighted_macro_f1_pilot_20261006/`.
- Persistent session: `credit-weighted-pilot-42`.
- Command: `env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/thuy/miniconda3/envs/env/bin/python -u scripts/run_credit_weighted_pilot.py`.
- Queue log: `.cache/credit_weighted_macro_f1_pilot_20261006/queue.log`.
- Arm logs: `output/credit_weighted_macro_f1_pilot_20261006/logs/{original,ctgan_full_dnn,ctgan_xonly_dnn}.log`.
- Records/predictions/histories: `output/credit_weighted_macro_f1_pilot_20261006/credit/acc/`.
- Selected checkpoints: `output/credit_weighted_macro_f1_pilot_20261006/credit/weight/`.
- `manifest.json` records the exact protocol, seed, package versions, class counts, input hashes, and generator provenance checks. `README.md` explains this namespace. Each run adds `pilot_protocol` and `evaluation_protocol`. `completed.json` is written only after all three commands succeed; final records and artifacts must still be verified.

Runner SHA256: `152d28ae0f3256199dcd4ce0932a67c06d53e356095cf3e8b03d277992fb9189`. Shared weighted helper SHA256: `5ebef8cbfde7f4bac17de2bd08611d7dc8bd8e40a063f7876ce65045578374e9`. The helper was changed only to accept Credit as an explicit dataset; its Census default remains unchanged. The previous helper was preserved in `.cache/credit_weighted_macro_f1_pilot_20261006/helper_before_credit_extension.py`. Transfer hashes and remote Python syntax checks passed. Production modeling sources were not replaced.

Launched October 6 at 11:51 a.m. Chicago time. The fresh process check found no active Python training/generation job. Approximately 29 GB RAM and 10.9 GB GPU memory were available; the root filesystem had approximately 11 GB free. The runner checks at least 1 GB free before preflight and before each arm. No completed research artifact was deleted to make space.

Separately, the Census weighted queue was found stopped, with a prior `OSError: [Errno 28] No space left on device` in its queue log. The Census artifacts were left intact; this launch does not imply that matrix is complete or resumed.

Startup verification confirmed all 19 required input/provenance files and package versions (torch 2.5.1, SDV 1.18.0, CTGAN 0.10.2). Real training has 246,971 negatives and 429 positives; development has 53 positives; test has 10. CTGAN full+DNN has 26,783 positives per 100,000 rows; CTGAN features-only+DNN has 225. The real-only arm was verified actively training at epoch 3. At epoch 2, development recall was 0.792453, precision 0.051597, binary F1 0.096886, with 814 predicted positives. These early diagnostics show positive predictions but poor precision; they are not a final fix or a test result. CTGAN arms remain queued.

## Verified final results

All three pilots finished October 6, 2026, at 12:34 p.m. Chicago time. Verification confirmed all three final records, existing selected checkpoints, identical before/after source partitions and synthetic inputs, the 9,992 real test source IDs and 10 positive targets, strict 0.5 predictions, recomputed F1/precision/recall, and maximal-development-macro-F1 checkpoint selection within the existing tolerance.

| Arm | Previous test binary F1 | Weighted test binary F1 | Precision | Recall | True positives | False positives | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 0.000000 | 0.263158 | 0.151515 | 1.000000 | 10 | 56 | 42 |
| CTGAN full-table + DNN | 0.516129 | 0.516129 | 0.380952 | 0.800000 | 8 | 13 | 25 |
| CTGAN features-only + DNN | 0.000000 | 0.454545 | 0.294118 | 1.000000 | 10 | 24 | 93 |

Selected development binary F1: real-only 0.348548, CTGAN full+DNN 0.645161, CTGAN features-only+DNN 0.522293. Development macro F1, not binary F1 or test performance, selected these checkpoints.

The combined weighting/macro-F1-selection procedure fixes the all-negative collapse for real-only and CTGAN features-only+DNN, but introduces false positives. CTGAN full+DNN has unchanged test binary F1. This is one seed with only 10 test frauds; it does not establish an isolated weighting effect or universally improved performance.
