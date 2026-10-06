# Census KDD weighted downstream pilot

Launched October 4, 2026, at 2:46 p.m. Chicago time. User requested one seed only: 42.

Three sequential runs use existing corrected inputs: real-only, CTGAN full-table features relabeled by DNN (`compare_dnn`), and CTGAN features-only relabeled by DNN (`dnn`). Both synthetic arms train on all 100,000 saved synthetic rows; no new generator or labeler is fitted.

The pilot uses the existing Census architecture, including dropout 0.6, and Adam with learning rate 0.001, batch size 128, up to 100 epochs, and patience 30. The output is raw logits for `BCEWithLogitsLoss`, with positive weight equal to each arm's actual training negative count divided by its positive count. Probabilities use sigmoid and predictions use the existing strict `probability > 0.5` rule. Checkpoints maximize real-development macro F1. Training batches remain unshuffled. Scaling and source splits use the existing corrected loader.

This pilot changes both class weighting and checkpoint selection. It checks the combined procedure; it cannot isolate the effect of weighting alone. Weighted losses use different weights across arms and should not be compared as common unweighted BCE values. Binary F1 remains distinct from the macro F1 selection metric.

Remote environment checks passed (torch 2.5.1, SDV 1.18.0, CTGAN 0.10.2). Preflight verifies disjoint source IDs, hashes inputs, checks generator fit-table provenance and 500-epoch/batch-500/CUDA/100,000-sample settings, and checks finite weighted-loss gradients on extreme logits. The transferred runner's SHA256 is `752d3bc257e3c261c9308f3ab9a269e6abc1b0a076b4c7edc390130e55b559ce`. Existing production sources and results were not replaced.

Remote repository: `/home/thuy/Research/minh_data_synth/TabularDA`.

- Session: `census-weighted-pilot-42`.
- Runner: `/home/thuy/miniconda3/envs/env/bin/python -u scripts/run_census_weighted_pilot.py`, with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.
- Queue log: `.cache/census_weighted_pilot_20261004/queue.log`.
- Output namespace: `output/census_weighted_pilot_20261004/`.
- Arm logs: `output/census_weighted_pilot_20261004/logs/seed42_{original,ctgan_full_dnn,ctgan_xonly_dnn}.log`.
- Predictions, checkpoint records, and epoch histories: `output/census_weighted_pilot_20261004/census_kdd/{acc,weight}/`.
- Successful queue completion: `output/census_weighted_pilot_20261004/completed.json` plus three completed `.run.json` records containing `pilot_protocol` metadata and existing prediction/checkpoint files. The shell's `.cache/.../exit_code` marker is not valid evidence; rely on these records and logs.

Initial live verification found the real-only arm training at epoch 1, with 153,031 negatives and 9,931 positives (`pos_weight=15.409425032725808`). Queue PID at launch: 3146369; first training child: 3146480. These are historical launch identifiers, not proof of current liveness or completion.

A subsequent live check reached epoch 6. The saved real-only epoch-4 development checkpoint had macro F1 0.671236, binary F1 0.427409, precision 0.283281, and recall 0.870096. Epoch 5 predicted 3,650 positives among 18,076 development rows. These are intermediate development diagnostics, not final pilot results or evidence of improvement over the unweighted real-only model. CTGAN arms were still queued at this check.

Interpret success using the selected development checkpoint's minority recall, binary F1, macro F1, and predicted-positive count, compared with the preserved unweighted baseline. Merely predicting some positives is insufficient evidence of useful classification. Reserve real test evaluation for the development-selected checkpoint. No pilot result is established by the launch itself.

## Queue repair and resume

The real-only arm completed all 100 epochs and final evaluation, selecting epoch 100. Its test binary F1 is 0.555556, macro F1 0.758615, precision 0.450161, and recall 0.725389. The queue then failed while finalizing extra pilot metadata because the `development_collapse_resolved` expression returned a NumPy Boolean that JSON cannot serialize. Neither CTGAN arm had started.

The expression now explicitly converts to a Python Boolean. A focused remote serialization check was added, and resume validates unchanged input hashes and skips completed records with verified protocol metadata and prediction/checkpoint files. The real-only metadata was recovered from its preserved log and predictions with `audit/repair_census_pilot_metadata_2026_10_04.py`: checks confirmed the full epoch history, selected development score, existing checkpoint, and saved prediction F1. No real-only retraining or new test evaluation was performed. The original script and run record were backed up under `.cache/census_weighted_pilot_20261004/` before repair.

The queue resumed October 4, 2026, at 7:25 p.m. Chicago time in session `census-weighted-pilot-42`, using the same command with `--resume`. Its log is `.cache/census_weighted_pilot_20261004/resume_queue.log`. Repaired runner SHA256: `e3c6ee7d995f974798cdfac6db5f06f00a017b2e6ddad734783c6decb89040d1`. CTGAN outcomes remain pending until their completed records are verified.

## Verified completion

All three runs finished October 4, 2026, at 7:53 p.m. Chicago time. On October 5, remote verification confirmed the completion marker, three protocol-bearing records, existing checkpoints, 9,523 unique test source IDs per prediction table, strict 0.5 decisions, recomputed test binary F1/precision/recall, development-selected checkpoint scores, and identical split manifests against preserved production baselines.

| Seed-42 arm | Selected epoch | Development binary F1 | Test binary F1 | Test precision | Test recall | Test positive predictions | Prior protocol test binary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real-only | 100 | 0.547973 | 0.555556 | 0.450161 | 0.725389 | 933 | 0.404639 |
| CTGAN full-table + DNN | 41 | 0.537949 | 0.542118 | 0.524194 | 0.561313 | 620 | 0.000000 |
| CTGAN features-only + DNN | 36 | 0.544076 | 0.542466 | 0.575581 | 0.512953 | 516 | 0.000000 |

The combined weighted-loss and macro-F1-selection procedure resolves the all-negative collapse in both tested CTGAN arms. This is a one-seed pilot; it does not isolate weighting from checkpoint selection or establish generalization across seeds and datasets. No original production result was overwritten.
