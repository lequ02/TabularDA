# Adult checkpoint-selection audit — October 15, 2026

Verified at 2026-10-15T13:06:05.656529-05:00 (Chicago).

All 134 nonduplicate retained Adult matrix records used development-loss checkpoint selection. Two duplicated real-only TabDDPM reference records were omitted from the counts. Histories came exclusively from `G:/SummerResearch_Retained_20261013_2221`. The 24 known-defective Adult configurations excluded from that retained snapshot were not re-evaluated.

Of 82 configurations passing the saved scientific-validity checks, 71 have a different epoch with strictly higher real-development binary F1. Of the 58 valid configurations in the comparison figures, 54 differ. The other 52 retained configurations are Adult TabDDPM results with the unresolved generator-source caveat; their history inventory does not clear that caveat.

## Additional test results

The three largest additional development-F1 gains among valid figure configurations were selected before evaluating their alternative checkpoints. The previously evaluated TVAE seed-43 generated-target synthetic case was excluded from this selection. All three preselected outcomes are reported below.

| Construction | Training mode | Seed | Epoch: loss → F1 | Old test binary F1 | Alternative test binary F1 | Change, points |
|---|---|---:|---:|---:|---:|---:|
| CTGAN Full-table + XGB | mix | 42 | 1 → 15 | 60.06% | 67.08% | +7.02 |
| CTGAN Full-table + RF | synthetic | 43 | 2 → 14 | 64.07% | 67.47% | +3.40 |
| CTGAN Features-only + XGB | mix | 42 | 1 → 23 | 65.74% | 68.67% | +2.93 |

The earlier TVAE seed-43 generated-target synthetic-only recovery improved test F1 from 54.36% to 63.59% (epoch 2 → 11). Its seed-42 counterpart selects epoch 1 under both rules and stays at 59.13%.

A difference in development optima is not proof of better test F1. The three newly tested alternatives, the previously tested TVAE alternative, and unchanged checkpoints have explicit result statuses in the full inventory. All remaining alternative checkpoints are marked untested. These selected cases do not estimate the average effect across all Adult configurations.

## Recovery and checks

Each selected checkpoint was recovered remotely by replaying the original training prefix on the existing generated table, with the retained source snapshot and matching package versions. Every train/development history field matched the original within 1e-8, and every state tensor at the original loss-selected epoch matched the saved original checkpoint exactly. The checkpoint with maximum development F1 was fixed before loading the held-out test partition. The original completed history, rather than a potentially longer training trajectory with F1-based stopping, defines the selection scope.

Independent checks recomputed F1 from saved predictions, matched test source IDs and true targets against the original predictions, and rechecked original run records, input hashes, weights, histories, and recovered artifact hashes. All three use the same 9,521 held-out rows and threshold 0.5. The original matrix records and comparison figures were preserved.

Higher F1 need not imply lower probability loss: for CTGAN full-table XGB mix seed 42, test F1 improves from 60.06% to 67.08%, while test loss increases from 0.413543 to 0.474590. The loss-selection rule was followed correctly; this audit evaluates its consequences for the reported F1 metric.

## Server execution

- Server environment: `/home/thuy/miniconda3/envs/env`; recorded original package versions checked for each replay.
- Session: `adult-devf1-audit-20261015`; sequential queue command: `bash output/adult_devf1_audit_20261015_v1/worker_queue.sh`.
- Output namespace and queue log: `output/adult_devf1_audit_20261015_v1/queue.log`.
- One replay at a time, one CPU thread, CPU affinity 0, niceness 19, GPU allocator cap 8% (about 0.9 GiB), 8 GiB minimum available RAM and 2 GiB minimum GPU headroom.
- Both pre-existing experiment PIDs 2312829 and 2317009 remained active after verification. No other job was interrupted or restarted.

## Evidence

- [All retained matrix configurations and evaluation status](../output/adult_devf1_audit_20261015_v1/adult_checkpoint_selection_audit.csv)
- [Audit counts and selection scope](../output/adult_devf1_audit_20261015_v1/summary.json)
- [Independent verification and three comparisons](../output/adult_devf1_audit_20261015_v1/verification.json)
- [Complete replay results](../output/adult_devf1_audit_20261015_v1/completed.json)
- [Predeclared candidate selection](../output/adult_devf1_audit_20261015_v1/selected_candidates.json)
- [Launch record](../output/adult_devf1_audit_20261015_v1/launch.json)

Each run's result, replay history, recovered checkpoint, and test predictions are preserved in the `jobs/` directory of the output namespace. Recovery source and the exact input specifications are preserved in its `payload.zip`, with local preparation files under `.cache/adult_devf1_audit_20261015/`.
