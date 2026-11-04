# Adult TVAE: test evaluation after development-F1 checkpoint selection

Completed October 15, 2026, at 12:43:48 p.m. Chicago. Verification finished at 12:45:11 p.m.

Scope: Adult, full-table TVAE with generated targets, synthetic-only downstream training, seeds 42 and 43. Select the highest real-development binary F1 within each original completed training history. Keep threshold 0.5, data, preprocessing, architecture, optimizer, and batches unchanged.

| Seed | Epoch selected by lowest dev loss | Test binary F1 | Epoch selected by highest dev F1 | Test binary F1 |
|---|---:|---:|---:|---:|
| 42 | 1 | 59.129565% | 1 | 59.129565% |
| 43 | 2 | 54.357298% | 11 | 63.590450% |
| Mean | | 56.743432% | | 61.360007% |

Seed 43 improves by 9.233151 percentage points. Its test accuracy is 83.342086%, macro F1 76.395525%, positive precision 66.522574%, and positive recall 60.905893%. The test confusion matrix is TN=6,550, FP=697, FN=889, TP=1,385, over the same 9,521 held-out rows as the original record.

## Recovery and validation

The original runner retained only the checkpoint with lowest development loss. Seed 42 needs no new training or evaluation because both criteria choose its saved epoch-1 checkpoint. Seed 43's epoch-11 weights were recovered by replaying the first 11 epochs remotely using the source snapshot retained in `G:/SummerResearch_Retained_20261013_2221/source/current`, the original saved synthetic table, the original prepared inputs, seed 43, CUDA, and matching recorded package versions.

Every train/development history field through epoch 11 matched the original history within 1e-8. At epoch 2, every model state tensor was exactly equal to the original saved loss-selected checkpoint. Epoch 11 had already been selected from the original development history before loading the test partition. Its development F1 is 65.801354%. Test F1 was independently recomputed from the new saved predictions; held-out source IDs and true targets exactly match the original saved predictions.

This evaluates the highest-development-F1 checkpoint in the original 33-epoch trajectory. It does not estimate what a separate run with F1-based stopping and a potentially longer trajectory would achieve. The existing experiment records and comparison figures retain their original loss-selection protocol.

All original input, weight, history, and run-record checks passed again after evaluation. The synthetic-table hash matches the October 13 artifact audit; this is not a new historical training-time hash claim.

## Execution and resources

- Server: `thuy@10.24.10.133`; repository `/home/thuy/Research/minh_data_synth/TabularDA`.
- Environment: `/home/thuy/miniconda3/envs/env`; PyTorch 2.5.1, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.5.2, SDV 1.18.0, CTGAN 0.10.2, RDT 1.13.2.
- Dedicated tmux session: `adult-tvae-devf1-20261015`; exited after successful completion.
- Output namespace: `output/adult_tvae_devf1_recovery_20261015_v1`.
- Command: `nice -n 19 taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/thuy/miniconda3/envs/env/bin/python -u recover_dev_f1.py`.
- Log: `output/adult_tvae_devf1_recovery_20261015_v1/worker.log`.
- One CPU thread, CPU affinity 0, GPU allocator cap 8% (about 0.9 GiB), guards requiring at least 8 GiB available RAM and 2 GiB GPU headroom during replay.
- Peak resident memory: 1,101,088 KiB (about 1.05 GiB). Observed total GPU process memory: 192 MiB; PyTorch peak allocated tensors: 18,849,280 bytes.
- Existing experiment PIDs 2312829 and 2317009 remained running throughout and after recovery. Their GPU memory remained 678 and 676 MiB. No other process was interrupted or restarted.

## Evidence

- [Scores and provenance](../output/adult_tvae_devf1_recovery_20261015_v1/result.json)
- [Independent verification](../output/adult_tvae_devf1_recovery_20261015_v1/verification.json)
- [Replayed train/development history](../output/adult_tvae_devf1_recovery_20261015_v1/replay_history.csv)
- [Recovered checkpoint](../output/adult_tvae_devf1_recovery_20261015_v1/recovered/adult_seed43_tvae_max_dev_f1_epoch11.weights.pth)
- [Saved test predictions](../output/adult_tvae_devf1_recovery_20261015_v1/recovered/adult_seed43_tvae_full_generated_synthetic.predictions.csv)
- [Launch record](../output/adult_tvae_devf1_recovery_20261015_v1/launch.json)
- [Worker log](../output/adult_tvae_devf1_recovery_20261015_v1/worker.log)

The isolated recovery script, exact input specification, and retained source snapshot remain on the server in the output namespace and locally under `.cache/adult_tvae_devf1_20261015/`.
