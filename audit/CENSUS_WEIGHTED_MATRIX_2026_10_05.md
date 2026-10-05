# Census KDD weighted-loss / macro-F1 matrix

This is the expanded evaluation requested after the verified seed-42 pilot. It is a separate protocol from the original loss-selected `corrected_v2` results.

The canonical remote output namespace is `output/census_kdd_weighted_macro_f1_20261005/` under `/home/thuy/Research/minh_data_synth/TabularDA`. Original production results remain under `output/corrected_v2/`; the original pilot remains under `output/census_weighted_pilot_20261004/`. Do not combine those original loss-selected records with the new weighted evaluation.

## Scope and exact procedure

There are 106 planned downstream configurations: 53 for seed 42 and 53 for seed 43. Each seed includes one real-only run and 26 synthetic constructions under both synthetic-only and mixed training. The constructions cover CTGAN/TVAE generated targets, full-table features relabeled by Gaussian NB, Categorical NB, PCA/GMM, RF, XGB, or DNN, and features-only generation with the same six labelers.

Three verified seed-42 pilot runs (real-only, CTGAN full-table+DNN synthetic-only, CTGAN features-only+DNN synthetic-only) are copied into the canonical namespace with verified artifact hashes and source-record provenance. They are not retrained. The remaining 103 configurations require new training.

- Objective: `BCEWithLogitsLoss`, with positive weight computed as actual training negative count / actual training positive count for each configuration, including concatenated counts for mixed runs.
- Checkpoint selection: maximum macro F1 on the existing real development partition; existing patience 30 and improvement tolerance are retained.
- Binary predictions: sigmoid probability strictly greater than 0.5. No threshold tuning.
- Existing Census architecture, BatchNorm/ReLU/dropout 0.6, Adam, learning rate 0.001, batch size 128, at most 100 epochs, unshuffled training batches, and existing seed handling.
- Existing corrected splits and real-training-fitted scaling; no generator or labeler refitting.
- Synthetic-only: all 100,000 saved rows. Mixed: all real training rows plus all 100,000 synthetic rows.

Binary F1 is distinct from the macro F1 used for checkpoint selection. Weighted test losses use arm-specific weights and should not be treated as a common unweighted BCE. Both weighting and checkpoint selection changed; the pilot did not isolate their effects.

## Provenance and operation

Runner: `scripts/run_census_weighted_matrix.py`, reusing `scripts/run_census_weighted_pilot.py` for the exact weighted training procedure. The pilot helper was extended only to accept explicit output/configuration arguments and validate mixed training row counts. Production modeling files were not replaced. The previous remote pilot helper was preserved in `.cache/census_kdd_weighted_macro_f1_20261005/pilot_runner_before_matrix_extension.py`.

The new namespace contains:

- `README.md`: protocol and interpretation.
- `manifest.json`: all configurations, planned/reused/new counts, package versions, hashed inputs, generator provenance checks, and copied pilot artifact provenance.
- `status.json`: completed, failed, and currently running configurations, with Chicago timestamps.
- `logs/`: individual configuration logs; actual failures remain visible and there are no automatic retries or fallback models.
- `census_kdd/acc/`: run records, epoch histories, selected-checkpoint predictions, and plots.
- `census_kdd/weight/`: selected checkpoints.
- `completed.json`: created only after all 106 records pass verification.
- `results/`: built from completed records using `scripts/build_corrected_results.py` for the exact Census-only matrix after successful completion.

New records include `evaluation_protocol` and per-arm class counts/weights. Imported records retain original training provenance and include `origin=verified_pilot_import` and source-record hashes. Original pilot artifacts remain intact.

Preflight validates all 52 synthetic tables' actual target counts against quality records, both classes, 100,000 rows, required predictor artifacts, all eight generator fit-table provenance records and 500-epoch/batch-500/CUDA settings, disjoint split source IDs, package versions, and the pilot's focused weighted-loss/serialization check. Execution is remote-only, with sequential downstream jobs and two CPU threads.

Intended persistent session: `census-weighted-matrix`. Queue command: `env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/thuy/miniconda3/envs/env/bin/python -u scripts/run_census_weighted_matrix.py`. Queue log: `.cache/census_kdd_weighted_macro_f1_20261005/queue.log`. Launch verification and current status are recorded below once confirmed.

## Launch

The remote preflight passed on October 5, 2026: all 106 planned configurations, 52 synthetic tables, and 190 input/provenance files were verified. The queue was launched at 9:58 a.m. Chicago time in session `census-weighted-matrix`. At launch, other active work was Intrusion seed-42 CTGAN generation and MNIST12 seed-43 CTGAN generation; neither was interrupted. Approximately 15 GB RAM was available and 10.3 GB GPU memory was free. Only one additional downstream job runs at a time.

Matrix runner SHA256: `b2d057f8b54c972067508a4690ac7e7f4f115df07af6734f1d318eecc29a2fec`. Weighted training helper SHA256: `77ebb82233c6a9942f46b800cac5b013a3ac8efa5f70b08ac7255c0d3b3e2908`. Remote source transfer hashes and Python syntax checks passed. The original pilot procedure is unchanged; the helper's new configuration arguments allow synthetic/mixed arms and a distinct output namespace.

A successful launch is not a completion claim. Use `status.json`, per-run records, and the final `completed.json` marker for progress. Imported pilot logs remain available in the original pilot namespace; the import ledger links their source records and preserves artifact hashes.

Startup verification at approximately 10:00 a.m. Chicago time confirmed the canonical manifest and all three copied pilot records, zero failures, and active training of `census_kdd_seed42_ctgan_full_generated_synthetic` at epoch 3. There were three verified completed records and 103 configurations left to train. Canonical manifest SHA256 at launch: `4e185c5a4b3aa8c487f36fbc0f008ed29468caf33e1609ca1cda51926ddba9d2`.
