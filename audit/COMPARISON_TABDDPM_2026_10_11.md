# TabDDPM comparison tables

Verified snapshot: **October 11, 2026, 11:03:37 a.m. Chicago**. Existing CTGAN/TVAE reports retain their October 10 source records, all of which were checked unchanged on the server during this update.

Both reports add seven TabDDPM construction rows: generated targets, full-table RF/XGB/DNN relabeling, and X-only RF/XGB/DNN labeling. Fresh real-only baselines appear in a separate `Real-only (TabDDPM run)` row, retaining their own provenance. The earlier baseline and every earlier CSV cell are unchanged. TabDDPM records use `tabddpm_500ep_20261010_v1`; CTGAN/TVAE namespaces are preserved. NB/PCA-GMM remain outside these tables.

Each report contains **72/112** completed selected TabDDPM records, **282** completed records overall, and **540** displayed metric values. Across both reports, **300 added metric values** were independently matched to verified TabDDPM records/sidecars; the previous **780** values remain unchanged. All **1,764** earlier CSV cells, including paper rows and metadata, are preserved. Sources for the previous 406 run records and 116 D² sidecars were rechecked remotely: **522 unchanged hashes**.

| Dataset/seed | Completed selected records per report | Planned |
|---|---:|---:|
| Adult 42 | 8 | 8 |
| Adult 43 | 8 | 8 |
| Housing 42 | 8 | 8 |
| Housing 43 | 8 | 8 |
| MNIST12 42 | 8 | 8 |
| MNIST12 43 | 0 | 8 |
| MNIST28 42 | 8 | 8 |
| MNIST28 43 | 8 | 8 |
| News 42 | 8 | 8 |
| News 43 | 7 | 8 |
| Covertype 42 | 1 | 8 |
| Covertype 43 | 0 | 8 |
| Census KDD 42 | 0 | 8 |
| Census KDD 43 | 0 | 8 |
| Total | **72** | **112** |

The full TabDDPM experiment has **195/346** verified completed records at this snapshot, including supplementary configurations. This is partial coverage. Covertype seed 42 generation stopped with `ValueError: Nonfinite Tab-DDPM loss at step 70869`; its seed-43 queue has not proceeded. Census KDD has not started at this snapshot. MNIST12 seed 43 and one News seed-43 selected arm remain pending. Missing results are blank, never zero or a substituted generator's score. No worker was restarted or interrupted.

Verification ran remotely with one CPU thread and reduced scheduling priority. It checked launch source hashes, all saved artifact hashes, prepared-partition file metadata and split IDs against the existing comparisons, saved prediction IDs and finite targets/predictions, and the minimum-development-loss or maximum-development-macro-F1 selected epoch. Canonical train-only full/X-only fit tables reproduce the saved generator fit hashes. Provenance confirms **500 complete epochs**, batch 500, CUDA, 1,000 diffusion timesteps, sampling batch 500, 100,000 rows per source, and `steps = 500 × updates_per_epoch`. These are unconditional joint/X-only constructions; the results are not a reproduction of native target-conditional TabDDPM.

The existing `scripts/build_corrected_results.py` aggregated a frozen copy of all 195 completed records under remote `.cache/comparison_tabddpm_20261011_110053/aggregate/`. Its expected keys were scoped to that completed snapshot; the 346-run production plan was not changed. Original run records, predictions, checkpoints, tables and source code stayed unchanged on the server.

Housing/News use saved R² and NMAEσ. D² absolute error was computed from the selected completed held-out predictions with the maintained `scripts/compute_regression_d2.py`, preserving originals in separate `.d2.json` sidecars. The update produced **58 distinct new regression sidecars** (four real-only records occur in both reports), which supply **62 added D² cells**. Each sidecar records the test median, median-baseline MAE, prediction MAE, rows, selected epoch and original-record/prediction/test-table hashes. No target normalization is recomputed during plotting.

The figures preserve the signed-log News axes, same-row R²/D² points, existing paper lines, Credit/Intrusion exclusions and documented leakage caveats. TabDDPM has its own metric colors. Figure height grows with the added rows to preserve readable row spacing. CSVs retain their 49 columns and now have 26 data rows: 23 configurations and three paper references. No TabDDPM paper reference is invented. Both CSVs were imported and inspected with the spreadsheet tool and their changed ranges were rendered for review.

## Refresh

Use the verified remote fetch workflow `.cache/remote_intrusion/fetch_tabddpm_comparison_20261011.py` before rebuilding the existing comparison builder. It captures completed records, verifies provenance/artifacts, aggregates the partial snapshot, computes regression sidecars and downloads `snapshot.zip`. Cached copies with different JSON formatting are backed up and replaced with the exact verified remote bytes; semantic differences fail explicitly. The builder requires the matching `audit/comparison_tabddpm_snapshot.json` and rejects unverified source records.

Original reports and builder: `.cache/remote_intrusion/comparison_before_tabddpm_20261011_105505/`. The remote verification, aggregate and ZIP remain in `.cache/comparison_tabddpm_20261011_110053/`. Experiment-artifact verification and D² computation ran remotely; local work was source edits and report preparation.

Evidence: [worker and launch inspection](comparison_tabddpm_preflight_2026_10_11.json), [verified records, generator provenance, sidecars and aggregate](comparison_tabddpm_snapshot.json), and [CSV preservation, added-value validation and final hashes](comparison_tabddpm_refresh_2026_10_11.json).
