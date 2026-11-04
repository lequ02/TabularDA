# TabDDPM comparison refresh — October 12, 2026

## Frozen report snapshot

The comparison tables, CSVs, and plots use a verified snapshot taken **October 12, 2026, 12:13:52 p.m. Chicago**. The CTGAN/TVAE source snapshot remains October 10 at 11:58 a.m. Chicago. Both reports contain **301 distinct completed run records** and **91/112 selected TabDDPM records**. The original 816-run corrected CTGAN/TVAE matrix and the original TabDDPM namespace are not complete.

The Census KDD TabDDPM recovery namespace, `tabddpm_census_kdd_500ep_fix_20261011_v1`, had **28/54 verified run records**: 27 new runs and the reused seed-42 real-only baseline. Fourteen new records are in the selected comparison scope: seed 42's seven selected synthetic constructions in each training mode (generated target plus full-table and X-only RF/XGB/DNN relabeling). A fresh seed-43 real-only baseline is also included. Thus the reports show Census KDD selected coverage of **8/8 for seed 42 and 1/8 for seed 43**. NB/PCA-GMM arms remain excluded from these comparison tables.

| Dataset | Seed 42 selected | Seed 43 selected | Both seeds |
|---|---:|---:|---:|
| Adult | 8/8 | 8/8 | 16/16 |
| Housing | 8/8 | 8/8 | 16/16 |
| Covertype | 1/8 | 1/8 | 2/16 |
| MNIST12 | 8/8 | 8/8 | 16/16 |
| MNIST28 | 8/8 | 8/8 | 16/16 |
| Census KDD | 8/8 | 1/8 | 9/16 |
| News | 8/8 | 8/8 | 16/16 |
| **Total per report** | **49/56** | **42/56** | **91/112** |

## Live state after the frozen snapshot

A read-only check at **12:20:54 p.m. Chicago** found no additional Census KDD run records: the namespace still had 28 completed records, including the reused seed-42 baseline. Seed-43 generation was active at **10,700/163,000 updates**. No completed seed-43 generated or relabeled results were added after the frozen snapshot.

The original TabDDPM namespace, `tabddpm_500ep_20261010_v1`, remains **240/346** complete. Its Covertype seed-42 and Census KDD seed-42 generation failures remain historical failures in that namespace. The separate corrected Covertype namespace had two verified baseline records (one reused, one fresh) and no generated results yet. At the live check its two workers were active; their latest logged progress was **127,600/514,000** updates for seed 42 and **97,900/514,000** for seed 43.

No experiment was launched, restarted, or interrupted for this refresh. TabDDPM retains the 500-epoch, batch-500, CUDA, 1,000-timestep, 100,000-rows-per-source settings. The constructions are unconditional joint/X-only generation, not native target-conditional TabDDPM.

## Validation and preserved conventions

- Rechecked **735 source-file hashes** referenced by the previous comparison CSVs against the remote repository.
- Verified all 28 Census run-record entries, their split manifests and test source IDs, finite saved predictions and epoch histories, and selected development epochs. The 27 newly produced runs' generator settings and saved provenance were checked; **114 referenced artifact files** were hash-verified. The reused seed-42 record matches the original record and split.
- Each CSV retains 49 columns and 26 rows. Each report gained **8 record links and 16 Census metric cells**; no previously populated cell changed. The D² sidecars and saved NMAEσ values are unchanged.
- Figures were visually reviewed. Binary F1 and macro F1 remain separate for Census KDD; missing seed-43 generated results remain blank. Existing paper references, same-row R²/D² panels, News symmetric-log scale, and documented leakage caveats remain. No TabDDPM paper reference or Housing paper-reference placeholder was added.

The previous report files and source snapshot were preserved in `.cache/remote_intrusion/comparison_before_refresh_20261012_1009/`. The exact coverage, hash, and artifact evidence is in [the refresh validation](comparison_tabddpm_refresh_2026_10_12.json) and [the source snapshot](comparison_tabddpm_snapshot.json).

## Current deliverables

- Synthetic-only: [table](../output/comparisons/dataset_pipeline_comparison.md) · [CSV](../output/comparisons/dataset_pipeline_comparison.csv) · [plot](../output/comparisons/dataset_pipeline_comparison.png)
- Mix: [table](../output/comparisons/dataset_pipeline_comparison_mix.md) · [CSV](../output/comparisons/dataset_pipeline_comparison_mix.csv) · [plot](../output/comparisons/dataset_pipeline_comparison_mix.png)

Earlier snapshots and the original TabDDPM experiment history remain in [the October 11 details](COMPARISON_TABDDPM_2026_10_11.md).
