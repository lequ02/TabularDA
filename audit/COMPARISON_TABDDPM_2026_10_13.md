# Comparison report refresh — October 13, 2026

## Frozen report snapshot

The comparison tables and plots use a verified TabDDPM snapshot taken **October 13, 2026, 10:51:22 a.m. Chicago**. The CTGAN/TVAE source snapshot remains October 10 at 11:58 a.m. Chicago. The synthetic-only report contains **319 distinct completed run records** and **109/112 selected TabDDPM records**; the mix report contains **317 distinct completed records** and **107/112 selected TabDDPM records**. The original 816-run matrix is not claimed complete.

| Corrected TabDDPM namespace | Verified records | New records | New selected report records |
|---|---:|---:|---:|
| `tabddpm_census_kdd_500ep_fix_20261011_v1` | 54/54 | 53, including one reused seed-42 baseline | 14 |
| `tabddpm_covertype_500ep_fix_20261011_v1` | 22/54 | 21, including one reused seed-42 baseline | 20 |

Coverage is per report and counts the real-only TabDDPM baseline among each seed's eight selected configurations. Census KDD is **8/8 for both seeds** in both reports. Covertype is **7/8 for seed 42 and 6/8 for seed 43** in synthetic-only, and **6/8 and 5/8** in mix. Adult, Housing, MNIST12, MNIST28, and News retain their prior 8/8 selected coverage for both seeds.

The refresh added 18 source-record links and 25 metric cells to synthetic-only, and 16 links and 23 metric cells to mix. Each CSV remains **26 rows × 49 columns**. All previously populated cells are unchanged. The reports retain the saved News NMAEσ values, paired R²/D² rows, metric definitions, and leakage caveats.

## Provenance and live status

The audit rechecked **750 prior report source hashes** and verified **331 artifact files** referenced by the corrected Census and Covertype records. It checked all 54 Census and 22 Covertype run records, their split manifests, saved predictions against test source IDs, finite epoch histories, selected development epochs where histories exist, generator fit-table hashes, and generator settings. The reused seed-42 baselines were matched to their original records and stored artifacts. Generators used 500 epochs, batch size 500, and 100,000 rows per source.

A live read-only check at **10:58:41 a.m. Chicago** found no new records since the frozen snapshot. Census KDD remained at 54/54 with no active worker process. Both corrected Covertype generators had completed 514,000/514,000 updates; downstream workers for seeds 42 and 43 were still active, and unfinished downstream configurations remain blank. No experiment jobs were launched, restarted, or interrupted.

The original `tabddpm_500ep_20261010_v1` namespace remains **240/346** complete. Its historical Covertype seed-42 generation failure and Census KDD seed-42 failure at update 13 remain separate from these corrected namespaces. The corrected rerun completions do not establish completion of the original full matrix.

## Validation and deliverables

The report builder loaded each linked run record and its saved metric values. The final CSV check verified every report source-record hash and 92 D² sidecar hashes per report. Both refreshed plots were visually reviewed.

- [Synthetic-only table](../output/comparisons/dataset_pipeline_comparison.md) · [CSV](../output/comparisons/dataset_pipeline_comparison.csv) · [Plot](../output/comparisons/dataset_pipeline_comparison.png)
- [Mix table](../output/comparisons/dataset_pipeline_comparison_mix.md) · [CSV](../output/comparisons/dataset_pipeline_comparison_mix.csv) · [Plot](../output/comparisons/dataset_pipeline_comparison_mix.png)
- [Verified source snapshot](comparison_tabddpm_snapshot.json)
- [Refresh validation and artifact hashes](comparison_tabddpm_refresh_2026_10_13.json)

The pre-refresh copies are preserved in `.cache/remote_intrusion/comparison_before_refresh_20261013_104312/`. The imported selected run records and updated snapshot are also archived in `.cache/remote_intrusion/comparison_tabddpm_refresh_20261013_snapshot.zip`.
