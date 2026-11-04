# Comparison report refresh — October 13, 2026, evening

This note preserves the **6:59 p.m. frozen snapshot** and **7:01 p.m. live check** as historical evidence. The Covertype seed-43 PCA/GMM mix record completed later and was included in the **7:34 p.m.** report refresh. See the [latest handoff](COMPARISON_HANDOFF.md) and [full-matrix status](FULL_MATRIX_STATUS_2026_10_13.md) for current counts.

## Frozen report snapshot

The synthetic-only and mix comparison tables and plots were rebuilt from a verified TabDDPM snapshot taken **October 13, 2026, 6:59:36 p.m. Chicago**. The CTGAN/TVAE source snapshot remains October 10 at 11:58 a.m. Chicago. Each report now contains **322 distinct completed run records** and **112/112 selected TabDDPM records**. The original 816-run matrix is not claimed complete.

| Corrected TabDDPM namespace | Verified records | New records | Newly selected report records |
|---|---:|---:|---:|
| `tabddpm_census_kdd_500ep_fix_20261011_v1` | 54/54 | 53, including one reused seed-42 baseline | 0 |
| `tabddpm_covertype_500ep_fix_20261011_v1` | 53/54 | 52, including one reused seed-42 baseline | 8 |

Census KDD and Covertype each have **8/8 selected configurations for both seeds** in both reports. The eight newly selected Covertype records filled three synthetic-only links/metric cells and five mix links/metric cells. Each CSV remains **26 rows × 49 columns**. Every previously populated cell is unchanged. Saved News NMAEσ values, paired R²/D² rows, metric definitions, and leakage caveats remain in place.

## Provenance and live status

The refresh reverified **784 prior report source files** and **440 saved artifact files** referenced by the corrected Census KDD and Covertype records. It checked all 54 Census and 53 available Covertype records, split manifests, saved predictions against test source IDs, finite metrics and epoch histories, development-selected checkpoints where histories exist, generator fit-table hashes, and generator settings. The reused seed-42 baselines were matched to their original records and artifacts. Generator provenance confirms 500 epochs, batch size 500, and 100,000 rows per generated source.

A final read-only check at **7:01:18 p.m. Chicago** found no records added since the frozen snapshot. Census KDD remained 54/54 with no worker process. Covertype remained 53/54; the generators had completed 514,000/514,000 updates for both seeds, while the downstream worker for seed 43 remained active. The sole missing run is the unselected seed-43 full PCA/GMM mix arm, which is excluded from these comparison tables. No jobs were launched, restarted, or interrupted.

The original `tabddpm_500ep_20261010_v1` namespace remains **240/346** complete. Its historical Covertype seed-42 generation failure and Census KDD seed-42 failure at update 13 remain separate from these corrected namespaces. The corrected reruns do not establish completion of the original full matrix.

## Validation and deliverables

The existing comparison builder checked linked records and split isolation for all 18 panels; no exact train/holdout overlaps were found. The post-build check verified all report-linked run-record and D² sidecar hashes, the unchanged previously populated CSV cells, and valid PNG files. Both plots were visually reviewed.

- [Synthetic-only table](../output/comparisons/dataset_pipeline_comparison.md) · [CSV](../output/comparisons/dataset_pipeline_comparison.csv) · [Plot](../output/comparisons/dataset_pipeline_comparison.png)
- [Mix table](../output/comparisons/dataset_pipeline_comparison_mix.md) · [CSV](../output/comparisons/dataset_pipeline_comparison_mix.csv) · [Plot](../output/comparisons/dataset_pipeline_comparison_mix.png)
- [Verified source snapshot](comparison_tabddpm_snapshot.json)
- [Refresh validation and artifact hashes](comparison_tabddpm_refresh_2026_10_13_evening.json)

The pre-refresh copies are preserved in `.cache/remote_intrusion/comparison_before_refresh_20261013_185559/`. The hash-verified selected records and snapshot are archived in `.cache/remote_intrusion/comparison_tabddpm_refresh_20261013_evening_snapshot.zip`.
