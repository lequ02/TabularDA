# Comparison reports — October 31, 2026 (Adult TabDDPM posterior fix)

**Latest best-validation-F1 results:** [valbest/README.md](valbest/README.md). All 28 selected Adult
TabDDPM evaluations are complete; this includes refreshed synthetic/mix reports, local scores and
before/after comparisons using best validation binary F1 on both sides.

This is the latest original-selection comparison (development-loss checkpoints). The earlier reports in
`comparisons/` (`synthetic.*`, `mix.*`) are kept unchanged as the historical October 13 version.

- [Synthetic-only table](reports/dataset_pipeline_comparison.md) · [CSV](reports/dataset_pipeline_comparison.csv) · [plot](reports/dataset_pipeline_comparison.png)
- [Mix table](reports/dataset_pipeline_comparison_mix.md) · [CSV](reports/dataset_pipeline_comparison_mix.csv) · [plot](reports/dataset_pipeline_comparison_mix.png)

## What changed

Only the 28 selected Adult TabDDPM cells changed in each report: generated target, plus full-feature and
X-only RF/XGB/DNN, for seeds 42/43. They now come from the posterior-fix rerun
`tabddpm_adult_500ep_fix_20261030_v1` instead of the held pre-fix records in `tabddpm_500ep_20261010_v1`
([rerun handoff](../../audit/TAB_DDPM_ADULT_RERUN_2026_10_30.md)).

- Both Adult real-only TabDDPM baselines are the same reused records.
- Every other score, and every source record path once resolved, is identical to a fresh rebuild of the
  October 13 reports.
- Before rebuilding, `scripts/workspace.py reports --selection original` was confirmed to reproduce every
  published score in `comparisons/synthetic.csv` and `mix.csv`.

## Provenance

- `comparison_tabddpm_snapshot_20261031.json`: a copy of `audit/comparison_tabddpm_snapshot.json` with only
  the 28 Adult selected records and their 4 generator provenances replaced. It also has an added
  `adult_recovery` block. The original snapshot is unchanged.
- The new records, generator provenance, `launch.json`, and queue scripts are archived in
  `adult/tabddpm_fix_20261030/`. All 213 files matched their remote SHA-256 (`report_evidence_20261031.json`),
  and each record passed the runner's `completed_run(..., 'corrected')` check remotely.
- `rebuild.py` regenerates `reports/` from the workspace root. It refuses to overwrite an existing `reports/`.
  `reports/artifact_hashes.json` records the output hashes.
- The only builder change is one conditional sentence in `scripts/plot_recent_comparison.py`. It appears only
  when a snapshot has `adult_recovery`, so earlier snapshots render as before.

The header's "TabDDPM snapshot: October 13" time still applies to the non-Adult TabDDPM records. The Adult
rerun verification time is stated in the TabDDPM note. The MNIST rerun README link in the tables is not
present in this post-swap workspace, as in the earlier reports.

## Before/after and new results CSV

- [Before/after figure](before_after/adult_tabddpm_before_after.png) · [table](before_after/adult_tabddpm_before_after.md) · [CSV](before_after/adult_tabddpm_before_after.csv). All 52 arms, binary F1, macro F1, and accuracy, before (held pre-fix) and after (posterior-fix rerun).
- New results in the `adult/scores.csv` column format: `adult/tabddpm_fix_20261030/scores.csv`. It holds the 52 new arms plus the 2 reused baselines, and was built from the hash-checked archived records. `adult/scores.csv` is unchanged and still holds the pre-fix scores.
- Regenerate both with `before_after.py` from the workspace root.
