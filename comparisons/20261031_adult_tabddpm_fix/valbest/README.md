# Best-validation-F1 results — October 31, 2026

All **28 selected Adult TabDDPM evaluations** finished on the research server by **11:59:35 a.m. Chicago**.
They use the posterior-fix generators and the earliest maximum development **binary F1**, within each
original completed training history. There were 27 downstream replays (491 epochs) and one unchanged
checkpoint; all four workers exited 0. Generators and labelers were reused.

- New Adult results for upload: [scores_valbest.csv](../../../adult/tabddpm_fix_20261030/scores_valbest.csv) — 28 arms, both seeds and modes, with exact local evidence paths.
- Full refreshed selection: [scores_valbest.csv](scores_valbest.csv) — 602 evaluations, replacing 28 historical Adult TabDDPM entries; the other 574 are unchanged.
- Synthetic-only: [table](dataset_pipeline_comparison_val_f1.md) · [CSV](dataset_pipeline_comparison_val_f1.csv) · [figure](dataset_pipeline_comparison_val_f1.png).
- Mix: [table](dataset_pipeline_comparison_mix_val_f1.md) · [CSV](dataset_pipeline_comparison_mix_val_f1.csv) · [figure](dataset_pipeline_comparison_mix_val_f1.png).
- Before/after, **both selected by best validation binary F1**: [figure](adult_tabddpm_before_after_valf1.png) · [table](adult_tabddpm_before_after_valf1.md) · [CSV](adult_tabddpm_before_after_valf1.csv). The CSV also includes the fixed runs' loss-selected scores.

Scope is generated targets and full-table/X-only RF, XGB and DNN, seeds 42/43, synthetic-only/mix.
NB and PCA/GMM are outside these 28 evaluations. The 54-record loss-selected rerun, including those
labelers and two reused real-only references, is retained separately in the parent report folder.
The main comparison's real-only baselines remain unchanged.

The corrected-vs-historical best-F1 comparison has mean binary-F1 change **+0.10 percentage points**
(range −1.11 to +2.11), mean macro-F1 change +0.03 points, and mean accuracy change −0.04 points.
These are paired observed differences across two seeds, not a significance claim.

## Evidence and validation

All 28 results record history agreement within `1e-8`, exact reproduction of the original checkpoint,
matching test IDs/targets, and independently checked test binary F1. Remote verification rechecked the
input, record, history, prediction and weight hashes. The archive holds 150 hash-verified files plus
verification and transfer manifests under [adult/tabddpm_fix_20261030/valbest](../../../adult/tabddpm_fix_20261030/valbest/).
Result JSON bytes and original remote paths are preserved; the scores CSV resolves their local files.

[validation.json](validation.json) confirms all 28 displayed Adult metric cells per report match their
new results, and every other score and resolved source matches the historical best-F1 reports.
[artifact_hashes.json](artifact_hashes.json) records generated output hashes.
MNIST12 TVAE collision and shared-holdout caveats remain. The historical pre-patch Adult reports remain
available as evidence; these new report cells use `tabddpm_adult_500ep_fix_20261030_v1`.

Rebuild from the workspace root with:

```powershell
.venv/Scripts/python.exe comparisons/20261031_adult_tabddpm_fix/rebuild_valbest.py
```

The builder uses the existing checkpoint-report recipe and refuses to replace an existing output folder.
Historical scores and figures remain unchanged. These 602 selected evaluations are not a full-matrix
completion count.
