# Leakage re-audit — October 31, 2026

Read-only remote re-audit of every namespace used by the current comparison reports, using `scripts/audit_run_leakage.py --data --scopes ...` on `thuy@10.24.10.133` (environment `/home/thuy/miniconda3/envs/env`, pandas 2.2.3, numpy 1.26.4). Run in tmux session `leakage-audit-20261030`, started 1:31 a.m. and finished 3:34 a.m. Chicago. Remote outputs: `.cache/leakage_audit_20261030/` (the first attempt, which stopped at the Intrusion text target, is preserved in `run1/`). No experiment file, job, or output was changed.

Scopes: `corrected_v2`, `census_kdd_weighted_macro_f1_20261005`, `mnist_head_fixed_20261009`, `news_log_v1`, `housing_no_faker_20261009`, `tabddpm_500ep_20261010_v1`, `tabddpm_census_kdd_500ep_fix_20261011_v1`, `tabddpm_covertype_500ep_fix_20261011_v1`. The Adult TabDDPM rerun namespace `tabddpm_adult_500ep_fix_20261030_v1` was launched during this audit ([handoff](TAB_DDPM_ADULT_RERUN_2026_10_30.md)) and is not covered.

## Verdict

**No evidence of holdout data entering generator fitting, labeler training data, checkpoint selection, or the real splits.** The strict unseen-feature guarantee still fails in two known/near-known places: MNIST12 TVAE (unchanged) and a handful of Census KDD rows.

| Check | Result |
| --- | --- |
| Records / groups | 1,403 `*.run.json` in 41 dataset/seed/namespace groups; all 41 groups completed |
| Source-ID disjointness (train/dev/test) | 0 overlaps in all groups |
| Real split files vs manifest hash and rows | 246/246 match |
| Real feature-row overlap between splits (raw and one-hot) | 0 in all 246 comparisons |
| Generator fit tables reproduced from the training split | 76/76 distinct fit hashes match |
| Dev checkpoint selection recorded | all records |
| Test prediction IDs equal the test split | all records |
| Synthetic tables compared with real dev and test | 681/681, all with identical one-hot columns |
| Synthetic features equal to real dev/test features | 113 table/split flags, below |

| Group | Tables flagged | Matching dev rows | Matching test rows |
| --- | ---: | --- | --- |
| MNIST12 TVAE, seed 42 (`corrected_v2`, `mnist_head_fixed_20261009`) | 13 each | 348–376 | 377–392 of 9,873 |
| MNIST12 TVAE, seed 43 (`corrected_v2` 7, `mnist_head_fixed_20261009` 13) | 20 | 308–370 | 367–439 of 9,873 |
| Census KDD TVAE X-only, seeds 42/43 (`corrected_v2`, weighted macro-F1) | 19 | 1–2 | 0 |
| Census KDD CTGAN X-only Gaussian NB, seed 42 (both namespaces) | 2 | 0 | 1 of 9,523 |

No CTGAN or TabDDPM MNIST table, and no Adult, Credit, Covertype, News, or Housing table, matched any dev/test feature row.

The MNIST12 result reproduces the [September 30 review](LEAKAGE_REVIEW_2026_09_30.md): independent collisions in the compressed binary image space, not holdout use. The Census KDD matches (inspected with `census_kdd_synthetic_matches_2026_10_31.py`) are five distinct real rows: four children aged 1–9 and one 71-year-old with nearly every field "Not in universe" or 0. Each pattern occurs once in its split and nowhere in training; the generator fit hashes show training-only fits. They are best read as chance collisions in a highly concentrated region, but they are strict unseen-feature exceptions like MNIST12.

## Script changes

`scripts/audit_run_leakage.py` gained `--scopes` (default unchanged); a column-mismatch error so differing synthetic/real columns cannot report false zero overlap; one-hot target parsing that keeps text labels (Intrusion) as text; and fit-hash reproduction that applies the News log target. The metadata-only full-table check against raw file bytes was removed because floating-point re-serialization made it fail for regression; `--data` reproduces each fit hash exactly instead.

Scope limits: this audit checks data overlap and provenance. It does not re-verify, in code, that labelers and scalers are fitted on training rows; see the October 13 scientific-validity review for those checks.

Evidence: [leakage_audit_2026_10_31.jsonl](leakage_audit_2026_10_31.jsonl) (SHA-256 `9f9c4829f9635c6f7d486117332d799d55974a50c15f861969c6045c8cac8172`, identical to the remote copy).
