# Local audit: what matters for the final results

**Retention scope expanded:** [the individual CSV and research-decision review](LOCAL_CSV_PURPOSE_REVIEW_2026_10_10.md) is now authoritative for scientific retention. The 406-run derivation and final-source verification below still apply, but final-set exclusion alone cannot justify retiring evidence of rejected alternatives, failures, corrections, or limitations. The whole-file catalog has separate scientific-retention annotations linked to that review.

October 10, 2026, Chicago time. This review uses only local files. No server connection, experiment execution, model loading, move, rename or deletion was performed. The local result audit passed at 2:08 p.m.; a subsequent member-by-member check verified the existing frozen snapshot archive.

This replaces the earlier remote-centered inventory as the basis for local organization. The retention question is: **does this file supply, substantiate, or reproduce a declared final result?** A dependency of an obsolete audit, pilot, or retired report is not automatically a final-result dependency.

## Exactly where 406 comes from

The current final real-data reports include **seven datasets**: Adult, Census KDD, Covertype, MNIST12, MNIST28, News, and California Housing. They have seeds 42 and 43.

For each dataset and seed, each of CTGAN and TVAE has these seven synthetic constructions:

| Construction | has_y during generator fitting | Target used downstream | Number |
|---|---:|---|---:|
| Full-table generated-target baseline | 1 | Generator's target | 1 |
| Full-table relabeling | 1 | RF, XGB, or DNN prediction | 3 |
| Features-only relabeling | 0 | RF, XGB, or DNN prediction | 3 |
| Total per generator | | | 7 |

Each construction is evaluated twice: synthetic-only training and mix training. Each dataset/seed also has one original-data-only baseline, shared by both report types.

```text
Per dataset/seed:
  2 generators × (1 generated + 2 has_y choices × 3 labelers) × 2 training modes
  + 1 original-only baseline
  = 28 + 1 = 29 downstream runs

All datasets/seeds:
  7 × 2 × 29 = 406 distinct downstream runs

By target construction:
  336 relabeled + 56 generated-target + 14 original-only = 406

By report type:
  196 synthetic + 196 mix + 14 shared original-only = 406
  210 synthetic-report records + 210 mix-report records − 14 shared = 406
```

There is no generated-target has_y=0 construction; multiplying every method by two has_y levels would overcount. The original-only baseline is not separately fitted for each generator, labeler, construction or report.

| Dataset | Synthetic-only runs | Mix runs | Real-only baselines | Distinct total |
|---|---:|---:|---:|---:|
| Adult | 28 | 28 | 2 | 58 |
| Census KDD | 28 | 28 | 2 | 58 |
| Covertype | 28 | 28 | 2 | 58 |
| MNIST12 | 28 | 28 | 2 | 58 |
| MNIST28 | 28 | 28 | 2 | 58 |
| News | 28 | 28 | 2 | 58 |
| California Housing | 28 | 28 | 2 | 58 |
| Total | 196 | 196 | 14 | 406 |

Five classification datasets contribute 290 runs; the two regression datasets contribute 116. Six datasets with these same arms would give 348; Housing contributes the additional 58 in the current seven-dataset reports.

Accuracy, binary F1, macro F1, R², saved NMAE and D² are measurements of these runs, not additional runs. The three published-paper reference rows in each comparison CSV are also not new runs. NB/PCA-GMM, Credit, Intrusion, dropout trials and the separate simulated benchmark are not part of this selected 406 set. This count is not the original full experiment matrix's count.

The current statistical report was revised by another chat during the audit. Its experiment membership stayed unchanged. Per training mode, its classification factorial table now contains **120 relabeled observations** (5 datasets × 2 seeds × 2 generators × 2 has_y levels × 3 labelers), plus **20 generated-target baseline observations** in a separate table. Ten classification real-only runs supply shared references. Across both training modes: 120×2 + 20×2 + 10 = 290. Regression contributes 56 synthetic + 56 mix + 4 original = 116.

### How the count was verified

- Read actual `*_record` columns from both current comparison CSVs. Each identifies 210 distinct records; their intersection consists of exactly 14 original-only baselines.
- Constructed the complete expected grid independently and compared path and configuration sets. There are no missing, extra or duplicate configurations.
- Read all 406 canonical local run records and their frozen copies; checked dataset, seed, method mapping and training mode using the inspected pure mapping in `src/modeling/constants.py`.
- Verified all 406 frozen SHA256s. Every canonical selected record is byte-identical to its frozen copy.
- Matched all 780 displayed metric cells to saved source-record/D² values; verified all 3,770 rows in the frozen score table against the same source set.
- Checked the current classification, generated-baseline, regression and D² input tables and all 76 current statistical deliverable-manifest hashes.

The exact 406 configurations are listed in `local_final_results_selected_runs_2026_10_10.csv`. The collection loop in `audit/statistical_328_20261010/collect_remote.py` independently describes the same dataset/seed/generator/construction/labeler/mode grid, but it was only read, never executed.

## The final outputs used as audit roots

The declared local final families are:

1. `output/comparisons/dataset_pipeline_comparison*`: current synthetic-only/mix CSVs, figures and explanatory Markdown.
2. `output/statistical_analysis_328_20261010/` and its PDF directory: the current explicit-factor synthetic-only/mix analyses and combined export, as identified by the latest `deliverable_manifest.json`. Membership is taken from that manifest, not from everything left in the directory.
3. `output/pdf/synthetic_benchmark/`: the separate standalone simulated report, its source, seven figures and supplied data/export package. Its 378 benchmark rows are separate from the real-data 406. The archived paper comparison contributes 42 relevant historical baseline rows, not another 406 real-data runs.

Current final results, their rebuilding inputs, their validity evidence and reusable experiment assets are distinct needs. A checkpoint may substantiate/replay an experiment without being read to render a figure. A frozen result CSV can reproduce an analysis without requiring the legacy training tables that happen to exist locally.

## Local files that matter

| Material | Final-result purpose | Recommended local disposition |
|---|---|---|
| The exact 406 selected records | Source metrics, settings, selected epochs, embedded split/provenance information | Keep one canonical immutable set; preserve original bytes and hashes. |
| The 116 D² sidecars | Regression metrics/normalization linked to original record hashes | Keep one canonical set with the source records. |
| Current comparison CSVs, figure source, three fit/source-verification JSONs and two current snapshot manifests | Rebuild and substantiate current comparisons | Keep; promote useful code out of cache and evidence out of the mixed audit folder. |
| Frozen `snapshot/scores.csv`, saved regression D² scores and snapshot JSON | Refit the current statistical analyses from exactly the published source set | Keep one canonical input set. |
| `analyze_explicit.py`, `build_explicit_report.py`, `verify_explicit.py`, current explicit results and the imported `inference` helper | Current statistical analysis/rendering/verification | Keep/promote. Earlier pooled and composite-condition analyses are outside the latest final package. |
| Three Census pilot source records and their import history | Three current runs originated in verified pilot results | Keep original identity/evidence. Other pilot outputs do not become final inputs merely through this import. |
| Current simulated per-run/summary/config tables, 42 archived paper baselines, builder/template and oracle package | Rebuild the separate simulated report and retain its caveats | Keep/promote; the full legacy SDGym code/data tree is not required to render this report. |
| Locally present assets actually named by selected runs: 174 selected weights and 24 fitted DNN labelers | Reuse/replay selected experiments | Keep as selected experiment assets, separate from report tables. |
| Applicable MNIST overlap findings and correction explanations | Interpret the final results accurately | Keep concise validity/history documentation and relevant evidence; archive the investigations themselves. |

Current experiment code also has small helpers under audit that should be promoted: the matrix explicitly invokes `audit/sdv_api_smoke.py`, and the MNIST/Housing worker/plan/task sources document the completed rerun procedures. Their useful code belongs with experiment/validation tools even though those files are not numerical report inputs. This is separate from preserving old generated outputs merely because those helpers once used them. The catalog retains these helpers and the small direct MNIST overlap/match evidence supporting the published caveat.

The three imported Census source records have Windows line endings locally. Their raw local hashes differ from the cited original hashes; converting CRLF to LF reproduces all three cited original byte hashes exactly. Preserve that restoration rule or a verified original-byte copy. This is not a new fit or a numerical result difference.

The latest explicit statistical source is authoritative for this review. The earlier `SEPARATE_ANALYSIS_HANDOFF.md` describes the previous analysis version and should not be used as the current model specification. The active report wrapper now invokes `build_explicit_report.py`.

## Local files outside the final result set

| Collection | Why it is outside the declared final workflows | Proposed treatment |
|---|---|---|
| Legacy data under `data/{adult,census,census_kdd,covertype,credit,mnist12,mnist28,news}` and unselected root data files: **16.07 GiB** | Final figures/analyses use corrected records and frozen result tables. Current dataset acquisition does not consume these legacy splits. None is a selected prepared/synthetic table path. | Remove from the active final-results repo by transferring useful historical content to a research archive; decide archive retention by scientific purpose. |
| Unselected/legacy files under `sdv trained model/`: **2.81 GiB** | They are outside selected record model references locally. Historical generator names/old-split variants are not current report inputs. | Archive useful historical checkpoints; consolidate proven duplicates. Keep the selected provenance files and selected labelers identified separately. |
| 682 unselected canonical run records | Outside the exact current 406 grid; includes superseded evaluations and broader-matrix methods not selected for the final reports | Archive by study/status. Keep scientifically meaningful unique evidence without presenting it as a current result. |
| `pilot_ctgan_v1`, Credit/News pilots and earlier local figures | Not selected by current final reports | Archive useful diagnostics/history. No requirement to retain their active paths solely for old audit scripts. |
| Earlier pooled/condition-factor statistical results and their build variants | Replaced in the current final analysis/package | Archive the meaningful earlier release; retire duplicate intermediate output. |
| `audit/results_review/`, most reconstruction exports, dated process/status snapshots, old refresh backups/checkers | Their consumers are historical investigation/report workflows | Retain a concise reconstruction/correction ledger and necessary evidence; archive or retire the rest. Their dependency graphs do not define the final repo. |
| Legacy SDGym code/data, old sample figures, exploratory notebooks, `April29.csv`, `final_results.xlsx` | Outside the inspected current final builders/input set | Label as historical/reference material; archive useful originals. Do not use historical maximum-test-score tables as corrected result sources. |
| Build previews, bytecode, notebook/test caches | Recreated from retained sources | Retire after source/deliverable preservation. |

The 682 excludes the three imported Census origin records: there are 1,091 canonical local run records in total, comprising 406 selected + 3 necessary origin records + 682 outside the final set. An unselected run can be scientifically useful without belonging in the active final-results tree.

The 16.07 + 2.81 = **18.88 GiB** figure is a local archive pool, not an unconditional deletion proposal or guaranteed space recovery. It deliberately excludes selected local checkpoints/labelers/provenance. A historical file may also be useful for a future project; the classification establishes its relationship to these final results rather than predicting all future use.

## Consolidation that preserves the selected results

There are two flat local copies of every selected run record: canonical output and the statistical frozen snapshot. Each set occupies **938,065,754 bytes (0.874 GiB)** and all 406 pairs match exact bytes.

`audit/statistical_328_20261010/remote_snapshot.zip` is **149,294,667 bytes (142.38 MiB)**. Every one of its 418 file members was checked against the extracted snapshot, including all 406 selected source hashes. It therefore supplies an already verified compressed evidence bundle. Its name should become descriptive of the local frozen source set rather than its transport origin.

Choose one canonical record store. Two workable local layouts are:

- One directory of immutable records, with redundant frozen copies/transport bundles consolidated after current readers are updated.
- One verified compressed provenance bundle with an explicit archive reader and hash index; published result tables remain small, accessible CSVs. This avoids needing both 0.874-GiB flat copies.

Neither option changes the original record bytes. The current figure/verification scripts expect files at existing paths; update those deliberate readers before retiring copies. The two direct `latest_comparison*.zip` files are current checker inputs today; the 13 older ZIPs at backup paths are historical exports. They are not all permanent final dependencies.

## Recommended local migration order

1. Document the seven-dataset, two-seed, RF/XGB/DNN final scope and the separate simulated study in a root README. Give historical/broader-matrix studies explicit status.
2. Put current useful CSVs in `artifacts/tables/{real_comparison,real_statistics,simulated}` with analysis/seed/purpose subfolders where needed. Keep their old-to-new names and source hashes in a catalog. Legacy training CSVs need not share the active final table hub.
3. Promote current report/analysis/validation sources and preserve one canonical provenance set, the three Census origins and applicable validity evidence. Keep selected checkpoints/labelers separately under models.
4. Archive legacy data/models, unselected studies and retired audit/report workflows. Update the retained code's imports/paths; deliberately retire old consumers instead of preserving their whole dependency trees.
5. Consolidate byte-identical record copies and verified historical exports. Update `.gitignore` and path-specific LFS rules with the actual moves. Check the final 406 grid, original hashes, saved regression metadata and final numerical tables again after migration.

No remote asset inventory or download is needed to make these local retention decisions. The local repo does not contain the 196 selected synthetic tables, 84 prepared views, 56 generators or any of the 406 held-out prediction CSVs. Those absences explain why the existing local final workflow relies on saved records/frozen scores. The unrelated legacy CSVs do not fill that role. Access to full experiment/re-evaluation assets can be documented separately when the remote part is revisited.

## Audit outputs and limits

`local_final_results_file_dispositions_2026_10_10.csv` classifies 10,143 local files by final purpose; it is a review catalog, not a deletion allowlist. The environment, Git object databases/worktrees, credential/config directories and nested GLRM/pyglrm repositories were excluded from recursive research inventory; operational access files were not opened. Current report changes by another chat were preserved.

The check compared saved scores/artifact bytes and metadata. It did not independently recompute model predictions, scan training-table distributions, or execute experiments locally. The snapshot archive check verifies content preservation, not the scientific validity of every original run.

Machine-readable evidence: `local_final_results_validation_2026_10_10.json`, `local_final_results_selected_runs_2026_10_10.csv`, and `local_final_results_snapshot_archive_verification_2026_10_10.json`. The read-only audit source is `local_final_results_audit_20261010.py`. These cleanup-investigation outputs are themselves intermediate materials; the eventual root documentation/catalog should replace them as the newcomer-facing entrypoint.
