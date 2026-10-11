# Repository organization review

**Local scope correction:** use [the individual CSV and research-decision review](LOCAL_CSV_PURPOSE_REVIEW_2026_10_10.md) for scientific retention and [the local final-results audit](LOCAL_FINAL_RESULTS_AUDIT_2026_10_10.md) for the exact 406-run derivation and final-source checks. Decision evidence can matter outside that grid. The remote inventory and earlier statistical source classifications below are historical; the active local analysis has since changed to explicit factors.

Reviewed October 10, 2026, Chicago time. The full remote file/reference/hash inventory is from 1:13 p.m.; namespace counts, CSV schemas, current builder evidence, environment versions and worker state were refreshed at 1:39 p.m. This is a proposed organization and retention policy. No experiment asset has been moved, deleted, renamed, rewritten or retrained.

This review supersedes the earlier cleanup plan's assumption that an old audit script makes its historical input path permanently necessary. A historical investigation can be retired from the active workflow while its important evidence is preserved.

## What the leakage audits are, and what belongs in the final repo

The September audits checked whether training and evaluation were independent: disjoint source IDs/feature groups, preprocessing fitted on training only, generator fitting on training only, development-based checkpoint selection, held-out prediction IDs, saved hashes and possible synthetic/holdout overlap. They are research-validity checks, not steps required to train a model or build every report.

There are two different findings to explain to a new reader:

- The original pipeline had real methodological defects, including train/test overlap, test-based checkpoint selection and evaluation preprocessing fitted independently. Those results are historical and must not be mixed with corrected results.
- The corrected MNIST12 TVAE tables have documented exact feature collisions with held-out examples. The investigation supported train-only generator fitting; the collisions are not proof that test data entered training. They remain an interpretation caveat, including for downstream reruns that reused those tables.

The final repo should retain a concise validity document, the correction/history ledger, split manifests, generator/model provenance, relevant overlap evidence, source snapshots/version information, and useful validation code. Keeping every dated status dump, exploratory script, duplicate record export and intermediate screenshot in the main tree is unnecessary.

| Current material | Actual role | Proposed disposition |
|---|---|---|
| `audit/AUDIT.md`, `audit/LEAKAGE_REVIEW_2026_09_30.md` | Findings tied to historical source versions and results | Summarize applicable findings in `docs/validity.md`; preserve originals and essential evidence in the investigation archive. Do not present their old completion counts as current. |
| `scripts/audit_run_leakage.py`, `scripts/audit_leakage_provenance.py` | Older, hardcoded audits of corrected_v2/pilot_ctgan_v1; the latter also checks the old SDGym simulated pipeline | Archive the original scripts with the investigated studies. Retain/adapt useful checks under validation entrypoints before calling them general current validators. Their loss-selection assumptions do not cover weighted Census. |
| `audit/comparison_rerun_verification_latest.json` | Read directly by the current real-data comparison builder for MNIST source verification | Move to durable provenance and update the builder. Required by the current implementation. |
| `audit/comparison_news_log_fit_table_hashes.json` | Read directly by the builder to validate log-target generator fit tables | Move to durable provenance and update the builder. Required by the current implementation. |
| `audit/comparison_housing_fit_table_hashes.json` | Read directly by the builder to validate corrected Housing generator fits | Move to durable provenance and update the builder. Required by the current implementation. |
| `audit/statistical_328_20261010/` | Contains the active separate-analysis source, frozen inputs, results and older presentations | Promote active analysis/build code and frozen input evidence. Archive superseded pooled/earlier presentation variants. The directory cannot be removed wholesale. |
| `audit/statistical_review_20261006/analyze_results.py`, `rebuild_rf_xgb_dnn.py` | Older code still used/copied by the current statistical workflow | Promote the needed helpers with the active analysis; update imports and package assembly. Other dated code does not inherit this requirement automatically. |
| `.cache/remote_intrusion/{refresh_comparison*,check_comparison*}.py` | Current report collection/verification tools, currently hidden in ignored cache | Promote the current tools into report/validation scripts. Archive obsolete checkers with their historical snapshots. |
| `audit/mnist_head_rerun_20261009/`, `audit/housing_faker_rerun_20261009/`, `audit/experiment_plan_20261007/` | Actual experiment workers/plans, despite being under audit | Promote reusable workers/task helpers to experiment scripts; retain frozen versions and completed execution evidence in provenance. |

No generic claim that all `audit/` files are disposable or all are mandatory is supported. Final location follows purpose and current use.

## Findings that determine the layout

1. **There is no root README.** The modeling README describes an earlier one-seed workflow and is insufficient as the project's entrypoint. A newcomer needs an overview, supported commands, current study list, data access instructions and clear distinctions between real-data, simulated, historical and diagnostic studies.
2. **The current reports have a definite source set.** The two comparison CSVs each cite 210 distinct records; their union is 406, exactly matching the statistical analysis's frozen source set. This was checked from the actual CSV source columns, not from archive membership. Comparison snapshot archives contain extra records and are not selection lists.
3. **Current results borrow older assets.** Selected records use prepared/generator assets from corrected_v2 and corrected_v2_seed42_mnist28_news, alongside the MNIST/Housing/News rerun namespaces. Result study and input/model study are not always the same. Folder age cannot determine whether an asset is useful.
4. **Three Census runs are verified pilot imports.** Preserve their original records, source hashes and import ledger. Duplicate pilot predictions/weights can share a verified stored copy after the location mapping and import checks are updated; the import does not require two physical copies forever. The production runner also imports the pilot helper's code, so that code must be promoted even if pilot outputs are archived.
5. **CSV does not mean one kind of data.** The remote inventory has 4,282 CSV files: prepared train/dev/test tables, synthetic tables, feature-only tables, predictions, epoch histories, summaries, audit findings and copied snapshots. Representative remote headers confirmed these different schemas. A shared table root needs purpose subfolders.
6. **Important assets are incorrectly hidden in cache.** News keeps fitted labelers, combined labeler tables and completion markers in a runtime cache. Other workers keep generated features there; the statistical workflow keeps frozen analysis inputs there. Move durable assets before making cache disposable.
7. **Modern real-data names already encode the important distinctions.** Dataset, seed, generator, full/X-only construction, labeler, row count and training mode are present where applicable. Preserve these names initially. Normalize ambiguous simulated and legacy names only after their meaning is established.
8. **The repo's Git rules need corresponding updates.** CSVs and pickles are broadly ignored, while Git LFS rules name specific old paths. A move can silently omit deliverable tables or stop applying LFS. Explicitly track small published tables/manifests/code and use the intended storage policy for large data/models; update path rules with the migration.

The checkpoint inventory is not a pile of disposable per-epoch saves: output has 1,207 selected weight files alongside 1,207 canonical run records; a separate preserved partial weight exists in cache. The earlier exact-byte investigation found 212 duplicate groups with 13.01 GiB of repeated logical content, but about 6.75 GiB of repeated allocated content after counting shared inodes once. Some copies are already hardlinked. These are opportunities to investigate after relocation, not promised deletion savings; retained consumers, uninspected links and historically distinct evidence reduce what can be reclaimed.

The refreshed counts are unchanged: corrected_v2 635 records; supplementary namespace 90; weighted Census 76; repaired MNIST 212; corrected Housing 58; News log 74; CTGAN pilot 45; three each in the Census/Credit/News diagnostic pilots. These are file counts, not independent observations or a claim that the original 816-run matrix is complete.

The maintained implementation has a useful existing separation worth keeping: `src/datasets.py` acquires source datasets; `src/commons/` prepares/splits them; `src/synthesize_data/` fits generators and labelers; `src/modeling/` trains/evaluates downstream models and writes run records; `simulated_eval/` implements the separate oracle benchmark. The root README should explain this sequence and link specific entrypoints. Prototype notebooks and old simulated helpers mixed into those folders require individual consumer review; a filename such as old/test/pilot is not enough to classify them.

## Recommended folder structure

Use one table hub. All useful CSVs are discoverable there, with purpose and study visible before opening a file. Keep a single canonical location for each durable artifact; report packages may contain documented export copies.

```text
README.md                         project overview and starting commands
docs/
  repository_map.md               what each folder/module does
  real_data_protocol.md           methods, splits, settings, metrics
  simulated_protocol.md           independent known-distribution design
  validity.md                     applicable findings and limitations
  history.md                      corrections, pilot imports, exclusions
src/                              maintained implementation
simulated_eval/                   maintained separate benchmark package
scripts/
  experiments/                    runners and worker/task helpers
  validation/                     current artifact/provenance checks
  analysis/                       statistical analysis source
  reports/                        figures, tables, PDFs, refresh tools
artifacts/
  tables/
    real/<asset-study>/<dataset>/seed_42/
      prepared/                   train/dev/test, raw and one-hot views
      synthetic/                  generated-target and relabeled tables
      features/                   shared features without targets
      test_predictions/           held-out predictions
      epoch_history/              per-epoch training/development history
    simulated/<study>/<dataset>/seed_42/
      prepared/                   original independent train/test tables
      synthetic/                  generated and relabeled tables
      features/                   feature-only generator samples
      evaluation_tables/          real-only baseline training table
    results/<report-or-analysis>/ tables used in published deliverables
    validation/<investigation>/   retained tabular findings
    historical/<study>/           useful legacy tables, clearly marked
  models/
    generators/<asset-study>/<dataset>/seed_42/
    labelers/<asset-study>/<dataset>/seed_42/
    downstream/<result-study>/<dataset>/seed_42/
  records/<result-study>/<dataset>/seed_42/
                                  immutable run records and D2 sidecars
  provenance/<study>/              manifests, hashes, frozen source, import ledgers
  catalog.csv                      artifact role, status, location and identity
reports/
  real_comparison/                 figures and explanatory text
  real_statistics/                 PDFs, editable reports, export package
  simulated_benchmark/             report and its source-linked exports
archive/
  investigations/                  one-off audits and essential supporting evidence
  legacy_code/                     superseded pipelines/notebooks
  reports/                         historically distinct report releases
.cache/                            recreatable downloads/build intermediates
```

`<study>` initially retains the existing namespace, avoiding accidental merging of incompatible protocols. A small study index explains its status and purpose. Stable report names point to explicitly identified releases/source sets; there should be no ambiguous collection of files called final/latest/new without a documented source.

Raw source assets belong with their dataset acquisition provenance; do not relabel legacy train/test CSVs as untouched raw source datasets. Do not change CSV payloads or convert formats as part of organization.

## Naming and location examples

| Existing file | Proposed location/name |
|---|---|
| `data/corrected_v2/adult/seed_42/adult_seed42_real_train_raw.csv` | `artifacts/tables/real/corrected_v2/adult/seed_42/prepared/adult_seed42_real_train_raw.csv` |
| `data/corrected_v2/adult/seed_42/adult_seed42_ctgan_full_rf_100k.csv` | `artifacts/tables/real/corrected_v2/adult/seed_42/synthetic/adult_seed42_ctgan_full_rf_100k.csv` |
| `output/corrected_v2/adult/acc/adult_seed42_ctgan_full_rf_synthetic.predictions.csv` | `artifacts/tables/real/corrected_v2/adult/seed_42/test_predictions/adult_seed42_ctgan_full_rf_synthetic.predictions.csv` |
| `data/corrected_v2/adult/seed_42/adult_seed42_ctgan_full_rf_100k.predictor.pkl` | `artifacts/models/labelers/corrected_v2/adult/seed_42/adult_seed42_ctgan_full_rf_100k.predictor.pkl` |
| `output/simulated_all_20260930/gaussian/seed_42/ctgan-joint-rf.csv` | `artifacts/tables/simulated/simulated_all_20260930/gaussian/seed_42/synthetic/gaussian_seed42_ctgan_full_rf_10k.csv` |
| `output/simulated_all_20260930/gaussian/seed_42/test.csv` | `artifacts/tables/simulated/simulated_all_20260930/gaussian/seed_42/prepared/gaussian_seed42_real_test_raw.csv` |

Use the same tokens across tables/models/records. `full` means a generator fitted on (X,Y), `xonly` means fitted on X; `generated` identifies retained generated targets. The mapping must follow `src/modeling/constants.py`, not substring guesses. Do not equate `census` with Census KDD; legacy `census` denotes Adult. Do not rename historical aliases without recording the original identity.

## Retention decisions

- **Current reported studies:** keep records, held-out predictions, D2/normalization evidence, split manifests, input hashes, source/protocol evidence and report/analysis inputs. Keep useful prepared/synthetic tables and trained generators/labelers/selected checkpoints as reusable research assets. Large assets need documented availability; every one need not be committed to the source repo.
- **Completed methods outside today's panels:** retain as distinct study extensions, not disposable files. NB/PCA-GMM and Credit/Intrusion exclusion from a figure is not deletion evidence. Mark known preprocessing defects on affected historical runs and prevent their promotion to validated current results.
- **CTGAN pilot:** a real full-budget historical study, not a smoke test. Archive its distinct records/protocol/evidence. Preserve unique useful models/tables; consolidate verified identical assets through the catalog. An obsolete leakage script's hardcoded scope does not make all pilot paths permanent active dependencies.
- **Census pilot:** keep immutable source records and import evidence; use verified shared locations for duplicate predictions/checkpoints. Move the production-used helper to a neutral experiment-helper location and update its consumers.
- **News/Credit pilots and dropout trials:** preserve concise diagnostic history and distinct result/evidence bundles outside the main report set. Old News scores must not be blended with fresh log-target scores.
- **Superseded MNIST/Housing/News results:** archive their defect/protocol explanation and unique historical evidence. Preserve any inputs/models still shared with retained studies. A repaired downstream evaluator does not imply generators were retrained.
- **Smoke outputs:** retain useful validation source plus a small completion/settings record. Retire bulky intermediate fixtures only after checking that they are reproducible and no retained benchmark/report uses them.
- **Repeated snapshots/backups:** preserve historically distinct content and its hashes. Consolidate byte-identical or exactly restorable copies. Obsolete checkers can be archived with their evidence; their old ZIP paths need not constrain the active repo forever. The previous 2.34-GiB proposal is a verified subset, not the ceiling on eventual cleanup.

Keep distinct original and standalone simulated studies. The archived paper-replication table is still an input to the simulated report; its pre-fix provenance limitations must follow it. Moving it under the current benchmark's name would misrepresent it.

## Migration mechanics and order

1. **Document the project and study index first.** Define current report sources, diagnostics, historical studies, dataset identities and each folder's purpose. Identify reusable code currently hidden in audit/cache. Preserve nested GLRM/pyglrm repositories; do not relocate them incidentally. Unexplained root files such as April29.csv/final_results.xlsx need an owner/provenance decision before retirement.
2. **Finish the migration catalog.** For every retained artifact, record old host/path/name, new relative path/name, role, study/dataset/seed/construction, hash, status, consumers and any retained counterpart. The CSV is the machine-readable mapping; history.md explains why studies changed. The generated review catalog supplies a starting point, including explicit unresolved cases.
3. **Promote the active code.** Move reusable analysis/report/validation/experiment tools with their needed helpers; update imports, root calculations and frozen-runtime creation. Move report build sources out of tmp. Preserve historical source snapshots unchanged. Both local and remote code can differ; inspect and preserve remote originals before deploying necessary files.
4. **Move current artifact families together, without rewriting original records.** Existing records contain absolute remote/runtime paths; D2 sidecars/snapshots hash those original record bytes. Introduce an explicit old-path-to-new-location lookup at historical record readers and fail if a mapping is absent. Preserve original records, sidecars and source hashes. Update future writers to use the new documented locations and relative paths. Do not silently search for substitute assets or rewrite old source hashes to match reorganized code.
5. **Move cache-resident durable assets and update their real consumers.** News's cache markers verify combined tables and predictors; retain their identities together. The task helpers' feature tables/metadata and frozen statistical snapshots need durable locations before old queues/runtimes are retired. Update current collection/check scripts; retire one-off checkers deliberately.
6. **Archive historical bundles and normalize proven names.** Keep old namespace/protocol/identity distinctions in the index. Apply the proposed simulated names with updates to benchmark writers, rescore readers, auditors and report builders. Do not introduce a new statistical design or regenerate experiments for a folder change.
7. **Consolidate copies last.** Verify bytes/line endings and collective snapshot coverage. Record counterpart locations and restoration limits; older ZIP container hashes are not recovered merely by preserving member bytes. Remove only the reviewed duplicate/intermediate allowlist after migration checks pass.

Implementation touchpoints include `src/modeling/constants.py`, `run_record.py`, output paths in classification/regression entrypoints, `CreateSyntheticData.py`, generator/labeler writers, matrix/resume runners, weighted-Census import verification, News cache helpers, MNIST/Housing workers, `build_corrected_results.py`, `compute_regression_d2.py`, `plot_recent_comparison.py`, current refresh/check tools, statistical collection/analysis/package code, and simulated benchmark/rescore/report code. Moving scripts into subfolders changes their root/import assumptions; this needs source edits, not a file-manager-only move.

## Validation and scope limits

The review catalog has 9,122 entries: all 4,282 remote CSVs, relevant saved models and provenance/record sidecars, the eight dynamically used News labeler completion markers, plus 122 local statistical CSVs. It proposes 7,565 locations and marks 1,557 entries for owner/purpose/location review; it is not an execution or deletion allowlist. Historical run defects from the October 9 review are included on corresponding records/predictions/checkpoints/epoch histories. The proposed destinations have no collisions within a host. The dependency closure contains 2,410 files reachable from selected records, D2 sidecars and the three current builder evidence files; this is a metadata graph, not an assertion that every file is necessary for plotting or that all dynamic/external consumers are captured.

Before actual migration, refresh working-tree changes and live server processes/queues. At 1:39 p.m. no experiment worker was observed and the six inspected tmux panes were dead. The active statistical report changed during this investigation; its source/output files were only read, and its latest separate synthetic/mix analysis must be preserved.

After migration, validate artifact hashes, explicit path resolution, source IDs, import provenance, current normalization and D2 linkage, selected record coverage and report/analysis numerical tables with existing applicable tools. Run experiment/data/model checks remotely; the local work here only inspects metadata/artifact files. Compare metric tables and protocol/selection settings, rather than claiming entire PDFs must have identical bytes after documented path changes. No retraining is needed to establish a location migration.

New preparation writes only review artifacts. Known hashes reuse the previous byte-level audit and must be refreshed at migration time. The comprehensive remote inventory predates the separate-analysis runtime added by another chat; current local statistical CSVs supplement it, and that runtime still requires a fresh full inventory before it can be moved. Remaining review cases are intentionally not guessed or declared unused.

Review artifacts: `repository_organization_file_catalog_2026_10_10.csv`, `repository_organization_review_validation_2026_10_10.json`, `repository_organization_fresh_remote_2026_10_10.json`, and the read-only preparation script. Supporting hashes/import/duplicate evidence remain in the earlier cleanup audit artifacts.
