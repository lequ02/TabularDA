# Local CSV purpose and research-decision review

The retention rule is now: preserve evidence needed to reproduce the published results **and** evidence needed to explain research choices, failures, corrections, and limitations. Being outside the 406 selected runs does not establish that a file is unnecessary. This review supplements the exact final-run audit and supersedes using its final-membership labels as scientific retention decisions.

No experiment files have been moved, renamed, or deleted. No remote inventory, training, generation, checkpoint replay, or numerical data diagnostics were performed for this review. Existing work on the statistical report was preserved.

## Coverage and inspection

The local scan at **2:30 p.m. Chicago, October 10, 2026** found **1,845 existing CSV files**: **1,577 research CSVs** and **268 installed Python-package CSVs**. Ignored files, `.cache`, legacy `SDGym`/`SDGym-research` data, and nested research directories were included. Git databases/worktrees, `node_modules`, credential/config directories, and symlink targets were excluded. No credential contents were opened.

Every discovered CSV was opened. For the research collection:

- **1,316 files** were parsed completely, including result tables, original historical epoch logs, comparison snapshots, statistical exports, and small validation tables.
- **261 larger files** received bounded structural inspection: their first CSV record and up to ten subsequent records. Their full row counts, distributions, and whole-file hashes are not claimed. This respects the instruction against local large-data diagnostics.
- Headers, observed study/configuration identities, inspection depth, structural issues, evidence links, proposed treatment, and proposed destination are recorded individually. Generic basename matches are distinguished from specific source references; a reader dependency alone is not a retention reason.
- **26 historical logs** have 13 header fields but 23 data fields. Preserve their original bytes. `audit/results_review/reconstruct.py` documents recovering explicitly serialized test-metric dictionaries; blindly treating those extra fields as ordinary header-aligned columns would misinterpret the evidence.

The [individual catalog](local_csv_purpose_catalog_2026_10_10.csv) includes all 1,845 files. Its environment entries mean “leave the installed environment intact”; they are not research cleanup candidates. Numeric, headerless package fixtures are outside the research schema interpretation; their catalog first-record preview is capped at 30 fields.

The new catalog, decision index, and dropout export are outputs of this investigation and were checked separately after creation. The scanner excludes its own `local_csv_purpose_*` outputs to avoid a recursively growing inventory.

## Classification by purpose

These counts are file paths, not independent experiments. Exact copies remain assigned the purpose of the study they document.

| Scientific role | CSV paths | Why retain or consolidate |
|---|---:|---|
| Current result, report input, analysis export, or rebuild evidence | 88 | Preserve canonical tables, source identities, metric definitions, and rebuild recipes. Publication/package copies can share a canonical source. |
| Decision or validity evidence | 191 | Preserve unique observations explaining changes, failures, exclusions, or caveats, even outside the main grid. Includes original historical logs. |
| Earlier statistical/reconstruction exports | 429 | Keep meaningful, labeled versions with their sources and recipes; consolidate repeated exports. Superseded inference is not current inference. |
| Historical report snapshots | 22 | Retain meaningful displayed states and corrections in a version archive; every intermediate refresh backup need not remain active. |
| Explicit hypothetical sensitivity scenarios | 10 | Keep labeled scenarios and their recipes; substituted/reference values are not newly observed experiments. |
| Separate exploratory simulated study | 2 | Preserve the separate 54-row/two-seed design and summary. Do not merge it into the seven-dataset production benchmark. |
| Smoke/validation summaries | 12 | Compact evidence that the checked implementation executed, with its exact configuration and checks. |
| Generated validation/sample tables | 547 | Separate execution validation from method-quality evidence. Review recreation recipes before retiring samples; retain unique regression fixtures where needed. |
| Legacy real-data tables requiring provenance review | 248 | Preserve pending an explicit historical run/split binding decision. Headers and filenames cannot prove safe deletion. Includes the unnamespaced Adult draft. |
| Legacy benchmark data/reference tables | 21 | Archive with their distinct legacy study or reference material, not as extra current observations. |
| Historical operational/provenance snapshots | 2 | Dated queue/source-history evidence, not current completion claims or method-quality results. |
| Earlier cleanup-investigation catalogs | 5 | Intermediate investigation materials. Replace them with the final migration/name catalog after the reorganization is complete. |
| **Research total** | **1,577** | |

For completely inspected small files, **457 groups** contain byte-identical CSVs, with **640 additional copies totaling 35,791,535 bytes (34.13 MiB)**. The catalog identifies a canonical candidate by exact SHA-256, rather than a matching basename. This is a consolidation opportunity, not authorization to remove paths today. Update explicit readers and preserve old-path aliases, study membership, and experiment identities first. Identical sample bytes do not make different labelers or runs the same experiment.

Large legacy tables were not hashed in this pass. No duplicate claim or space-recovery promise is made for them. The older 18.88-GiB data/model archive pool is not a deletion list.

## Evidence that explains alternatives and changes

The [decision index](local_csv_purpose_decision_index_2026_10_10.csv) records 15 topics, supported conclusions, limits, and concrete source paths. It includes:

| Topic | Evidence to preserve | Accurate interpretation |
|---|---|---|
| Housing and News dropout 0/0.1/0.4 | Completed-trial JSON, protocol/source snapshot, and the new [12-observation CSV](local_csv_purpose_dropout_evidence_2026_10_10.csv) | Housing favors 0.4 by mean development MSE. News favors 0.1 by that criterion but test R² remains near zero. Do not describe News 0.1 as losing to 0.4. |
| News log-target pilot | Three pilot CSVs, verified result JSON, original records/hashes, and pilot/rerun notes | All three MAEs improved; R² did not uniformly improve. Log training and BatchNorm removal were combined. The user excluded all three pilot runs from final-rerun reuse. |
| News raw-target failures and negative labels | Saved prediction/checkpoint/labeler diagnosis JSONs and controls | Preserve failed configurations. Correct inverse arithmetic does not prevent an unrestricted regressor from predicting negatives; downstream BatchNorm amplification was independently diagnosed. Clipping was not validated as a repair. |
| Housing Faker defect | Generator metadata/replay/coordinate-control evidence and rerun specification | Explains correcting geographical metadata. Coordinate-only probes are diagnostic counterfactuals, not corrected generator experiments. |
| Census all-negative collapse | Eleven checkpoint summaries, 369 history rows, output-weight bounds, and weighted-pilot evidence | Explains replacing the combined objective/checkpoint-selection protocol. It does not isolate the effect of weighting, dropout, or selection individually. |
| Credit scope | Successful and collapsed historical arms, three weighted-pilot outcomes, and minority counts | The weighted pilot improved two arms. Only ten positive holdout cases motivate scope caution; exclusion is not evidence of unfavorable results. |
| MNIST output heads | 871-run preprocessing table, preserved probes, fixed-head plan/source, and old records | Explains fresh downstream fits. Head correction is separate from generator provenance and feature-collision caveats. |
| NB/PCA repairs | Binary-feature-collapse flags, twelve-run PCA repair table, and repair notes | Identifies implementation defects. These findings do not establish that every NB/PCA method is inherently inferior. |
| Historical aggregation and leakage caveats | Original epoch logs, filename/source manifests, April29 source/reconciliations, and collision evidence | Explains replacing 2025 maximum-test-score/mixed-condition results and retaining current caveats. Historical and corrected observations must stay separate. |
| Statistical specification | Current explicit-factor equations/checks, prior handoffs, and two previous-version planned-contrast CSVs | The current verifier still compares the previous four contrast effects with the new analysis. Preserve those small references without preserving every old report dependency as active. |
| Legacy versus current simulation; Tab-DDPM status | Distinct specifications, paper/supplement comparison, validation evidence, and implementation notes | Smoke execution is not production quality. Deferred/not-run paths must not be described as empirically rejected alternatives. |

For dropout, the mean real-development MSE across seeds 42/43 is:

| Dataset | Dropout 0 | Dropout 0.1 | Existing dropout 0.4 |
|---|---:|---:|---:|
| California Housing | 0.71533452 | 0.67957270 | **0.65489252** |
| News | 68,472,179.48 | **67,951,915.95** | 69,171,250.89 |

The dropout CSV copies saved values and provenance from `audit/dropout_real_results_2026_10_10.json`. It has **eight separate trial observations plus four existing baseline references**. The four baseline references are already among the 406 main runs; the export does not add twelve independent runs. No test score selected a checkpoint or dropout candidate in this review, and no main-report scores were replaced.

Some important evidence is JSON, JSONL, Markdown, source code, or a checkpoint rather than CSV. The individual CSV catalog is therefore accompanied by the decision index and a local decision-asset reference map. In addition to explicit paths, **58 local diagnostic source records were matched to the saved diagnostic SHA-256 values**, protecting historical failures and controls from being dismissed as unselected. The older whole-file catalog now has a separate scientific-retention annotation, so “outside the final grid” cannot be mistaken for “unnecessary.” Asset absence locally does not mean the original experiment never produced it; full remote asset retention remains a later investigation.

## What to do with leakage audits

Keep the concise validity/correction ledger and evidence behind caveats in the final repository's evidence area. For example, keep the MNIST12 collision counts, direct confirmation, train-only fitting investigation, source/hash links, and bounded interpretation. The September 30 audit checked the then-available 313 real runs; that historical coverage must not be advertised as verification of all later namespaces.

Most launch/status dumps, downloaded investigation copies, previews, and old scanner outputs are intermediate materials. Preserve a meaningful compressed historical release where useful; they do not all need permanent active paths. Keep a checker if it is still part of the maintained verification recipe. Moving relevant diagnoses out of the generic `audit/` pile makes their scientific role visible without requiring a newcomer to read every operational snapshot.

## Proposed organization and names

Use one discoverable table hub with distinct studies and purposes, rather than mixing raw training data and statistical results in a flat CSV directory:

```text
README.md                         # repository map, current scope, how to rebuild
docs/
  methods/                        # protocols and metric definitions
  decisions/                      # alternatives, failures, changes, limitations
  history/                        # concise chronological correction ledger
artifacts/
  tables/
    real_comparison/final/
    real_statistics/final/{synthetic,mix}/
    simulated_benchmark/final/
    <decision-study>/evidence/
    <validation-study>/validation/
    <historical-study>/history/
  provenance/                     # immutable records, hashes, source snapshots
  reports/                        # current shareable reports and figures
data/
  prepared/<study>/<dataset>/seed_<seed>/
  synthetic/<study>/<dataset>/seed_<seed>/
  archive/<legacy-study>/          # only retained historical table bundles
models/<study>/<dataset>/seed_<seed>/
scripts/                          # maintained experiment, build and check entrypoints
archive/                          # meaningful previous releases and large history bundles
```

Every catalog row has an unambiguous proposed destination; **these are review proposals, not executed moves**. Proposed table filenames use lowercase study, optional training mode, and purpose, with a short stable old-path identifier for historical releases that share a basename. The catalog preserves every original name and path. A human-facing canonical example is `news_dropout__per_run.csv`; its companion README states the seeds, candidate settings, selection criterion, original source names/hashes, and conclusion. Unknown legacy seed/configuration identities must stay explicitly unknown rather than being invented in a new filename.

Raw/prepared/generated row tables stay under `data/`; useful result, diagnostic, and decision-summary CSVs share `artifacts/tables/`. This satisfies a unified place to find useful tables without obscuring the difference between an observation table and a training input.

For an alternative-path study, preserve a coherent evidence bundle: the question, frozen protocol/configuration/source, all candidate outcomes, checkpoint-selection rule/history evidence, source/test identities and hashes, supported conclusion, and limitations. Retain checkpoints and row-level outputs where replay/reanalysis is still a stated purpose. Do not discard unfavorable outcomes, keep only successful candidates, or select maximum test scores. Large artifacts whose only purpose was temporary execution may be archived or retired only after that purpose is resolved and the retained bundle is verified.

## Migration sequence and validation

1. Agree on the scientific inventory represented by the catalog: current results, decision evidence, separate studies, validation, history, and provenance-review holds. Resolve the 248 legacy real-data table bindings before any destructive cleanup of them.
2. Write the root map and study/decision READMEs, then promote useful evidence and maintained build/check sources. Use the name map as the migration record. Preserve original source bytes and hashes.
3. Move canonical CSVs into the table hub and update their deliberate readers/writers. Historical absolute paths embedded in immutable records remain original provenance; use an explicit old-to-new location map rather than rewriting record contents or silently substituting assets.
4. Archive meaningful prior study/report releases. Retire obsolete consumers intentionally, so their dependency trees do not force every intermediate to remain active. Keep small cross-version verification references where the maintained checker needs them.
5. Consolidate verified duplicate bytes after verifying canonical destinations and aliases. Recheck exact numerical tables, all source hashes, saved News normalization, report scope/labels, decision exports, and the 406-run identity grid. No training or reevaluation is needed for a path-only migration.

The main grid remains exactly **7 datasets × 2 seeds × [2 generators × (1 generated-target baseline + 2 has_y values × 3 labelers) × 2 training modes + 1 real-only baseline] = 406**. Decision evidence, diagnostics, and separate simulated studies are not extra main-grid runs. Each training-mode report cites 210 records and shares the same fourteen real-only baselines: **210 + 210 − 14 = 406**.

The unresolved work is explicit: the local audit establishes each CSV's observed contents and scientific role, but it does not cryptographically bind every large historical data table to a past run or authorize deleting it. The next cleanup stage can migrate the reviewed summaries/evidence and consolidate verified small duplicates while keeping those provenance holds intact.

## Review artifacts

- `local_csv_purpose_catalog_2026_10_10.csv`: individual inventory, observed structure/scope, scientific purpose, evidence, duplicate candidate, and proposed path.
- `local_csv_purpose_decision_index_2026_10_10.csv`: fifteen alternatives/corrections with accurate conclusions and limits.
- `local_csv_purpose_dropout_evidence_2026_10_10.csv`: twelve saved observations, including the four shared main baseline references.
- `local_csv_purpose_inspection_2026_10_10.json`: raw bounded/full inspection metadata.
- `local_csv_purpose_validation_2026_10_10.json`: coverage, classification and byte-duplicate counts.
- `local_csv_purpose_decision_assets_2026_10_10.json`: decision sources and identified local referenced assets.
- `local_csv_purpose_audit_20261010.py`: inspection/catalog and saved-evidence export recipe; it does not execute experiments.

These review outputs are themselves working materials. The final reorganization should retain a concise canonical catalog, decision ledger, and migration record instead of accumulating successive cleanup audits indefinitely.
