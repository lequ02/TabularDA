# Phased local repository cleanup

Prepared October 11, 2026, Chicago time. This is a plan, not a new deletion authorization. No remote inventory or experiment execution is part of these phases. The only changes made while preparing this plan were writing this document. Existing changes from other chats remain intact.

## Target and retention rule

The finished repository should contain the maintained research code, current paper/report sources and outputs, one canonical set of numerical inputs and immutable experiment evidence, and compact evidence explaining decisions and validity limitations. A newcomer should be able to find each through the root README and a single file catalog.

For each retained file, record at least one concrete reason:

1. It supplies a current result, exhibit, or maintained computation.
2. It is needed to rebuild or verify that result, including source records, configurations, split/provenance information, and saved metric definitions.
3. It uniquely supports a research decision, correction, failure diagnosis, exclusion, or limitation.
4. It is required for an explicitly retained experiment replay, diagnostic replay, or regression fixture.
5. It is a necessary operational input for maintained code or the installed local environment.

Dependencies of a retired workflow can be retired with that workflow. Being outside the final experiment grid is not a deletion reason. An archive is subject to the same rules: it should preserve a necessary historical evidence bundle, not become a second copy of everything removed from active folders.

Use four final dispositions: **keep canonical**, **keep compact historical evidence**, **delete after verified migration**, and **hold pending a specific provenance decision**. A hold must identify the unresolved question and the evidence needed to resolve it; it is not a permanent archive classification.

## Starting state

- Completed: 231 listed local duplicate/redundant files were deleted, totaling 2,516,489,972 bytes (2.344 GiB). The deletion receipt and old-to-retained mappings remain available.
- The October 10 CSV catalog covered 1,577 research CSVs and 268 installed-package CSVs. Its 640 extra small, byte-identical CSV copies are consolidation candidates, not a deletion allowlist. They total 35,791,535 bytes.
- At least 37 CSV paths in the newer two-contrast analysis and AAAI paper folders are outside that catalog. Fifteen two-contrast result CSVs currently have exact mirrors between audit and output, totaling 301,128 redundant bytes. Other recent additions must be checked during the inventory refresh.
- The selected real-data grid remains 406 distinct runs: seven datasets, two seeds, and 29 configurations per dataset/seed. The separate simulated benchmark has 378 method rows. Neither number includes every scientifically useful decision study.
- The prior audit identified 248 legacy real-data CSVs needing a historical provenance decision and 547 generated validation/sample CSVs. Those are separate review queues.
- The prior 18.88-GiB legacy data/model pool was an archive-review estimate before the recent duplicate deletions. Refresh it; it is neither a current deletion estimate nor promised space recovery.

## Phase 1 — Freeze the current scope and refresh the catalog

**Purpose:** define what the minimal repository must support before altering dependencies.

- Include current comparisons, the new two-contrast analysis, the separate simulated benchmark, the AAAI paper draft/exhibits/template inputs, and evidence behind research choices. The earlier explicit-factor report is a historical-version candidate; its role must be reconciled with current cross-version checks before it is retired.
- Refresh the local file inventory against current manifests and readers. Include ignored CSVs and newly added report/paper assets, and reconcile deleted paths with the receipt. Open and classify every new research CSV; preserve the distinction between complete small-file inspection and bounded inspection of large tables.
- Identify maintained experiment, analysis, report, and validation entry points. Distinguish code still invoked by them from historical audit scripts and notebooks. Check imports, configured paths, constructed filenames, package contents, and embedded source/hash references rather than relying on literal-name searches alone.
- Create one final catalog with original path/name, study/configuration, role, retained reason, canonical destination, hash where verified, reader/writer, proposed action, and unresolved question. Keep immutable historical record contents unchanged.
- Establish two reproducibility commitments explicitly: rebuilding published tables/reports from saved evidence, and any retained model/diagnostic replay. Report rebuilding alone does not establish that every training table or checkpoint is necessary.

**Exit condition:** every current deliverable and maintained entry point has named inputs; every proposed cleanup candidate has a scientific disposition or an explicit hold. Recent paper and analysis work is included.

## Phase 2 — Reduce historical studies to necessary evidence bundles

**Purpose:** decide what is worth keeping independently of which old script once read it.

- Consolidate the existing 15 decision topics into a concise decisions ledger. For each, preserve the question, exact protocol/configuration/source, candidate identities and outcomes, selection rule, original source/hash links, supported conclusion, and limitations.
- Preserve unfavorable and failed alternatives as well as successful ones. For dropout, keep the eight new trials and four existing baseline references without counting those references as new runs. Housing favors 0.4 by mean development MSE; News favors 0.1 by that criterion. Preserve that distinction.
- Keep the three Census imported origin records and import history. Preserve their tested original-byte restoration rule and the link to the production records.
- Keep concise leakage/collision/correction evidence and relevant original observations. Retain a checker only when it remains part of maintained verification or is necessary to reproduce an important diagnosis. Repeated process listings, transfer logs, scanner outputs, and report-refresh snapshots need a specific retention reason.
- Review the 682 nonselected canonical records by study and purpose. Some substantiate corrections or alternatives. Keep the necessary candidate set for each retained conclusion; discard redundant exports or unrelated execution residue only after the evidence set is complete.
- Review the 547 validation/sample CSVs. Keep the necessary checks/configurations and unique fixtures. Retire generated samples used only for temporary smoke execution when their exact rows are not needed to reproduce a scientific claim or retained regression check. A seed alone does not prove exact reproducibility across package versions.
- For the 248 legacy tables, bind each candidate to historical run/split/encoding evidence where possible. A filename, similar header, or missing current reader is insufficient. Preserve a required original input or a verified reconstructible representation; delete unrelated/redundant variants only after that decision. Leave unresolved cases on a named hold.

**Exit condition:** each retained historical study has a coherent, minimal evidence bundle. Each file outside those bundles is either deletable for a recorded reason or held for a specific unresolved issue.

## Phase 3 — Establish the canonical folders and names

**Purpose:** make useful material discoverable and stop creating parallel copies.

Recommended layout, relative to D:\SummerResearch:

```text
README.md
docs/
  methods/                  protocols and metrics
  decisions/                alternatives, corrections, limitations
  catalog.csv               canonical file inventory
  path_map.csv              original names/paths -> retained locations
src/                        maintained experiment implementation
scripts/
  reports/                  maintained report/analysis builders
  validation/               maintained checks
simulated_eval/             separate known-distribution implementation
papers/aaai27_relabeling/    editable paper sources and template material
artifacts/
  tables/
    real_comparison/
    real_two_contrasts/
    simulated_benchmark/
    paper_exhibits/
    decisions/<study>/
    validation/<study>/
  provenance/<release>/     canonical records, sidecars, hashes, source snapshots
  reports/<report>/         current PDFs, figures, and editable reports
data/<study>/<dataset>/     only inputs required by retained workflows
models/<study>/<dataset>/   only assets with an explicit replay/evidence purpose
archive/<study-or-release>/ compact necessary historical evidence
```

- Result, decision, and diagnostic CSVs belong in the table hub. Raw/prepared/synthetic row tables remain under data; that distinction should be visible.
- Use lowercase table names such as `real_two_contrasts__classification_contrasts.csv`, `real_two_contrasts__synthetic__f1_macro_matched_scores.csv`, and `news_dropout__per_run.csv`. Record dataset/seed/method in per-run filenames when known; keep them as columns for aggregate tables. Do not invent missing historical identities.
- Keep one canonical copy of each logical table and source record. A meaningful version with different data, scope, or statistical specification is a distinct artifact, not a duplicate.
- Prefer one immutable, directly readable directory for the 406 published records and 116 D2 sidecars. It fits existing record-based code and avoids introducing an archive-reading framework. If compressed storage is later chosen, use an explicit indexed reader and verify every original member; do not keep both representations without a reason.
- Promote maintained builders/checks from audit, tmp, output, or cache into scripts. Keep package copies only where a standalone published deliverable actually requires them; generate those packages from the canonical sources.
- Update deliberate readers and writers in small batches. Keep historical record bytes and original asset hashes unchanged; resolve relocated historical paths through the documented path map. Preserve study/run identities even where multiple records refer to identical table bytes.
- Update necessary .gitignore exceptions and path-specific Git LFS rules as part of the move. The current blanket CSV ignore would otherwise omit required tables in the new hub and paper folder. Preserve nested repository boundaries and credential exclusions.

**Exit condition:** each moved asset has a verified retained destination and name map; maintained readers use it; the root README explains every top-level area. Active work and experiment settings remain unchanged.

## Phase 4 — Retire obsolete workflows and their dependency trees

**Purpose:** remove the requirement to keep whole old investigations alive.

- Designate one maintained builder and verifier for each current report family. Retire superseded pooled/composite-factor builders, reconstruction variants, obsolete launch scripts, and exploratory notebooks after their useful code and evidence have been extracted.
- Resolve the actual current two-contrast dependencies: its builder and remote input preparation still read the earlier statistical snapshot; its verifier reads earlier planned contrasts and checks the earlier full deliverable manifest. Move the shared frozen inputs and necessary cross-version references to canonical locations before retiring the old tree.
- Replace obsolete whole-release preservation checks with explicit checks of the retained historical evidence and current results where appropriate. Preserve the scientific comparison they establish; do not simply comment out failing assertions.
- Resolve the current paper builder's numerical, template, and simulated-coverage dependencies. Its source package alone does not include the frozen numerical inputs, so deleting the repository inputs would break rebuilding.
- For historical comparison checkers, retain the necessary check/evidence in the maintained workflow or formally retire the checker. Only then consider removing the remaining October 7/current transport ZIPs. These were intentionally excluded from the completed deletion.
- Keep original historical epoch logs where their unusual serialized metrics are needed to explain corrections. Keep one parser/recipe, rather than all its repeated intermediate exports.
- Preserve a compact, meaningful earlier statistical release and its specification when needed to explain the transition. Do not retain every earlier rendering, wrapper, and package solely because a retired guard once read it.

**Exit condition:** no retained workflow requires a retired path. Current and historical statistical claims remain distinguishable and verifiable from the retained evidence.

## Phase 5 — Delete verified duplicates and disposable intermediates

**Purpose:** remove files whose information already exists in the canonical repository.

- Consolidate the 640 prior small-CSV duplicate candidates and the 15 newly verified two-contrast mirrors, reconciling any overlapping groups in the refreshed catalog. Update readers first; keep study memberships and original-path mappings.
- Consolidate the two flat copies of the 406 selected records after the readers are changed. The prior exact-copy audit measured 938,065,754 bytes (0.874 GiB) per set; recheck current bytes before claiming that saving. Likewise consolidate duplicated sidecars, frozen score exports, current transport bundles, and packages where no independent delivery requirement remains.
- Remove build previews, compiler outputs, and generated exports once the retained source and current deliverable cover their purpose. Keep deliberately supplied shareable formats; do not assume every PDF/HTML/PNG pair is redundant.
- Remove compiled caches only after confirming their source survives. Review notebook checkpoints for unique edits; orphan bytecode and unsaved notebooks are not automatic trash.
- Consolidate the cleanup investigation itself into the final catalog, decisions ledger, path map, and deletion receipts. Retire obsolete scanner scripts, inventories, and repeated cleanup plans only when their essential conclusions and restoration/hash mappings are preserved in those final records.
- Delete only exact reviewed files. Recheck keeper existence/hashes immediately before deletion, preserve a receipt, and remove empty directories only after confirming their remaining purpose. Do not delete whole old/pilot/audit/cache directories by label.

**Exit condition:** all removed files have a recorded canonical replacement or a verified disposable purpose, current outputs are unchanged, and the migration/deletion ledger is complete.

## Phase 6 — Prune large data and checkpoints by explicit purpose

**Purpose:** achieve the smallest defensible retained repository, rather than just a neater layout.

- Refresh sizes after the preceding phases. Review the legacy data/model pool and locally retained experiment assets file-by-file using the Phase 2 decisions.
- Keep a large artifact only for a maintained computation, necessary original-input provenance, an explicitly retained replay, or a diagnosis requiring that artifact. The fact that a result record mentions a checkpoint does not automatically require the full checkpoint for report rendering.
- Selected checkpoints and labelers remain protected until that replay/evidence decision is recorded. The earlier audit identified 174 selected weights and 24 fitted DNN labelers locally; verify the current set rather than making a blanket selected/unselected rule.
- Keep checkpoints or row-level outputs that uniquely establish retained failures or repairs when compact saved evidence cannot substantiate the diagnosis. Do not retain every failed checkpoint merely because an old diagnostic could load it.
- Retire unique execution-only assets only after the retained evidence satisfies the declared reproducibility scope. Do not claim that source plus a seed can recreate an original checkpoint byte-for-byte.
- Local absence of selected inputs, generators, or prediction CSVs is not proof that the experiment never produced them, and unrelated legacy CSVs cannot replace them. Document the current local replay limitations. Do not use assumed remote copies as justification for a local deletion; remote retention verification remains a separate future task.
- Keep meaningful archival bundles only where their contents satisfy a retention reason. Compressing or moving unreviewed files inside the same repository is not a resolved deletion decision and does not justify a promised disk saving.

**Exit condition:** every remaining large asset has a named use; every unresolved one has an explicit hold. The final local reproducibility statement accurately describes what can and cannot be replayed.

## Phase 7 — Verify the minimal repository and close the cleanup

- Verify the exact 406-run identity grid, source hashes, 116 D2 sidecar bindings, and current numerical report/paper tables. Preserve distinct metric definitions and read saved News normalization; do not recompute scores or change research settings during a path migration.
- Confirm that every decision-ledger conclusion has the original candidate evidence it cites, including failures and unfavorable alternatives. Historical tables, hypothetical sensitivity scenarios, and paper reference values remain labeled separately from completed corrected runs.
- Check maintained imports, report input/output paths, package manifests, name maps, and available existing verifiers. Use lightweight local checks of saved bytes/values and source references. Any needed numerical analysis, model replay, generation, training, or experiment test belongs on the server in a separately authorized execution task.
- Rebuild or repackage current report artifacts through existing builders only within their permitted local scope; compare numerical content and source hashes. Original run/source hashes remain fixed, while moved-package and path-manifest hashes may legitimately need regeneration.
- Ensure required CSVs and provenance are actually tracked or intentionally provided through the documented artifact mechanism. Keep installed environment contents, private credentials, SSH host records, and nested repository histories outside this cleanup.
- Finish with one root map, one catalog, one path map, one decisions ledger, and necessary deletion/provenance receipts. Every active top-level folder and retained file has a stated purpose. Publish the list of any unresolved holds instead of implying that they have been proven necessary or safe to delete.

**Exit condition:** a new person can locate current sources, tables, reports, evidence, and retained replay assets; the declared outputs rebuild/verify from the retained material; no unresolved legacy dependency is hidden inside a blanket cleanup claim.

## Space estimates and execution order

The only completed saving is **2.344 GiB**. The previously verified extra flat record set represents about **0.874 GiB** of possible further consolidation, and the prior small-CSV copies plus newly identified mirrors represent about **34.4 MiB** before fresh group reconciliation. Do not extrapolate a deletion saving from the old large-data/model pool.

Start with Phases 1 and 2, perform Phase 3 in small coherent batches, and complete Phase 4 before deleting its retired dependencies. Execute Phase 5 only for verified batches. Phase 6 provides the largest potential reduction but depends on resolved scientific retention questions. Phase 7 closes the work. No further deletion is authorized by merely writing this plan.

## Existing evidence used for this plan

- [CSV purpose and decision review](/D:/SummerResearch/audit/LOCAL_CSV_PURPOSE_REVIEW_2026_10_10.md)
- [Selected result and provenance audit](/D:/SummerResearch/audit/LOCAL_FINAL_RESULTS_AUDIT_2026_10_10.md)
- [Individual CSV catalog](/D:/SummerResearch/audit/local_csv_purpose_catalog_2026_10_10.csv)
- [Decision index](/D:/SummerResearch/audit/local_csv_purpose_decision_index_2026_10_10.csv)
- [Completed deletion receipt](/D:/SummerResearch/audit/deletion_receipt_2026_10_10.jsonl)
- [Current two-contrast report description](/D:/SummerResearch/output/statistical_two_contrasts_20261010/README.txt)
- [Current paper source description](/D:/SummerResearch/papers/aaai27_relabeling/README.md)

The October 10 audits describe their own earlier snapshots. The additions identified while preparing this plan are bounded checks of the newer paper/two-contrast folders, not a claim of a new exhaustive whole-repository audit.
