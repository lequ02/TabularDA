# Investigated experiment cleanup plan

Revised October 10, 2026, after the remote dependency investigation completed at 1:13:51 p.m. Chicago and subsequent focused checks. No experiment asset was deleted, moved, rewritten, loaded, or retrained. Only investigation evidence, file lists, and copies of small historical report snapshots were written locally.

The earlier folder-level proposal is withdrawn. In particular, pilot_ctgan_v1 is not a disposable 6.21-GiB pool. Names, timestamps, and absence from today's comparison figure do not establish disuse.

## What was actually checked

- Inventoried 16,477 remote files across data, generators, output, cache, audit, scripts, source, and simulated evaluation, without following directory symlinks or inspecting credentials.
- Parsed 3,204 JSON files without parse errors, including 1,207 canonical output run records and 861 copied records in cache snapshots. These are file counts, not independent observations or a claim that every planned experiment completed.
- Traced saved paths/import provenance, provenance/quality/metric sidecars, small result tables, and source/document references. Then inspected dynamic cache and queue paths in their actual runners.
- Hashed 19.03 GiB of suspected duplicate assets remotely, finding 212 exact duplicate groups. A duplicate remains necessary if a retained record or runner uses its path.
- Verified all three Census imported source-record hashes and copied predictions/checkpoints, and all six CTGAN pilot generator provenance records against 500 epochs, batch 500, CUDA, and 100,000 samples.
- Inspected all 16 local report ZIPs member-by-member using SHA256, all 216 loose backup records, and same-size local checkpoint/archive pairs. Local work inspected artifact bytes only; no data distributions, model execution, or experiment tests ran locally.

No missing top-level saved *_path dependencies were identified in the canonical remote records. This is a filesystem/reference finding, not a replay or independent validation of every score.

## Pilot files that must remain

| Collection | Actual use | Decision |
|---|---|---|
| pilot_ctgan_v1 | 45 full-budget historical downstream runs reference 21 synthetic tables, 6 generators, 18 fitted labelers, 45 selected checkpoints, and 45 prediction files. Both leakage-audit scripts still inspect this namespace. | Keep the experiment and evidence bundle. |
| census_weighted_pilot_20261004 | Three runs were imported into the weighted production matrix. Production records cite original pilot records/SHA256s; its import ledger records original artifacts. The October 6 statistical snapshot also cites pilot records. | Keep original import provenance, results, and supporting artifacts. |
| credit_weighted_macro_f1_pilot_20261006 | Three completed weighted-loss/macro-F1 diagnostic runs document the minority-class investigation. They reuse corrected_v2 inputs and generators. | Keep the distinct diagnostic procedure and evidence, despite exclusion from current panels. |
| news_log_pilot_20261008_0048 | Three completed log-target diagnostic runs; historical figures and verification evidence use them. Their inputs/generators remain in corrected_v2/news/seed_43. | Preserve pilot results and shared dependencies. Fresh news_log_v1 panel use does not retire evidence. |
| Simulated smoke collections | Separate implementation-check evidence; some small collections have no canonical run-record schema. | Retain separately; no production interpretation or bulk deletion. |

The file-level pilot decision table records **552 files to keep**, with namespace, size, reason, and literal-reference count. A zero count does not override dynamic namespace use or membership in a retained evidence bundle.

Concrete counterexamples: the pilot Covertype RF predictor is byte-identical to production copies but its two pilot run records use its original path. Several pilot MNIST28 tables are identical to repaired-MNIST inputs, but their pilot records still use the original paths. Equal bytes do not authorize breaking saved provenance paths.

## Caches and older inputs that remain needed

The News runner constructs labeler paths dynamically in its runtime's .cache/news_log_v1/seed_<seed>/. Completion markers require both the combined prediction table and fitted predictor to exist and match hashes. Eight predictor files had no literal JSON path references, but source inspection proves they are used. Keep these files, combined tables, markers, and frozen runtime source. Evidence: scripts/run_news_log_experiment.py, lines 194-209.

The full_completion_20261007 runner also reads .cache/full_completion_20261007/features/<dataset>_seed<seed>_<generator>_xonly.csv and its JSON before relabeling. Eight feature tables total approximately 456 MiB. Retiring them requires a decision to retire that labeling path, not a cache purge. Evidence: audit/experiment_plan_20261007/experiment_task.py, lines 197-200.

Keep downloaded source caches needed by the dataset fetchers; prepared splits, generators, fitted labelers, selected weights, predictions, sidecars, saved News normalization, audit findings, and runtime snapshots supporting retained studies. Later reruns share some earlier assets.

News/MNIST/Housing runtimes link data/output/generator directories to shared repository directories. Recursive cleanup must never follow those directory links.

## Concrete proposed local cleanup

The proposal contains **231 exact file paths**, approximately **2.34 GiB of logical bytes**. Execution requires a fresh hash/job/change check and preservation of all listed counterparts and restoration evidence. This investigation has not performed deletion.

| Action | Files | Logical GiB | Evidence and retained copy |
|---|---:|---:|---|
| Remove ignored duplicate legacy checkpoint copies | 4 | 0.439 | Two MNIST12 and two Covertype pickle pairs match SHA256; retain adjacent current copies. Targeted source/notebook searches found no consumer of those old model directories. |
| Consolidate loose mix_refresh_backup_20261005 records | 216 | 0.665 | Every backup is a run.json and matches its retained canonical file after newline normalization. Every original byte sequence was reproduced exactly by converting retained LF text to CRLF. Preserve original/keeper SHA256s and the tested restoration rule. |
| Replace eleven redundant historical report ZIPs with preserved snapshots/member maps | 11 | 1.239 | Historical record/sidecar versions have retained byte-identical sources. Unique snapshot metadata was copied and hash-verified; each member's retained location/hash is recorded. Keep current ZIPs and canonical records. |

The four proposed checkpoint paths, relative to the local repository:

    sdv trained model/mnist12/old_split/mnist12_synthesizer_onlyX.pkl
    sdv trained model/mnist12/old_split/mnist12_synthesizer.pkl
    sdv trained model/covertype/old/covertype_synthesizer_onlyX.pkl
    sdv trained model/covertype/old/covertype_synthesizer.pkl

ZIP coverage was checked collectively: **738 distinct historical record/sidecar versions** are already present in retained canonical files or protected current archives. Coverage does not depend on another ZIP that is also proposed for removal.

Two October 7 ZIPs are **excluded** because .cache/remote_intrusion/verify_refreshed_comparisons.py, lines 28-33, reads them directly. Keep their original paths. Keep historical CSV/JSON/figure/source backups read by older comparison checkers. Only specific files are proposed; do not remove whole backup directories.

ZIP **member and snapshot bytes** remain exactly recoverable through the member manifest. Repacking need not recreate original ZIP container bytes or whole-ZIP hashes. If those exact old containers must remain, exclude the eleven ZIPs and limit the proposal to about **1.10 GiB**.

The two MNIST28 ZIP copies match bytes but both are tracked archive artifacts, including the historical Git LFS checkpoint; neither is proposed for deletion. MNIST28 old pickle checkpoints differ in size from their current counterparts. Covertype's old TVAE checkpoint is also outside the verified duplicate copies. These folders must not be removed wholesale.

## Remote decision

No remote production/pilot namespace or runtime cache is in the deletion proposal. The preliminary static-reference shortlist was manually reclassified: eight News predictors are dynamic dependencies, twelve Housing data copies remain held as validation fixtures, and one small preserved partial Census checkpoint matches its completed canonical counterpart but remains a separate review candidate. Directory-wide deletion is unsupported.

At the snapshot, no experiment training worker was observed; all six inspected tmux panes were dead. MNIST/Housing/News completion statuses and final logs corroborated their finished reruns. Full-completion queue logs ended successfully; the older experiment-repair log contained a failure and remains protected. Recheck immediately before execution. The server had approximately 416 GiB free.

## Execution and validation, if requested

1. Use the CSV/JSON proposal as an allowlist. Refresh processes/queues and relevant working-tree changes. Verify candidate/keeper hashes, resolve absolute paths and links, and check current consumers.
2. Verify preserved ZIP snapshots/member maps and exact loose-backup restoration. Keep canonical records/current archives immutable; stop on changed or missing counterparts. Exclude ZIPs if original container bytes are still required.
3. Remove only listed files. Preserve pilot bundles, runtime cache fits/markers, shared inputs, selected checkpoints, manifests, logs, and unrelated changes.
4. Confirm unchanged retained record coverage, Census imports, report source hashes, normalization/D2 sidecars, and required paths. Use applicable existing checkers; do not run obsolete historical assertions against changed modern panels or retrain models.
5. Record deleted paths, actual allocated space recovered, retained counterparts, and restoration instructions. Logical sizes do not guarantee an equivalent free-space increase.

## Review artifacts

- [Exact proposed files](experiment_cleanup_proposed_files_2026_10_10.csv) and [review manifest](experiment_cleanup_review_manifest_2026_10_10.json).
- [Pilot file decisions](experiment_cleanup_pilot_file_decisions_2026_10_10.csv) and [pilot/import verification](experiment_cleanup_pilot_verification_2026_10_10.json).
- [Remote dependency/hash audit](experiment_cleanup_dependency_audit_2026_10_10.json) and [manually classified cache duplicates](experiment_cleanup_remote_duplicate_candidates_2026_10_10.json).
- [Local duplicate hashes](experiment_cleanup_local_duplicate_hashes_2026_10_10.json), [backup restoration evidence](experiment_cleanup_loose_backup_audit_2026_10_10.json), [ZIP member audit](experiment_cleanup_backup_archive_audit_2026_10_10.json), [archive retention decision](experiment_cleanup_archive_retention_plan_2026_10_10.json), and [member restoration map](experiment_cleanup_archive_member_manifest_2026_10_10.json).
- Read-only audit sources: investigate_experiment_cleanup_20261010.py (remote); investigate_cleanup_backup_archives_20261010.py (local artifacts); prepare_cleanup_manifest_20261010.py (local review artifacts).

Reference discovery covers inspected repository files and documented runners, not uninspected external consumers. Explicit imports, dynamic runtime paths, historical checkers, and evidence bundles were reviewed. No file is proposed merely because a text search found zero matches.
