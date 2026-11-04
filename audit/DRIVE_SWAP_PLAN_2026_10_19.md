# Plan: full research archive on G, working research folder on D

Prepared October 19, 2026, Chicago time. This document plans the move; it
does not authorize launching experiments or change the existing protocols.

## Proposed destinations

| Purpose | Current location | Final location |
|---|---|---|
| Complete original workspace, including hidden files, history, old results, environments and caches | `D:/SummerResearch` | `G:/SummerResearch_Full` |
| Organized results plus a complete, maintained execution package | `G:/SummerResearch_Retained_20261013_2221` | `D:/SummerResearch` |

`G:/summer_research` already exists. Windows path matching is case-insensitive,
so `G:/SummerResearch` would collide with that unrelated existing folder.
Use the distinct `SummerResearch_Full` name after checking it is unoccupied.
Do not create a second `SummerResearch` directory inside either destination.

The remote execution repository remains
`/home/thuy/Research/minh_data_synth/TabularDA`, using
`/home/thuy/miniconda3/envs/env`. D is the code, results, and remote-control
workspace; production computation continues on the research server.

## Findings from the current inspection

- The original D workspace has 124,173 regular files and about 55.983 GiB
  of logical file contents, excluding three external directory links.
- The organized G folder has 10,516 regular files and about 35.117 GiB.
- D has approximately 0.15 GiB free; G has approximately 790.42 GiB free.
  These figures must be refreshed immediately before execution.
- The largest D components are `.git` (22.378 GiB), `data` (16.177 GiB),
  `output` (7.107 GiB), the local `summer_research` environment (5.093 GiB),
  and `sdv trained model` (2.373 GiB).
- The organized folder includes valuable historical source snapshots, but
  `source/current/scripts` contains only the result aggregator and comparison
  plotter. It lacks the main matrix runner and other operational support.
  There is no top-level working `src`, `scripts`, or standalone benchmark.
- The original working tree has modified and untracked files. New work
  includes `output/statistical_anova_gdrive_20261019`; the October 18 archive
  cannot be assumed to contain subsequent work.
- The saved validation-best input inventory references 192 prepared-real CSV
  paths representing 36 saved content hashes and 527 synthetic CSV paths
  representing 521 saved hashes. None of those exact paths maps to a CSV
  currently present under D. These counts cover that comparison inventory,
  not every historical experiment. The 16 GiB of old D data is not a substitute
  for the corrected input tables on the server.
- Three directory links point outside ordinary archive file contents:
  `tmp/statistical_328_csv_replication/node_modules`,
  `audit/results_review/node_modules`, and
  `.cache/remote_intrusion/artifact_tabddpm_20261011/node_modules`.
  Their targets must be inventoried without recursively copying those targets.

These are logical-size measurements, not a guarantee of allocated disk space
or transfer duration. Current file changes, filesystem allocation, and any
additional inputs affect the final space calculation.

## What the new D folder must contain

Keep the current dataset-first result organization: `adult`, `census_kdd`,
`credit`, `covertype`, `mnist12`, `mnist28`, `news`, `california_housing`, and
`comparisons`. Keep the historical `source` snapshots and all validation-best
records, selected weights, predictions, histories, source maps, and caveats.
The 602 alternative evaluations remain distinguishable from the original
loss/F1-selected results.

Add the following small execution package directly at the project root:

| Item | Required contents and reason |
|---|---|
| `src/` | Complete maintained source, including preparation/splitting, generator and labeler implementations, downstream models, CLI entry points, and bundled TabDDPM modules. Preserve the current uncommitted source changes. |
| `scripts/` | Existing matrix, resume, dataset-specific, aggregation, metric-sidecar, and plotting entry points, including any helpers they import. Preserve all current small scripts rather than selecting only one runner. |
| `simulated_eval/` | Its code, README, requirements, and four BIF oracle files. Keep it distinct from real-data experiments. |
| Necessary recovery/report source | Restore validation-best workers, batch helper, inventory/hash metadata, and report recipes to usable runtime filenames and relative paths. Preserve archived historical worker versions separately. |
| `audit/` | Small operational dependencies such as `sdv_api_smoke.py`, relevant current protocol/provenance notes, transfer receipts, and required report/analysis rebuilding sources. Trace dependencies instead of copying all 1.3 GiB indiscriminately. |
| Environment instructions | Root requirements plus the actual server Python/package versions, platform/CUDA information, and a tested setup/launch guide. Do not relabel partially pinned requirements as a complete lockfile. |
| Project instructions and paths | `AGENTS.md`, suitable ignore rules, a README with exact entry points, and one explicit mapping of working root, full archive, remote repository, namespaces, and input locations. |
| Current unfinished work | Preserve ongoing source, paper/report edits, and new analysis outputs identified during the final review; do not leave another chat's only working files solely in the cold archive. |
| Remote login support | Preserve the dedicated credential at the existing ignored local path and the SSH host records, with appropriate permissions. Credentials never enter a report, Git commit, or general export. |

Keep the 22 GiB original Git directory in the full G archive. The D workspace
must retain its usable connection and branch ancestry through an independent,
filtered clone, rather than an unrelated newly initialized repository. Do not
borrow Git object storage from G. Review ignore rules for the newly top-level
dataset artifacts before any commit. Preserve GLRM/pyglrm nested history
unchanged in the full copy; only bring their working code to D if the selected
workflows actually import it.

The old local Python environment stays in the comprehensive archive. Create
and validate a smaller, usable environment at `D:/SummerResearch/.venv` after
the final root is in place. Record and install the necessary local dependencies
separately from the full experiment requirements. Keep the existing full
server environment for production execution.

## Git and venv requirements, clarified after inspection

Both a connected Git repository and a usable local venv are acceptance
requirements for the new D workspace. A plain copy of the results archive
would not supply either one.

Git inspection found the current branch is `backup`, tracking `origin/backup`.
Three remotes are configured: `origin` for `lequ02/TabularDA`, `2nd-repo` for
`thuydt02/DA`, and `code-only` for `lequ02/TabularDA_code`. Configuration is
not proof of live authentication or synchronization; those checks remain
part of readiness. No cached branches were found under the `code-only`
remote during this inspection, so do not silently change the primary remote
to that repository.

`git count-objects -v` reports 11,348,274 KiB (about 10.82 GiB) of garbage
files in the object database, plus approximately 9.31 GiB of valid packs.
Preserve the original directory on G; do not clean it destructively to make
the swap work. The fresh D clone should omit that garbage and avoid retrieving
historical data blobs unnecessarily.

The Git preparation procedure is:

1. Check remote access, branch tips, and local-only commits/refs without
   changing the working tree. Preserve all local changes and unpublished work.
2. Prepare a blob-filtered clone from the same primary remote, retaining the
   selected branch ancestry. Use a sparse code checkout so tracked large
   artifacts are not downloaded just to populate the working tree. Verify
   server support and actual size; do not silently perform a full large clone
   if filtering cannot be honored.
3. Preserve the three remote definitions and the intended upstream tracking.
   Overlay the reviewed current source changes, and explicitly retain any
   unpublished commits needed for ongoing work. All original history and refs
   also remain in the verified comprehensive G archive.
4. Keep curated result/model files outside the code checkout selection and
   protect them with appropriate ignore rules. Check ordinary status/diff,
   branch tracking, and authenticated remote reads. Publishing commits is a
   separate action; no force-push or remote history rewrite is part of the swap.
5. Confirm D uses its own object database, not a worktree/alternate that needs
   G connected. Historical blob inspection may download additional objects
   from the remote on demand, so the initial size is not a permanent cap.

The existing local environment uses Python 3.11.3, based on `D:/python`,
which exists outside the folder being swapped. Its large packages include
TensorFlow (about 1,433 MiB), PyTorch (1,131 MiB), JAX runtime (222 MiB),
and XGBoost (170 MiB). These measurements explain much of the old size;
they do not imply every package is needed for local reporting work.

The venv preparation procedure is:

1. Audit the maintained local workflows and create a pinned local requirements
   file covering their actual dependencies: metadata/report processing,
   plotting, and the selected statistical-analysis tools. Keep full training
   requirements and server versions separately documented.
2. Resolve/download required local packages using G for any staging/cache
   space, given the limited free space on C and D. Check the installation-size
   budget before cutover; do not install the whole historical environment.
3. Recreate `.venv` at its final D location using the verified base Python.
   Venv scripts embed paths, so preparing it on G and merely moving it to D
   is not the portability strategy.
4. Set the editor/launch configuration to the new interpreter and ignore
   `.venv/` in Git. Check the interpreter path, package consistency, imports,
   and required non-training local utilities. Check experiment entry points
   against the existing remote environment.
5. Measure the resulting venv and Git sizes. Both must pass functionality and
   capacity checks before the essential-G recovery copy is deleted. No exact
   reduced size is promised before this validation.

Reference behavior: Git supports filtered object retrieval and sparse
checkouts ([clone documentation](https://git-scm.com/docs/git-clone)); its
object report defines garbage and KiB sizes
([object-count documentation](https://git-scm.com/docs/git-count-objects)).
Python documents recreating venvs because of their embedded absolute paths
([venv documentation](https://docs.python.org/3.11/library/venv.html)).

## Inputs and the meaning of runnable

The default plan makes D a complete code/results workspace for the existing
remote execution model. Every required input must have an explicit location,
expected hash, namespace, and recovery instruction. No experiment may discover
a missing input halfway through training.

1. Enumerate inputs from manifests and recorded provenance for the supported
   current pipelines, the historical checkpoint recoveries, and any unfinished
   work. Verify their live server locations and saved hashes before cutover.
2. Retain small schemas, split manifests/source IDs, preprocessing metadata,
   data-acquisition definitions, source hashes, environment records, and the
   input registry on D.
3. Preserve the exact real partitions and synthetic tables on the server.
   Keep existing original D data in the full G copy. Explicitly identify
   server-only files: moving the D folder does not back up those files.
4. Make selective restoration/deployment use those declared paths and hashes.
   Fail if an expected table is missing or changed. Do not silently substitute
   the old D CSVs or regenerate a synthetic table for historical replay.
5. If an independent, offline copy of all execution inputs on D is required,
   measure and deduplicate that input set first and add it to the capacity
   calculation. This is a different size requirement; the current 35 GiB
   results folder alone is not such a copy. A server-only input backup on G
   likewise requires its own measured transfer rather than an implied claim.

The expected baseline after the swap is approximately 35.1 GiB plus the small
code/operational package and selected newer work. Removing 56.0 GiB from D and
returning 35.1 GiB would yield roughly 21 GiB free before those additions.
Use actual allocated-space measurements and retain a practical free-space
reserve before committing to any additional local data.

## Execution sequence and gates

### 1. Freeze the scope and establish an inventory

- Confirm the two final paths and that no destination conflicts with an
  existing folder. Recheck capacity and filesystem capabilities, including
  hidden files, long paths, directory links, and credential permissions.
- Record file counts, sizes, paths, hashes, and directory-link targets for
  the full source. Include ignored, untracked, and hidden content; Git's
  tracked-file list is not a full inventory.
- Save the current Git state without resetting, cleaning, committing, or
  modifying another chat's work. Record the root and nested repository states.
- Check local receivers, analysis jobs, editors, and relevant chats. Inspect
  live remote jobs/queues as required, but leave unrelated remote experiments
  running. Arrange a short local period without writes for final verification
  and cutover. New files must be incorporated, not skipped as late changes.
- Run the migration procedure from a neutral directory outside
  `D:/SummerResearch`. Keep its receipts on G so the workspace being replaced
  is not its own execution/logging dependency.

Gate: a stable, measured copy scope and a complete list of work that must
remain active after the swap.

### 2. Prepare the working package on G

- Add the execution package above to the existing organized folder on G.
  Keep dataset/result paths intact; add standard code folders only where
  needed. Do not copy another entire repository inside `source/`.
- Preserve historical source and JSON records byte-for-byte. Make maintained
  scripts use configured roots and the existing artifact index where the
  tidy artifact names differ from execution filenames.
- Fix report/recovery assumptions about the old retained-G location and old
  `output/<namespace>` layout. Record relocation in an explicit path map;
  do not rewrite historical provenance to pretend it originated on D.
- Preserve both original checkpoint selection and the separate maximum
  development-metric recovery workflow. Do not change the default matrix's
  stopping rule, seeds, epochs, sample counts, or quality gates as part of
  making it portable.
- Confirm required imported code and packaged assets are present. Capture
  the server environment metadata and validate the input registry.

Gate: this folder contains the source and operational assets required to
launch each supported workflow in the existing remote environment.

### 3. Copy the full D workspace to G

- Copy to a clearly named temporary destination such as
  `G:/SummerResearch_Full.incoming`; check that the name is unoccupied.
- Preserve the complete source tree, including `.git`, `.cache`, ignored
  files, original data/results, the old environment, nested repositories,
  and unfinished work. Do not use a destructive mirror or an operation that
  deletes source files before verification.
- Handle the three directory links explicitly. Do not follow them into
  unrelated global dependency trees. Record any link whose external target
  is unavailable or cannot be represented on the destination filesystem.
- During the local write freeze, recopy changed/new files and recheck the
  inventory. Verify all regular-file bytes with SHA-256 and verify link
  representation separately. Confirm Git and nested-repository integrity
  from the copy without changing their history.
- When verification passes, rename the incoming directory to
  `G:/SummerResearch_Full` on G and save the verification receipt there.

Gate: every in-scope original D file is preserved and verified on G; all
exceptions are explicit. A successful copy command is not the gate.

### 4. Validate readiness before freeing D

- Check imports, CLI help, configuration/path resolution, schemas, and
  artifact lookup. Perform experiment-related checks on the server using
  isolated temporary paths and the established environment.
- Check representative saved CTGAN/TVAE/TabDDPM and downstream/labeler
  artifacts can be found and loaded with their appropriate source versions.
  Do not fit generators, launch queues, or run extra test evaluations merely
  to validate a filesystem move.
- Rebuild/check selected reports from saved completed records in a separate
  destination. Verify metric values, membership, labels, and source references;
  keep validation-best and original results distinct.
- Trace any missing helper/configuration to its actual source and add only
  that dependency. Do not invent unsupported runner flags or claim a dry-run
  mode exists if it does not.
- Ensure active paths do not require the old D contents or the old G retained
  folder name. Historical provenance paths remain explanatory metadata.

Gate: the full backup and the enhanced working package both pass readiness
checks. Until then, original D remains intact.

### 5. Release space on D and populate the working folder

- After the full-copy gate passes, remove only the original D contents
  already verified in the full G archive. Validate every absolute removal
  target. Preserve the project-root name and necessary small login/bootstrap
  files so the final workspace remains `D:/SummerResearch`.
- Renaming the old D folder on D would not free space, so it cannot solve
  the capacity constraint. The verified full G copy is the recovery source.
- Copy the enhanced essential folder from G into `D:/SummerResearch`.
  Retain the G source throughout the copy. Reconcile any intentionally kept
  bootstrap files by checksum/permissions instead of blindly overwriting them.
- Check that no obsolete old-workspace files remain outside the approved
  working-folder inventory. Apply the final root mapping, preserve the ignored
  credential location, finish the independent connected code clone, and
  recreate `.venv` at its final D location.

Gate: D contains the complete planned working set and enough actual free
space; the essential source on G remains available for recovery.

### 6. Verify the final D location

- Verify all copied essential artifacts against their manifests and all new
  working-code assets against their transfer hashes. Verify declared metadata
  edits against their explicit expected new hashes.
- Recheck the 602 alternative result records, their reused-artifact links,
  original result coverage, score tables, and all report links. Confirm that
  the 1,144 reported values and 1,144 delta values remain unchanged.
- Confirm active source, configuration, report rebuilding, remote connection,
  and input preflight work from the final D root. Ensure success does not
  depend on a leftover original file on D or the old retained-G folder name.
- Confirm current unfinished work is present, the credential remains ignored,
  and the root/nested original history is still intact in the full G copy.
- Confirm the D Git clone retains the intended remotes/upstream and does not
  depend on G for object storage. Confirm `.venv` uses its final D interpreter
  path and passes its local dependency/utility checks; the full remote
  experiment environment must remain usable as well.
- Update active instructions and handoff references with the final roots,
  keeping historical records immutable. Save final counts, hashes, environment
  information, input locations, caveats, and Chicago timestamps.

Gate: the final D workspace is usable and all original files remain recoverable.

### 7. Remove the superseded G working copy

Only after the final-D gate passes, remove
`G:/SummerResearch_Retained_20261013_2221` and migration-only temporary files.
Validate resolved deletion targets. Preserve every file added to G's working
copy during preparation in the verified D copy before removing it.

Final state: one full original workspace on G and one organized working
folder on D, with no extra directory wrapping. Keep a compact migration
receipt in each location.

## Recovery rules

- Before the D cleanup gate: abandon an incomplete destination and continue
  using the untouched original D workspace.
- After D space has been released: the verified full G folder restores the
  original workspace; the retained essential-G source restores the compact
  one. Leave those copies in place if any verification fails.
- If D fills unexpectedly, a file/hash differs, a code/input dependency is
  missing, or an active writer changes the source, stop at that gate and
  resolve the cause. Preserve partial transfer evidence and completed artifacts;
  do not substitute inputs, weaken the experiment, or delete the recovery copy.
- Existing remote jobs remain governed by their original commands, source,
  and output namespaces. The local drive swap must not interrupt them or
  deploy new code into files they are using.

## Completion criteria

The move is finished only when the complete original workspace is verified
on G; the organized folder and required current execution package are verified
at `D:/SummerResearch`; every declared input is available at its documented
location; current unfinished work is preserved; the workflow passes readiness
checks; and the obsolete G source can be removed without breaking active paths.

An offline, all-inputs-on-D package is not implied by remote-backed readiness.
If that is the desired requirement, its exact data inventory and capacity
must pass the same gates before the swap is executed.
