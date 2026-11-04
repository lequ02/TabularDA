# Census KDD TabDDPM rerun

Launched **October 11, 2026, 6:16:36 p.m. Chicago** on `thuy@10.24.10.133`.
This is the authorized recovery from
[the seed-42 numerical failure](TAB_DDPM_CENSUS_KDD_DIAGNOSIS_2026_10_11.md).
The [patch register](TAB_DDPM_PATCHES.md) records the fix and its rationale.

## Queue and settings

Persistent tmux session: **`tabddpm-census-kdd-fix`**.
Output namespace: **`tabddpm_census_kdd_500ep_fix_20261011_v1`** under the
normal remote `data/`, `sdv trained model/`, and `output/` directories.

One worker runs this order:

1. Seed 42: reuse the validated weighted real-only baseline; fit/sample/relabel
   the full-table and X-only generators; evaluate all 26 synthetic/mix arms.
2. Seed 43: train its weighted real-only baseline; fit/sample/relabel the
   full-table and X-only generators; evaluate all 26 synthetic/mix arms.

The complete inventory is four generator fits, 26 synthetic tables, and
54 downstream records, including **53 new runs and one reused baseline**.
RF/XGB/DNN downstream comparisons precede supplementary NB/PCA-GMM arms
under the existing runner's ordering.

Each generator starts from zero and retains **500 epochs, batch size 500,
1,000 diffusion timesteps, and 100,000 samples per source**. There are
162,962 training rows, giving 326 batches per epoch and **163,000 updates
per fit**, including each epoch's final short batch.

Downstream settings remain batch 128, learning rate 0.001, at most 100 epochs,
patience 30, and the corrected Census `census-weighted` evaluation protocol
with development `f1_macro` selection. The exact per-seed queue was checked
before launch: 57 commands cover two preparations, two generation pipelines,
and 53 new downstream runs. The reused baseline supplies the remaining
record in the 54-record expected matrix. This dispatch check did not train
models or count queued commands as completed results.

## Source and artifact preservation

The worker reuses the already validated frozen corrected source at:

`/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_rerun_20261011/repo`

All source hashes and the runner hash were verified. No additional production
source file was changed or copied. Separate processes receive the Census
namespace through `CORRECTED_RUN_NAMESPACE`, so sharing the frozen code with
the Covertype workers does not share their outputs or Python state. Keep
this snapshot until all dependent queues and provenance needs are resolved.

Corrected `src/` source hash:
`ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc`.
The validated per-variable logsumexp correction is present in both that
snapshot and the main remote source. Its full/X-only Census checks passed
for seeds 42/43, including finite gradients and all diffusion timesteps,
as documented in the diagnosis. No production budget was shortened.

Both seeds' real train/development/test tables and split manifests were
copied from `tabddpm_500ep_20261010_v1`, after validating every saved hash,
row count, split random state 42, and disjoint source IDs. Copied hashes were
checked again. Rows per seed are train 162,962, development 18,076, test 9,523.

The seed-42 original record is linked into the new output inventory. Its
actual original source hash, artifact paths, and values remain unchanged.
All artifact hashes, settings, weighted protocol, and split manifest were
revalidated. Only the generator's categorical-posterior module differs from
that baseline's source; real-only training does not use it. The new launch
manifest explicitly records this provenance and reuse. Failed original
generation logs and original tmux panes are preserved.

## Initial verified progress

At **6:17:16 p.m. Chicago**, the queue parent PID 1503552 and generation child
PID 1503639 were live. Seed 42's full-table generator reached **100 / 163,000
updates**, with finite loss **1.933374**, beyond its previous update-13
failure. Seed 43 is queued after all seed-42 stages.

No new Census generator fit or downstream result had completed. The sole
completed record in this namespace was the reused original seed-42 baseline.
Both Covertype rerun workers remained active and their shared frozen source
hashes were unchanged. GPU memory was 859 MiB used with 10,306 MiB free;
available RAM was 23,473 MiB.

## Following the run

Worker log:

`/home/thuy/Research/minh_data_synth/TabularDA/output/tabddpm_census_kdd_500ep_fix_20261011_v1/logs/workers/tabddpm-census-kdd-fix.log`

Generation logs are under the same output namespace:

- `logs/census_kdd/seed_42/census_kdd_seed42_tabddpm_generate.log`
- `logs/census_kdd/seed_43/census_kdd_seed43_tabddpm_generate.log`

Each downstream arm has its own stage log in the corresponding seed folder.
Remote `output/<namespace>/launch.json` records the exact command, settings,
source and worker hashes, input hashes, baseline reuse, and dispatch checks.

The thin worker queue is in ignored remote
`.cache/tabddpm_census_kdd_rerun_20261011/worker.py`; it calls the maintained
`run_experiment`, `methods_for`, and `run_classifier` functions. Local copies,
the launch manifest, validation, and the initial status are in the matching
`.cache/` directory. Its `inspect_remote.py` can be sent through the dedicated
SSH connection to inspect current processes, stage logs, and source hashes.

Failures stop the queue with their real exception. There are no automatic
retries. Check live processes and saved artifacts before any manual restart;
the generator entry point refuses to overwrite partial artifacts. A launched
queue or dead pane does not establish completion: verify model/sample
provenance and completed `*.run.json` records before aggregation.
