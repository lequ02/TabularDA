# Covertype TabDDPM rerun

Launched **October 11, 2026, 11:51 a.m. Chicago** on `thuy@10.24.10.133`.
The [numerical diagnosis](TAB_DDPM_COVERTYPE_DIAGNOSIS_2026_10_11.md) describes
the corrected per-variable posterior normalization.

Namespace: **`tabddpm_covertype_500ep_fix_20261011_v1`** under the remote
repository's `data/`, `sdv trained model/`, and `output/` directories.

| Persistent tmux worker | Queue |
| --- | --- |
| `tabddpm-covertype-fix42` | Reuse verified original baseline; full-table fit/sample/relabel; X-only fit/sample/relabel; all 26 synthetic/mix downstream arms. |
| `tabddpm-covertype-fix43` | Train original baseline; full-table fit/sample/relabel; X-only fit/sample/relabel; all 26 synthetic/mix downstream arms. |

Both workers run concurrently with existing Census KDD and MNIST12 queues,
bringing the live experiment worker count back to four. Failed original
Covertype logs and the old tmux session remain intact.

## Settings and source isolation

Each generator fit retains 500 complete epochs, batch 500, 1,000 diffusion
timesteps, and 100,000 synthetic rows per source. Covertype's 513,911 training
rows give 1,028 updates per epoch and **514,000 updates per fit**. Both full
and X-only models start at update zero; the failed model has no intermediate
checkpoint. There are four planned generator fits and 26 synthetic tables.

Downstream settings remain batch 128, learning rate 0.001, at most 100 epochs,
patience 30, and development-loss selection. All six classification labelers,
both feature sources, and synthetic/mix training are queued. The complete
two-seed evaluation inventory is 54 records: **53 new runs plus one reused
real-only baseline**.

To preserve the active workers' source and future subprocesses, the rerun
uses this frozen remote source snapshot:

`/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_rerun_20261011/repo`

Only the categorical-posterior computational line differs in its `src/`
Python files. The maintained runner's remote-root guard was minimally
extended to allow snapshots nested inside the remote repository. The main
local runner contains this same change; it still rejects the local laptop.
The existing CUDA smoke and public dispatch checks passed against the
snapshot. No new production entry point, dependency, fallback, or retry was
introduced. A thin per-seed queue is in ignored remote `.cache/.../worker.py`;
it uses `run_experiment(..., stage='generators')`, `methods_for`, and
`run_classifier` from the maintained runner.

The snapshot links its data/model/output roots to the normal remote roots;
all new artifacts use the repair namespace. Its original source files were
preserved under `.cache/tabddpm_covertype_rerun_20261011/originals/` before
replacement. Transferred files and every copied source file were hash-checked.
Do not edit the frozen snapshot while its jobs are active.

Corrected snapshot source hash:
`ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc`.
The active main remote source remains the original launch hash:
`f5daf5ff979a57d5e7c538e37e2e0a26d7fe92f44312309c700f59e61f87351b`.

## Reused baseline and inputs

Both seeds' six real split tables and manifests were copied from
`tabddpm_500ep_20261010_v1`, after verifying every saved table hash, row count,
split random state 42, and disjoint source IDs. Copied contents were rechecked.

Seed 42's completed original record is linked into the new output inventory.
Its record and all artifact hashes, split manifest, seed, protocol, and
downstream settings were validated. Its actual original source hash and
artifact paths are preserved without rewriting provenance. All source
modules are identical except the categorical diffusion posterior, which the
real-only baseline does not use. `launch.json` explicitly records this reuse;
do not claim that this baseline was trained using the corrected source.

## Initial verified progress

At **11:52:26 a.m. Chicago**, both tmux panes and their training children were
live. Seed 42's full-table generator reached **100 / 514,000 updates**, with
finite loss 1.143597. Seed 43's original baseline had started epoch 1/100;
its generators and downstream comparison arms were queued behind it.

No new rerun fit or downstream result had completed. The sole completed
record in the repair inventory was the reused original seed-42 baseline.
The other two workers remained live, and their source hash was unchanged.
GPU memory was 1,077 MiB used of 11,264 MiB; available RAM was 24,300 MiB.

Follow-up at **11:55:03 a.m. Chicago**: seed 42 reached **1,300 / 514,000
updates** with finite loss 0.536899; seed 43 reached baseline epoch 6/100.
All four experiment workers remained live and main remote source remained
unchanged. GPU memory was 1,073 MiB used, with 10,092 MiB free.

## Following the queues

Worker logs:

- `/home/thuy/Research/minh_data_synth/TabularDA/output/tabddpm_covertype_500ep_fix_20261011_v1/logs/workers/tabddpm-covertype-fix42.log`
- `/home/thuy/Research/minh_data_synth/TabularDA/output/tabddpm_covertype_500ep_fix_20261011_v1/logs/workers/tabddpm-covertype-fix43.log`

Stage logs are under `logs/covertype/seed_42/` and `logs/covertype/seed_43/`
within that same output namespace. Each seed has
`covertype_seed<seed>_tabddpm_generate.log` and individual downstream logs.

The authoritative remote `output/<namespace>/launch.json` records exact
commands, settings, source hashes, input hashes, validated baseline reuse,
and validation evidence. Local copies and the initial status snapshot are
in `.cache/tabddpm_covertype_rerun_20261011/`. Its `inspect_remote.py` can be
sent over the dedicated SSH connection to inspect current logs/processes.

Failures stop the corresponding queue and remain visible. There are no
automatic restarts. Check live processes and existing artifacts before any
manual restart; the generator entry point refuses to overwrite partial
artifacts. Completion requires verified models, samples, and downstream
`*.run.json` records, not just a launched queue or an exited pane.
