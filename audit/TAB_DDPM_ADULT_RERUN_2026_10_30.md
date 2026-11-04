# Adult TabDDPM posterior-fix rerun

Launched **October 30, 2026, 10:22:11 p.m. Chicago** on `thuy@10.24.10.133`.

This reruns the **52 held Adult TabDDPM records** (`hold_pre_patch_multicategorical_source` in
`cleanup_review_20261013`). Their generators were trained before the October 11 categorical-posterior
normalization fix ([patch register](TAB_DDPM_PATCHES.md)). The held originals in
`tabddpm_500ep_20261010_v1` are preserved unchanged.

## Namespace, source, inputs

- Output namespace: **`tabddpm_adult_500ep_fix_20261030_v1`** under the normal remote `data/`,
  `sdv trained model/`, and `output/` directories.
- Source: the validated frozen corrected snapshot
  `.cache/tabddpm_covertype_rerun_20261011/repo`, `src/` hash
  `ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc`. This is the same hash as main remote
  `src/`. The per-variable `log_sum_exp_by_classes` fix was checked before launch. No production source file
  was changed.
- Packages: torch 2.5.1+cu124, scikit-learn 1.5.2, xgboost 2.1.4, pandas 2.2.3, numpy 1.26.4. These match
  the Census KDD rerun.
- Real train/dev/test raw and one-hot tables and the split manifests for seeds 42/43 were copied from
  `tabddpm_500ep_20261010_v1`. Each was checked against `scripts/checkpoint_recovery/input_hashes.json`
  before and after copying, and against its split manifest (rows, hashes, split random state 42, disjoint
  IDs).
- The two real-only baselines (`adult_seed{42,43}_real_original.run.json`, which pass the current validity
  checks) are symlinked into the new namespace. Their record and artifact hashes were rechecked. Real-only
  training does not use the posterior module.
- Settings are unchanged: 500 epochs, batch 500, 1,000 diffusion timesteps, 100,000 samples per source,
  and the six labelers. Downstream uses batch 128, learning rate 0.001, at most 100 epochs, patience 30,
  development-loss selection, and the corrected/standard protocol.
- Plan: 4 generator fits, 26 synthetic tables, and 54 records (52 new runs plus 2 reused baselines). The
  planned 52 arm names were asserted to equal exactly the 52 held record names.

## Queue design

Six tmux sessions:

| Session | Role |
| --- | --- |
| `tabddpm-adult-fix-gen42`, `tabddpm-adult-fix-gen43` | One seed each, in parallel: `run_experiment(stage='generators')`, which runs the prepare-only check, then fits, samples, and labels full, then fits, samples, and labels X-only. |
| `tabddpm-adult-fix-eval1` … `eval4` | Shared downstream pool over all 52 arms (`nice 5`). |

Pool rules (`eval_worker.py`):
- An arm becomes ready when its table's `.quality.json` exists. The generator writes that file after the
  CSV. As a result, the 28 full-table arms start while the X-only fits are still training.
- Each worker claims one arm at a time through an atomic `O_EXCL` claim file, taking the longest expected
  arm first (generated-target mix, then generated-target synthetic, then mix, then synthetic). This keeps the
  workers finishing together.
- Launches are spaced at least 60 s apart. A launch also requires at least 6 GiB of available RAM and at
  least 1.5 GiB of free GPU memory, because the host is shared.
- Each arm goes through the maintained `run_classifier`, which refuses to overwrite a record.
- A failure stops that worker with the real exception and saves the traceback to `state/failed/<arm>`.
  There are no retries. If a generation worker exits nonzero (`state/gen_seed<seed>.exit`), the pool stops
  once only that seed's arms remain.

At 10:23 p.m., both fits had passed update 300/34,500 with finite loss (seed 42: 1.970773; seed 43:
1.952581). Each fit was running at about 5–6 updates/s, CPU-bound and sharing the GPU (77% utilization)
with the Pareto_MTL jobs. That is roughly 1h45m per fit, so the full-table arms should start around
12:15 a.m. and the X-only tables should land around 2 a.m. The rough finish is about 2:30 a.m. Chicago,
October 31. This is an estimate, not a measurement of completion.

## Following the run

- Worker logs: `output/tabddpm_adult_500ep_fix_20261030_v1/logs/workers/<session>.log`.
- Stage logs: `output/tabddpm_adult_500ep_fix_20261030_v1/logs/adult/seed_<seed>/`.
- Launch manifest: `output/tabddpm_adult_500ep_fix_20261030_v1/launch.json`.
- Queue scripts and claim state: remote `.cache/tabddpm_adult_rerun_20261030/` (local copy in the same
  `.cache/` path).

At launch, other jobs on the host were two Pareto_MTL GPU jobs (another project) and the
`leakage-audit-20261030` scan. None were interrupted. A launch is not completion: verify completed
`*.run.json` records, provenance, and selected epochs before using any score. Then refresh the reports and
replace the held Adult TabDDPM cells only with verified records from this namespace.

## Completion — October 31, 2026

Verified at **10:39 a.m. Chicago**:
- Both generation workers and all four downstream workers exited with code 0.
- All 52 arms were claimed, with no failure tracebacks.
- The last pool worker drained at **2:25:51 a.m.**; the four workers finished within 3 m 16 s of each other.
- All four generator fits reached 34,500/34,500 updates with finite final loss: seed 42 full 1.142665, seed 43 full 1.299063.
- Each generator's provenance records 34,280 training rows and 100,000 samples.

The namespace has **54/54 records**: 52 new runs plus the 2 linked baselines. Each of the 52 new records
passed the maintained `completed_run(..., 'corrected')` check, which confirms:
- matching dataset, seed, mode, and method;
- a selected development epoch;
- prediction and weight files are present;
- the standard evaluation protocol;
- the source hash equals the snapshot's;
- every recorded artifact hash still matches.

Before these scores replace the held cells, refresh the reports with the existing builders.

## Report refresh — October 31, 2026

New original-selection comparison reports with these records are in
[comparisons/20261031_adult_tabddpm_fix](../comparisons/20261031_adult_tabddpm_fix/README.md). The earlier
reports are preserved. Records, generator provenance, and queue scripts are archived locally in
`adult/tabddpm_fix_20261030/` (213 files, hash-verified).

## Best-development-F1 checkpoint recovery — October 31, 2026

All 28 selected report configurations (generated targets and full/X-only RF/XGB/DNN, seeds 42/43,
synthetic-only/mix) finished remotely by **11:59:35 a.m. Chicago**. Four pooled workers ran in
`tabddpm-adult-fix-valbest1`–`4`; each exited 0. There were 27 replays, 491 total replayed epochs,
and one unchanged selected checkpoint. Generators and labelers were reused.

The earliest maximum development binary F1 was selected within each saved completed history. Every
replayed history matched within `1e-8`, original checkpoint tensors reproduced exactly, and selected
development F1 and saved test predictions passed the worker's checks. Remote export rechecked all
recorded input/output hashes. All 150 exported files matched their SHA-256 after local transfer;
results, weights, predictions, replay histories and logs are in `adult/tabddpm_fix_20261030/valbest/`.
The queue, deployed worker and inventory are retained there as executed.

[Refreshed best-F1 reports](../comparisons/20261031_adult_tabddpm_fix/valbest/README.md) replace only these
28 evaluations in the 602-evaluation report selection; the other 574 are unchanged. The standalone
`adult/tabddpm_fix_20261030/scores_valbest.csv` contains the 28 corrected arms. NB/PCA-GMM were outside
this recovery scope. The historical best-F1 results and all loss-selected results remain preserved.
