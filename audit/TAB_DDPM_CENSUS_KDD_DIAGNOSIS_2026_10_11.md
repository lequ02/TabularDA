# Census KDD TabDDPM failure and fix

Investigated October 11, 2026, 5:09–5:20 p.m. Chicago. This is Census KDD,
not the legacy `census` alias for Adult. No Census production rerun was
launched during this investigation.

## Failure

The original `tabddpm-3` queue completed the Census KDD seed-42 real-only
baseline, then failed in full-table TabDDPM training with:

`ValueError: Nonfinite Tab-DDPM loss at step 13`

The planned fit was 500 epochs, batch 500, 326 updates per epoch, or
163,000 updates. The generation log ended at 11:55:45 a.m. Chicago:

`output/tabddpm_500ep_20261010_v1/logs/census_kdd/seed_42/census_kdd_seed42_tabddpm_generate.log`

No generator checkpoint, sample, or synthetic evaluation was saved. The
queue stopped before seed 43. This queue retained the original frozen
source; the Covertype repair had been launched from a separate corrected
snapshot to preserve the other active experiments.

## Reproduction and numerical cause

A remote replay used the actual training table, seed 42, original model and
optimizer settings, shuffled batch-500 loader, and the learning-rate schedule
for the actual 163,000-update budget. It reproduced the recorded first loss
13.285660 and the nonfinite loss at exactly zero-based update 13.

The training table contained 162,962 rows, 13 numerical columns, and 25
categorical variables including the target. Transformed numerical values
were finite and ranged approximately from -5.19934 to 5.19934. All earlier
parameter gradients and the weights entering update 13 were finite.

At the failed update, the Gaussian loss was finite (1.517713), but the
categorical loss was NaN. The model posterior became nonfinite for batch
row 499 at diffusion timestep 0, specifically `PEFNTVTY` (father's country
of birth), which has 42 observed training states. Its input log-probabilities
were finite.

This is the same cumulative-subtraction numerical defect identified in
[Covertype](TAB_DDPM_COVERTYPE_DIAGNOSIS_2026_10_11.md). The old posterior
normalization computes each group's mass by subtracting two cumulative
sums in float32; a small increment can disappear through rounding. The
resulting nonfinite posterior propagates into the categorical loss.

For the same weights, batch, and random draws at update 13, replacing that
normalization with direct per-variable logsumexp produced finite loss
**2.887054** and finite parameter gradients. No learning-rate, category,
sample-count, epoch-budget, or loss-objective change was needed.

## Fix and validation

The existing one-line correction in
`src/synthesize_data/tabddpm_core/gaussian_multinomial_diffsuion.py` calls
`log_sum_exp_by_classes(unnormed_logprobs, self.slices_for_classes)` instead
of `sliced_logsumexp(unnormed_logprobs, self.offsets)`.

The corrected core passed 100-update remote training checks for both seeds
and both full/X-only sources on Census KDD training data. Losses, checked
parameter gradients, and final weights were finite. Separate country-of-birth
corruption checks covered all 1,000 diffusion timesteps; category probability
sums differed from one by at most 4.77e-7. These short checks used a smoke
budget; they are not 500-epoch production results.

The existing `audit/tabddpm_smoke.py` categorical regression now checks both
the Covertype and Census KDD schemas. The focused posterior, normalization,
loss, and gradient checks passed on CPU and CUDA. No new test file or
production abstraction was added.

At 5:20:05 p.m. Chicago, the validated correction was deployed to the main
remote source. Only the core, its vendoring README, and the existing smoke
check were transferred. The original files were preserved under:

`/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_census_kdd_diagnosis_20261011/originals`

The main source hash changed from
`f5daf5ff979a57d5e7c538e37e2e0a26d7fe92f44312309c700f59e61f87351b`
to
`ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc`.
The latter matches the corrected Covertype snapshot. All active experiment
processes were verified to use that independent snapshot, whose source
contents remained unchanged. No active worker was stopped or restarted.
The deployed main files then passed the focused CPU/CUDA regression.

## Recovery

Both seeds' saved split hashes were validated: train 162,962, development
18,076, and test 9,523 rows. The seed-42 real-only baseline's saved source,
all artifact hashes, split manifest, batch 128, learning rate 0.001, 100-epoch
budget, and `census-weighted` evaluation protocol were verified. It can be
reused with its actual original provenance preserved; the sole changed
production module is a generator module that real-only training does not use.

Restart Census KDD seeds 42 and 43 using the corrected main source and a
distinct namespace, such as `tabddpm_census_kdd_500ep_fix_20261011_v1`.
Copy only the verified prepared splits/manifests; preserve the original
failed logs and real-only baseline. Reuse that baseline explicitly rather
than changing its source hash. Train seed 43's original baseline, then run
both seeds' full-table and X-only fits and all existing downstream arms.
Retain 500 epochs, batch 500, 100,000 samples per source, and the Census
weighted protocol. Each generator starts from zero because no intermediate
checkpoint was saved. Full-budget stability remains to be established.

Local evidence is in ignored `.cache/tabddpm_census_kdd_diagnosis_20261011/`:
`preflight.json`, `replay_failure.json`, `validation.json`, `regression.log`,
`recovery_inputs.json`, `deployment.log`, and `deployed_regression.log`.
Remote `deployment.json` records transferred hashes, preserved originals,
and the unaffected active snapshot jobs.

Update: the authorized Census KDD queue launched at **6:16 p.m. Chicago**,
with seed 42 followed by seed 43, using the corrected frozen source and
unchanged production budgets. At 6:17 p.m., seed 42 had reached 100/163,000
updates with finite loss. See [the rerun handoff](TAB_DDPM_CENSUS_KDD_RERUN_2026_10_11.md)
for the persistent session, logs, provenance, and queue order.
