# TabDDPM patch register

Documented October 11, 2026. All dates and times below are Chicago time.
This is the central index for the TabDDPM integration changes and numerical
repairs. Add future entries here with the affected file, cause, change,
validation evidence, and deployment status.

## October 11: categorical posterior normalization

**File:** `src/synthesize_data/tabddpm_core/gaussian_multinomial_diffsuion.py`,
`GaussianMultinomialDiffusion.q_posterior`.

**Change:** one computational line now uses the existing
`log_sum_exp_by_classes(unnormed_logprobs, self.slices_for_classes)` instead
of `sliced_logsumexp(unnormed_logprobs, self.offsets)`.

**Rationale:** the old implementation obtained a categorical variable's
normalizing mass by subtracting two cumulative sums in float32. A small
increment after a much larger accumulated mass could round away, producing
an incorrect normalizer or `log(0)`. Direct logsumexp within each variable
computes the same mathematical normalization without that subtraction.
The categorical posterior, loss, and gradients then remain finite in the
reproduced failure cases. The learning rate, feature schema, diffusion
schedule, objective, training budget, and sample count remain the agreed
research settings. The patch uses an existing helper and adds no dependency,
clipping, retries, or exception handling.

**Observed failures and evidence:**

- Covertype seed 42 failed at update 70,869 of 514,000. Valid-input CPU/GPU
  probes reproduced posterior precision errors and nonfinite categorical
  losses. The exact historical model state was not saved, so the diagnosis
  establishes a mechanism matching the failure rather than an exact replay
  of that historical update.
- Census KDD seed 42 failed at update 13 of 163,000. A replay reproduced the
  same update and traced the nonfinite model posterior to `PEFNTVTY`, with
  42 categorical states, at diffusion timestep 0. Inputs, weights, and all
  earlier gradients were finite. With identical weights, batch, and random
  draws, the correction changed the NaN loss to **2.887054**, with finite
  gradients.
- For each dataset, remote checks covered seeds 42/43, full-table/X-only
  sources, 100 training updates per case, parameter gradients, final weights,
  and corrupted inputs across all 1,000 diffusion timesteps. These are
  focused validation runs, not completed 500-epoch production fits.

**Deployment:** the corrected Covertype snapshot was launched at 11:51 a.m.
in `tabddpm_covertype_500ep_fix_20261011_v1`. The original Census queue still
used the frozen original source and therefore retained the defect. At
5:20:05 p.m., after the original workers stopped, the fix was deployed to
the main remote source with backups and transferred-content verification.
Active Covertype snapshots were preserved. Their corrected `src/` hash and
the newly corrected main `src/` hash are both
`ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc`.
These source hashes cover `src/` Python files; runner hashes are recorded
separately in launch/deployment manifests.

Detailed evidence: [Covertype diagnosis](TAB_DDPM_COVERTYPE_DIAGNOSIS_2026_10_11.md),
[Covertype rerun](TAB_DDPM_COVERTYPE_RERUN_2026_10_11.md), and
[Census KDD diagnosis and deployment](TAB_DDPM_CENSUS_KDD_DIAGNOSIS_2026_10_11.md).
The latter includes the original-file backup location and evidence paths.
Census KDD had not been relaunched at that deployment snapshot.

At **6:16 p.m. October 11**, the authorized Census KDD seed-42/43 recovery
queue was launched from the existing corrected snapshot in namespace
`tabddpm_census_kdd_500ep_fix_20261011_v1`. No further production source
patch was required. See [the Census rerun handoff](TAB_DDPM_CENSUS_KDD_RERUN_2026_10_11.md)
for exact queues, settings, reuse provenance, and initial progress.

## October 11: remote snapshot runner guard

**File:** `scripts/run_corrected_matrix.py`, `run_matrix`.

**Change:** the remote-only guard permits source snapshots nested inside
`/home/thuy/Research/minh_data_synth/TabularDA`, instead of requiring the
repository root itself.

**Rationale:** the repaired Covertype queue needed to use the maintained
runner from a frozen source copy while the original Census/MNIST workers
continued. Keeping separate source copies preserves each run's code
provenance and protects future subprocesses in active queues. The local
laptop remains outside the permitted path.

**Validation/status:** the snapshot's public dispatch produced all 54
expected Covertype downstream configurations with unchanged budgets and
paths, and its CUDA smoke check passed. This runner change is in the main
local file and the corrected remote snapshot. It was not included in the
three-file main remote numerical-fix deployment; the main remote runner
continues to accept its normal repository root. See the
[rerun handoff](TAB_DDPM_COVERTYPE_RERUN_2026_10_11.md).

## October 11: focused regression coverage

**File:** existing `audit/tabddpm_smoke.py`, `check_categorical_posterior`.

**Change:** check both Covertype's 44 binary variables plus seven-class
target and Census KDD's categorical cardinalities, including the 42-state
country-of-birth variable. Assert finite posteriors, per-variable
probabilities summing to one, finite loss, and finite gradients under valid
rare corruptions at timestep 0.

**Rationale:** finite raw input and a successful initial training update
did not prevent these failures. The regression exercises the numerical
conditions that exposed them. It extends the existing check instead of
adding a separate test suite.

**Validation/status:** both schemas passed remotely on CPU and CUDA, both
before deployment and against the main deployed code. The expanded check
was deployed alongside the core and its vendoring README. See the
[Census evidence](TAB_DDPM_CENSUS_KDD_DIAGNOSIS_2026_10_11.md).

## October 10: integration changes retained in the main files

| File / change | Rationale |
| --- | --- |
| `tabddpm.py`: explicit epoch budgets and recorded row/batch/update counts | Implement the agreed 500 complete epochs; avoid interpreting 500 as optimizer updates. Include the final short batch. |
| `main.py`: mutually exclusive epoch/step arguments; `run_corrected_matrix.py`: generator selection and `generator_epochs` | Expose the experiment through the existing entry points while preserving CTGAN/TVAE defaults. |
| `tabddpm.py`: prepared-split verification and existing-artifact rejection | Fit on the verified real training partition and preserve partial/completed research artifacts. |
| `tabddpm.py`, `main.py`, runner: corrected News log-target routing and one inverse transform on export | Keep generation, relabeling, and raw-share evaluation consistent with the corrected News protocol. |
| `tabddpm.py`: saved feature/generator hashes and shared feature tables | Establish provenance and give labelers identical generated features within each source. |
| `synthesizer.py`: exempt actual TabDDPM checkpoints from the SDV no-Faker check | TabDDPM uses its own encoder/normalizer; the SDV transformer check still applies to SDV models. |
| Vendored core: unconditional sampling, device-correct absent loss tensors, nonfinite-sample failure | Support the joint/X-only construction and pure categorical/numerical tables; surface invalid generated values. |
| Runner: RF/XGB/DNN evaluation order | Make the agreed primary pilot results available before supplementary NB/PCA-GMM evaluations. |

The [original launch handoff](TAB_DDPM_LAUNCH_2026_10_10.md) records the
integration checks, production settings, source hashes, backups, and
preserved corrections from earlier runs. The vendored core's
[README](../src/synthesize_data/tabddpm_core/README.md) lists its local changes.
