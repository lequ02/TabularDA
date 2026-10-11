# TabDDPM training budget and Covertype numerical diagnosis

Verified October 11, 2026, 11:31 a.m. Chicago. Production namespace:
`tabddpm_500ep_20261010_v1`. These diagnostics do not constitute completed
production Covertype fits or downstream results.

## Original training budget

The [ICML 2023 paper](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf)
lists 5,000/10,000/20,000 optimizer iterations in Table 1 and 30,000-step runtime
examples in Appendix G. Training updates and diffusion timesteps are different.
Released [Adult](https://github.com/yandex-research/tab-ddpm/blob/main/exp/adult/ddpm_cb_best/config.toml)
and [California Housing](https://github.com/yandex-research/tab-ddpm/blob/main/exp/california/ddpm_cb_best/config.toml)
configurations both specify 30,000 updates and batch size 4,096. Thus there is no
single paper epoch count to reproduce.

Our fixed pilot uses batch size 500 and `500 * ceil(training_rows / 500)` updates:

| Dataset | Training rows | Updates per epoch | Planned updates per fit |
| --- | ---: | ---: | ---: |
| California Housing | 14,861 | 30 | 15,000 |
| News | 28,543 | 58 | 29,000 |
| Adult | 34,280 | 69 | 34,500 |
| MNIST12 | 54,094 | 109 | 54,500 |
| MNIST28 | 54,094 | 109 | 54,500 |
| Census KDD | 162,962 | 326 | 163,000 |
| Covertype | 513,911 | 1,028 | 514,000 |

The official [loader](https://github.com/yandex-research/tab-ddpm/blob/main/lib/data.py)
includes the final short batch. On the paper's splits, 30,000 updates correspond
to approximately 4,286 Adult passes and 7,500 Housing passes. Our 500 passes
therefore provide less exposure on these small datasets, despite Adult having
more optimizer updates. Epochs or update counts alone cannot establish underfit.
Keep the active pilot's settings; any longer-budget study belongs in a distinct
configuration, with development utility used for selection and test held out.

Our fixed two-layer, unconditional classification construction also differs
from the paper's class-conditional, tuned architecture. A larger epoch count
alone would not make this a reproduction of the published baseline.

## Failure and evidence

Covertype seed 42 full-table training stopped at zero-based step 70,869 of
514,000 planned updates. Its last printed loss at step 70,800 was 0.417535.
The exception was `ValueError: Nonfinite Tab-DDPM loss at step 70869` in
`output/tabddpm_500ep_20261010_v1/logs/covertype/seed_42/covertype_seed42_tabddpm_generate.log`.
No full-table generator or intermediate model/optimizer checkpoint was saved.
The queued seed 43 fit has not started.

Remote checks used Python 3.10.21, PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3,
and scikit-learn 1.5.2 in `/home/thuy/miniconda3/envs/env`. All 513,911 real
training rows were finite and without missing values. The ten transformed
numerical columns were finite, with range approximately [-5.19934, 5.19934].
The full-table categorical schema has 44 binary indicators followed by a
seven-class target, with valid encoded values. No development/test data were
used in these diagnostics.

The posterior in `q_posterior` used `sliced_logsumexp`, which computes a group
sum by subtracting two cumulative sums. When a t=0 corruption changes the
target, its unnormalized probability can be approximately 5.9e-6 after the
preceding binary groups contribute roughly 44. Float32 rounding loses the
small increment, giving an incorrect normalizer or `log(0) = -inf`.
Posterior infinities propagate into the categorical loss. In particular,
`compute_Lt` multiplies the unused KL term by zero at t=0; zero times a NaN
remains NaN. This cumulative-subtraction implementation is inherited from
the [vendored upstream commit](https://github.com/yandex-research/tab-ddpm/blob/b476257dd460b778ba09eb97f7a51d6490fa17f8/tab_ddpm/utils.py).

Deterministic valid-input tests established both precision error and a
nonfinite loss with finite input values/model logits:

- A single actual training row with only its target corrupted gave a float32
  normalizer of -11.465069 instead of the direct/float64 value -12.040940.
- With finite logits +/-8, a CPU case produced a NaN categorical loss.
- A CUDA batch of 45 valid corruption cases produced nonfinite true
  posteriors/losses in five cases, including the target-only corruption.
- Direct per-variable logsumexp made all these posteriors, losses, and
  checked gradients finite; categorical probabilities summed to one.

This proves a numerical defect capable of the observed failure. The exact
historical batch and model state were not saved, so it does not prove that
the same arithmetic path was the sole cause at step 70,869.

## Minimal fix and validation

One computational line in `src/synthesize_data/tabddpm_core/gaussian_multinomial_diffsuion.py`
now calls the existing `log_sum_exp_by_classes(unnormed_logprobs, self.slices_for_classes)`
instead of cumulative subtraction. This computes the same mathematical
normalization directly within each categorical variable. No learning rate,
epoch count, sample count, feature schema, loss objective, clipping, retries,
or exception handling changed. The vendoring README records the numerical
change. The existing `audit/tabddpm_smoke.py` includes one focused regression
for this categorical schema, normalization, loss, and gradients.

The modified code was validated remotely in an isolated cache:

`/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_diagnosis_20261011/candidate`

The existing complete smoke check passed on CPU and CUDA. Real-data checks
ran 100 updates each for full-table/X-only sources at seeds 42 and 43 with
batch 500 and 1,000 diffusion timesteps. Losses, all checked parameter
gradients, and final weights were finite. Separate corrupted-input checks
covered every diffusion timestep; maximum category probability-sum error
was 5.96e-7. These are validation runs, not production results.

The active remote main files were not replaced. At 11:31 a.m. Chicago,
workers 3/4 remained live, worker 1 was finished, and worker 2 remained failed.
The live source hash still matched the launch:
`f5daf5ff979a57d5e7c538e37e2e0a26d7fe92f44312309c700f59e61f87351b`.
The candidate core hash was
`4b69d9cfcdc378bd11f5532c9176abda40c893a102e886c56f39b4effa4a92ac`.

Local evidence is in ignored `.cache/tabddpm_covertype_diagnosis_20261011/`:
`preflight.json`, `posterior_reproduction.json`, `scan_posterior.json`,
`candidate_manifest.json`, `candidate_smoke.log`, `candidate_stress.log`, and
`final_remote_check.json`. Remote `stress_result.json` records the real-data
checks. Source/test changes are in the main local files; diagnostic tooling
does not add production entry points or dependencies.

## Restart requirements

Covertype has not been restarted. Before restarting, check live queues and
resources and isolate the corrected source from workers still using the
frozen launch code. Preserve failed logs and the completed real-only
baseline, validating source/split provenance before reuse. Run both seeds'
full-table and X-only fits at the agreed 500 epochs and batch 500; each fit
has 514,000 updates. The failed generator must start at update zero because
there is no resumable checkpoint. Record a distinct repair namespace and
the corrected source hash rather than silently treating its artifacts as
having the original launch's source provenance. Full-budget stability
remains to be established by those production fits.

Update: both seeds were launched in parallel at 11:51 a.m. Chicago using
the corrected source snapshot. See [the rerun handoff](TAB_DDPM_COVERTYPE_RERUN_2026_10_11.md)
for exact queues, provenance, validation, and initial progress.
