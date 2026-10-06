# Tab-DDPM extension

Implemented on `codex/tabddpm-supervised` in the isolated worktree
`D:/SummerResearch/.worktrees/tabddpm-supervised`.

The official pinned diffusion core is reused with one unconditional DataFrame
adapter. The adapter receives no separate target argument: fitting on X alone
cannot reinsert y. Full-table fitting includes y explicitly, as categorical for
classification and numerical for regression. The existing real-data labelers,
quality reports, predictor artifacts, and downstream models are reused.

All labelers receive the same saved feature sample for each source. As in the
existing pipeline, predictors are fitted independently for each labeling call;
this extension adds no predictor cache. DNN development gates remain mandatory.
No class rebalancing, retries, fallback models, or test-based selection is added.

The current California Housing source changes were copied into this worktree to
retain the current nine-dataset scope. Original working-tree changes and remote
running experiments are untouched.

## Remote commands

Run only on the research server, after checking live queues and resources.
Use a fresh namespace; it contains the same shared split protocol, with artifacts
separate from active corrected runs. Existing prepared splits can be copied into
that namespace only after verifying their manifests and file hashes. If a
manifest is absent, the existing dataset preparer creates the partitions.

```sh
export CORRECTED_RUN_NAMESPACE=corrected_v2_tabddpm
/home/thuy/miniconda3/envs/env/bin/python scripts/run_corrected_matrix.py \
  --generators tabddpm --dataset adult --seed 42

# Classifier-only resume uses the existing completion-record check.
/home/thuy/miniconda3/envs/env/bin/python scripts/run_corrected_matrix.py \
  --generators tabddpm --dataset adult --seed 42 --stage classifiers --resume

# Entire nine-dataset, two-seed extension, when scheduled:
/home/thuy/miniconda3/envs/env/bin/python scripts/run_corrected_matrix.py --generators tabddpm

/home/thuy/miniconda3/envs/env/bin/python scripts/build_corrected_results.py \
  --generators tabddpm --runs output/corrected_v2_tabddpm \
  --out output/corrected_v2_tabddpm/results
```

CTGAN/TVAE remain the default generators. `--generators ctgan tvae tabddpm`
selects the combined matrix: 1,326 planned downstream records, counting real-only
once. Tab-DDPM alone expects 454 records, including 18 real-only baselines.
The builder still requires every selected planned record to be complete.
Generation refuses to replace an existing Tab-DDPM generator; classifier-only
resume does not refit it. Training checkpoints do not support mid-fit resume.

Tab-DDPM defaults: 20,000 optimizer steps, training/sample batch size 500,
1,000 diffusion timesteps, MLP widths 256/256, dropout 0, time embedding 128,
quantile numerical normalization, ordinal categorical codes with multinomial
diffusion, cosine schedule, AdamW at 0.001 with linear decay and weight decay
0.00001. Sample final weights with the ancestral sampler. Save the fitted
transforms, schema, CPU weights, parameters, and loss history beside provenance.
The upstream small-integer numerical rounding policy is retained, including
binary MNIST pixels. The generated regression target returns to original units.

`--tabddpm-steps` specifies a different declared optimizer budget. These steps
are not epochs and are not equivalent to CTGAN/TVAE's unchanged 500 epochs.
Production cost/quality is not yet validated; no production run is launched by
this implementation task. Tiny smoke settings are not production results.

Focused check (remote, small CPU fixture):

```sh
OMP_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python audit/tabddpm_smoke.py
# Optional identical small check on CUDA, after checking GPU availability:
OMP_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python audit/tabddpm_smoke.py cuda
```

The check covers absence of target dependence, seed propagation, mixed and
single-type schemas, classification target decoding, regression units,
non-divisible sample batches, saved-model reload, the existing RF relabeling
path, and method/count registration. It does not establish full-scale GPU
performance, downstream utility, or passage of production DNN quality gates.

## Verification completed

October 4, 2026, approximately 3:39 p.m. Chicago time:

- The focused smoke check passed on CPU and CUDA using the existing remote
  Python 3.10.21 / PyTorch 2.5.1 / scikit-learn 1.5.2 environment.
- A temporary tiny CPU integration check verified full/X-only orchestration,
  shared features within each labeler group, generated-target baseline artifacts,
  rejection of changed prepared-input hashes, and refusal to overwrite models.
  Labeler dispatch was observed in that integration check; actual RF labeling
  for both tasks was covered by the smoke check.
- The synthesis, matrix runner, and result builder command-line entry points
  accept Tab-DDPM. Local syntax and focused whitespace checks passed.
- Validation was isolated under the remote repository's
  `.cache/tabddpm_supervised_validation_20261004/`. Transfer hashes were checked.
  Logs were copied into this worktree's ignored `.cache/tabddpm-*-smoke.log`
  and `.cache/tabddpm-integration.log`.

No production runs, dependency installations, shared-source replacements, or
changes to existing experiment results were made. Full-scale performance and
production DNN quality gates remain untested.
