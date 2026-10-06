# Tab-DDPM with supervised target synthesis: implementation plan

Prepared October 4, 2026, approximately 10:06 a.m. Chicago time. This is a source-reviewed implementation proposal, not an implemented or experimentally validated extension. No training, installation, deployment, or modification of existing experiment artifacts was performed.

## Decision and feasibility

Implement two independently fitted, unconditional Tab-DDPM generators: one for the joint table `(X, y)` and one for `X` alone. Train the existing supervised labelers on real training data and apply them to saved generated features. This preserves the three constructions used by our CTGAN/TVAE experiments.

The architecture supports this: `MLPDiffusion` can disable target conditioning, while the diffusion combines Gaussian numerical and multinomial categorical variables. The change is in input construction, training/sampling plumbing, decoding, and repository integration; it does not require a new diffusion objective. Use the MLP implementation for this extension. [Upstream denoiser](https://github.com/yandex-research/tab-ddpm/blob/main/tab_ddpm/modules.py), [diffusion implementation](https://github.com/yandex-research/tab-ddpm/blob/main/tab_ddpm/gaussian_multinomial_diffsuion.py).

**Deleting the target file or setting `is_y_cond=false` alone is insufficient.** The upstream loader inserts classification targets into categorical inputs and regression targets into numerical inputs whenever conditioning is disabled. Add an explicit `feature_source=full|xonly` option independent of conditioning; only `full` inserts the target. [Upstream dataset construction](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/utils_train.py).

Upstream's documented classification setup is target-conditioned. It samples a conditioning class and produces features given that class. Keep that distinction explicit: our primary generated-target baseline is an **unconditional joint Tab-DDPM**, rather than a reproduction of the usual conditional classification experiment. A conditional baseline could be a separately named later extension, but is outside this matched matrix. [Upstream configuration description](https://github.com/yandex-research/tab-ddpm/blob/main/CONFIG_DESCRIPTION.md).

Mathematical and architectural feasibility is established by source inspection. Compatibility with our installed packages, runtime at production scale, and downstream utility remain to be verified remotely.

## Current experimental scope

Use the current local `audit/CORRECTED_RUN_SPEC.md` and `scripts/run_corrected_matrix.py`, including their ongoing changes, as the design authority:

- Classification: Adult, genuine Census KDD, Credit, Covertype, Intrusion, MNIST12, MNIST28.
- Regression: News and **California Housing**, added October 4. Housing uses all eight numerical features and the original `MedHouseVal` units.
- Experiment seeds: 42 and 43, sharing the same reserved source split within each dataset. These measure model randomness, not independent holdouts.
- Do not introduce upstream datasets or splits. The legacy `census` alias is not an additional Census KDD dataset.
- The current comparison figure displays five datasets and selected methods. That display subset is not the full experimental dataset list. Do not choose the new study's datasets from favorable displayed scores.

The remote runner inspected today still lists eight datasets and handles only News as regression. California Housing is a local extension, not evidence of remote deployment or completed results. Verify and selectively deploy that extension before including Housing in remote runs; coordinate with its originating work rather than overwriting shared files.

## Matched experiment arms

For each dataset/seed, fit two models and save one 100,000-row sample from each:

| Construction | Generator fit data | Synthetic features | Synthetic target | Proposed run key |
| --- | --- | --- | --- | --- |
| Full-table baseline | Real training `(X, y)` | Saved joint sample's X | Same sample's generated y | `tabddpm` |
| Full-table relabeling | Same joint fit | Exactly the baseline sample's X | Existing real-trained labeler prediction | `tabddpm_compare_<labeler>` |
| Proposed X-only method | Real training X only | Saved independent X-only sample | Existing real-trained labeler prediction | `tabddpm_<labeler>` |

Classification labelers remain GaussianNB, CategoricalNB, PCA/GMM, RF, XGB, and DNN. Regression labelers remain PCA/GMM, RF, XGB, and DNN. Use the real-data settings and gates in the existing modules; do not import simulated-benchmark settings.

Fit each labeler once per dataset/seed and reuse that fitted predictor for both feature sources where provenance and existing fitting settings match. Reuse existing predictors only after checking fit/split hashes, preprocessing, settings, and seed. Every labeler within a source group receives identical saved features. Save feature hashes before attaching targets, and check that labeling preserves row order and every feature value. Classification uses the existing hard prediction rule; regression uses the existing predicted continuous target. Do not introduce probability sampling, class balancing, or tuned thresholds.

Evaluate every synthetic table twice: synthetic-only training, and all real training rows plus the same 100,000 synthetic rows. Reuse one verified real-only baseline per dataset/seed/protocol, rather than counting it again for each generator.

| Task | Tables added per dataset/seed | Downstream runs added per dataset/seed |
| --- | ---: | ---: |
| Classification | `1 + 2*6 = 13` | `13*2 = 26` |
| Regression | `1 + 2*4 = 9` | `9*2 = 18` |

Across seven classification and two regression datasets, two seeds add **36 generator fits, 218 labeled synthetic tables, and 436 downstream evaluations**. The existing CTGAN/TVAE plan is 890 downstream runs; the combined three-generator plan is **1,326**, counting real-only once. These are planned counts, not completion claims. A standalone Tab-DDPM collection with its own 18 real-only records would contain 454 downstream records.

## Preserve the evaluation protocol

Read verified prepared raw/one-hot splits and manifests rather than regenerating them during a new generator invocation. Record the source namespace when inputs come from an existing corrected or recovery namespace. Validate actual manifest source IDs, hashes, row counts, duplicate policy, and rare-class protection; do not assume exact split fractions. MNIST12/28 must share source-image IDs and retain their existing transformations and feature types. Keep the legacy MNIST split separate.

Fit generator transforms and labeler preprocessing on real train only. Numerical/categorical roles come from the dataset schema, not integer-looking values or one-hot column names. In particular, preserve the current numerical representation of MNIST pixels for the primary comparison. Decode generated data into the existing raw feature schema, then use the existing training-fitted encoding for labelers and downstream inputs. Do not use target-dependent categorical encodings in the X-only generator.

Keep our dataset-specific downstream neural models, batch size 128, learning rate 0.001, maximum 100 epochs, patience 30, and minimum real-development-loss checkpoint selection. Fit downstream scaling on real training features even for synthetic-only arms. Test is evaluated only after restoring the selected checkpoint. Upstream CatBoost/MLP evaluation scripts and `--change_val` must not replace this protocol.

Retain all existing metrics: binary/macro F1 for binary tasks, macro F1 for multiclass tasks, MNIST accuracy, and regression MSE/MAE/R². Include Credit precision, recall, and PR-AUC. State each plot's metric explicitly; current panel metrics and the result builder's primary metrics are not identical.

The October 4 Census diagnosis identifies collapse in early checkpoints selected by the existing loss rule. Preserve that rule and disclose the limitation. Any later common classifier repair needs a distinct protocol version and matched reruns across generators; do not repair only the Tab-DDPM arms or choose checkpoints using test scores.

## Implementation sequence

### 1. Pin and isolate upstream code

Pin an exact upstream commit and retain its license. Add only the necessary diffusion/MLP implementation and narrowly scoped adapter, with a recorded patch hash; avoid bringing in upstream baseline repositories and evaluators. Use explicit package imports so upstream `lib` and `utils` do not collide with our modules.

Prefer compatibility with `/home/thuy/miniconda3/envs/env`. It currently has Python 3.10.21, PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, and SciPy 1.15.3. `tomli` is installed; `libzero`, `rtdl`, and `category-encoders` were not in the package listing. The MLP is implemented in the upstream source, so determine actual imports before adding dependencies. Do not install the whole historical requirements file or downgrade the shared environment. [Upstream requirements](https://github.com/yandex-research/tab-ddpm/blob/main/requirements.txt).

Check concrete API incompatibilities, including the upstream quantile transform's floating-point `subsample=1e9` against our scikit-learn version. If needed, use the equivalent integer value as a documented compatibility patch. Keep transformation semantics intact. [Upstream preprocessing](https://github.com/yandex-research/tab-ddpm/blob/main/lib/data.py).

### 2. Add a narrow Tab-DDPM adapter

Proposed new module: `src/synthesize_data/tabddpm_adapter.py`. Its responsibilities are explicit full/X-only input construction, training-fitted transforms, fit/sample/save/load, schema decoding, and provenance.

- Disable `is_y_cond` in both primary modes. Training supplies an empty conditioning dictionary, and the features-only generator accepts no real targets as model inputs or transformation inputs.
- For full classification tables, treat y as an additional categorical variable. For full regression tables, treat y as an additional numerical variable. Track its location and decoding explicitly in a saved schema.
- For X-only mode, include no target dimension and no class embedding. Separate task metadata from diffusion categorical cardinalities so classification targets are not accidentally counted as input categories.
- Support numerical-only, categorical-only, and mixed inputs through the same original Gaussian/multinomial objective. Preserve categorical cardinalities and mappings from train.
- Implement unconditional sampling without reading class frequencies or synthesizing conditioning labels. Return X only in X-only mode; extract generated y only when the saved schema says it exists.
- Decode with saved transforms, not newly fitted transforms at sampling time. Restore category labels, column names/order, and original regression target units. Keep legitimate integer/domain handling explicit and documented; no new data-dependent clipping or filtering.
- Save the architecture, diffusion configuration, transforms, schema, and weights as a reloadable artifact bundle. Save final and EMA weights if retained; select one sampling policy in advance.

Upstream sampling reads real labels, assumes a leading numerical target for regression, and does not explicitly extract a jointly generated categorical classification target. Its high-level script therefore needs replacement/targeted adaptation for these modes, rather than direct reuse. [Upstream sampler](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/sample.py).

Pass seed 42/43 explicitly to training, transforms, and sampling. Upstream `pipeline.py` omits the training seed argument, leaving the training function's default at zero. Our runner must not inherit that behavior. [Pipeline](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/pipeline.py), [training](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py).

### 3. Attach existing supervised labelers

Add a narrowly scoped entry point accepting an already generated feature table; reuse predictor fitting/prediction code from `synthesizer.py` and the existing labeler modules without invoking an SDV fit or sample. Preserve real-data PCA/GMM numerical-column selection, train-fitted encoding, DNN development gates, and predictor artifacts. Avoid a general generator-framework refactor.

### 4. Register methods and records

Touch only the integration points needed:

| Existing file | Required extension |
| --- | --- |
| `src/modeling/constants.py` | Explicit `tabddpm` / `tabddpm_` mapping, synthetic paths, and a Tab-DDPM artifact path; preserve CTGAN/TVAE names and `.pkl` behavior. |
| `src/modeling/run_record.py` | Recognize generated-target Tab-DDPM records without expecting a predictor; verify the complete generator bundle and save its hashes/configuration. |
| `src/modeling/__main__.py` | Extend method choices if constrained. Preserve downstream architectures and selection behavior. |
| `scripts/run_corrected_matrix.py` | Add opt-in generator selection and Tab-DDPM dispatch; retain the current CTGAN/TVAE default and correct per-task labelers. |
| `scripts/build_corrected_results.py` | Add an explicit three-generator matrix selection/expected set; preserve current two-generator completion requirements. |
| `scripts/plot_recent_comparison.py` | Add separately identified Tab-DDPM comparisons only when verified records exist; keep metric definitions and missing results explicit. |

The current `methods_for()` treats every non-CTGAN generator as TVAE; `method_parts()` treats unrecognized prefixes as CTGAN. Both require explicit handling before using new keys. The current record writer also assumes SDV model paths and treats only CTGAN/TVAE as generated-target baselines.

Use a new extension namespace such as `corrected_v2_tabddpm`, with immutable links/copies of verified prepared inputs and explicit provenance. Keep all current output directories intact. A combined report can read compatible records from multiple namespaces; it must reject conflicting splits/protocols and count a reused real-only baseline once.

### 5. Freeze a generator-specific training budget

Match the research design and evaluation budget while documenting Tab-DDPM's own optimization settings. CTGAN/TVAE retain 500 epochs and batch size 500. Tab-DDPM training is specified in optimizer steps, which are different from its diffusion timesteps. An arbitrary 500-step run is not equivalent to 500 epochs.

Proposed starting production configuration for profiling: MLP hidden widths `[256, 256]`, time embedding 128, dropout 0, 1,000 diffusion timesteps, cosine schedule, numerical MSE, native multinomial objective, AdamW learning rate 0.001, weight decay 0.00001, training batch size 500, and **20,000 optimizer steps**. These are proposed new settings, not previously authorized or verified research defaults. Use the same configuration and training budget for full and X-only fits; only schema-dependent input/output dimensions change. Sample exactly 100,000 rows with ancestral sampling and a fixed documented batch size established by resource profiling. Prefer final weights initially to match the upstream pipeline; do not choose between final/EMA using test utility.

Before production, freeze the configuration based on train/development evidence and measured resource costs, with no test-score tuning. Record effective passes `steps*batch_size/n_train`, wall time, and peak RAM/GPU use. A strict 500-pass alternative would need roughly `500*ceil(n_train/500)` steps and could be extremely expensive for Intrusion. Treat equal-pass or equal-compute comparisons as separately specified analyses, not an unannounced change. This budget decision is the main remaining design choice.

## Focused remote validation and rollout

1. **Compatibility and schema smoke checks:** Run remotely in a separate smoke namespace. Cover mixed classification, numerical-only regression, and categorical-only input. Verify finite loss/sample values, target inclusion/exclusion, category decoding, regression-unit round trips, and exact reload behavior. Confirm no transform is fitted on dev/test.
2. **Concrete regression check for the proposed method:** With X and seed fixed, changing/permuting y must leave the X-only generator input, loss, and seeded sample unchanged. Full-mode schema must include y. Verify requested seeds reach training and repeated sampling uses saved transforms. This directly detects accidental target reinsertion/conditioning.
3. **Small end-to-end checks:** Adult and California Housing cover both tasks; MNIST12 checks binary numerical pixels and the shared image split. Use existing labeler gates and downstream loader, and exercise a baseline plus one relabeled arm in both training modes. Tiny checks are never production results. Include a non-divisible sample batch case and generated-target provenance check.
4. **Resource assessment:** Profile full and X-only fit/sampling before extrapolating to high-dimensional MNIST28 and full Intrusion. Today the host has one GTX 1080 Ti with 11 GB VRAM and 31 GiB RAM, with active Intrusion generation and Covertype/MNIST28 evaluation jobs and nearly full swap. Low instantaneous VRAM use is not permission to start another costly fit. Check live jobs, queues, sessions, and latest logs again immediately before execution; schedule after existing work or use the established queue.
5. **Production:** Deploy only reviewed necessary files, compare remote contents first, preserve replaced originals, and verify transferred hashes. Launch through persistent tmux/queue; record command, dataset, seed, configuration hash, log, and namespace. Resume only verified compatible checkpoints/samples/records. Save optimizer, scheduler, RNG, and progress state if training resume is supported; a final denoiser weight file alone is not a training-resume checkpoint.
6. **Completion and reporting:** Aggregate completed `.run.json` records through the extended existing builder. Preserve missing/failed/running distinctions, per-seed results, label counts/target ranges, source IDs, overlap checks, predictor/generator hashes, and development checkpoint selection. Compare joint generated targets against joint relabeling first to isolate target replacement, then compare joint relabeling against X-only relabeling to assess the separate generator fit.

Success means the X-only generator is demonstrably independent of y, all three constructions use the same verified real partitions and evaluation protocol, and the new artifacts are complete and auditable. It does not require the proposed method to win or every labeler gate to pass; failures must remain visible with their actual cause.
