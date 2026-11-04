# MNIST12 TabDDPM relabeling diagnosis

Read-only remote diagnosis completed October 12, 2026, at approximately **12:37 p.m. Chicago**. Namespace: `tabddpm_500ep_20261010_v1`. Both seeds and both feature sources were inspected. No production source, saved samples, checkpoints, scores, queues, or running jobs were changed; no fitting or generation was performed.

The verified failure is that real-trained classifiers replace useful generated feature–target associations with severely imbalanced predictions on generated inputs. The generated pixel distribution differs substantially from real inputs, but the 73% generated-target result establishes that it retains predictive structure. DNN labeler standardization amplifies unexpected foreground in nearly blank pixels; this is a contributing sensitivity, not a sufficient explanation of the full relabeling failure. RF/XGB and a raw-input real-trained downstream DNN also fail to preserve generated associations.

## Follow-up: why useful generated targets can coexist with poor relabeling

The initial explanation emphasized pixel distortion too broadly. The user's objection is correct: training a classifier on synthetic `(X, generated Y)` and obtaining 73% real-test accuracy demonstrates useful transferable structure. This is different from applying a fixed real-trained classifier to synthetic X and replacing Y with its predictions.

A second bounded remote check inspected the first 5,000 joint samples and first 2,000 real development rows per seed. Results:

- All probed RF/XGB/DNN relabeled feature values match the generated-target baseline exactly by pixel name and row position. Saved predictor inference reproduces their stored targets. No column-order, row-order, or target-decoding fault was observed in these probes.
- Relabeling changes **78.2–80.0%** of joint generated targets for RF, **82.7–83.4%** for XGB, and **87.1–87.9%** for DNN. These are disagreement rates, not synthetic-label error rates.
- To examine digit signal separately from the overall background, class-average pixel images were centered by subtracting their respective table's overall pixel means. Under cosine similarity, **all ten generated classes match their corresponding real class most closely**, for both seeds. Matching-class similarities range from 0.429–0.688 for seed 42 and 0.341–0.693 for seed 43. This is a class-average descriptive check; it does not assign true labels to individual synthetic images.
- The saved **real-only downstream DNN**, using raw pixels and the same large architecture as the synthetic-trained evaluator, scores **95.75%/95.65%** on the real-development probes but agrees with generated targets on only **14.92%/17.36%** of joint synthetic rows. It assigns approximately 74–75% of those rows to digit 2. Thus the failure on synthetic inputs is not unique to the small standardized DNN labeler.
- The saved **generated-target synthetic-trained downstream DNN** scores **74.05%/75.10%** on the same real-development probes, consistent with its saved **73.19%/73.26%** real-test scores. It agrees with its synthetic training targets on **42.16%/42.34%** of the probed rows. Checkpoints were selected on real development loss, so high synthetic training accuracy is not guaranteed.

The supported interpretation is that useful digit-related signal survives alongside a changed pixel/background distribution. A classifier fitted on generated inputs with generated targets can learn associations that transfer to real digits. A fixed classifier fitted only on real inputs can instead respond incorrectly to the changed pixel combinations, and hard relabeling overwrites much of the useful original association. These two directions of transfer need not have the same performance. Which individual pixel interactions dominate RF/XGB failure remains unresolved; background masking alone did not establish a complete repair.

The second script/output are `.cache/mnist12_tabddpm_diagnosis_20261012/transfer_check.py` and `transfer_results.txt`. This follow-up used CPU inference only, with no fitting, generation, saved-table modification, or experimental restart.

## Completed downstream scores

Accuracy percentages from the completed run records and current comparison tables:

| Synthetic-only construction | Seed 42 | Seed 43 |
| --- | ---: | ---: |
| Real-only baseline | 95.7 | 95.6 |
| Joint generated targets | 73.2 | 73.3 |
| Joint features + RF | 55.0 | 56.0 |
| Joint features + XGB | 54.2 | 52.3 |
| Joint features + DNN | 14.8 | 24.5 |
| X-only features + RF | 76.7 | 68.8 |
| X-only features + XGB | 64.1 | 67.8 |
| X-only features + DNN | 25.8 | 29.7 |

For mix training, X-only RF reaches 91.5% for both seeds and X-only XGB reaches 92.0%/93.2%, while X-only DNN reaches 55.3%/73.8%. Joint DNN mix reaches 37.2%/60.4%. All 100,000 synthetic rows are included with 54,094 real training rows, so synthetic rows comprise approximately 64.9% of the mixture.

## Pixel-distribution evidence

MNIST12 preparation produces binary pixels. The diagnosis used all 54,094 real training rows for reference statistics and the first 5,000 saved synthetic rows per source and seed. These are bounded descriptive probes, not fresh experiment results or randomized representativeness guarantees.

The 51 pixels whose real-training nonzero frequency is less than 1% have an aggregate mean of **0.001528** (0.153% foreground). Generated values remain binary and within [0, 1], but their frequencies differ substantially:

| Table | Zero-pixel fraction | Mean foreground pixels per image | Mean at the 51 rare pixels |
| --- | ---: | ---: | ---: |
| Real training | 80.97% | 27.41 | 0.00153 |
| Joint seed 42 | 54.05% | 66.17 | 0.38550 |
| X-only seed 42 | 52.91% | 67.81 | 0.40208 |
| Joint seed 43 | 54.05% | 66.16 | 0.38265 |
| X-only seed 43 | 55.62% | 63.91 | 0.34730 |

Thus the issue is excessive foreground and altered pixel patterns, rather than values escaping their permitted range. Per-image maximum absolute standardized values have a median of approximately **164.46** and a 90th percentile of **232.58**, using real-training pixel means and population standard deviations.

## Saved labeler checks and diagnostic intervention

Saved checkpoints were loaded on CPU with one thread. Their predictions reproduced all probed saved synthetic targets exactly for RF/XGB/DNN, both seeds and sources. The first 2,000 real development rows give RF **95.0%**, XGB **95.6%**, and DNN **94.30%/94.35%** accuracy. The saved DNN reports over the complete development split give **94.20%/94.43%**. This supports a failure on generated inputs rather than an inability to classify real digits.

The DNN fits its pixel mean and scale on real training data (`src/synthesize_data/dnn_labeler.py`). A foreground value at a pixel with tiny nonzero training variance can therefore become an extreme standardized input. Its mean maximum class probability on the synthetic probes is 96.5–97.4%; this is confidence, not measured synthetic-label accuracy.

As an in-memory diagnostic only, the 51 rare pixels were set to zero in each synthetic probe. This changed **75.74–85.08%** of DNN predictions, while real-development accuracy under the same intervention stayed at **94.1–94.4%**. RF predictions changed only **5.60–8.38%**, and XGB predictions **16.10–23.28%**. This isolates a substantial DNN sensitivity to the anomalous background. It does not establish that blanking pixels would repair downstream utility; other pixel distortions remain, and no repaired table was saved or evaluated downstream.

Full saved target counts show the resulting label collapse. Seed-42 joint DNN assigns 61,259 rows to digit 3, 22,744 to digit 1, zero to digit 4, two to digit 8, and three to digit 9. Seed-43 joint DNN assigns 74,017 rows to digit 5 and zero to digit 8. Joint RF seed 42 assigns 36,777 rows to digit 2 and zero to digit 1. Joint XGB seed 42 assigns only 149 rows to digit 1.

On joint probes, original labeler predictions agree with the generator's targets only 12.1–12.9% for DNN, 20.0–21.8% for RF, and 16.6–17.3% for XGB. Generated targets are not ground truth; these agreement rates show how extensively relabeling replaces the learned joint associations, not which labels are correct.

## Generator mechanism: observed routing and remaining uncertainty

`src/synthesize_data/create_synthetic_data/mnist12.py` declares no categorical pixel columns. `src/synthesize_data/tabddpm.py` consequently applies a normal-output quantile transform and Gaussian diffusion to the binary pixels, followed by inverse transformation and nearest-value rounding. Only the joint target uses categorical diffusion. The saved fits use 500 epochs and 54,500 updates with the specified unconditional construction.

This numerical treatment of sparse binary pixels is a concrete candidate for the generator failure. The present diagnosis verifies the distorted generated distribution and its downstream consequences; it does **not** isolate quantile transformation, Gaussian diffusion, rounding, model capacity, or sampling as the sole originating cause. A change of representation would require a separate controlled experiment with a new namespace, rather than altering these completed records.

## Reproduction evidence

The bounded diagnostic script and complete output are retained in `.cache/mnist12_tabddpm_diagnosis_20261012/inspect.py` and `results.txt`. It reads existing run records, prepared inputs, samples, and predictor artifacts from the remote repository. The environment was `/home/thuy/miniconda3/envs/env/bin/python`, NumPy 1.26.4, pandas 2.2.3, PyTorch 2.5.1+cu124, and scikit-learn 1.5.2. Live processes and resources were checked first; three unrelated generator fits remained active. No GPU inference was used.

Source records: `output/tabddpm_500ep_20261010_v1/mnist12/acc/*.run.json`. Current score tables: `output/comparisons/dataset_pipeline_comparison.md` and `dataset_pipeline_comparison_mix.md`.
