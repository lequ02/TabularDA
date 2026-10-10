# Housing DNN failure diagnosis — October 8, 2026

Completed read-only investigation: 2026-10-08T17:41:54.327365-05:00 (Chicago). All computation and checkpoint replay ran on the research server with two CPU threads. Existing jobs were inspected and left running; no training, generation, checkpoint replacement, clipping, or experiment-score changes occurred.

The main cause is an SDV metadata error: both Housing coordinates were classified as private geographical fields and anonymized with globally distributed Faker values. The DNN labeler extrapolated on those invalid inputs and supplied severely corrupted training targets. The downstream network then amplified rare real inputs through BatchNorm in several of the worst failures. The saved affine inverse target transform is correct; classification logits, a missing sigmoid, and an inverse-transform bug do not explain these results.

## 1. Scope, evidence, and validation

Inspected all eight fitted Housing generator objects (CTGAN/TVAE × full/X-only × seeds 42/43), replayed all eight saved DNN labelers on all 100,000 synthetic rows and on real development/test data, recomputed metrics from all 58 distinct selected Housing downstream prediction files, and replayed all 16 DNN-relabeled downstream checkpoints plus both originals and eight generated-target synthetic/mix controls. Verified feature order, saved feature means/scales, target means/scales, source-ID alignment, selected checkpoints, and both seeds’ shared prepared tables.

Labeler CSV replay agrees within 0.000090568 dataset units; downstream checkpoint replay agrees within 0.000219043. Replays use CPU float32; training used GPU float32 and CSV serialization. Checkpoint replay reproduces the saved downstream predictions and their extreme values. After all probes, 140 remote record/checkpoint/table hashes still match the initial inspection.

Machine-readable evidence: [full replay](housing_dnn_replay_2026_10_08.json), [coordinate and activation mechanisms](housing_dnn_mechanism_2026_10_08.json), [controls and additional probes](housing_dnn_controls_2026_10_08.json), [checkpoint selection checks](housing_dnn_selection_checks_2026_10_08.json). All 16 DNN downstream checkpoints also reproduce the recorded selected development loss and satisfy the minimum-development-loss selection rule within its 1e−5 tolerance. The files retain source, checkpoint, predictor, generator, and synthetic-table SHA-256 values and diagnostic script hashes. Diagnostic helpers are under `.cache/remote_intrusion/diagnose_housing*.py`.

## 2. Confirmed generator bug: coordinates were anonymized, not learned

`src/synthesize_data/synthesizer.py:get_metadata` calls `Metadata.detect_from_dataframe(data)` and overrides categorical fields only. The Housing class lists Latitude/Longitude among numerical PCA/GMM columns, but that list does not override SDV’s detected metadata. Every saved generator metadata object contains `Latitude: {sdtype: latitude, pii: true}` and `Longitude: {sdtype: longitude, pii: true}`.

Every fitted transformer is `rdt.transformers.pii.anonymizer.AnonymizedFaker`, with provider `geo` and function `latitude` or `longitude`, empty function arguments, and no output sdtypes. Installed RDT `_transform` returns `None`, dropping the input column. `_reverse_transform` calls the Faker provider to generate new coordinates. Installed Faker samples longitude over approximately [−180, 180] and latitude over [−90, 90]. The actual saved CTGAN/TVAE model transformers list six learned features, plus MedHouseVal for full-table fits; neither coordinate appears. `enforce_min_max_values=True` on the synthesizer does not constrain these Faker outputs to California.

Real training latitude is [32.55, 41.95], longitude [−124.35, −114.31]. In every 100,000-row DNN table, 94,854 latitudes and 97,328 longitudes lie outside their respective real training ranges. Only **134 rows (0.134%)** lie within both coordinate ranges; 99.866% fail at least one. Observed latitude spans −89.9967 to 89.9980. Both coordinates lose their relationship to the other features and to generated targets.

This flaw applies to every synthetic Housing construction using these fits, including RF, XGB, and generated-target baselines. A decent utility score in another arm does not validate the generated geography or make this a fair comparison of labelers on faithfully generated California data.

## 3. Target normalization and the output head were checked directly

Real targets are all positive, from 0.14999 to 5.00001. The labeler fits its target scaling on real training targets: mean **2.0664193630**, population standard deviation **1.1499680281**. Saved parameters exactly match recomputation with the original float32 convention. Regression output is a single unrestricted linear value `z`; it is not a classification probability or a vector of classification logits. The saved inverse is exactly:

```text
y_hat = 1.1499680281 * z + 2.0664193630
```

Real standardized training targets range from approximately −1.6665 to 2.5510. Negative standardized outputs can therefore be normal and still inverse-transform to positive prices. A physical prediction becomes negative only when `z < −1.7969363605`. A linear regression head has no mathematical nonnegativity or maximum-value constraint; positive training targets alone do not impose one. Here, the reason for crossing that boundary is verified gross input extrapolation, rather than an unexplained sign error.

Example: seed-42 CTGAN full synthetic row 52,132 has latitude −89.08866 and longitude 177.54623. The saved labeler outputs `z ≈ −37.06344`; the correct affine inverse produces **−40.555355**, matching its saved CSV. Replacing only the two coordinates with the real training medians (34.26, −118.5), with the same weights, scaling, and other six features, gives **4.35035**. Its positive extreme row changes from **161.96037** to **5.34815** under the same coordinate-only probe.

The labeler itself performs well on real data: seed 42 development/test R² is 0.80066/0.81136; seed 43 is 0.79119/0.80660. Each seed’s four source arms use the same fitted real-data labeler behavior. The quality gate checks real development data, so it never checks whether generated features are on the labeler’s training domain.

| Seed | Generator/source | Saved synthetic target range | Negative targets / 100,000 | Outside real target range | Negatives after coordinate-only probe | Outside range after probe |
|---:|---|---:|---:|---:|---:|---:|
| 42 | ctgan_full | -40.555 to 161.960 | 56,884 | 94,687 | 0 | 1,444 |
| 42 | ctgan_xonly | -40.405 to 162.213 | 56,853 | 94,564 | 0 | 1,580 |
| 42 | tvae_full | -39.842 to 161.217 | 56,782 | 94,562 | 0 | 405 |
| 42 | tvae_xonly | -39.948 to 161.576 | 56,786 | 94,528 | 0 | 259 |
| 43 | ctgan_full | -60.618 to 108.736 | 75,811 | 98,057 | 0 | 903 |
| 43 | ctgan_xonly | -60.780 to 110.272 | 75,728 | 98,059 | 0 | 1,183 |
| 43 | tvae_full | -60.698 to 109.698 | 75,746 | 98,061 | 0 | 285 |
| 43 | tvae_xonly | -60.765 to 109.119 | 75,736 | 98,048 | 0 | 298 |

These are counterfactual probes of a frozen predictor, not corrected samples or new experiment results. They remove all negative labels across all eight tables and reduce out-of-range labels from 94.5–98.1% to 0.259–1.580%. Remaining overshoots show that merely fixing coordinates is not proof that all other synthetic joint structure is valid. Over 99.992% of centered synthetic target squares come from out-of-range labels: the target variance supplied to squared-error training is dominated by corrupted labels. This measures target variation, not a decomposition of the final fitted training loss.

The downstream loader standardizes **features only**, using real training statistics. Housing targets stay in their saved raw dataset units, and the downstream linear output is evaluated directly. There is no downstream inverse-target transform that could create the huge predictions. Saved source IDs and real target values align, and recomputed R²/MAE reproduce the report.

## 4. Downstream amplification: actual hidden-unit trace

The downstream `DNN_News` model used for Housing has four Linear → ReLU → BatchNorm → Dropout blocks and a linear scalar output. BatchNorm uses stored running means/variances in evaluation mode. Both regression training and validation set their modes correctly; dropout is disabled during checkpoint replay.

The worst seed-42 TVAE full+DNN mix checkpoint has test R² **−31.15662**. Source row **1914** has true target **5.00001**, but the downstream checkpoint predicts **−390.1318**. This single row accounts for **85.900%** of the total squared error (MSE 44.0301). Its 395.13 absolute error contributes only about 0.0957 to the 4,128-row MAE, which helps explain the less dramatic NMAE of 1.00955.

This is a real feature outlier: AveRooms 141.9091 (56.54 real-training standard deviations), AveBedrms 25.6364 (51.09 standard deviations). It is not a negative real target. The original seed-42 labeler predicts **5.10955** on this exact real row; the failure is in the subsequent network trained on the synthetic labels.

At first-layer BatchNorm unit 479, the real test row has a post-ReLU activation of **14.9631**. The saved running mean is **0.00003327**, running variance **0.0000019573**, learned gamma **1.20284**, beta **−0.196732**, and epsilon **0.00001**. The actual evaluation formula is:

```text
BN(a) = gamma * (a − running_mean) / sqrt(running_variance + eps) + beta
BN(14.9631) ≈ 5204.70
```

The saved effective gain is **347.85**. The last 4,096 synthetic rows activate this unit on just one row, at most 0.05676, while real training rows reach 27.0443 and development rows reach only 1.88182. Other implicated first-layer units have variances near zero and similarly large gains. The amplified features propagate through later layers to the saved negative scalar output. Direct CPU replay reproduces every step and the final prediction.

Mixing does not guarantee BatchNorm represents the real data: the corrected loader concatenates real rows first and synthetic rows last and uses `shuffle=False`. Every epoch ends with 782 synthetic batches. With momentum 0.1, the prior running-statistic contribution is attenuated by `(0.9)^782 ≈ 1.65e−36`. This exposes a concrete way for the synthetic distribution to dominate evaluation statistics even after the network has seen real rows.

In seed-43 CTGAN full+DNN mix, source 1914 predicts **−252.60751**, accounting for **91.499%** of squared error. One first-layer unit has variance 1.20e−29 and turns activation 14.34866 into 3431.04; another is completely inactive on development and on the last 4,096 synthetic rows, but activates on the worst test row.

## 5. All DNN downstream arms and limits of causal probes

| Seed | Generator/source | Training | Test R² | Development R² | Largest-error prediction | Share of squared error in that row |
|---:|---|---|---:|---:|---:|---:|
| 42 | ctgan_full | mix | -3.6928 | -0.5337 | 126.654 | 55.8% |
| 42 | ctgan_full | synthetic | -8.6490 | -2.1544 | 175.147 | 53.1% |
| 42 | ctgan_xonly | mix | -28.7112 | -0.7112 | -391.857 | 93.8% |
| 42 | ctgan_xonly | synthetic | -10.3008 | -1.3262 | 215.431 | 69.3% |
| 42 | tvae_full | mix | -31.1566 | -1.2930 | -390.132 | 85.9% |
| 42 | tvae_full | synthetic | -5.2685 | -2.8837 | 74.166 | 13.5% |
| 42 | tvae_xonly | mix | -14.0128 | -1.7642 | 265.135 | 79.7% |
| 42 | tvae_xonly | synthetic | -7.5704 | -3.1929 | 110.516 | 23.0% |
| 43 | ctgan_full | mix | -11.8317 | 0.1422 | -252.608 | 91.5% |
| 43 | ctgan_full | synthetic | -0.0248 | 0.4233 | 53.580 | 40.7% |
| 43 | ctgan_xonly | mix | -3.4750 | 0.2034 | -135.493 | 78.0% |
| 43 | ctgan_xonly | synthetic | 0.3466 | 0.3818 | -10.528 | 6.5% |
| 43 | tvae_full | mix | -0.8830 | 0.0938 | 65.969 | 34.9% |
| 43 | tvae_full | synthetic | 0.1504 | 0.3393 | 24.599 | 8.0% |
| 43 | tvae_xonly | mix | -0.6366 | 0.0609 | 58.256 | 30.7% |
| 43 | tvae_xonly | synthetic | 0.3967 | 0.3917 | 6.796 | 1.2% |

In-memory variance probes isolate the contribution of first-layer units originally below variance 1e−5. For seed-42 TVAE full+DNN mix, assigning those units variance 0.001 changes the worst prediction from −390.13 to −189.35 and test R² from −31.16 to −7.70, with every model weight unchanged. For seed-43 CTGAN full+DNN mix it changes −252.61 to −86.75 and R² from −11.83 to −1.58. These values demonstrate amplification; they are not a validated repair or reportable scores.

BatchNorm is **not the sole explanation for every arm**. Some floors worsen outcomes; seed-43 TVAE full/X-only mix has no first-layer variance below 1e−5 and still underperforms. Replacing the entire first-layer running statistics with statistics measured on real training features worsens all six probed mix checkpoints (for example −31.16 to −432.53 in seed-42 TVAE full+DNN mix), because the already-trained later network depends on its existing activation distribution. This negative diagnostic result is retained in the controls JSON. Neither a variance floor nor post-hoc real-data calibration repairs training on corrupted targets.

Seed-42 DNN downstream development R² is already negative in every arm; early stopping selects the best saved development loss available, not a checkpoint guaranteed to improve on real-only training. Removing the worst test row diagnostically leaves seed-42 TVAE full+DNN mix at R² −3.54. The fault is broader than a single bad test row. Test rows must not be dropped or used to tune thresholds.

RF synthetic targets stay within the real target range in the checked controls. XGB can also produce negative outputs here (roughly 1,900–3,100 rows in the four full-source controls), but its range is only about −1.20 to 8.54, much smaller than the DNN labeler’s −60.78 to 162.21. Both use different extrapolation behavior; positive real targets do not by themselves forbid negative XGB/DNN regression outputs. Their better downstream utility does not remove the shared coordinate flaw.

## 6. Interpretation and corrective work

The metadata error is directly established in every saved fitted generator, and coordinate-only frozen-model probes establish the source of the vast majority of extreme labels. The contribution of unstable BatchNorm to selected catastrophic downstream predictions is numerically demonstrated. The exact share of each cause in the full utility gap, and the performance of a corrected pipeline, require new controlled experiments; these read-only probes cannot supply that result.

A proper repair must explicitly declare Housing coordinates as numerical SDV fields before fitting and ensure fitted transformers actually learn both coordinates. Existing generator checkpoints cannot be repaired by changing their metadata after fitting, because the coordinates were absent from model training. Refit/resample/relabel and downstream evaluation belong in a new output namespace with the same seeds, splits, budgets, and selection criteria. Separately test downstream normalization/batch ordering on valid samples to distinguish its contribution. Clipping targets, dropping real outliers, or modifying normalization at report time would conceal the cause and change the design.

No repair or rerun was launched by this investigation. Both comparison reports keep the observed Housing scores and now explicitly flag the coordinate flaw; Credit and Intrusion panels/tables/source columns were removed as requested. News remains restricted to the three new-pipeline pilot runs and missing arms remain empty.
