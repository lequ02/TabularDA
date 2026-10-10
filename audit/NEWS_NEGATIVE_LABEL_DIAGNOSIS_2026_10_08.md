# News negative synthetic labels: saved-model forensic diagnosis

Checked October 8, 2026, starting 5:13 p.m. Chicago time. Training, generation, labeler tuning, clipping, and report refresh remain paused at the user's request. This investigation performed read-only inference remotely, using existing saved predictors and tables, with two CPU threads. No experiment outputs or model parameters were changed.

## Finding

The negatives are reproduced by the saved regression labelers. No incorrect target inverse transformation, classification decoding, feature-column ordering, or CSV feature/target alignment was found in the checked paths. Positive observed targets do not constrain these fitted model outputs to remain positive: the DNN ends with an unconstrained affine layer, and the XGB regressor adds signed learned corrections in raw share units.

This conclusion covers 16 saved DNN/XGB tables: seeds 42 and 43, CTGAN and TVAE, full-table and features-only sources, each with 100,000 rows. It does not claim an audit of every other labeler or every historical namespace.

## DNN: exact calculation for a negative label

The saved seed-43 CTGAN full-table DNN uses real-training target mean 3399.20263671875 and population standard deviation 12450.6923828125. These exactly match the training targets after their documented float32 conversion. The saved feature means and scales also match the real training features exactly, in the saved feature order.

The forward target transform is `(shares - mean) / std`; its inverse is `mean + std * network_output`. This inverse restores the target's units, rather than adding an independently chosen prediction rule. Applying the forward and inverse transforms to all real training targets gives maximum absolute round-trip error 0.015625 shares from float32 arithmetic. It does not create negative true training labels.

The smallest transformed training target is -0.27269187569618225, corresponding to four shares. Zero shares corresponds to network output -0.27301314113351344. Thus a negative standardized output is ordinary for a target below the training mean; negative *shares* require output below the zero boundary.

For synthetic row index 81342 (zero-based), the final layer's hidden-unit contributions are:

| Quantity | Value |
|---|---:|
| Sum of positive output-weight contributions | 0.01572493277490139 |
| Sum of negative output-weight contributions | -0.3302644193172455 |
| Learned output bias | -0.08937634527683258 |
| Manual final-layer dot product plus bias | -0.40391584977624007 |
| Replayed single-row network output | -0.4039158225059509 |
| Inverse in float64 arithmetic | -1629.829017853539 shares |
| Inverse in float32 arithmetic | -1629.8291015625 shares |
| Saved CSV target | -1629.8291 shares |

For example, hidden unit 49 has activation 0.5864260196685791 and output weight -0.10193556547164917, contributing -0.05977766960859299 to the standardized output. Hidden ReLUs make activations nonnegative, but final-layer weights and bias can be negative. The final regression output has no sigmoid, softmax, ReLU, or other range constraint. The labeler contains neither BatchNorm nor LayerNorm.

This row's hidden activations are finite and modest (maximums 1.4312 and 0.6173 in the two hidden layers). The negative prediction is already implied by the learned final layer before inverse scaling. Float64 inverse arithmetic preserves it; this is not a rounding-induced sign error.

## XGB: independent confirmation with no inverse scaling

News XGB fits the unchanged real-training share counts using `XGBRegressor`, objective `reg:squarederror`. Its saved artifact is marked regression and has no label encoder or target scaler. The synthesis path writes `model.predict(synthetic_features)` directly to the target column.

For seed-43 CTGAN full-table synthetic row index 37265, the actual tree leaves sum as follows:

`3399.2024 base score + 6897.3555321477 positive corrections - 16088.730174690001 negative corrections = -5792.172242542301 shares`.

The saved model's raw margin and ordinary prediction are both -5792.1708984375; the CSV contains -5792.171. Small differences from the manual sum arise from rounded tree-dump values and floating-point accumulation. The model has 100 trees, and this row's cumulative prediction first crosses below zero after tree 23. Tree 10 alone contributes -1469.27014 shares, and tree 24 contributes -1436.02393 shares. Tree leaves represent signed residual corrections, rather than averages of positive original share counts.

TVAE full-table row 22111 independently gives `3399.2024 + 5481.4855821595 - 18564.62669023 = -9683.9387080705`, matching model prediction -9683.9384765625 and saved target -9683.938.

## Original training rows also expose the problem

All 28,543 real training targets are positive, ranging from four to 843,300 shares. All 3,172 real development targets are positive, ranging from 49 to 233,400 shares. Nevertheless, replaying the same seed-43 labelers on these original features yields:

| Labeler | Negative real-train predictions | Negative real-dev predictions | Example original row |
|---|---:|---:|---|
| DNN | 62 | 5 | Source ID 31037: true 5900, predicted -49433.6211 |
| XGB | 264 | 48 | Source ID 952: true 560, predicted -3878.5571 |

Consequently the negatives are not restricted to generated feature rows. Every negative synthetic row in all 16 checked tables lies within the observed real-training minimum/maximum for each individual feature. That does not establish valid joint feature combinations or prove an absence of distribution shift, but it rules out marginal range violations as a necessary explanation here.

The very large DNN error on source ID 31037 has an additional observable input issue: the prepared real training row contains `n_non_stop_words=1042`, `n_unique_tokens=701`, and `n_non_stop_unique_tokens=650`. After the correctly fitted feature scaling these inputs are approximately 169 standard deviations above their respective training means; seed-43 hidden-layer maxima become 45.19 and 14.59. These unusual real feature values help explain the magnitude for this specific row. They are not established as the cause of the other negative predictions, and this investigation does not establish where those source feature values originated.

## Why these checkpoints were accepted

The seed-43 DNN checkpoint was selected at epoch 3; training stopped after epoch 33. Its saved development R² 0.016591012477874756 and standardized development MSE 0.44205838441848755 both replay exactly. The existing quality gate checks aggregate development R² above zero and above the mean baseline. It does not check positivity of regression predictions. A slightly positive aggregate R² can therefore pass while individual predictions violate the physical target domain.

Affine target standardization does not impose positivity. For fixed mean and scale, standardized squared error equals squared error in shares divided by the squared scale. The inverse is internally consistent, and its role is to restore units. Neither squared-error loss nor the currently fitted output representations prevent the negative predictions demonstrated above.

## Complete negative-label counts

Each entry below counts negatives among 100,000 saved synthetic rows. CPU replay reproduces every negative count.

| Seed | Generator | Feature source | DNN negatives | XGB negatives |
|---|---|---|---:|---:|
| 42 | CTGAN | Full table | 224 | 2039 |
| 42 | CTGAN | Features only | 22 | 1012 |
| 42 | TVAE | Full table | 10 | 789 |
| 42 | TVAE | Features only | 16 | 913 |
| 43 | CTGAN | Full table | 307 | 1312 |
| 43 | CTGAN | Features only | 240 | 1306 |
| 43 | TVAE | Full table | 26 | 1160 |
| 43 | TVAE | Features only | 38 | 1001 |

## Verification and evidence

- Checked saved table and predictor hashes; verified real raw input hashes against each split manifest.
- Replayed all 1.6 million synthetic targets from saved predictors. Maximum absolute discrepancy across all tables is 0.012500000011641532 shares, consistent with float32 calculation and decimal CSV serialization. All negative counts match.
- Verified regression mode, absence of classification mappings, target exclusion from features, saved feature-column order, and XGB sanitized feature names.
- Verified exact DNN feature and target scaling against real training data, and manually decomposed negative DNN and XGB outputs.
- Confirmed prepared News raw and one-hot training/development tables are identical, including row order and labels. Source inspection shows X and y are extracted from the same CSV without row sorting; News has no categorical encoding step that rearranges rows.
- Replayed saved DNN development loss and R² for both seeds and both feature sources; all match exactly.
- Compared relevant local/remote source hashes. DNN, synthesis caller, constructor, one-hot helper, and dataset loader hashes match. Ensemble and News constructor bytes differ only in line endings: decoded source text is identical. No code was deployed.
- Remote environment: PyTorch 2.5.1, NumPy 1.26.4, pandas 2.2.3, XGBoost 2.1.4.

Detailed evidence: [saved-model replay](news_negative_label_replay_2026_10_08.json) and [alignment/checkpoint validation](news_negative_label_alignment_2026_10_08.json).

Relevant source: `src/synthesize_data/dnn_labeler.py` (target transform, output layer, inverse, quality gate), `src/synthesize_data/ensemble.py` (raw-target XGB fit/predict), `src/synthesize_data/synthesizer.py` (feature-order alignment), and `src/synthesize_data/create_synthetic_data/CreateSyntheticData.py` (same-table X/y extraction).

Primary references: [PyTorch Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) defines the affine output; [XGBoost model tutorial](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/model.html) describes additive tree predictions and signed leaf scores. The numerical findings above come from the saved repository artifacts, not from these general references.

## Implication for the paused rerun

The separate downstream BatchNorm amplification diagnosis remains valid; these labeler negatives arise through a different mechanism. Changing only the downstream loss cannot change existing negative synthetic labels. Plain log cannot accept them. Yeo–Johnson can represent negative labels, but that alone does not correct their invalid physical meaning. No clipping, deletion, relabeling, checkpoint replacement, or new experiment has been performed. Any domain-preserving modeling change would need to be specified consistently as an experimental design change rather than selectively tuned to the failing arms.
