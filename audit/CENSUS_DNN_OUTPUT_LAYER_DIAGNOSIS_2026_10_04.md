# Census KDD selected-checkpoint output-layer diagnosis

Inspected eleven saved seed-42 downstream checkpoints remotely on October 4, 2026, at 1:39 a.m. Chicago time. No fitting, retraining, threshold tuning, or changes to completed artifacts were performed. Checkpoint hashes and output-layer parameters are in [census_dnn_output_layer_diagnosis_2026_10_04.json](census_dnn_output_layer_diagnosis_2026_10_04.json).

This provides a more specific mechanism for the zero-F1 behavior than the October 3 training-history diagnosis. The Census downstream architecture applies ReLU to the last hidden layer, then dropout, a linear output layer, and sigmoid. Its last hidden representation is therefore nonnegative. In evaluation mode dropout is inactive.

Let `h >= 0` be that last hidden representation, `w` the 32 output weights, and `b` the output bias. The logit is `z = w·h + b`. If all output weights are nonpositive, `z <= b`, so every possible input satisfies `p <= sigmoid(b)`. A negative bias then makes the fixed 0.5 positive decision impossible, independent of the number of positive rows or test-set sampling.

| Selected seed-42 arm | Epoch | Positive output weights / 32 | Global probability upper bound |
|---|---:|---:|---:|
| CTGAN full + RF | 1 | 0 | 0.425845 |
| CTGAN full + XGB | 1 | 0 | 0.425701 |
| CTGAN X-only + RF | 1 | 0 | 0.424871 |
| CTGAN X-only + XGB | 1 | 0 | 0.424544 |
| CTGAN full + DNN | 1 | 0 | 0.427609 |
| CTGAN X-only + DNN | 1 | 1 | No bound from weight signs alone |
| TVAE generated targets | 1 | 0 | 0.425252 |
| TVAE full + DNN | 1 | 0 | 0.425344 |
| TVAE X-only + DNN | 1 | 0 | 0.425279 |
| Real-only | 14 | 1 | No bound from weight signs alone |
| CTGAN generated targets | 5 | 7 | No bound from weight signs alone |

Eight of the nine failing plotted synthetic arms are mathematically incapable of positive predictions under the existing threshold. This includes five of six CTGAN relabeled arms. The exception, CTGAN X-only + DNN, has one positive output weight but all saved real-test probabilities are still below 0.5 (maximum 0.468895). No global impossibility claim applies to that exception.

The bound need not be reached by an observed input; for example the TVAE generated-target maximum observed score is below its global bound. CTGAN full + RF's recorded maximum, 0.4258454, reaches its bound to numerical precision.

The histories show that training continued for 32 epochs and later learned positive predictions. Minimum unweighted real-development binary-cross-entropy loss selected the epoch-1 state instead. Thus the reported complete collapse is the behavior of an early selected checkpoint, while later histories show poor generalization in loss. These are related but distinct findings.

The likely optimization driver is majority-class dominance in the unweighted binary loss. For output weight `w_j`, the minibatch gradient is the mean of `(p-y) h_j`; many negatives initially favor pushing the bias and weights downward. The exact learned sign pattern is verified. The causal roles of imbalance, dropout, minibatch order, feature distribution, and checkpoint selection have not been separated through controlled training experiments. Positive counts alone do not guarantee an adequate output decision boundary.

No change to the Census architecture, training objective, thresholds, or selection rule was made in this diagnosis. A future repair should be evaluated as a common binary-classification protocol, preserving current scores and using training/development data for decisions.
