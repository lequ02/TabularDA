# Compact F1 summary

Descriptive summary only: hybrid cells average the scores of RF, XGB and DNN as well as seeds 42/43. This is not a selected labeler or an ensemble. Macro F1 (0–1), arithmetic mean of seeds 42 and 43, using the same validation-best checkpoints as the ANOVA. Hybrids use RF, XGB or DNN to predict targets. Full: generator fitted on (X,Y); X-only: generator fitted on X. Bold indicates the highest score and underline the second-highest distinct score per dataset and training mode, across all displayed synthetic methods. Ranking uses unrounded values; all exact ties share the same rank. Real-only scores are excluded from ranking. Emphasis indicates numerical rank, not statistical significance. News and California Housing are regression tasks, so F1 does not apply. Credit has no validation-best evaluation in this selection. Adult TabDDPM and MNIST12 TVAE retain the documented source/collision caveats.

## Synthetic-only training

| Method | Adult | Census KDD | Covertype | MNIST12 | MNIST28 |
|---|---:|---:|---:|---:|---:|
| Real only | 0.7925 | 0.7563 | 0.5258 | 0.9559 | 0.9792 |
| CTGAN | 0.7651 | 0.7105 | 0.4648 | 0.4861 | 0.5506 |
| TVAE | 0.7526 | 0.7121 | 0.4727 | 0.9298 | 0.9421 |
| TabDDPM | 0.7849 | 0.7095 | 0.5233 | 0.7340 | 0.3853 |
| **Hybrids** | **────** | **────** | **────** | **────** | **────** |
| CTGAN: full (labeler mean) | 0.7880 | 0.7497 | 0.6643 | 0.9118 | 0.8937 |
| TVAE: full (labeler mean) | 0.7889 | 0.7405 | 0.6178 | **0.9375** | <u>0.9558</u> |
| TabDDPM: full (labeler mean) | **0.7931** | 0.7503 | **0.7052** | 0.4154 | 0.3496 |
| CTGAN: X-only (labeler mean) | 0.7904 | <u>0.7528</u> | 0.6619 | 0.9011 | 0.8976 |
| TVAE: X-only (labeler mean) | <u>0.7913</u> | 0.7338 | 0.5930 | <u>0.9357</u> | **0.9565** |
| TabDDPM: X-only (labeler mean) | 0.7902 | **0.7571** | <u>0.7023</u> | 0.5206 | 0.2075 |

## Mixed training

| Method | Adult | Census KDD | Covertype | MNIST12 | MNIST28 |
|---|---:|---:|---:|---:|---:|
| Real only | 0.7925 | 0.7563 | 0.5258 | 0.9559 | 0.9792 |
| CTGAN | 0.7772 | 0.6842 | 0.5155 | 0.6997 | 0.9279 |
| TVAE | 0.7574 | 0.7586 | 0.4941 | 0.9441 | 0.9637 |
| TabDDPM | 0.7807 | 0.6916 | 0.4908 | 0.8883 | 0.8825 |
| **Hybrids** | **────** | **────** | **────** | **────** | **────** |
| CTGAN: full (labeler mean) | 0.7904 | <u>0.7691</u> | 0.6527 | 0.9371 | 0.9622 |
| TVAE: full (labeler mean) | 0.7882 | 0.7691 | 0.6537 | **0.9454** | <u>0.9651</u> |
| TabDDPM: full (labeler mean) | 0.7926 | 0.7657 | <u>0.7004</u> | 0.7214 | 0.7634 |
| CTGAN: X-only (labeler mean) | **0.7939** | 0.7685 | 0.6512 | 0.9362 | 0.9614 |
| TVAE: X-only (labeler mean) | 0.7915 | **0.7709** | 0.6250 | <u>0.9453</u> | **0.9657** |
| TabDDPM: X-only (labeler mean) | <u>0.7928</u> | 0.7677 | **0.7008** | 0.8125 | 0.6708 |
