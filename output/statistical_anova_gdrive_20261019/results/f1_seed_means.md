# F1 averaged over seeds 42 and 43

Macro F1 (0–1), arithmetic mean of seeds 42 and 43, using the same validation-best checkpoints as the ANOVA. Hybrids use RF, XGB or DNN to predict targets. Full: generator fitted on (X,Y); X-only: generator fitted on X. Bold indicates the highest score and underline the second-highest distinct score per dataset and training mode, across all displayed synthetic methods. Ranking uses unrounded values; all exact ties share the same rank. Real-only scores are excluded from ranking. Emphasis indicates numerical rank, not statistical significance. News and California Housing are regression tasks, so F1 does not apply. Credit has no validation-best evaluation in this selection. Adult TabDDPM and MNIST12 TVAE retain the documented source/collision caveats.

## Synthetic-only training

| Method | Adult | Census KDD | Covertype | MNIST12 | MNIST28 |
|---|---:|---:|---:|---:|---:|
| CTGAN | 0.7651 | 0.7105 | 0.4648 | 0.4861 | 0.5506 |
| TVAE | 0.7526 | 0.7121 | 0.4727 | 0.9298 | 0.9421 |
| TabDDPM | 0.7849 | 0.7095 | 0.5233 | 0.7340 | 0.3853 |
| **Hybrids ↓** | **────** | **────** | **────** | **────** | **────** |
| CTGAN + RF (full) | 0.7838 | 0.7376 | 0.6181 | 0.9009 | 0.8750 |
| CTGAN + XGB (full) | 0.7863 | 0.7569 | 0.6468 | 0.9146 | 0.8862 |
| CTGAN + DNN (full) | 0.7939 | 0.7547 | 0.7280 | 0.9197 | 0.9200 |
| CTGAN + RF (X-only) | 0.7868 | 0.7466 | 0.6225 | 0.8886 | 0.8812 |
| CTGAN + XGB (X-only) | 0.7895 | 0.7545 | 0.6507 | 0.9048 | 0.8894 |
| CTGAN + DNN (X-only) | <u>0.7950</u> | 0.7574 | 0.7126 | 0.9098 | 0.9223 |
| TVAE + RF (full) | 0.7829 | 0.7290 | 0.5809 | 0.9307 | 0.9477 |
| TVAE + XGB (full) | 0.7897 | 0.7433 | 0.5876 | **0.9421** | <u>0.9601</u> |
| TVAE + DNN (full) | 0.7940 | 0.7494 | 0.6850 | 0.9397 | 0.9596 |
| TVAE + RF (X-only) | 0.7890 | 0.7221 | 0.5543 | 0.9283 | 0.9489 |
| TVAE + XGB (X-only) | 0.7918 | 0.7313 | 0.5531 | <u>0.9412</u> | **0.9606** |
| TVAE + DNN (X-only) | 0.7933 | 0.7479 | 0.6718 | 0.9376 | 0.9601 |
| TabDDPM + RF (full) | 0.7906 | 0.7401 | 0.6777 | 0.6041 | 0.5260 |
| TabDDPM + XGB (full) | **0.7956** | 0.7556 | 0.6810 | 0.4976 | 0.4548 |
| TabDDPM + DNN (full) | 0.7931 | 0.7553 | <u>0.7568</u> | 0.1446 | 0.0679 |
| TabDDPM + RF (X-only) | 0.7862 | 0.7492 | 0.6588 | 0.7068 | 0.2959 |
| TabDDPM + XGB (X-only) | 0.7910 | **0.7636** | 0.6892 | 0.6100 | 0.2505 |
| TabDDPM + DNN (X-only) | 0.7936 | <u>0.7586</u> | **0.7588** | 0.2449 | 0.0762 |

## Mixed training

| Method | Adult | Census KDD | Covertype | MNIST12 | MNIST28 |
|---|---:|---:|---:|---:|---:|
| CTGAN | 0.7772 | 0.6842 | 0.5155 | 0.6997 | 0.9279 |
| TVAE | 0.7574 | 0.7586 | 0.4941 | 0.9441 | 0.9637 |
| TabDDPM | 0.7807 | 0.6916 | 0.4908 | 0.8883 | 0.8825 |
| **Hybrids ↓** | **────** | **────** | **────** | **────** | **────** |
| CTGAN + RF (full) | 0.7860 | 0.7685 | 0.5663 | 0.9296 | 0.9569 |
| CTGAN + XGB (full) | 0.7931 | 0.7694 | 0.6246 | 0.9411 | 0.9663 |
| CTGAN + DNN (full) | 0.7922 | 0.7694 | <u>0.7672</u> | 0.9405 | 0.9633 |
| CTGAN + RF (X-only) | 0.7926 | 0.7647 | 0.5703 | 0.9284 | 0.9577 |
| CTGAN + XGB (X-only) | <u>0.7955</u> | 0.7698 | 0.6224 | 0.9397 | 0.9624 |
| CTGAN + DNN (X-only) | 0.7937 | 0.7710 | 0.7608 | 0.9406 | 0.9641 |
| TVAE + RF (full) | 0.7828 | 0.7624 | 0.5994 | 0.9392 | 0.9597 |
| TVAE + XGB (full) | 0.7900 | <u>0.7740</u> | 0.6143 | **0.9494** | <u>0.9688</u> |
| TVAE + DNN (full) | 0.7917 | 0.7709 | 0.7475 | 0.9476 | 0.9670 |
| TVAE + RF (X-only) | 0.7882 | 0.7656 | 0.5630 | 0.9395 | 0.9603 |
| TVAE + XGB (X-only) | 0.7914 | **0.7742** | 0.5666 | <u>0.9490</u> | **0.9688** |
| TVAE + DNN (X-only) | 0.7950 | 0.7730 | 0.7453 | 0.9473 | 0.9678 |
| TabDDPM + RF (full) | 0.7877 | 0.7613 | 0.6653 | 0.7696 | 0.9309 |
| TabDDPM + XGB (full) | **0.7971** | 0.7707 | 0.6708 | 0.9111 | 0.9300 |
| TabDDPM + DNN (full) | 0.7928 | 0.7651 | 0.7651 | 0.4836 | 0.4293 |
| TabDDPM + RF (X-only) | 0.7899 | 0.7622 | 0.6467 | 0.9149 | 0.9343 |
| TabDDPM + XGB (X-only) | 0.7930 | 0.7713 | 0.6813 | 0.9271 | 0.6350 |
| TabDDPM + DNN (X-only) | 0.7953 | 0.7697 | **0.7744** | 0.5956 | 0.4431 |
