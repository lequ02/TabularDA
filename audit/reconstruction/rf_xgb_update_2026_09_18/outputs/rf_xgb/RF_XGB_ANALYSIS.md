# Joint CTGAN/TVAE versus RF/XGBoost labeling

Adult aliases are counted once. Only ctgan, tvae, rf and xgb target methods are included: 40 scores, comprising two joint baselines and eight decoupled configurations per dataset. RF/XGB are target predictors; synthetic features still come from CTGAN or TVAE. The family comparison averages both feature generators and both target-inclusion settings equally within each dataset, then weights the four datasets equally.

## Is the low TVAE point consistent with the paper?

It is Covertype, not a separate failed seed. Our TVAE macro-F1 is 0.2972 at its test maximum and 0.2672 at the final epoch. The paper reports 0.433 for TVAE and 0.324 for CTGAN on Covertype. Weak performance on this task is qualitatively consistent, but our TVAE score is substantially lower and the CTGAN/TVAE ranking reverses. The paper averages multiple downstream classifiers; our historical score uses a DNN and a maximum over test evaluations. These are not controlled replications of the same protocol. The point should be investigated, not removed just for being low. [Paper: Table 6 and evaluation protocol](https://arxiv.org/pdf/1907.00503).

```text
 method paper_macro_f1 local_macro_max local_macro_end
  CTGAN          0.324         0.48352         0.47484
   TVAE          0.433         0.29723         0.26725
```

## Family comparison and sensitivity

```text
                        analysis datasets joint_mean rf_xgb_mean    gain
              Maximum test score        4    0.63608     0.80129 0.16521
               Final-epoch score        4    0.61774     0.78801 0.17027
     Maximum, Covertype excluded        3    0.71798     0.86557 0.14758
 Final epoch, Covertype excluded        3    0.69997     0.85077 0.15080
 relative_gain   lower_95 upper_95  p_value
       0.25972  0.0044996  0.32591 0.046722
       0.27564 -0.0202493  0.36079 0.065420
       0.20555 -0.1403679  0.43553 0.158227
       0.21543 -0.1936690  0.49527 0.200311
```

Tests are two-sided paired t tests across dataset differences. Primary test-maximum comparison is unadjusted. End scores and exclusions are exploratory sensitivities. Covertype exclusion is reported because the historical methods used inconsistent test versions. It cannot certify that the remaining data are leakage-free.

## Individual labeler means by dataset

```text
   dataset y_synth macro_max macro_end
     Adult   ctgan   0.77433   0.76653
 Covertype   ctgan   0.48352   0.47484
   MNIST12   ctgan   0.48160   0.47039
   MNIST28   ctgan   0.44536   0.39162
     Adult      rf   0.77522   0.74642
 Covertype      rf   0.59962   0.59334
   MNIST12      rf   0.90582   0.90440
   MNIST28      rf   0.90961   0.90573
     Adult    tvae   0.74615   0.72065
 Covertype    tvae   0.29723   0.26725
   MNIST12    tvae   0.92485   0.92169
   MNIST28    tvae   0.93563   0.92894
     Adult     xgb   0.77358   0.72524
 Covertype     xgb   0.61730   0.60614
   MNIST12     xgb   0.91076   0.90922
   MNIST28     xgb   0.91841   0.91360
```

## Same-feature-generator comparisons

```text
                          analysis datasets joint_mean rf_xgb_mean     gain
 CTGAN features versus joint CTGAN        4    0.54620     0.78999 0.243786
   TVAE features versus joint TVAE        4    0.72596     0.81259 0.086626
 relative_gain  lower_95 upper_95  p_value  p_holm
       0.44633 -0.082274  0.56985 0.097668 0.19534
       0.11933 -0.125402  0.29865 0.284401 0.28440
```

Each comparison averages RF and XGB and both target-inclusion settings with its own feature generator. Two secondary p values are Holm-adjusted. This prevents the pooled comparison from hiding a much stronger CTGAN baseline deficit.

The same comparisons without Covertype (Holm adjustment within this additional sensitivity family):

```text
                     analysis datasets joint_mean rf_xgb_mean     gain
 CTGAN features, no Covertype        3    0.56710     0.84217 0.275079
  TVAE features, no Covertype        3    0.86887     0.88896 0.020083
 relative_gain    lower_95 upper_95  p_value   p_holm
      0.485065 -3.1856e-01 0.868719 0.184357 0.184357
      0.023114  5.4938e-05 0.040112 0.049747 0.099494
```

## Leave-one-dataset-out sensitivity

```text
          analysis datasets joint_mean rf_xgb_mean    gain relative_gain
     Without Adult        3    0.59470     0.81025 0.21556       0.36246
 Without Covertype        3    0.71798     0.86557 0.14758       0.20555
   Without MNIST12        3    0.61370     0.76562 0.15192       0.24754
   Without MNIST28        3    0.61795     0.76372 0.14577       0.23589
 lower_95 upper_95  p_value
  0.19201  0.23910 0.000644
 -0.14037  0.43553 0.158227
 -0.14453  0.44836 0.158254
 -0.13783  0.42937 0.157525
```

## Mixed-model sensitivity

Original random-effects structure:
```text
 contrast            estimate         SE    df   lower.CL  upper.CL t.ratio
 rf_xgb_minus_joint 0.1652061 0.03683965 37.02 0.09056339 0.2398489   4.484
 p.value
  0.0001

Degrees-of-freedom method: kenward-roger 
Confidence level used: 0.95 
```

Labeler random effect retained:
```text
 contrast            estimate         SE    df   lower.CL  upper.CL t.ratio
 rf_xgb_minus_joint 0.1652061 0.07697665 45.67 0.01022987 0.3201824   2.146
 p.value
  0.0372

Degrees-of-freedom method: kenward-roger 
Confidence level used: 0.95 
```

These configurations are not independent experiment seeds. Small numbers of dataset/generator/labeler levels and any singular fits limit interpretation.

## Figures

See ../figures/19_rf_xgb_distribution.png, 20_rf_xgb_dataset_comparison.png and 21_rf_xgb_sensitivity.png; PDF versions are alongside them. All three are also included in ../legacy_artifacts_report.html.

## Interpretation limits

Selecting RF/XGB after seeing their strong scores makes this an exploratory restricted-family result. Known test-fitted preprocessing, test-maximum selection, uncertain split/checkpoint provenance, related MNIST tasks and differing Covertype test versions remain. A positive result here does not prove leakage-free superiority. The original all-methods analysis and abstract are retained unchanged.
