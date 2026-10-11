# Separate real-data 328 analyses

Synthetic-only and mix now have independent fits on the same verified selected
experiments. Neither analysis contains a training-data factor or pooled effect.
Each has 140 macro-F1 scores: five classification tasks, two seeds, two generators
and seven observed conditions. Real-only runs stay outside the ANOVA.

`has_y` is 0 for generator fitting on X and 1 for fitting on (X,Y).
`condition` is generated Y or RF/XGB/DNN crossed with has_y. It is broader than
has_y; generated Y has no has_y=0 cell. The previous three-level grouping factor
was removed. The public analysis CSVs use has_y and condition consistently.

The fitted equations, independently within each training analysis, are:

```text
macro_F1 ~ dataset * generator * condition
macro_F1 ~ dataset + generator * condition
macro_F1 ~ dataset * condition + generator
macro_F1 ~ generator * condition + (1 | dataset)
```

The mixed model uses all 140 scores, rather than averaging labelers into three
categories before fitting. REML fits converged without warnings. Four planned
contrasts use equal generator weights; paired tests use five dataset means.
Holm families are separate within each analysis: four planned contrasts, twelve
generator/labeler/has_y contrasts, and nineteen GG-corrected repeated ANOVA terms.
Confidence intervals are unadjusted. Ordinary Type II ANOVA p-values are nominal.

Relabeling mean gains: synthetic-only +0.116319 macro F1 (+17.38% relative),
mix +0.067729 (+9.04%). Mixed-model Holm p-values are 1.17e-12 and 2.64e-9;
dataset-level Holm p-values are 0.139687 and 0.199729. The latter limit claims of
consistent improvement across tasks. has_y gains are small and nonsignificant.
News relabeling improves R2 but worsens NMAE/D2; Housing improves all three means.

Each report has six pages. Every section begins with a question, method and
conclusion. Actual fitted equations, factor definitions, levels, bold significant
rows and significance stars are included. The repeated-measures ANOVA is a
supporting appendix about variation between seeds on the same tasks. It does not
replace the five-dataset check. Residual diagnostics are supplied in the package.
The MNIST12 TVAE holdout-feature collision caveat remains in both reports.

Source data and experiments were not changed or rerun. A read-only check at
October 10, 2026, 1:35:35 p.m. Chicago found all 406 remote records and 116 D2
sidecars unchanged from the input snapshot. The template remains unchanged.
The two six-page PDFs and twelve-page combined PDF passed rendered visual review;
94 significance rows and 616 analysis score values passed focused validation.

Current tables are in the output's synthetic/ and mix/ directories. Old pooled
tables are historical and excluded from the refreshed package. frozen_inputs/
preserves the original CSV schemas for provenance, without introducing factors.
The preceding report is backed up under
`.cache/statistical_328_before_separate_20261010_132303/`.

Remote execution directory:
`/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_separate_20261010`.
Credit, Intrusion, NB/PCA-GMM and simulated experiments remain outside scope.
