# Regression ANOVAs and Holm explanation

This revision adds R2 and D2 ANOVAs to the explicit-factor 328 reports. It retains
the classification fits, planned contrasts, source scores, experimental settings
and separate synthetic-only/mix analyses. Each report is now eight pages; the
combined report contains the same sixteen pages.

## Why an 11.63-point gain does not pass Holm

Synthetic-only relabeling minus generated targets averages +0.116319 macro F1,
or 11.63 percentage points (17.38% relative). The five dataset gains are Adult
3.53, Census KDD 3.28, Covertype 11.76, MNIST12 21.58 and MNIST28 18.02 points.
Their standard error is 3.71 points; paired t=3.136 with four residual df gives
raw p=0.034977. The unadjusted 95% interval is [1.33,21.93] points.

Holm covers four planned contrasts within each training analysis. Its first
cutoff is .05/4=.0125. The smallest raw p, .034922, fails that cutoff, so none
passes. The relabeling comparison is second in the sorted family; its adjusted
p is max(4*.0349216487,3*.0349766348)=.139687. A positive unadjusted interval
does not contradict an adjusted p above .05. Gains are large but variable across
only five tasks; seeds and synthetic rows do not provide more independent tasks.
Mix relabeling has raw p=.056598 and Holm p=.199729.

## Regression equations and scope

For each training analysis, fit separate outcomes with categorical factors:

```r
R2 ~ dataset + generator * has_y * labeler * seed
D2 ~ dataset + generator * has_y * labeler * seed
```

The complete News/Housing relabeled grid has 48 scores per outcome, 25 columns,
rank 25 and 23 residual df. Generated targets have no labeler and no has_y=0
case; they stay outside these fits. There is no composite treatment factor.
Saved R2 and verified absolute-error D2 sidecars retain their original units and
denominators. NMAE remains descriptive.

Within each task, also fit:

```r
R2 ~ seed + generator * has_y * labeler
D2 ~ seed + generator * has_y * labeler
```

There are 24 scores, rank 13 and 11 residual df per within-task model. A full
four-factor fit within one task has 24 coefficients for 24 scores and zero
residual df. The explicit within-task equation keeps seed additive; its residual
contains omitted seed interactions. No fallback model is substituted.

All regression ANOVA/contrast p-values are nominal exploratory results, with
unadjusted model intervals. Only two fixed tasks and two seeds on shared test
partitions are available; these results do not establish effects on new tasks.

## Findings

| Training | Outcome | Nominal pooled treatment main effects |
| --- | --- | --- |
| Synthetic-only | R2 | None below .05 |
| Synthetic-only | D2 | generator p=.016431; labeler p=.047519 |
| Mix | R2 | generator p=.042267; labeler p=.049865 |
| Mix | D2 | labeler p=.000530 |

No pooled treatment interaction is below .05; pooled has_y and seed are
nonsignificant for both outcomes and training analyses. A significant term
indicates a difference among choices, not improvement over generated targets.
TVAE minus CTGAN reduces pooled synthetic D2 by .012773 and mix R2 by .007917.
Within Housing mix, has_y=1 minus 0 raises D2 by .006637 (nominal p=.013980).

| Training / task | Relabeled - generated R2 | Relabeled - generated D2 |
| --- | --- | --- |
| Synthetic-only / News | +.024584 | -.017843 |
| Synthetic-only / Housing | +.120828 | +.144799 |
| Mix / News | +.019244 | -.020846 |
| Mix / Housing | +.121201 | +.138166 |

These baseline gains are descriptive. Squared-error R2 can improve while
absolute-error D2 worsens, as happens on News. Housing improves both outcomes.
Signed pooled and task effects, confidence intervals, complete within-task
ANOVAs and baseline identities are supplied as CSVs.

## Validation and preservation

All 406 selected run records and 116 D2 sidecars still matched remote hashes at
October 10, 2026, 2:53:24 p.m. Chicago. Numerical refits ran remotely with one
thread in `.cache/statistical_328_regression_20261010`, using existing verified
inputs. No experiment was launched or changed.

Focused validation checks 808 exported score values, all observed factor cells,
12 regression model ranks, 60 signed regression effects, 202 independently
reproduced classification/regression F statistics and 164 significance rows or
cells. Both eight-page reports were rendered and visually reviewed. The combined
PDF is the same two analyses, concatenated. Source scores, classification results
and MNIST12 TVAE holdout-feature collision caveats are preserved.

Previous deliverables are backed up in
`.cache/statistical_328_before_regression_anova_20261010_145303/`. Publication
replaces only report-owned files and adds regression tables. The separate
`real_data_328_csv_replication.zip` from concurrent work is preserved unchanged;
its earlier contents do not include these newly added regression ANOVAs.
