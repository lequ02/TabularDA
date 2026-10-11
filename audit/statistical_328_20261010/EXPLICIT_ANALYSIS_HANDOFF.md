# Real-data 328 analysis with explicit factors

The classification specification below remains current. Regression was extended
with R2/D2 ANOVAs and signed task effects; current reports are eight pages each.
See [the regression extension and current validation](REGRESSION_ANALYSIS_HANDOFF.md).
Page counts and validation totals farther below describe the earlier revision.

This revision supersedes the pooled and composite-factor analyses. Synthetic-only
and mix remain independent. The combined real-data report contains those same
two analyses; it is not a third experiment or fit.

The primary equation in each report is exactly:

```r
macro_F1 ~ dataset + generator * has_y * labeler * seed
```

All factors are categorical, including seeds 42 and 43. The complete relabeled
grid contains five datasets, two generators, two has_y values, three labelers and
two seeds: 120 scores. has_y is 0 for fitting the generator on X, 1 for fitting
on (X,Y). labeler is RF, XGB or DNN. All 28 model columns are identifiable;
residual df is 92. Type II ANOVA uses sum-to-zero contrasts.

Twenty generated-target scores per training analysis stay outside this factorial
fit: they have no labeler and no has_y=0 combination. They are matched to
relabeled runs by dataset, seed and generator for explicit baseline comparisons.
The same comparator is not counted as independent observations when reused.
Ten real-only classification scores per training analysis are shared references.

Two predefined supporting equations are also reported:

```r
macro_F1 ~ dataset + seed + generator * has_y * labeler
macro_F1 ~ dataset * (has_y * labeler) + generator * has_y * labeler + seed
```

These have ranks 17 and 37. Nested-model tests compare adding seed interactions
(11 df) or dataset/has_y/labeler interactions (20 df). These are explicit checks,
not error-handling substitutes. The previous random-intercept mixed fit and
repeated-measures appendix have been removed from the current deliverables.

The requested ANOVA detects labeler differences (nominal p=1.0336e-5 for
synthetic-only, 7.8518e-6 for mix). Generator, has_y, seed and treatment interactions
are nonsignificant. Seed interactions jointly add no detected effect (p=.999786
and .999874). Dataset interactions are strong (p=1.8647e-9 and 2.9276e-32).
These ordinary-model tests remain exploratory because heldout samples are shared,
residuals are nonnormal and treatment effects depend on dataset.

Relabeling averages are unchanged: synthetic-only +.116319 macro F1 (+17.38%
relative), mix +.067729 (+9.04%). Paired inference uses five dataset means, with
Holm over the same four planned contrasts: adjusted p=.139687 and .199729.
Confidence intervals are unadjusted. Twelve per-generator/labeler/has_y baseline
comparisons have their own Holm family in each training analysis. No such baseline
comparison passes adjusted p<.05. has_y effects are small and nonsignificant.

Regression scores stay descriptive and retain their original R2/NMAE/D2 units,
saved normalization and sidecar provenance. Housing improves all three mean
metrics; News improves R2 but worsens NMAE and D2. MNIST12 TVAE holdout-feature
collision caveats and shared-test-partition limitations remain in both reports.

All 406 selected source records and 116 D2 sidecars were rechecked remotely at
October 10, 2026, 1:54:04 p.m. Chicago and still match the frozen inputs. No
experiment was run or modified. The numerical refit ran in the isolated remote
directory `.cache/statistical_328_explicit_20261010/` using one numerical thread.
Every ANOVA F was independently checked by fitting the restricted design: 74 F
checks passed. Input/export checks cover 616 analysis score values, exact matched
baseline identities, observed-cell coverage, model rank, unchanged contrast
effects and 68 significance rows. All twelve final PDF pages were rendered and
visually reviewed, including detailed ANOVA/model-check pages.

Each training report has six pages with a question, method/equation and conclusion
at the start of each section. HTML, Markdown, editable Rmd, raw frozen inputs,
statistical tables, scripts, provenance and validation are in the package.
Optional R code is equivalent in specification but has not been executed.

Previous deliverables are preserved under
`.cache/statistical_328_before_explicit_20261010_135353/`. Current output folders
contain only this revision, so obsolete pooled/composite-factor tables cannot be
mistaken for the current fits. Credit, Intrusion, NB/PCA-GMM and simulated data
remain outside scope. This does not claim completion of the broader experiment
matrix.
