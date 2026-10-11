There are TWO analyses: synthetic-only and mix, each with its own factorial fit.
The combined real_data_328_analysis PDF/HTML contains both; it is not a third analysis.
Primary equation, with categorical seed:
macro_F1 ~ dataset + generator * has_y * labeler * seed
This complete grid contains RF/XGB/DNN relabeled runs only: 120 scores per analysis.
Twenty generated-target scores are explicit matched baseline comparisons. Real-only
scores are references. No nonexistent generated-target/has_y=0 cell is estimated.
No composite treatment factor is used. Current tables are only in synthetic/ and mix/.
The former mixed model and repeated-measures appendix are removed from these reports.
Original 328 template structure: ANOVA, model checks, estimated means and effect contrasts.
Numerical fits ran remotely in Python; optional R code has not been executed.
Bold and stars use the p-value stated for each table. Confidence intervals are unadjusted.
Frozen input files preserve the historical schema verbatim as evidence, not model factors.
R2 and D2 have separate exploratory ANOVAs on 48 relabeled scores per training analysis:
outcome ~ dataset + generator * has_y * labeler * seed (rank 25, residual df 23).
Within each regression task: outcome ~ seed + generator * has_y * labeler
(rank 13, residual df 11). A within-task full four-factor fit would have zero residual df.
Regression ANOVA and model contrast p-values are nominal, not Holm-adjusted.
Generated-target improvements are descriptive comparisons outside those ANOVAs.
Only News and Housing are available; these tests do not establish effects across new tasks.
All 406 records and 116 D2 sidecars remain unchanged; see latest_source_verification.json.
Remote cache: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010
Regression cache: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_regression_20261010
Never run experiments locally. Prior deliverables are preserved in the audit backup.
