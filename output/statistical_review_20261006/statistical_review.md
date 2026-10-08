# CTGAN/TVAE versus CTGAN/TVAE + labeler

Corrected statistical report | October 6, 2026 | existing results only

**This report compares CTGAN and TVAE with their own RF, XGBoost and DNN relabeling arms.** The original generator keeps generated targets; a hybrid replaces them with targets predicted by a labeler trained on real training data. CTGAN and TVAE remain separate in the method tables.

There are two separate factorial ANOVA tables: **real data** and **simulated data**. Each includes **dataset, generator, labeler and generator-input inclusion**. A separate approach contrast tests generated targets versus hybrid targets. Real-data ANOVA also includes synthetic-only versus mixed training.

**Approach** asks whether replacing generator targets helps: original CTGAN/TVAE versus the average RF/XGB/DNN hybrid (1 df). **Labeler** asks whether RF, XGBoost and DNN differ within the hybrid family (2 df). Labeler excludes the original baseline because that baseline has no relabeler. The two factors answer different questions.

**Generator-input inclusion means full (X,Y) versus X-only fitting.** X is included in both; the variable being included or excluded is Y, as confirmed by the user. The ANOVA column is named x_inclusion to match the requested factor list; its contrast is full minus X-only.

## Table 1. Original generators versus hybrids

| Study | RF/XGB/DNN construction | Hybrid gain pp | Dataset wins | Paired-t p | Exact p |
| --- | --- | --- | --- | --- | --- |
| Real macro F1 | Full-table hybrid | +9.61 | 4/4 | 0.0314 | 0.1250 |
| Real macro F1 | X-only hybrid | +9.53 | 4/4 | 0.0229 | 0.1250 |
| Real macro F1 | Both sources averaged | +9.57 | 4/4 | 0.0266 | 0.1250 |
| Simulated accuracy | Full-table hybrid | +5.37 | 7/7 | 0.0233 | 0.0156 |
| Simulated accuracy | X-only hybrid | +5.24 | 7/7 | 0.0266 | 0.0156 |
| Simulated accuracy | Both sources averaged | +5.30 | 7/7 | 0.0248 | 0.0156 |

Each gain compares against the matched original CTGAN/TVAE baseline. Dataset means receive equal weight; seeds, generators and modes are averaged within dataset. The three-labeler mean summarizes a method family, not an ensemble. P-values above are unadjusted; Holm adjusts the full-hybrid, X-only-hybrid and source comparisons separately within each study. The both-source row is descriptive and is not another independent primary test.

**Scope:** RF/XGB/DNN are the user-requested study family. This restriction follows inspection of earlier results, so the analysis is exploratory rather than prospectively preregistered. Conclusions concern these three labelers. The earlier broader analysis is retained in the audit archive.

The frozen snapshot contains **632 real-data records and 378 simulated rows** at 12:14 p.m. Chicago on October 6. No research models were trained. The analysis follows the factor exploration in D:/Rprojects/research_data_synthesis while using only the corrected production snapshot.


---

# Real data: generator versus generator + labeler

Macro F1 multiplied by 100. Full hybrid uses features from full (X,Y) generator fitting; X-only hybrid uses a separately fitted features-only generator. Both replace targets using the same named labeler. Gains are percentage points versus the original generator, not relative percentages.

| Generator | Labeler | Original score | Full hybrid | Full gain | X-only hybrid | X-only gain | N tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CTGAN | RF | 64.51 | 74.62 | +10.11 | 75.46 | +10.95 | 4 |
| CTGAN | XGBoost | 64.51 | 75.65 | +11.14 | 75.31 | +10.80 | 4 |
| CTGAN | DNN | 64.51 | 80.41 | +15.90 | 80.27 | +15.76 | 4 |
| TVAE | RF | 71.85 | 74.94 | +3.10 | 74.53 | +2.69 | 3 |
| TVAE | XGBoost | 71.85 | 75.78 | +3.94 | 75.19 | +3.34 | 3 |
| TVAE | DNN | 71.85 | 81.25 | +9.40 | 81.16 | +9.31 | 3 |

The baseline is repeated in this display for readability; it appears only once per generator/seed/mode in the ANOVA. Every hybrid is matched to the same generator, dataset, seed and training mode. Zero and negative results are retained.

CTGAN coverage: Adult, weighted Census KDD, Covertype and MNIST28. TVAE coverage: Adult, Covertype and MNIST28; weighted Census lacks complete full/X-only triplets. Adult and Covertype have seeds 42/43, MNIST28 seed 42, matched Census CTGAN seed 42. Both training modes are averaged. Mixed training adds 100,000 synthetic rows to all real training rows, rather than a fixed 50/50 mix. Task coverage differs, so these CTGAN/TVAE marginal scores are not a direct generator ranking.

**CTGAN + RF/XGB/DNN:** the both-source average gain is +12.44 points. Gains are measured relative to CTGAN generated targets on matched tasks.

**TVAE + RF/XGB/DNN:** the both-source average gain is +5.30 points. Gains are measured relative to TVAE generated targets on matched tasks.

These summaries do not select a winning labeler on each test set. Generator-specific, labeler-specific and dataset-specific scores and matched tests are available in the companion CSV tables.


---

# Simulation: generator versus generator + labeler

Accuracy in percent. Full hybrid uses features from full (X,Y) generator fitting; X-only hybrid uses a separately fitted features-only generator. Both replace targets using the same named labeler. Gains are percentage points versus the original generator, not relative percentages.

| Generator | Labeler | Original score | Full hybrid | Full gain | X-only hybrid | X-only gain | N tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CTGAN | RF | 79.55 | 88.31 | +8.75 | 88.31 | +8.76 | 7 |
| CTGAN | XGBoost | 79.55 | 88.46 | +8.91 | 88.44 | +8.89 | 7 |
| CTGAN | DNN | 79.55 | 88.28 | +8.73 | 88.31 | +8.75 | 7 |
| TVAE | RF | 86.21 | 88.05 | +1.84 | 87.77 | +1.56 | 7 |
| TVAE | XGBoost | 86.21 | 88.28 | +2.07 | 88.05 | +1.84 | 7 |
| TVAE | DNN | 86.21 | 88.12 | +1.90 | 87.87 | +1.66 | 7 |

The baseline is repeated in this display for readability; it appears only once per generator/seed/mode in the ANOVA. Every hybrid is matched to the same generator, dataset, seed and training mode. Zero and negative results are retained.

Both generators cover all seven simulated datasets, seeds 42/43: Gaussian, Grid, Ring, Asia, Alarm, Child and Insurance. All methods use synthetic-only downstream training. Production settings were 300 generator epochs and 10,000 training, test and synthetic rows each.

**CTGAN + RF/XGB/DNN:** the both-source average gain is +8.80 points. Gains are measured relative to CTGAN generated targets on matched tasks.

**TVAE + RF/XGB/DNN:** the both-source average gain is +1.81 points. Gains are measured relative to TVAE generated targets on matched tasks.

These summaries do not select a winning labeler on each test set. Generator-specific, labeler-specific and dataset-specific scores and matched tests are available in the companion CSV tables.


---

# Table 2. Real-data factorial ANOVA

Balanced repeated coverage is Adult seeds 42/43, Covertype seeds 42/43 and MNIST28 seed 42: **3 datasets, 5 dataset/seed units, 140 scores**. Every unit has CTGAN/TVAE, one original plus six RF/XGB/DNN hybrid constructions per generator, and both training modes. Weighted Census remains in matched method comparisons; its incomplete TVAE block prevents entry into this balanced ANOVA.

| Source | Df (num / den) | Sum Sq | Mean Sq | F value | Pr(>F) | Adj. p |
| --- | --- | --- | --- | --- | --- | --- |
| **Dataset** | **2 / 2** | **25182.06** | **12591.03** | **4615.61** | **0.0002 \*\*\*** | **0.0084** |
| Generator | 1 / 2 | 30.96 | 30.96 | 1.98 | 0.2947 | 1.0000 |
| Approach | 1 / 2 | 1284.99 | 1284.99 | 643.29 | 0.0016 \*\* | 0.0589 |
| Labeler | 2 / 4 | 983.70 | 491.85 | 308.75 | <0.0001 \*\*\* | 0.0739 |
| Input inclusion | 1 / 2 | 0.22 | 0.22 | 0.27 | 0.6531 | 1.0000 |
| Training mode | 1 / 2 | 665.40 | 665.40 | 229.00 | 0.0043 \*\* | 0.1432 |
| Dataset x Generator | 2 / 2 | 535.40 | 267.70 | 17.12 | 0.0552 . | 1.0000 |
| Dataset x Approach | 2 / 2 | 401.31 | 200.66 | 100.45 | 0.0099 \*\* | 0.3056 |
| Generator x Approach | 1 / 2 | 226.40 | 226.40 | 316.30 | 0.0031 \*\* | 0.1101 |
| Dataset x Generator x Approach | 2 / 2 | 391.85 | 195.92 | 273.71 | 0.0036 \*\* | 0.1238 |
| Dataset x Labeler | 4 / 4 | 1713.41 | 428.35 | 268.89 | <0.0001 \*\*\* | 0.0812 |
| Generator x Labeler | 2 / 4 | 2.75 | 1.38 | 0.40 | 0.6970 | 1.0000 |
| Dataset x Input inclusion | 2 / 2 | 39.95 | 19.98 | 24.66 | 0.0390 \* | 1.0000 |
| Generator x Input inclusion | 1 / 2 | 2.05 | 2.05 | 0.43 | 0.5801 | 1.0000 |
| Labeler x Input inclusion | 2 / 4 | 3.12 | 1.56 | 0.14 | 0.8747 | 1.0000 |
| Approach x Training mode | 1 / 2 | 157.33 | 157.33 | 104.31 | 0.0095 \*\* | 0.3024 |

Significance codes beside Pr(>F): \*\*\* p < 0.001; \*\* p < 0.01; \* p < 0.05; . p < 0.10. Stars use raw p. **Bold rows** have adjusted p < 0.05. Adj. p applies Greenhouse-Geisser (GG), then Holm across all 39 model effects. Df is numerator / denominator; Sum Sq and Mean Sq are uncorrected. Each repeated-measures effect has its own error stratum, so there is no single pooled residual row.

Dataset is a fixed between-unit factor; seed is nested within dataset. Generator and constructions repeat within each dataset/seed unit. **Approach = original versus hybrid**; **Labeler = RF versus XGBoost versus DNN within hybrids**. Input inclusion compares full (X,Y) with X-only fitting. The complete CSV contains every interaction.

**Interpretation.** The RF/XGB/DNN approach main effect has raw p = 0.0016 and model-wide Holm p = 0.0589; it does not survive correction. Inclusion has p = 0.6531. Labeler Holm p = 0.0739; generator-by-approach Holm p = 0.1101. These tests are conditional on the fixed benchmark datasets.

Only **two residual seed degrees of freedom** estimate repeat variability. The covariance estimate and corrected tests are fragile, especially for the singleton MNIST28 group. These are conditional tests on fixed datasets and holdouts, not estimates of test-row uncertainty or evidence from five independent datasets.


---

# Table 3. Simulated-data factorial ANOVA

All seven datasets have seeds 42/43: **7 datasets, 14 dataset/seed units, 196 scores**. Each unit contains CTGAN/TVAE and one original plus six RF/XGB/DNN hybrid constructions per generator. Fourteen real-only reference rows are outside the ANOVA, yielding 210 selected simulated results in total. The complete 378-row production snapshot is preserved.

| Source | Df (num / den) | Sum Sq | Mean Sq | F value | Pr(>F) | Adj. p |
| --- | --- | --- | --- | --- | --- | --- |
| **Dataset** | **6 / 7** | **3584.78** | **597.46** | **149.57** | **<0.0001 \*\*\*** | **<0.0001** |
| **Generator** | **1 / 7** | **22.14** | **22.14** | **21.78** | **0.0023 \*\*** | **0.0275** |
| **Approach** | **1 / 7** | **675.37** | **675.37** | **195.92** | **<0.0001 \*\*\*** | **<0.0001** |
| **Labeler** | **2 / 14** | **1.28** | **0.64** | **13.23** | **0.0006 \*\*\*** | **0.0254** |
| Input inclusion | 1 / 7 | 0.65 | 0.65 | 1.32 | 0.2882 | 1.0000 |
| Dataset x Generator | 6 / 7 | 48.91 | 8.15 | 8.02 | 0.0073 \*\* | 0.0807 |
| **Dataset x Approach** | **6 / 7** | **458.26** | **76.38** | **22.16** | **0.0003 \*\*\*** | **0.0050** |
| **Generator x Approach** | **1 / 7** | **293.02** | **293.02** | **146.58** | **<0.0001 \*\*\*** | **0.0001** |
| **Dataset x Generator x Approach** | **6 / 7** | **268.47** | **44.75** | **22.38** | **0.0003 \*\*\*** | **0.0050** |
| **Dataset x Labeler** | **12 / 14** | **5.80** | **0.48** | **10.00** | **<0.0001 \*\*\*** | **0.0058** |
| Generator x Labeler | 2 / 14 | 0.10 | 0.05 | 0.86 | 0.4457 | 1.0000 |
| Dataset x Input inclusion | 6 / 7 | 5.35 | 0.89 | 1.81 | 0.2269 | 1.0000 |
| Generator x Input inclusion | 1 / 7 | 0.68 | 0.68 | 1.12 | 0.3244 | 1.0000 |
| Labeler x Input inclusion | 2 / 14 | 0.01 | 0.00 | 0.19 | 0.8257 | 1.0000 |
| Dataset x Labeler x Input inclusion | 12 / 14 | 0.33 | 0.03 | 1.84 | 0.1389 | 1.0000 |
| Dataset x Generator x Labeler | 12 / 14 | 1.01 | 0.08 | 1.48 | 0.2402 | 1.0000 |
| Dataset x Generator x Input inclusion | 6 / 7 | 4.93 | 0.82 | 1.35 | 0.3475 | 1.0000 |
| Generator x Labeler x Input inclusion | 2 / 14 | 0.01 | 0.01 | 0.21 | 0.8156 | 1.0000 |
| Dataset x Generator x Labeler x Input inclusion | 12 / 14 | 0.40 | 0.03 | 1.37 | 0.2860 | 1.0000 |

Significance codes beside Pr(>F): \*\*\* p < 0.001; \*\* p < 0.01; \* p < 0.05; . p < 0.10. Stars use raw p. **Bold rows** have adjusted p < 0.05. Adj. p applies Greenhouse-Geisser (GG), then Holm across all 19 model effects. Df is numerator / denominator; Sum Sq and Mean Sq are uncorrected. Each repeated-measures effect has its own error stratum, so there is no single pooled residual row.

Dataset is a fixed between-unit factor; seed is nested within dataset. Generator and constructions repeat within each dataset/seed unit. **Approach = original versus hybrid**; **Labeler = RF versus XGBoost versus DNN within hybrids**. Input inclusion compares full (X,Y) with X-only fitting. The complete CSV contains every interaction.

**Interpretation.** The RF/XGB/DNN approach main effect has raw p = <0.0001 and model-wide Holm p = <0.0001; it survives correction. Inclusion has p = 0.2882. Labeler Holm p = 0.0254; generator-by-approach Holm p = 0.0001. These tests are conditional on the fixed benchmark datasets.


---

# Target inclusion and approach are different questions

The source comparison holds generator and labeler fixed and subtracts the X-only hybrid from the full-table hybrid. It does not subtract an original generator. Both hybrids can gain substantially over the original while differing little from each other.

| Labeler | Real full - X pp | Real exact p | Sim full - X pp | Sim exact p |
| --- | --- | --- | --- | --- |
| RF | -0.26 | 0.5000 | +0.14 | 0.4688 |
| XGBoost | +0.44 | 0.8750 | +0.13 | 0.5625 |
| DNN | +0.07 | 1.0000 | +0.11 | 0.7656 |

**Real data, RF/XGB/DNN:** mean full-minus-X-only = +0.08 points; 95% paired-t interval [-1.68, +1.84]; exact p = 0.8750.

**Simulation, RF/XGB/DNN:** mean full-minus-X-only = +0.12 points; 95% paired-t interval [-0.23, +0.48]; exact p = 0.7188.

For RF/XGB/DNN, real full-hybrid gain is +9.61 points and X-only gain is +9.53, so their difference is +0.08. Simulation gives +5.37 minus +5.24 = +0.12. The small source difference is not an arithmetic error and does not mean the hybrid gain itself is 0.08 points.

## Which statistical question does each test answer?

**Factorial ANOVA:** does a factor or interaction change mean scores relative to experiment-seed repeat variability on these fixed datasets? Dataset is an explicit factor and dataset interactions are estimated. Normality, common covariance across dataset groups and independent RNG repeats are assumptions; real repeats share one held-out split.

**Dataset-level paired comparisons:** is the average improvement consistent across the observed tasks? Seeds are averaged within dataset, so the primary real comparison has four task units and simulation seven. Exact sign-flip p-values assume independent task units and symmetric zero-centered differences under the null. Paired-t p-values assume approximately normal task differences.

The tests have different denominators and different scopes. Small conditional ANOVA p-values do not make new datasets or establish general superiority. Likewise, a nonsignificant inclusion effect is not proof of equivalence for every dataset or labeler.


---

# Simulation: utility versus distribution fidelity

The simulated benchmark supplies known joint distributions and Bayes decisions. It therefore tests more than downstream accuracy. RF/XGB/DNN full-table relabeling increases average macro F1 by 6.83 points and Bayes agreement by 7.33 points; these are correlated secondary outcomes, not additional independent confirmations.

DNN full-table relabeling: each point averages seeds 42/43 for one dataset and generator. All fourteen accuracy gains are positive; eight L_test changes are negative.

![DNN full-table relabeling: each point averages seeds 42/43 for one dataset and generator. All fourteen accuracy gains are positive; eight L_test changes are negative.](D:/SummerResearch/output/statistical_review_20261006/utility_and_density.png)

For RF/XGB/DNN, average L_syn improves by about 0.323 nats per row while L_test changes by -0.132, with 95% interval [-0.405, +0.141] and exact p = 0.3125. There is no demonstrated improvement in density-refit fidelity. X-only L_test changes by -0.288, p = 0.2031.

Hard relabeling can recover a useful decision boundary while removing real target noise. It cannot recover feature modes absent from the generated features. Insurance CTGAN full+DNN retains approximately 48.8% oracle-impossible rows despite approximately 94.8% downstream accuracy.

L_test scores a density refitted to each training table; it is not generator likelihood. Mixed-data densities are exact. BN scores use historical log(p + 1e-8) once on joint probability, so they do not satisfy the normalized-density KL identity. Oracle probabilities and Bayes decisions remain exact.


---

# Exclusions, sensitivities and paper claims

**Credit:** held out of the primary classification analysis because the real holdout contains only ten positives. Including it raises the RF/XGB/DNN full-minus-X-only effect from +0.08 to +3.44 points. In the separate Credit ANOVA, approach raw p = <0.0001, Holm p = 0.0028; inclusion raw p = 0.0001, Holm p = 0.0036. Both survive correction in that sensitivity. The changed conclusion is driven by a rare-class holdout and severe X-only failures; it does not establish a robust source preference.

**Intrusion:** no completed generator-versus-hybrid comparisons enter the frozen snapshot. **MNIST12:** replaces MNIST28 in sensitivity analysis, rather than counting related source images as independent tasks. **Census:** old unweighted records are separate; weighted CTGAN comparisons remain, while incomplete TVAE cells are excluded from complete-triplet inference. All RF/XGB/DNN outcomes, including negative ones, remain in scope.

**News regression:** only one dataset/seed block exists, so News is separate from classification and has no benchmark-level ANOVA. All three labelers worsen normalized MAE versus generated targets. For DNN, full-table R2 improves by 0.0485 while normalized MAE worsens by 0.0364; X-only R2 improves by 0.0445 while normalized MAE worsens by 0.0486. Saved test-set normalization was used.

## What the current evidence supports

RF/XGB/DNN produce consistent utility gains against their original generators on the observed classification tasks, with generally larger gains against CTGAN. This supports a claim about flexible discriminative relabeling with these three methods. Including Y in generator fitting has no clear average utility advantage. Utility gains need not improve distribution fidelity.

Real-only superiority is not established. RF/XGB/DNN full-table hybrids average +1.11 real macro-F1 points relative to the real-only student (exact p = 0.875) and -0.048 simulated accuracy points (p = 0.4688). The real reference averages synthetic-only and mixed arms; it does not alone establish an augmentation benefit.

Across four real task units, the smallest exact two-sided sign-flip p is 0.125. Simulated full and X-only hybrid gains have raw p = 0.0156 and three-comparison Holm p = 0.0469. The corresponding Holm-adjusted paired-t p is 0.0699, showing sensitivity to the test. Grouping Gaussian/Grid/Ring into one family raises exact p to 0.0625. The requested restriction was made after seeing earlier results; this is exploratory evidence within RF/XGB/DNN.

## A simple next step

Present both original-generator baselines and all three selected labelers in the paper. Choose a primary labeler using training/development evidence. Before expanding to Tab-DDPM, establish what generated features add beyond direct teacher evaluation and teacher-labeled resampling of real training features. Additional independent datasets address generalization; extra seeds address repeat variability. No new experiments were run for this report.


---

# References, provenance and reproducibility

The requested reference project was read at D:/Rprojects/research_data_synthesis. Its final_328_project.Rmd explores dataset, generator (tvae), labeler (y_synth) and target inclusion (has_y), including dataset-by-labeler interactions. Its later deduplicated_analysis_2026_09_18/README.md, analyse.R and legacy_artifacts.R flag baseline/source aliasing, correlated methods and sensitivity to labeler dependence.

This report follows that factor structure while addressing the absent baseline X-only cells through nested construction contrasts. It does not import historical test maxima, the duplicated Adult/Census routes, hypothetical paper-score substitutions or leaked records. Real production results use development-selected checkpoints. Historical files are references, not additional independent observations.

## Exact statistical design

Each generator has seven observed constructions: one original generated-target baseline; three full-table relabeling arms; three X-only relabeling arms. An orthonormal construction basis partitions six contrast df into approach (1), hybrid labeler (2), target inclusion within hybrids (1), and labeler-by-inclusion (2). Generator and real training-mode contrasts are crossed with this basis. Labelers exist within the hybrid family; there is one baseline per generator/seed/mode.

Type III sums of squares test equal-dataset marginal effects in the between/within model. Every term uses its own seed-within-dataset error projection. Multi-df within contrasts receive GG corrections; Holm covers the complete 39-term real and 19-term simulated tables. Interactions, error strata and coverage are saved in results_rf_xgb_dnn. These conditional tests retain RF/XGB/DNN and original baselines without treating a shared baseline as three observations.

## Files and validation

The report is available as PDF, editable Markdown and standalone HTML with embedded figures. The audit folder contains the frozen snapshot, rebuild_rf_xgb_dnn.py, eleven restricted-family CSVs, both complete ANOVAs, method scores, task-level gains, Credit sensitivity, and independent checks. Earlier broader outputs remain archived. Statistics ran remotely: NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3 and statsmodels 0.15.0. Local work prepared report artifacts only.

Validation checked all source/table hashes, construction-basis orthogonality, exact score reconstruction, every ANOVA sum of squares against an independently restricted least-squares model, every one-dimensional F against statsmodels OLS, no duplicate cells, and no fabricated baseline arms. Referenced predictions/checkpoints and production budgets were checked; scores were not independently recomputed from every prediction file. No test-row bootstrap was performed.

## Method documentation

The afex reference documents Type III between/within ANOVA and GG correction. Demsar describes comparisons across datasets. The paired dataset tests and conditional ANOVAs answer different inferential questions.

https://search.r-project.org/CRAN/refmans/afex/html/aov_car.html

https://search.r-project.org/CRAN/refmans/afex/html/afex_aov-methods.html

https://www.jmlr.org/papers/v7/demsar06a.html

Frozen snapshot SHA-256: 387162270373aeb557892c2fcfcb351d6067dd1cc67aa373c2f8b27d1fb5beae
