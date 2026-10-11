# 1. Two questions, seven tasks

**Research question:** Does relabeling improve utility, and does fitting the generator with Y help?

**Method:** Use verified completed CTGAN/TVAE runs, seeds 42/43 and RF/XGB/DNN. Keep synthetic-only and mix separate.

**Finding:** Classification gains favor relabeling but do not pass Holm across the two questions. Average has_y effects are small. Regression depends on the task and metric.

| Design choice | Definition |
| --- | --- |
| Classification tasks | Adult, Census KDD, Covertype, MNIST12, MNIST28. Outcome: macro F1 for all five. |
| Regression tasks | News and California Housing. Outcomes: saved R2, NMAE and absolute-error D2, separately. |
| Relabeled | Average both has_y settings, both generators and RF/XGB/DNN; keep generated X and predict Y. |
| Generated-target comparator | Fit each generator to (X,Y), retaining its generated Y. Match dataset, seed and generator. |
| has_y=1 versus has_y=0 | Compare generator fitted to (X,Y) versus fitted to X only. Both arms receive predicted targets. |
| Training regimes | Synthetic-only: 100,000 synthetic rows. Mix: all real training rows plus 100,000 synthetic rows. |
| Coverage | 406 selected completed records: 290 classification and 116 regression, including 14 shared real-only references. |

```text
C_relabel[d] = mean(relabel[d]) - mean(generated[d])
```

```text
C_has_y[d] = mean(relabel[d, has_y=1]) - mean(relabel[d, has_y=0])
```

Means give equal weight to seeds, generators and labelers. Classification inference then gives each of the five datasets equal weight. Each generated baseline is a shared comparator, not six independent observations.

The two questions were selected after earlier analyses of these results. This redo is exploratory; it is not a newly preregistered confirmatory analysis. It excludes Credit, Intrusion, NB/PCA-GMM, simulated benchmarks and active TabDDPM runs.

# 2. Classification: gains and evidence

**Research question:** Are either of the two average macro-F1 contrasts supported across tasks?

**Method:** Two-sided paired t-tests on five dataset differences, df=4. Holm2 covers both questions within a regime; Holm4 covers all four tests across both regimes.

**Finding:** Relabeling gains 11.63 points for synthetic-only and 6.77 for mix; all five task gains are positive. Neither relabeling nor has_y passes the Holm2 threshold of .05.

| Training | Contrast | Gain (points) | Raw p | Holm2 p | Holm4 p |
| --- | --- | --- | --- | --- | --- |
| Synthetic-only | Relabeled - generated | +11.63 | 0.0350 | 0.0700 | 0.1399 |
| Synthetic-only | has_y=1 - has_y=0 | +0.36 | 0.4575 | 0.4575 | 0.9149 |
| Mix | Relabeled - generated | +6.77 | 0.0566 | 0.1132 | 0.1698 |
| Mix | has_y=1 - has_y=0 | +0.14 | 0.7182 | 0.7182 | 0.9149 |

| Task | Synthetic relabel | Mix relabel | Synthetic has_y | Mix has_y |
| --- | --- | --- | --- | --- |
| Adult | +3.53 | +1.71 | -0.69 | -0.75 |
| Census KDD | +3.28 | +4.80 | +0.15 | -0.06 |
| Covertype | +11.76 | +13.63 | +1.87 | +1.44 |
| MNIST12 | +21.58 | +12.03 | +0.62 | +0.07 |
| MNIST28 | +18.02 | +1.70 | -0.17 | -0.01 |

Every change in the second table is in macro-F1 percentage points; positive favors relabeling or has_y=1. The has_y contrast is positive on three of five synthetic-only tasks and two of five mix tasks.

Synthetic-only mean F1: relabeled .7855 versus generated .6692. Mix: .8166 versus .7488. The corresponding relative improvements are 17.38% and 9.04%; relative percentages and percentage points are different quantities.

Raw synthetic relabel p=.0350 is below .05; its Holm2 p=.0700 is above .05. Holm changes the inferential threshold, not the gain. No detected has_y effect does not establish equivalence.

# 3. Regression: News and Housing

**Research question:** Do the same constructions improve regression utility?

**Method:** Compare native-unit means across seeds, generators and RF/XGB/DNN separately for each task. These baseline contrasts are descriptive; no seven-task pooled score is computed.

**Finding:** Housing relabeling improves R2, NMAE and D2. News relabeling improves R2 but worsens NMAE and D2. has_y effects are smaller and have mixed directions.

**Synthetic-only**

| Task / metric | Generated | Relabeled | has_y=0 | has_y=1 | Relabel change | has_y change |
| --- | --- | --- | --- | --- | --- | --- |
| News / R2 | -0.0235 | 0.0011 | 0.0027 | -0.0004 | +0.0246 | -0.0031 |
| News / NMAE | 0.2546 | 0.2592 | 0.2597 | 0.2587 | +0.0046 | -0.0010 |
| News / D2 | 0.0099 | -0.0079 | -0.0098 | -0.0060 | -0.0178 | +0.0038 |
| Housing / R2 | 0.6730 | 0.7938 | 0.7946 | 0.7930 | +0.1208 | -0.0015 |
| Housing / NMAE | 0.4146 | 0.3036 | 0.3049 | 0.3022 | -0.1111 | -0.0027 |
| Housing / D2 | 0.4595 | 0.6043 | 0.6025 | 0.6061 | +0.1448 | +0.0036 |

**Mix**

| Task / metric | Generated | Relabeled | has_y=0 | has_y=1 | Relabel change | has_y change |
| --- | --- | --- | --- | --- | --- | --- |
| News / R2 | -0.0196 | -0.0003 | 0.0002 | -0.0008 | +0.0192 | -0.0010 |
| News / NMAE | 0.2494 | 0.2548 | 0.2551 | 0.2544 | +0.0054 | -0.0006 |
| News / D2 | 0.0300 | 0.0092 | 0.0079 | 0.0104 | -0.0208 | +0.0025 |
| Housing / R2 | 0.6676 | 0.7888 | 0.7905 | 0.7872 | +0.1212 | -0.0033 |
| Housing / NMAE | 0.4137 | 0.3077 | 0.3103 | 0.3052 | -0.1060 | -0.0051 |
| Housing / D2 | 0.4607 | 0.5989 | 0.5956 | 0.6022 | +0.1382 | +0.0066 |

Higher R2 and D2 is better; lower NMAE is better. The change columns retain native signs: negative NMAE change is improvement. R2 uses squared error against a test-mean baseline; D2 uses absolute error against a test-median baseline.

NMAE = saved MAE / saved SD_test(Y), with population SD (ddof=0). Saved record normalization and verified D2 sidecars are read directly; denominators are not recomputed. News log-target runs are scored in original units; Housing uses no-Faker runs.

Only two fixed regression tasks and two seeds are available. Descriptive improvement on Housing does not establish a population effect on new regression tasks. Testing multiple regression outcomes would also require an explicit family if used for joint significance claims.

# 4. Exploratory factor checks

**Research question:** What do factorial models add beyond the two research contrasts?

**Method:** Refit categorical Type II ANOVAs using sum-to-zero coding. Generated targets stay outside the relabeled factorial grid. All p-values on this page are nominal and exploratory.

**Finding:** Labeler choices differ in the classification model and treatment effects vary strongly by task. These tests do not establish improvement over generated targets.

```text
macro_F1 ~ dataset + generator * has_y * labeler * seed
```

Classification: 120 relabeled scores per regime, model rank 28 and residual df 92. The ANOVA assumes independent, homoskedastic residuals; shared heldout rows and task-specific effects limit its interpretation.

| Classification term | Synthetic nominal p | Mix nominal p |
| --- | --- | --- |
| dataset | 1.17e-49 | 5.36e-52 |
| generator | 0.7746 | 0.9987 |
| has_y | 0.6671 | 0.8459 |
| labeler | 1.03e-05 | 7.85e-06 |
| seed | 0.8926 | 0.8584 |

| Supporting nested test | Synthetic nominal p | Mix nominal p |
| --- | --- | --- |
| seed interactions | 0.9998 | 0.9999 |
| dataset interactions | 1.86e-09 | 2.93e-32 |

```text
R2 or D2 ~ dataset + generator * has_y * labeler * seed
```

Regression pooled checks: 48 relabeled scores per outcome/regime, rank 25, residual df 23. The two tasks remain fixed study tasks; pooled coefficients are not evidence about a population of regression tasks.

| Training / outcome | Generator p | has_y p | Labeler p | Seed p |
| --- | --- | --- | --- | --- |
| Synthetic-only / R2 | 0.2947 | 0.7332 | 0.0879 | 0.3036 |
| Synthetic-only / D2 | 0.0164 | 0.4608 | 0.0475 | 0.4740 |
| Mix / R2 | 0.0423 | 0.5589 | 0.0499 | 0.7202 |
| Mix / D2 | 0.0503 | 0.2605 | 0.0005 | 0.5071 |

Complete ANOVA tables and within-task regression models are supplied as CSVs. Within each regression task: outcome ~ seed + generator * has_y * labeler, rank 13 and residual df 11. A full four-factor within-task model is saturated and has no residual degrees of freedom.

Nominal ANOVA p-values support exploration, not additional adjusted headline claims. A significant labeler term means differences among RF/XGB/DNN, not superiority to generated targets.

# 5. Uncertainty and exact sensitivity checks

**Research question:** How precise are the gains, and do small-sample checks agree?

**Method:** Show pointwise 95% paired-t intervals and Bonferroni simultaneous intervals for both questions within each regime. Enumerate all 32 sign patterns; compute exact Wilcoxon signed-rank p-values.

**Finding:** Simultaneous intervals for all four effects contain zero. Exact two-sided relabeling p=.0625 in each regime, before multiplicity adjustment.

| Training / contrast | Pointwise 95% CI | Simultaneous 95% CI (2) |
| --- | --- | --- |
| Synthetic-only / Relabeled - generated | [+1.33, +21.93] | [-1.33, +24.60] |
| Synthetic-only / has_y=1 - has_y=0 | [-0.85, +1.56] | [-1.16, +1.88] |
| Mix / Relabeled - generated | [-0.31, +13.85] | [-2.14, +15.68] |
| Mix / has_y=1 - has_y=0 | [-0.85, +1.13] | [-1.11, +1.38] |

Intervals are in macro-F1 percentage points. Simultaneous intervals use t_(1-.05/(2*2),4); they are Bonferroni intervals, not an inversion of Holm. The package also includes simultaneous intervals for all four tests.

| Training / contrast | Exact sign-flip p | Exact Wilcoxon p | Sign-flip Holm2 |
| --- | --- | --- | --- |
| Synthetic-only / Relabeled - generated | 0.0625 | 0.0625 | 0.1250 |
| Synthetic-only / has_y=1 - has_y=0 | 0.6250 | 0.8125 | 0.6250 |
| Mix / Relabeled - generated | 0.0625 | 0.0625 | 0.1250 |
| Mix / has_y=1 - has_y=0 | 0.7500 | 1.0000 | 0.7500 |

With five nonzero task effects, the minimum attainable two-sided exact p is 2/32=.0625. The prior normal-approximation Wilcoxon value for the all-positive relabeling effects was .0431; the exact value is .0625. Neither exact check rejects at .05.

The paired-t test assumes independent task-level differences with a suitable normal model; exact sign-flip and signed-rank checks assume independent differences symmetric about zero under the null. The task methods were not randomly assigned, so this is a symmetry-based sensitivity analysis, not a design-based randomization test.

MNIST12 and MNIST28 share a task family and prepared split, and MNIST12 TVAE tables retain documented holdout-feature collisions. Both facts weaken broad task-generalization claims. Additional experiment seeds do not supply additional independent datasets.

Two observed seeds share heldout partitions; the reported intervals do not quantify uncertainty from resampling new test rows. The results show observed gains with limited evidence about generalization, not evidence that the two constructions are equivalent.

# 6. Is Holm required for contrasts?

**Research question:** Does using a contrast automatically require Holm correction?

**Method:** Distinguish the individual estimand from the family of claims. Use the two-question family specified for this redo; also report the four-test family when drawing conclusions across both training regimes.

**Finding:** No. A contrast is a way of defining a comparison; multiplicity comes from making several inferential claims. Holm is a defensible choice for the two related questions, not the only valid procedure.

| Situation | Implication |
| --- | --- |
| One prespecified primary hypothesis | Ordinary p-value and interval can be used for that single claim; no additional contrast adjustment is needed. |
| Several related claims | Define the family before interpreting significance. Holm controls the chance of at least one false rejection within that family, under valid marginal tests. |
| Prespecified or orthogonal contrasts | Planning improves focus but does not automatically give familywise error control. Correlated contrasts do not invalidate Holm. |
| Your two questions in one regime | Holm2. The smaller p must first be <=.025. Synthetic relabel raw p=.0350 misses this cutoff; mix raw p=.0566 also misses it. |
| Claims across both regimes | Holm4 covers the four reported classification tests. Separate model fits and shared data do not by themselves justify separate testing families. |
| Descriptive or exploratory results | Unadjusted p-values may be shown if labeled clearly; they should not be presented as familywise-controlled discoveries. |
| Current results already examined | Restricting to two questions now is transparent and useful, but cannot retroactively create prespecification. Retain two-sided tests in this redo. |

The previous report tested four construction contrasts per regime. This report tests only relabeled versus generated and has_y=1 versus 0, as requested. It does not add separate full-relabel-versus-generated or X-only-relabel-versus-generated significance claims.

Holm2 and Holm4 are family-specific results, not interchangeable thresholds to choose after viewing p-values. If the scientific claim concerns either question in either regime, use Holm4. A joint conjunctive claim requiring all component tests to succeed is a different hypothesis and would need its own stated decision rule.

**Provenance and checks:** October 10, 2026, 11:35 PM Chicago. All 522 record/sidecar hashes matched; 3,770 frozen metric values matched source records. Eighteen models were refitted remotely with one numerical thread; all 202 ANOVA F statistics were independently reproduced by restricted least squares. No experiment, source record, split or research setting was changed.

References: R stats p.adjust documentation (stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html); NIST, Bonferroni method (itl.nist.gov/div898/handbook/prc/section4/prc473.htm); SciPy, wilcoxon documentation (docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
