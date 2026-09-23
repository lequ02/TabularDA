# Adult counted once: historical reanalysis

This folder is separate from the old analysis. `analyse.R` reads two frozen April 29 CSV snapshots and regenerates its own outputs. It does not modify the original files or refit the synthesis/prediction models. Input paths and SHA-256 hashes are recorded in `inputs/source_manifest.json`.

Run in PowerShell:

```powershell
& 'D:\R-4.5.1\bin\Rscript.exe' --vanilla 'D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18\analyse.R'
```

Required R packages: ggplot2, lme4, lmerTest, emmeans, car and MASS. An optional first argument supplies another input directory containing the same two CSVs. Outputs are replaced only within this analysis folder. `analyse.R` automatically runs the readable companions `legacy_artifacts.R`, `rf_xgb_analysis.R` and `paper_comparison.R`.

## What changed

Adult, Census and Census_Kdd are averaged within each method and counted as one dataset, based on the reconstructed April code and data routing. The other tasks are Covertype, MNIST12 and MNIST28. This removes duplicate dataset weighting; it **does not remove leaked observations** or establish that Census_Kdd is an independent KDD evaluation.

Each dataset receives equal weight. Method-family means give each recorded configuration equal weight. The joint baseline pools CTGAN and TVAE equally. The main comparison includes every recorded labeling method; the historical exclusion of Gaussian methods is a sensitivity analysis. End-of-training scores provide another sensitivity check against reporting the maximum over test evaluations.

## Results

| Comparison | Mean macro-F1 gain | Relative gain | Dataset-level p | 95% interval for gain |
|---|---:|---:|---:|---:|
| All methods, test maximum | 0.0587 | 9.22% | 0.172 | -0.0457 to 0.1631 |
| Gaussian excluded, test maximum | 0.1176 | 18.49% | 0.071 | -0.0185 to 0.2537 |
| All methods, final epoch | 0.0654 | 10.58% | 0.188 | -0.0572 to 0.1880 |
| Gaussian excluded, final epoch | 0.1251 | 20.26% | 0.082 | -0.0295 to 0.2798 |

Tests use the four paired dataset differences, with two-sided t tests. These are exploratory comparisons: four tasks provide limited evidence, and the two MNIST tasks are related. No adjustment is applied to these sensitivity comparisons.

Matched target inclusion gives with-y minus without-y = **-0.0024**, p = **0.584**, 95% interval **[-0.0149, 0.0101]**. A nonsignificant result does not establish equivalence or justify a claim that excluding y has no cost.

For continuity, the script reproduces the historical mixed model and its three Sidak-adjusted contrasts, including the old third contrast without assigning it a new interpretation. That model reports a restricted-family gain p = 0.0011; adding labeling-method dependence gives p = 0.1191. Model sensitivity and the absence of independent seed replicates prevent treating the smaller p as conclusive evidence. See `outputs/statistical_results.txt` for estimates, diagnostics and R session details.

The original-data reference outperforms every recorded synthetic configuration on each of the four tasks. The gains above are against the pooled joint-synthesis baseline, not gains over original-data training or proof of superiority over both CTGAN and TVAE separately.

## Figures and tables

All figures are supplied as PNG and vector PDF in `outputs/figures`:

1. Dataset means: all methods versus the historical Gaussian exclusion.
2. All 22 synthesis configurations, including their scores.
3. Decoupled-minus-joint gains by dataset.
4. Observed, matched target-inclusion effects.
5. Dataset-level sensitivity intervals.
6. Historical mixed-model residuals.
7. Residual normality plot.
8. Original-data reference versus joint baselines and TVAE with RF/XGBoost targets.
9. The attached-style method-distribution boxplot, using the old Gaussian exclusion.
10. Method-distribution boxplot including all recorded methods.
11. Scores and mean lines by feature generator and target inclusion.
12. Scores and mean lines by dataset, feature generator and target inclusion.
13. Target-method marginal means and confidence intervals.
14. Original LM four-panel diagnostics.
15. Box-Cox likelihood profile.
16. Transformed LM four-panel diagnostics.
17. Target-inclusion marginal means by feature generator.
18. Contrasts between labeler groups, with adjusted confidence intervals.
19. Distribution comparison restricted to joint CTGAN/TVAE versus RF/XGBoost targets.
20. Joint baselines and RF/XGBoost labeling means by dataset.
21. Restricted-family sensitivity intervals.
22. Every paper dataset: final CTGAN/TVAE baseline comparisons using the matching metric, with missing results visible.
23. All-methods, no-Gaussian and RF/XGBoost gains before/after borrowing the paper's TVAE Covertype score.
24. RF/XGBoost borrowed-score sensitivity intervals.

## Paper comparison and borrowed-score sensitivity

Open [the paper-comparison report](outputs/paper_comparison/paper_comparison_report.html) for the complete table, figures and hypothetical analysis. [The detailed analysis](outputs/paper_comparison/PAPER_COMPARISON_AND_SENSITIVITY.md) describes metric matching, missing datasets, all historical versions, and statistical limits. Input snapshots and hashes are in `inputs/paper_comparison_manifest.json`.

Final Adult comparison uses binary F1, Covertype uses macro-F1, and MNIST uses accuracy (micro-F1 equals accuracy here). The three Adult aliases are averaged once. No genuine final Census-KDD, Credit, Intrusion or News baseline is available; the older workbook's Credit CTGAN scores are preserved as historical, audit-flagged evidence. March/April 2 baseline records repeat 80 metric cells from April 29 and do not add independent replications.

Replacing only TVAE Covertype with the paper's **0.433** increases the pooled joint baseline from **0.6361 to 0.6531**. The RF/XGBoost mean stays **0.8013**; its gain becomes **0.1482 (22.70%)**, with a mechanical paired-dataset **p = 0.0520**, versus **0.1652 (25.97%), p = 0.0467** for the recorded results. The corresponding historical no-Gaussian family gain is **15.41%, p = 0.0940**; the all-methods gain is **6.39%, p = 0.2109**.

This is an explicitly hypothetical hybrid of differing evaluation protocols, not a corrected experiment or leakage adjustment. Maximum and final-epoch sensitivities both borrow the same paper number because the paper provides no local-style epoch endpoints. Recorded results, the current abstract, and all prior analyses remain unchanged.

## RF/XGBoost-only comparison

The requested restricted comparison excludes categorical, PCA-GMM and Gaussian methods. RF/XGBoost assign targets to CTGAN- or TVAE-generated features. The joint mean is **0.6361**, versus **0.8013** for RF/XGBoost labeling: **+0.1652 macro-F1 (25.97%)**, paired dataset **p = 0.0467**, 95% interval **[0.0045, 0.3259]**. Final-epoch scores give **p = 0.0654**; excluding Covertype because of its inconsistent test versions gives **p = 0.1582**. This borderline significance is fragile and does not establish leakage-free superiority. Selecting stronger methods after inspecting results makes this an exploratory restricted-family analysis.

The low TVAE point is Covertype (**0.2972**, compared with the paper's **0.433**). Its CTGAN/TVAE ranking reverses relative to the paper. The paper averages downstream classifiers; these historical results use DNN test maxima. See [the full restricted analysis](outputs/rf_xgb/RF_XGB_ANALYSIS.md) for paper context, generator-matched comparisons, leave-one-dataset-out checks, and mixed-model sensitivity.

Open [the complete artifact report](outputs/legacy_artifacts_report.html) to browse figures and the restored tables/model outputs. [ARTIFACT_COVERAGE.md](ARTIFACT_COVERAGE.md) maps each artifact category from `final_328_project.Rmd` to its replacement. The restored exploratory models retain the historical Gaussian exclusion. Rank-deficient/saturated models, singular fits, unavailable tests, and extrapolated combinations are reported explicitly. Unit-leverage joint-baseline observations are omitted from some LM diagnostic panels by R and reported in `run_log.txt`.

CSV files expose collapsed method scores, dataset gains, sensitivity summaries, model contrasts and target-inclusion pairs. The revised title and abstract are in `revised_title_and_abstract.md`. As requested, the abstract amends the original poster wording and uses its historical Gaussian-excluded comparison and mixed-model target-inclusion contrast (p = 0.9991 after counting Adult once); the expanded analyses remain in this report.

## Trust limits

This analysis cannot calculate leakage-adjusted performance from summary scores. The audit found differing Covertype test versions, preprocessing fitted to test data, selection of maximum test scores, and incomplete run/split provenance. Removing Adult duplication changes statistical weighting, not the underlying predictions. Confirmatory results require generator fitting and target-predictor fitting on training data only, preprocessing learned from training data, model selection on validation data, and evaluation on one fixed untouched real test set with independent repeated seeds.

## Labeled paper/replication figure

Figure 25 shows the paper, our final epoch, and our test maximum for CTGAN and TVAE on matching metrics. Regenerate this figure independently with `Rscript plot_paper_comparison.R`; the full analysis also creates it. Missing final datasets are noted in the caption rather than plotted as zero.

## Boxplots matching the original style

Figures 26 and 27 compare recorded results with the hypothetical paper TVAE Covertype substitution. Figure 26 excludes categorical and PCA-GMM; Figure 27 retains the six methods shown in the reference figure. Both count Adult once and show approach means, dashed red lines, and a red difference arrow. Regenerate with `Rscript plot_paper_substitution_boxplots.R`.

## Generator-separated comparison

Figure 28 compares CTGAN-based methods on the left and TVAE-based methods on the right. It uses a user-specified joint TVAE Covertype maximum of 0.49. Figure details, provenance, statistics and interpretation limits are in [GENERATOR_COMPARISON_NOTES.md](GENERATOR_COMPARISON_NOTES.md). Reproduce with `Rscript plot_generator_comparison.R`.

## Four-panel comparison

Figure 29 labels the complete CTGAN/TVAE-to-RF/XGB pipelines and adds target-inclusion and pooled e2e/hybrid panels. Details and statistics: [FOUR_PANEL_COMPARISON_NOTES.md](FOUR_PANEL_COMPARISON_NOTES.md). Reproduce with `Rscript plot_four_panel_comparison.R`.

Figure 29 layout: e2e/hybrid top left; generator training inputs top right; CTGAN-based bottom left; TVAE-based bottom right. [Comparison statistics and both-generator task breakdown](FOUR_PANEL_STATISTICS.md).

The pooled Figure 29 panels retain eight task/generator points per group (four tasks times CTGAN/TVAE). The input panel is labeled Feature generator input, with Features only and Features + target. Inference still uses four task pairs; plotted generator results are related within task.
