# Coverage of the old final analysis

Reference: `D:\Rprojects\research_data_synthesis\final_328_project.Rmd`, cross-checked against `328_project.Rmd`. These are the April final-analysis artifacts. The earlier March/April 2 reports use older data and macro/micro averages; they are preserved in their original folders rather than mixed into the April reanalysis.

Every category of table, model and figure in the old final report has a replacement below. Numeric values and point counts intentionally change because three Adult aliases now count once. All plots have PNG and PDF versions in `outputs/figures`; model/table outputs are under `outputs/legacy_tables`. Run only `analyse.R`; it sources `legacy_artifacts.R` automatically.

| Old final report artifact | New artifact |
|---|---|
| Input preview and unique-value summary (lines 103–105) | `01_input_preview`, `02_factor_inventory`, `03_observed_combinations` tables |
| Labeler scores by generator, faceted by has_y (120–125) | `11_legacy_generator_factors` |
| Dataset trajectories, faceted by has_y and generator (128–133) | `12_legacy_dataset_factors` |
| Full-interaction LM and ANOVA (139–162) | `05_full_interaction_*` tables; saturation/aliases visible |
| Dataset-blocked factor LM, summary, ANOVA and aliases (170–204) | `06_factor_blocking_*` tables; unavailable Type III test recorded |
| Dataset-blocked method LM and Type I/III ANOVAs (209–229) | `07_method_blocking_*` tables |
| Dataset-by-labeler LM, summaries and ANOVAs (242–263) | `08_dataset_labeler_interaction_*` tables |
| Four-panel original-model diagnostics (255–257) | `14_legacy_lm_diagnostics` |
| Box-Cox profile and selected transformation (269–290) | `15_legacy_boxcox_profile`, `09_boxcox_*` tables |
| Transformed-model summary, ANOVAs and diagnostics (293–326) | `10_boxcox_model_*` tables, `16_legacy_transformed_diagnostics` |
| Initial mixed model, ANOVAs and intervals (363–386) | `11_initial_mixed_model_*` tables |
| Combined approach/has_y models with/without labeler random effects (435–483) | `12_combined_groups_labeler_random_*`, `13_combined_groups_historical_*` tables |
| Combined-group marginal means, custom contrasts and pairwise contrasts (491–537) | `15_group_marginal_means`, `16_group_pairwise_tukey`, `17_group_custom_sidak` |
| LM old/new labeler contrasts (551–601) | `18_lm_decoupled_minus_joint` |
| Conditional labeler means and contrasts by has_y (606–658) | `19_conditional_labeler_means`, `20_conditional_approach_contrasts`; extrapolations flagged |
| Target-inclusion marginal means and contrasts by generator (694–731) | `21_target_means_by_generator`, `22_target_with_minus_without`, figure `17_legacy_target_marginal_means` |
| Bonus labeler mixed model, intervals, ANOVAs and group contrasts (743–784) | `14_labeler_mixed_model_*`, `23_labeler_group_contrasts`, figure `18_legacy_labeler_group_contrasts` |
| Labeler marginal-mean plot with confidence intervals (791–831) | `13_legacy_labeler_marginal_means`, `24_labeler_marginal_means` |
| Attached boxplot: jitter, approach colors, overall mean circles/lines, difference arrow (837–937) | `09_legacy_method_distribution`; all-methods companion `10_all_method_distribution` |
| Rendered report containing the analysis outputs | `outputs/legacy_artifacts_report.html`; individual figures also export as vector PDF |

## Deliberate corrections

- No Adult alias is treated as an independent dataset. The restricted models use 72 rows (18 methods × four tasks), and the all-methods plots use 88 rows (22 × four).
- Overall-mean circles in the distribution plot represent approach means, matching the old visual; they are explicitly labeled to avoid interpreting them as individual method means. The improvement is calculated from the displayed means rather than hard-coded as 11%.
- Missing joint-generator configurations are not filled in. Conditional model extrapolations are flagged; observed matched pairs remain the preferred target-inclusion comparison.
- Target contrast signs now consistently mean with-y minus without-y. Labeler-group contrasts compare group averages; the old unnormalized weights compared sums.
- Custom contrasts use Sidak adjustment, because Tukey adjustment is intended for pairwise comparisons. Pairwise comparisons retain Tukey adjustment.
- Mixed-model intervals use the simpler, explicit Wald method, rather than the old profile method. Variance-component bounds are not available under this method; variance estimates remain in model summaries.
- The original Type III treatment coding is retained for artifact continuity and labeled exploratory. Saturation, aliasing, singularity and diagnostics cannot certify generalization or absence of leakage.

These artifacts supplement the dataset-level comparisons and trust limits in README.md. No old project artifact is modified.

## Additional RF/XGBoost comparison

`rf_xgb_analysis.R` now adds the requested joint CTGAN/TVAE versus RF/XGBoost-target comparison, excluding categorical, PCA-GMM and Gaussian configurations. Its tables and report are in `outputs/rf_xgb`; figures 19–21 show distributions, dataset means and sensitivity intervals. These supplement all 18 existing figures. The original broad-family tables and current abstract remain unchanged.

## Paper comparison and separate hypothetical results

`paper_comparison.R` adds the complete final CTGAN/TVAE paper comparison, with all eight paper datasets and explicit missingness. It preserves named aliases and earlier March/April 2/older-workbook records in a separate evidence table. Figures 22–24 and `outputs/paper_comparison` show matching metrics and a what-if substitution of 0.433 for TVAE Covertype. Actual method scores and all 21 previous figures remain unchanged; the borrowed-score tables are labeled hypothetical.
