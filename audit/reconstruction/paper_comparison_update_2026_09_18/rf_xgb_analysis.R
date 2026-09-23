# Run analyse.R, which sources this companion after collapsing the Adult aliases.
# RF/XGB generate targets; their features still come from CTGAN or TVAE.
# No categorical, PCA-GMM or Gaussian configuration is included here.

# 1. Select the requested comparison ---------------------------------------
focus <- scores[scores$y_synth %in% c("ctgan", "tvae", "rf", "xgb"), ]
stopifnot(nrow(focus) == 40, all(table(focus$dataset) == 10))
focus_dir <- file.path(output_dir, "rf_xgb")
dir.create(focus_dir, showWarnings = FALSE)
write.csv(focus, file.path(focus_dir, "selected_method_scores.csv"), row.names = FALSE)

# Four datasets, each with two joint baselines and eight RF/XGB configurations.
# Average configurations within dataset first; datasets then get equal weight.
focus_summary <- function(data, metric, name) {
  comparison <- dataset_comparison(data, metric)
  test <- t.test(comparison$gain)
  data.frame(analysis = name, datasets = nrow(comparison),
             joint_mean = mean(comparison$score.old), rf_xgb_mean = mean(comparison$score.new),
             gain = mean(comparison$gain), relative_gain = mean(comparison$gain) / mean(comparison$score.old),
             lower_95 = test$conf.int[1], upper_95 = test$conf.int[2], p_value = test$p.value)
}
focus_summaries <- rbind(
  focus_summary(focus, "macro_max", "Maximum test score"),
  focus_summary(focus, "macro_end", "Final-epoch score"),
  focus_summary(focus[focus$dataset != "Covertype", ], "macro_max", "Maximum, Covertype excluded"),
  focus_summary(focus[focus$dataset != "Covertype", ], "macro_end", "Final epoch, Covertype excluded"))
focus_gains <- dataset_comparison(focus, "macro_max")
write.csv(focus_summaries, file.path(focus_dir, "summary.csv"), row.names = FALSE)
write.csv(focus_gains, file.path(focus_dir, "dataset_gains.csv"), row.names = FALSE)

# 2. Resolve the pooled result into individual labelers and generators -------
method_means <- aggregate(cbind(macro_max, macro_end) ~ dataset + y_synth, focus, mean)
write.csv(method_means, file.path(focus_dir, "labeler_means.csv"), row.names = FALSE)
generator_tests <- do.call(rbind, lapply(c(0, 1), function(generator) {
  result <- focus_summary(focus[focus$tvae == generator, ], "macro_max",
                          if (generator == 0) "CTGAN features versus joint CTGAN" else "TVAE features versus joint TVAE")
  result
}))
# These two secondary comparisons share data; report Holm-adjusted p values.
generator_tests$p_holm <- p.adjust(generator_tests$p_value, method = "holm")
write.csv(generator_tests, file.path(focus_dir, "generator_matched_comparisons.csv"), row.names = FALSE)
generator_no_covertype <- do.call(rbind, lapply(c(0, 1), function(generator) {
  focus_summary(focus[focus$tvae == generator & focus$dataset != "Covertype", ], "macro_max",
                if (generator == 0) "CTGAN features, no Covertype" else "TVAE features, no Covertype")
}))
generator_no_covertype$p_holm <- p.adjust(generator_no_covertype$p_value, method = "holm")
write.csv(generator_no_covertype, file.path(focus_dir, "generator_matched_no_covertype.csv"), row.names = FALSE)

# Leave one dataset out at a time, exposing whether one task drives the gain.
leave_one_out <- do.call(rbind, lapply(unique(as.character(focus$dataset)), function(dataset) {
  focus_summary(focus[focus$dataset != dataset, ], "macro_max", paste("Without", dataset))
}))
write.csv(leave_one_out, file.path(focus_dir, "leave_one_dataset_out.csv"), row.names = FALSE)

# 3. Historical mixed-model sensitivity, restricted to the selected methods --
focus_model <- fit_model(focus)
focus_labeler_model <- fit_model(focus, include_labeler = TRUE)
focus_mixed <- as.data.frame(summary(contrast(emmeans(focus_model, ~ group),
  list(rf_xgb_minus_joint = c(-1, .5, .5))), infer = TRUE))
focus_labeler_mixed <- as.data.frame(summary(contrast(emmeans(focus_labeler_model, ~ group),
  list(rf_xgb_minus_joint = c(-1, .5, .5))), infer = TRUE))
write.csv(focus_mixed, file.path(focus_dir, "historical_mixed_model_contrast.csv"), row.names = FALSE)
write.csv(focus_labeler_mixed, file.path(focus_dir, "labeler_random_model_contrast.csv"), row.names = FALSE)
writeLines(c("Historical-model sensitivity; not independent seed replicates.",
             capture.output(summary(focus_model)), capture.output(summary(focus_labeler_model)),
             paste("Singular:", isSingular(focus_model), isSingular(focus_labeler_model))),
           file.path(focus_dir, "mixed_model_summaries.txt"))

# 4. Requested distribution plot and per-dataset comparison -----------------
plot_focus <- focus
plot_focus$y_synth <- factor(plot_focus$y_synth, levels = c("ctgan", "tvae", "rf", "xgb"))
approach_means <- aggregate(macro_max ~ approach, plot_focus, mean)
mean_points <- merge(unique(plot_focus[c("y_synth", "approach")]), approach_means, by = "approach")
primary <- focus_summaries[1, ]
colors <- c(old = "#9575CD", new = "#4DB6AC")
p <- ggplot(plot_focus, aes(y_synth, macro_max)) +
  geom_boxplot(aes(fill = approach), outlier.shape = NA, alpha = .7) +
  geom_point(aes(color = approach), alpha = .65,
             position = position_jitter(width = .12, height = 0, seed = 328)) +
  geom_point(data = mean_points, aes(fill = approach), shape = 21, size = 3.5, color = "black") +
  geom_hline(yintercept = approach_means$macro_max, color = "firebrick", linetype = "dashed") +
  annotate("segment", x = 2, xend = 3, y = primary$joint_mean, yend = primary$rf_xgb_mean,
           arrow = grid::arrow(length = grid::unit(.12, "inches")), color = "red") +
  scale_fill_manual(values = colors) + scale_color_manual(values = colors) +
  labs(title = "Joint CTGAN/TVAE versus RF/XGBoost targets",
       subtitle = sprintf("Approach means %.4f vs %.4f; gain %.2f%%; dataset-level p = %.3f",
                          primary$joint_mean, primary$rf_xgb_mean, 100 * primary$relative_gain, primary$p_value),
       x = "Target generation method", y = "Maximum test macro-F1", color = "Approach", fill = "Approach",
       caption = "Adult counted once. RF/XGB labels use CTGAN or TVAE features. Circles show approach means.") +
  theme(legend.position = "bottom")
save_plot(p, "19_rf_xgb_distribution", 11, 7)
p <- ggplot(method_means, aes(y_synth, macro_max, color = y_synth)) + geom_point(size = 3) +
  facet_wrap(~ dataset) + scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c(ctgan = "#9575CD", tvae = "#5e3c99", rf = "#4DB6AC", xgb = "#087f8c")) +
  labs(title = "Joint baselines and RF/XGBoost labeling by dataset", x = "Target method", y = "Mean maximum test macro-F1",
       caption = "RF/XGB means average feature generators and target-inclusion settings. Covertype test versions differ.") +
  theme(legend.position = "none")
save_plot(p, "20_rf_xgb_dataset_comparison", 11, 7)
p <- ggplot(focus_summaries, aes(gain, analysis)) + geom_vline(xintercept = 0, color = "grey60") +
  geom_segment(aes(x = lower_95, xend = upper_95, yend = analysis)) + geom_point(size = 3) +
  labs(title = "RF/XGBoost comparison: sensitivity intervals", x = "Macro-F1 gain with 95% interval", y = NULL,
       caption = "Paired dataset t intervals. Covertype exclusion checks known test-version inconsistency, not its low score.")
save_plot(p, "21_rf_xgb_sensitivity", 12, 5)

# 5. Trace the TVAE low point to the paper's Covertype result ----------------
# Paper values are from the user's Table 6, verified against arXiv 1907.00503.
paper_comparison <- data.frame(method = c("CTGAN", "TVAE"), paper_macro_f1 = c(.324, .433))
local_covertype <- focus[focus$dataset == "Covertype" & focus$approach == "old", ]
paper_comparison$local_macro_max <- local_covertype$macro_max[match(tolower(paper_comparison$method), local_covertype$method)]
paper_comparison$local_macro_end <- local_covertype$macro_end[match(tolower(paper_comparison$method), local_covertype$method)]
write.csv(paper_comparison, file.path(focus_dir, "covertype_paper_context.csv"), row.names = FALSE)

as_markdown_table <- function(table) {
  lines <- capture.output(print(table, row.names = FALSE, digits = 5))
  c("```text", lines, "```", "")
}
report <- c("# Joint CTGAN/TVAE versus RF/XGBoost labeling", "",
  "Adult aliases are counted once. Only ctgan, tvae, rf and xgb target methods are included: 40 scores, comprising two joint baselines and eight decoupled configurations per dataset. RF/XGB are target predictors; synthetic features still come from CTGAN or TVAE. The family comparison averages both feature generators and both target-inclusion settings equally within each dataset, then weights the four datasets equally.", "",
  "## Is the low TVAE point consistent with the paper?", "",
  "It is Covertype, not a separate failed seed. Our TVAE macro-F1 is 0.2972 at its test maximum and 0.2672 at the final epoch. The paper reports 0.433 for TVAE and 0.324 for CTGAN on Covertype. Weak performance on this task is qualitatively consistent, but our TVAE score is substantially lower and the CTGAN/TVAE ranking reverses. The paper averages multiple downstream classifiers; our historical score uses a DNN and a maximum over test evaluations. These are not controlled replications of the same protocol. The point should be investigated, not removed just for being low. [Paper: Table 6 and evaluation protocol](https://arxiv.org/pdf/1907.00503).", "",
  as_markdown_table(paper_comparison),
  "## Family comparison and sensitivity", "", as_markdown_table(focus_summaries),
  "Tests are two-sided paired t tests across dataset differences. Primary test-maximum comparison is unadjusted. End scores and exclusions are exploratory sensitivities. Covertype exclusion is reported because the historical methods used inconsistent test versions. It cannot certify that the remaining data are leakage-free.", "",
  "## Individual labeler means by dataset", "", as_markdown_table(method_means),
  "## Same-feature-generator comparisons", "", as_markdown_table(generator_tests),
  "Each comparison averages RF and XGB and both target-inclusion settings with its own feature generator. Two secondary p values are Holm-adjusted. This prevents the pooled comparison from hiding a much stronger CTGAN baseline deficit.", "",
  "The same comparisons without Covertype (Holm adjustment within this additional sensitivity family):", "",
  as_markdown_table(generator_no_covertype),
  "## Leave-one-dataset-out sensitivity", "", as_markdown_table(leave_one_out),
  "## Mixed-model sensitivity", "",
  "Original random-effects structure:", as_markdown_table(focus_mixed),
  "Labeler random effect retained:", as_markdown_table(focus_labeler_mixed),
  "These configurations are not independent experiment seeds. Small numbers of dataset/generator/labeler levels and any singular fits limit interpretation.", "",
  "## Figures", "",
  "See ../figures/19_rf_xgb_distribution.png, 20_rf_xgb_dataset_comparison.png and 21_rf_xgb_sensitivity.png; PDF versions are alongside them. All three are also included in ../legacy_artifacts_report.html.", "",
  "## Interpretation limits", "",
  "Selecting RF/XGB after seeing their strong scores makes this an exploratory restricted-family result. Known test-fitted preprocessing, test-maximum selection, uncertain split/checkpoint provenance, related MNIST tasks and differing Covertype test versions remain. A positive result here does not prove leakage-free superiority. The original all-methods analysis and abstract are retained unchanged.")
writeLines(report, file.path(focus_dir, "RF_XGB_ANALYSIS.md"))
