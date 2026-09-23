# Companion to analyse.R: artifact coverage of final_328_project.Rmd.
# Uses already-collapsed scores. These are exploratory historical models.
# Gaussian exclusion is retained here to match the old final report.

# 1. Tables and report helpers ---------------------------------------------
legacy_dir <- file.path(output_dir, "legacy_tables")
dir.create(legacy_dir, showWarnings = FALSE)
report_sections <- list()
record <- function(name, object) {
  report_sections[[name]] <<- capture.output(print(object))
  writeLines(report_sections[[name]], file.path(legacy_dir, paste0(name, ".txt")))
  if (is.data.frame(object) || is.matrix(object)) {
    write.csv(object, file.path(legacy_dir, paste0(name, ".csv")), row.names = TRUE)
  }
  invisible(object)
}
record_model <- function(name, model) {
  record(paste0(name, "_summary"), summary(model))
  if (inherits(model, "lm")) record(paste0(name, "_aliases"), alias(model))
  record(paste0(name, "_anova_type_I_or_Satterthwaite"),
         tryCatch(anova(model), error = function(e) conditionMessage(e)))
  record(paste0(name, "_anova_type_III"),
         tryCatch(car::Anova(model, type = "III"), error = function(e) conditionMessage(e)))
  if (inherits(model, "merMod")) {
    record(paste0(name, "_singular"), lme4::isSingular(model))
    record(paste0(name, "_wald_intervals"), confint(model, method = "Wald"))
  }
}

d <- restricted
for (column in c("dataset", "method", "has_y", "tvae", "approach")) d[[column]] <- factor(d[[column]])
labeler_order <- c("ctgan", "tvae", "categorical", "pca_gmm", "rf", "xgb")
d$y_synth <- factor(d$y_synth, levels = labeler_order)
record("01_input_preview", head(d))
record("02_factor_inventory", data.frame(
  column = names(d), unique_count = vapply(d, function(x) length(unique(x)), integer(1)),
  values = vapply(d, function(x) paste(sort(unique(x)), collapse = ", "), character(1))))
record("03_observed_combinations", as.data.frame(table(d$has_y, d$tvae, d$y_synth)))

# 2. Recreate the attached distribution plot, with current means ------------
# Mean circles are APPROACH means repeated at each category, as in the old plot.
# They are not individual labeler means. Boxes are descriptive, not independent runs.
distribution_plot <- function(data, label) {
  data$y_synth <- factor(data$y_synth, levels = c(labeler_order, "gauss"))
  means <- aggregate(macro_max ~ approach, data, mean)
  mapping <- unique(data[c("y_synth", "approach")])
  mean_points <- merge(mapping, means, by = "approach")
  old_mean <- means$macro_max[means$approach == "old"]
  new_mean <- means$macro_max[means$approach == "new"]
  record(paste0("04_approach_means_", label), means)
  colors <- c(old = "#9575CD", new = "#4DB6AC")
  ggplot(data, aes(y_synth, macro_max)) +
    geom_boxplot(aes(fill = approach), outlier.shape = NA, alpha = .7) +
    geom_point(aes(color = approach), alpha = .6, size = 1.7,
               position = position_jitter(width = .13, height = 0, seed = 328)) +
    geom_point(data = mean_points, aes(fill = approach), shape = 21,
               size = 3.5, color = "black") +
    geom_hline(yintercept = c(old_mean, new_mean), color = "firebrick", linetype = "dashed") +
    annotate("segment", x = 2, xend = 3, y = old_mean, yend = new_mean,
             arrow = grid::arrow(length = grid::unit(.12, "inches")), color = "red") +
    scale_fill_manual(values = colors) + scale_color_manual(values = colors) +
    labs(title = "Distribution of macro_max by y_synth Method",
         subtitle = sprintf("%s; approach means %.4f vs %.4f; relative gain %.2f%%",
                            label, old_mean, new_mean, 100 * (new_mean - old_mean) / old_mean),
         x = "Synthesis Method (y_synth)", y = "Maximum test macro-F1", fill = "Approach", color = "Approach",
         caption = "Adult counted once. Circles and dashed lines show approach means; points are configurations.") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
}
save_plot(distribution_plot(d, "Gaussian excluded"), "09_legacy_method_distribution", 10, 7)
save_plot(distribution_plot(scores, "All recorded methods"), "10_all_method_distribution", 11, 7)

# 3. The two old factor-exploration figures ---------------------------------
save_plot(ggplot(d, aes(y_synth, macro_max, color = tvae, group = tvae)) +
            geom_point() + stat_summary(fun = mean, geom = "line") + facet_wrap(~ has_y) +
            labs(title = "Labeler scores by feature generator and target inclusion", x = "Target method",
                 y = "Maximum test macro-F1", color = "TVAE features") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom"),
          "11_legacy_generator_factors", 10, 6)
save_plot(ggplot(d, aes(y_synth, macro_max, color = dataset, group = dataset)) +
            geom_point() + stat_summary(fun = mean, geom = "line") +
            facet_wrap(~ has_y + tvae, labeller = label_both) +
            labs(title = "Labeler scores across datasets and configurations", x = "Target method",
                 y = "Maximum test macro-F1", color = "Dataset") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom"),
          "12_legacy_dataset_factors", 11, 8)

# 4. Old model-development sequence, including failures/aliases -------------
# Missing cells are not fabricated. Saturated/rank-deficient models are diagnostics.
full <- lm(macro_max ~ has_y * tvae * dataset * y_synth, data = d)
blocking <- lm(macro_max ~ dataset + has_y * tvae * y_synth, data = d)
method_blocking <- lm(macro_max ~ dataset + method, data = d)
interaction <- lm(macro_max ~ y_synth * dataset + has_y + tvae, data = d)
record_model("05_full_interaction", full)
record_model("06_factor_blocking", blocking)
record_model("07_method_blocking", method_blocking)
record_model("08_dataset_labeler_interaction", interaction)

# Save the original four-panel LM diagnostic artifact in both formats.
save_diagnostics <- function(model, name, title) {
  for (format in c("png", "pdf")) {
    path <- file.path(figure_dir, paste0(name, ".", format))
    if (format == "png") png(path, width = 1800, height = 1400, res = 180, bg = "white")
    else pdf(path, width = 10, height = 1400 / 180)
    par(mfrow = c(2, 2), oma = c(0, 0, 2, 0))
    plot(model, which = c(1, 2, 3, 5), sub.caption = "")
    mtext(title, outer = TRUE)
    dev.off()
  }
}
save_diagnostics(interaction, "14_legacy_lm_diagnostics", "Exploratory dataset-by-labeler model")

# 5. Box-Cox profile, transformed model and its diagnostic panel ------------
profile <- MASS::boxcox(interaction, lambda = seq(-2, 2, .02), plotit = FALSE)
lambda <- profile$x[which.max(profile$y)]
profile_table <- data.frame(lambda = profile$x, log_likelihood = profile$y)
record("09_boxcox_profile", profile_table)
record("09_boxcox_selected_lambda", lambda)
save_plot(ggplot(profile_table, aes(lambda, log_likelihood)) + geom_line() +
            geom_vline(xintercept = lambda, linetype = "dashed", color = "firebrick") +
            labs(title = "Exploratory Box-Cox profile", subtitle = sprintf("Selected lambda = %.2f", lambda),
                 x = "Lambda", y = "Profile log likelihood",
                 caption = "Transformation selected on the same score table; not an independent validation."),
          "15_legacy_boxcox_profile")
d$transformed <- if (abs(lambda) < 1e-6) log(d$macro_max) else (d$macro_max^lambda - 1) / lambda
transformed_model <- lm(transformed ~ y_synth * dataset + has_y + tvae, data = d)
record_model("10_boxcox_model", transformed_model)
save_diagnostics(transformed_model, "16_legacy_transformed_diagnostics", "Box-Cox transformed exploratory model")

# 6. Mixed-model summaries, ANOVAs, intervals and group comparisons ---------
initial <- lmer(macro_max ~ approach * has_y + (1 | dataset) + (1 | tvae) + (1 | y_synth),
                data = d, REML = FALSE)
labeler_fit <- lmer(macro_max ~ y_synth + (1 | dataset) + (1 | tvae), data = d, REML = FALSE)
record_model("11_initial_mixed_model", initial)
record_model("12_combined_groups_labeler_random", labeler_model)
record_model("13_combined_groups_historical", old_model)
record_model("14_labeler_mixed_model", labeler_fit)
group_emm <- emmeans(old_model, ~ group)
record("15_group_marginal_means", as.data.frame(summary(group_emm)))
record("16_group_pairwise_tukey", as.data.frame(summary(pairs(group_emm, adjust = "tukey"), infer = TRUE)))
record("17_group_custom_sidak", as.data.frame(old_tests))

# Label names determine contrast weights; never rely on an implicit factor order.
labeler_emm <- emmeans(interaction, ~ y_synth)
joint_weights <- ifelse(labeler_order %in% c("ctgan", "tvae"), -.5, .25)
record("18_lm_decoupled_minus_joint", as.data.frame(summary(
  contrast(labeler_emm, list(decoupled_minus_joint = joint_weights)), infer = TRUE)))
conditional <- emmeans(interaction, ~ y_synth | has_y)
conditional_table <- as.data.frame(summary(conditional))
conditional_table$unsupported_combination <- with(conditional_table,
  has_y == "0" & y_synth %in% c("ctgan", "tvae"))
record("19_conditional_labeler_means", conditional_table)
conditional_contrast <- as.data.frame(summary(
  contrast(conditional, list(decoupled_minus_joint = joint_weights)), infer = TRUE))
conditional_contrast$uses_extrapolated_baseline <- conditional_contrast$has_y == "0"
record("20_conditional_approach_contrasts", conditional_contrast)
target_emm <- emmeans(interaction, ~ has_y | tvae)
record("21_target_means_by_generator", as.data.frame(summary(target_emm)))
record("22_target_with_minus_without", as.data.frame(summary(
  contrast(target_emm, list(with_y_minus_without_y = c(-1, 1))), infer = TRUE)))

group_weights <- list(
  rf_xgb_minus_joint = c(-.5, -.5, 0, 0, .5, .5),
  categorical_pca_minus_joint = c(-.5, -.5, .5, .5, 0, 0),
  rf_xgb_minus_categorical_pca = c(0, 0, -.5, -.5, .5, .5))
bonus <- as.data.frame(summary(contrast(emmeans(labeler_fit, ~ y_synth), group_weights,
                                       adjust = "sidak"), infer = TRUE))
record("23_labeler_group_contrasts", bonus)
save_plot(ggplot(bonus, aes(estimate, contrast)) + geom_vline(xintercept = 0, color = "grey60") +
            geom_segment(aes(x = lower.CL, xend = upper.CL, yend = contrast)) + geom_point(size = 3) +
            labs(title = "Exploratory contrasts between labeler groups", x = "Mean macro-F1 difference", y = NULL,
                 caption = "Group averages, not sums; three Sidak-adjusted contrasts. Historical method family."),
          "18_legacy_labeler_group_contrasts", 11, 5)

# 7. Marginal-mean figures from the original LM interaction model -----------
emm_table <- as.data.frame(summary(labeler_emm))
emm_table$approach <- ifelse(emm_table$y_synth %in% c("ctgan", "tvae"), "old", "new")
record("24_labeler_marginal_means", emm_table)
save_plot(ggplot(emm_table, aes(y_synth, emmean, color = approach)) +
            geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = .15) + geom_point(size = 3) +
            geom_line(aes(group = approach)) + scale_color_manual(values = c(old = "orangered", new = "steelblue")) +
            labs(title = "Estimated marginal means for target methods", x = "Target method", y = "Adjusted macro-F1",
                 subtitle = "Historical dataset-by-labeler LM; exploratory model-based estimates", color = "Approach") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom"),
          "13_legacy_labeler_marginal_means", 10, 6)
target_table <- as.data.frame(summary(target_emm))
save_plot(ggplot(target_table, aes(has_y, emmean)) +
            geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = .12) + geom_point(size = 3) +
            facet_wrap(~ tvae, labeller = label_both) +
            labs(title = "Model-based target-inclusion means by feature generator", x = "Target included (has_y)",
                 y = "Adjusted macro-F1", caption = "LM extrapolates to unsupported joint-baseline combinations; use matched pairs for observed effects."),
          "17_legacy_target_marginal_means", 11, 6)

# 8. Browsable report: figures plus every printed table/model ----------------
escape_html <- function(text) {
  text <- gsub("&", "&amp;", text, fixed = TRUE)
  text <- gsub("<", "&lt;", text, fixed = TRUE)
  gsub(">", "&gt;", text, fixed = TRUE)
}
html <- c('<!doctype html><html><head><meta charset="utf-8"><title>Historical artifact reanalysis</title>',
          '<style>body{font:16px sans-serif;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}pre{overflow:auto;background:#f5f5f5;padding:15px;font-size:13px}h2{margin-top:40px}</style></head><body>',
          '<h1>Old-report artifacts with Adult counted once</h1>',
          '<p>Four nominal tasks. Historical Gaussian exclusion applies to legacy models. These exploratory models do not override the dataset-level sensitivity results in README.md or establish absence of leakage.</p>',
          '<p>Rank deficiency, saturated models, singular fits and unavailable tests are recorded rather than concealed. Type III LM tests retain the old treatment coding. Mixed-model intervals here are Wald intervals; variance-component bounds are unavailable under that method. Unsupported conditional means are flagged. Target contrasts use with-y minus without-y; labeler-group contrasts use averages rather than the old unnormalized sums.</p>')
for (figure in sort(list.files(figure_dir, pattern = "\\.png$"))) {
  html <- c(html, paste0('<h2>', escape_html(figure), '</h2><img src="figures/', figure, '">'))
}
for (name in names(report_sections)) {
  html <- c(html, paste0('<h2>', name, '</h2><pre>',
                        paste(escape_html(report_sections[[name]]), collapse = "\n"), '</pre>'))
}
writeLines(c(html, '</body></html>'), file.path(output_dir, "legacy_artifacts_report.html"))
