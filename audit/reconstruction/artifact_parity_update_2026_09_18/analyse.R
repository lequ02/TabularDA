# Reanalysis of historical April 29 results. Run: Rscript --vanilla analyse.R
# Outputs are regenerated inside this folder; old project files are never edited.
# Optional argument: an alternative directory containing the two input CSVs.

required <- c("ggplot2", "lme4", "lmerTest", "emmeans", "car", "MASS")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install required packages: ", paste(missing, collapse = ", "))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(lmerTest))
suppressPackageStartupMessages(library(emmeans))

# 1. Paths and input checks -------------------------------------------------
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) != 1) stop("Run this script with Rscript.")
root <- dirname(normalizePath(sub("^--file=", "", script_arg)))
args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args)) args[1] else file.path(root, "inputs")
output_dir <- file.path(root, "outputs")
figure_dir <- file.path(output_dir, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

raw <- read.csv(file.path(input_dir, "April29_macro_max_mfa.csv"), stringsAsFactors = FALSE)
stopifnot(nrow(raw) == 132, !anyNA(raw), all(raw$macro_max >= 0 & raw$macro_max <= 1))
stopifnot(!anyDuplicated(raw[c("dataset", "method")]))
stopifnot(all(table(raw$dataset) == 22))

# The pivot has four metadata rows. Recover end scores without selecting test maxima.
pivot <- read.csv(file.path(input_dir, "April29.csv"), header = FALSE,
                  check.names = FALSE, stringsAsFactors = FALSE, na.strings = "")
raw$macro_end <- NA_real_
for (i in seq_len(nrow(raw))) {
  column <- which(pivot[1, ] == "test_f1_macro_end" &
                  pivot[2, ] == "synthetic" & pivot[3, ] == raw$dataset[i])
  row <- which(pivot[, 1] == raw$method[i])
  stopifnot(length(column) == 1, length(row) == 1)
  raw$macro_end[i] <- as.numeric(pivot[row, column])
  max_column <- which(pivot[1, ] == "test_f1_macro_max" &
                      pivot[2, ] == "synthetic" & pivot[3, ] == raw$dataset[i])
  stopifnot(abs(as.numeric(pivot[row, max_column]) - raw$macro_max[i]) < 1e-12)
}
stopifnot(!anyNA(raw$macro_end))

# 2. Count the three Adult aliases once ------------------------------------
# April code routed Census_Kdd to Census, and Census contains Adult data.
# Average the three aliases within each method, then weight each dataset equally.
raw$dataset[raw$dataset %in% c("Census", "Census_Kdd")] <- "Adult"
scores <- aggregate(cbind(macro_max, macro_end) ~
                      dataset + method + has_y + tvae + approach + y_synth,
                    data = raw, FUN = mean)
stopifnot(nrow(scores) == 88, all(table(scores$dataset) == 22))
restricted <- scores[!grepl("gauss", scores$method), ]
stopifnot(nrow(restricted) == 72)
write.csv(scores, file.path(output_dir, "method_scores.csv"), row.names = FALSE)

# 3. Describe the gain using datasets as the comparison units --------------
# Primary: include every recorded method. Restricted: historical Gaussian exclusion.
# Baseline is CTGAN and TVAE pooled equally, not a win over each separately.
dataset_comparison <- function(data, response) {
  means <- aggregate(data[[response]], data[c("dataset", "approach")], mean)
  names(means)[3] <- "score"
  result <- reshape(means, idvar = "dataset", timevar = "approach", direction = "wide")
  result$gain <- result$score.new - result$score.old
  result
}
cases <- list(All_methods_max = scores, No_Gaussian_max = restricted,
              All_methods_end = scores, No_Gaussian_end = restricted)
comparisons <- list()
summaries <- list()
for (name in names(cases)) {
  response <- if (grepl("_end$", name)) "macro_end" else "macro_max"
  comparison <- dataset_comparison(cases[[name]], response)
  test <- t.test(comparison$gain) # two-sided; four dataset differences
  comparisons[[name]] <- comparison
  summaries[[name]] <- data.frame(
    analysis = name, proposed_mean = mean(comparison$score.new),
    baseline_mean = mean(comparison$score.old), gain = mean(comparison$gain),
    relative_gain = mean(comparison$gain) / mean(comparison$score.old),
    lower_95 = test$conf.int[1], upper_95 = test$conf.int[2], p_value = test$p.value)
}
summary_table <- do.call(rbind, summaries)
rownames(summary_table) <- NULL
write.csv(summary_table, file.path(output_dir, "sensitivity_summary.csv"), row.names = FALSE)
write.csv(comparisons$All_methods_max, file.path(output_dir, "dataset_gains.csv"), row.names = FALSE)

# 4. Reproduce the old mixed model, and check labeling-method dependence -----
# These models are sensitivity analyses, not repeated-seed experiment estimates.
fit_model <- function(data, include_labeler = FALSE) {
  for (column in c("dataset", "tvae", "y_synth")) data[[column]] <- factor(data[[column]])
  data$group <- factor(ifelse(data$approach == "old", "joint",
                              ifelse(data$has_y == 0, "decoupled_no_y", "decoupled_with_y")),
                       levels = c("joint", "decoupled_no_y", "decoupled_with_y"))
  formula <- macro_max ~ group + (1 | dataset) + (1 | tvae)
  if (include_labeler) formula <- update(formula, . ~ . + (1 | y_synth))
  lmer(formula, data = data, REML = FALSE)
}
old_model <- fit_model(restricted)
labeler_model <- fit_model(restricted, include_labeler = TRUE)
contrasts <- list(with_y_minus_no_y = c(0, -1, 1),
                  decoupled_minus_joint = c(-1, .5, .5),
                  historical_third_contrast = c(1, -2, 1))
old_tests <- summary(contrast(emmeans(old_model, ~ group), contrasts, adjust = "sidak"),
                     infer = c(TRUE, TRUE))
labeler_test <- summary(contrast(emmeans(labeler_model, ~ group),
                                list(decoupled_minus_joint = c(-1, .5, .5))),
                       infer = c(TRUE, TRUE))
write.csv(as.data.frame(old_tests), file.path(output_dir, "historical_model_contrasts.csv"), row.names = FALSE)
write.csv(as.data.frame(labeler_test), file.path(output_dir, "labeler_model_contrast.csv"), row.names = FALSE)

# 5. Compare target inclusion only in observed, matched configurations ------
# Match dataset, feature generator and labeler; omit joint baselines.
decoupled <- scores[scores$approach == "new", ]
target_pairs <- reshape(decoupled[c("dataset", "tvae", "y_synth", "has_y", "macro_max")],
                        idvar = c("dataset", "tvae", "y_synth"), timevar = "has_y",
                        direction = "wide")
stopifnot(nrow(target_pairs) == 40, !anyNA(target_pairs))
target_pairs$gain_with_y <- target_pairs$macro_max.1 - target_pairs$macro_max.0
target_dataset <- aggregate(gain_with_y ~ dataset, target_pairs, mean)
target_test <- t.test(target_dataset$gain_with_y)
write.csv(target_pairs, file.path(output_dir, "target_inclusion_pairs.csv"), row.names = FALSE)
write.csv(target_dataset, file.path(output_dir, "target_inclusion_dataset_gains.csv"), row.names = FALSE)

# 6. Figures: always save both PNG and vector PDF ---------------------------
theme_set(theme_minimal(base_size = 12))
save_plot <- function(plot, name, width = 9, height = 5.5) {
  ggsave(file.path(figure_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 180, bg = "white")
  ggsave(file.path(figure_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}
dataset_order <- c("Adult", "Covertype", "MNIST12", "MNIST28")
scores$dataset <- factor(scores$dataset, levels = dataset_order)
comparison_plot <- rbind(
  transform(comparisons$All_methods_max, analysis = "All recorded methods"),
  transform(comparisons$No_Gaussian_max, analysis = "Gaussian excluded"))
comparison_long <- reshape(comparison_plot, varying = c("score.old", "score.new"),
                            v.names = "score", timevar = "approach",
                            times = c("Joint CTGAN/TVAE", "Decoupled"), direction = "long")
p <- ggplot(comparison_long, aes(dataset, score, color = approach, group = dataset)) +
  geom_line(color = "grey70") + geom_point(size = 3) + facet_wrap(~ analysis) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title = "Dataset means after counting Adult once", x = NULL, y = "Maximum test macro-F1", color = NULL,
       caption = "Historical scores; Gaussian exclusion changes the method-family comparison.") +
  theme(legend.position = "bottom")
save_plot(p, "01_dataset_comparison")

scores$configuration <- paste(ifelse(scores$tvae == 1, "TVAE", "CTGAN"),
                              scores$y_synth, ifelse(scores$has_y == 1, "with y", "without y"), sep = " / ")
p <- ggplot(scores, aes(dataset, reorder(configuration, macro_max, mean), fill = macro_max)) +
  geom_tile(color = "white") + geom_text(aes(label = sprintf("%.3f", macro_max)), size = 3) +
  scale_fill_gradient(low = "#f1f5f9", high = "#155e75", limits = c(0, 1)) +
  labs(title = "Every recorded synthesis configuration", x = NULL, y = NULL, fill = "Macro-F1",
       caption = "Feature generator / target method / target included during generator training. Scores are test maxima.")
save_plot(p, "02_configuration_heatmap", 11, 10)

p <- ggplot(comparison_plot, aes(dataset, gain, fill = analysis)) +
  geom_hline(yintercept = 0, color = "grey50") + geom_col(position = "dodge") +
  labs(title = "Decoupled minus joint generation by dataset", x = NULL, y = "Difference in macro-F1", fill = NULL) +
  theme(legend.position = "bottom")
save_plot(p, "03_dataset_gains")

p <- ggplot(target_pairs, aes(y_synth, gain_with_y, color = factor(tvae))) +
  geom_hline(yintercept = 0, color = "grey50") + geom_point(position = position_dodge(width = .35), size = 2.5) +
  facet_wrap(~ dataset) + labs(title = "Matched effect of including the training target", x = "Target predictor",
                               y = "With-y minus without-y macro-F1", color = "TVAE features",
                               caption = "Positive favors including y. Observed pairs only; these are not independent seed replicates.") +
  theme(legend.position = "bottom")
save_plot(p, "04_target_inclusion", 10, 7)

p <- ggplot(summary_table, aes(gain, reorder(analysis, gain))) +
  geom_vline(xintercept = 0, color = "grey50") +
  geom_segment(aes(x = lower_95, xend = upper_95, yend = analysis), linewidth = .7) +
  geom_point(size = 3) + labs(title = "Sensitivity of the average dataset gain", x = "Macro-F1 difference, with 95% interval", y = NULL,
                             caption = "Two-sided paired t intervals across four tasks; the two MNIST tasks are related.")
save_plot(p, "05_sensitivity_intervals", 10, 5)

diagnostic <- data.frame(fitted = fitted(old_model), residual = residuals(old_model))
p <- ggplot(diagnostic, aes(fitted, residual)) + geom_hline(yintercept = 0, color = "grey50") +
  geom_point() + labs(title = "Historical-model residuals", x = "Fitted macro-F1", y = "Residual",
                      caption = "Restricted method family. Diagnostics do not establish independence or absence of leakage.")
save_plot(p, "06_model_residuals")
p <- ggplot(diagnostic, aes(sample = residual)) + stat_qq() + stat_qq_line() +
  labs(title = "Historical-model residual normality", x = "Theoretical quantile", y = "Residual quantile")
save_plot(p, "07_model_qq")

# Original-data reference and selected interpretable pipelines.
original_columns <- which(pivot[1, ] == "test_f1_macro_max" & pivot[2, ] == "original")
original <- data.frame(dataset = as.character(unlist(pivot[3, original_columns])),
                       macro_max = as.numeric(unlist(pivot[pivot[, 1] %in% "none", original_columns])))
original$dataset[original$dataset %in% c("Census", "Census_Kdd")] <- "Adult"
original <- aggregate(macro_max ~ dataset, original, mean)
stopifnot(nrow(original) == 4, !anyNA(original))
selected <- scores[scores$approach == "old" |
                     (scores$tvae == 1 & scores$y_synth %in% c("rf", "xgb")),
                   c("dataset", "configuration", "macro_max")]
selected <- rbind(selected, transform(original, configuration = "Original data"))
write.csv(selected, file.path(output_dir, "original_and_selected_pipelines.csv"), row.names = FALSE)
p <- ggplot(selected, aes(macro_max, reorder(configuration, macro_max, mean))) +
  geom_point(aes(color = configuration == "Original data"), size = 3) +
  facet_wrap(~ dataset) + scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("#155e75", "#b45309"), guide = "none") +
  labs(title = "Selected pipelines and the original-data reference", x = "Maximum test macro-F1", y = NULL,
       caption = "Original-data training performs best on all four tasks. Historical evaluation limitations still apply.")
save_plot(p, "08_original_data_reference", 12, 7)

# Restore the final old report's exploratory tables and figures separately.
source(file.path(root, "legacy_artifacts.R"), local = TRUE)

# 7. Concise written results and revised abstract ---------------------------
# Keep the abstract aligned with the poster's original method selection/model.
poster_result <- summaries$No_Gaussian_max
target_p <- old_tests$p.value[old_tests$contrast == "with_y_minus_no_y"]
title <- "Decoupled Synthesis for Structured Data"
abstract <- sprintf(paste(
  "Generating useful synthetic tabular data remains challenging because datasets contain mixed variable types, imbalanced classes, and complex distributions.",
  "This study investigates whether separating feature generation from target generation improves synthetic data utility for downstream model prediction.",
  "We propose a decoupled pipeline that first generates synthetic features using a generative method, such as a conditional tabular generative adversarial network or tabular variational autoencoder.",
  "A target predictor, such as a random forest or gradient boosting model, is trained on the original data and then assigns labels to the synthetic features.",
  "We also examine whether including the target variable during feature generator training affects downstream performance.",
  "Evaluation uses four datasets, with deep neural networks trained exclusively on synthetic data and tested on real observations.",
  "Performance is measured using macro F1 scores.",
  "Compared with standard joint generation, the decoupled approach increases the average macro F1 score by %.4f, corresponding to a relative improvement of %.2f percent.",
  "Including the target during feature synthesis does not provide a statistically significant improvement, with a p value of %.4f.",
  "These findings support decoupled synthesis as a promising strategy for improving predictive utility and suggest that omitting targets during feature generation may simplify the pipeline.",
  "Future work will expand validation and investigate regression."),
  poster_result$gain, 100 * poster_result$relative_gain, target_p)
writeLines(c(paste0("# ", title), "", abstract), file.path(root, "revised_title_and_abstract.md"))

sink(file.path(output_dir, "statistical_results.txt"))
cat("COUNTING ADULT ONCE: HISTORICAL REANALYSIS\n\n")
print(summary_table, row.names = FALSE)
cat("\nDataset-level gains, all methods:\n"); print(comparisons$All_methods_max)
cat("\nDataset-level gains, historical Gaussian exclusion:\n"); print(comparisons$No_Gaussian_max)
cat("\nOriginal mixed model, Gaussian excluded, three Sidak-adjusted contrasts:\n"); print(old_tests)
cat("\nModel with labeling-method random effects:\n"); print(labeler_test)
cat("\nMixed-model singular fits:", isSingular(old_model), isSingular(labeler_model), "\n")
cat("\nObserved target-inclusion dataset differences, all labelers:\n"); print(target_dataset); print(target_test)
cat("\nLIMITATIONS: test maxima; test-fitted scaling; different Covertype test versions; incomplete run hashes; related MNIST tasks; no seed replication.\n")
cat("\nSession:\n"); print(sessionInfo())
sink()
cat("Saved tables, 18 PNG/PDF figure pairs, legacy HTML report, statistical results and revised abstract to", root, "\n")
