# Run analyse.R: compare joint CTGAN/TVAE baselines with paper Table 6.
# Paper substitution is a what-if calculation, NEVER a corrected replication.

# 1. Read every historical baseline without mixing versions -----------------
paper_dir <- file.path(output_dir, "paper_comparison")
dir.create(paper_dir, showWarnings = FALSE)
paper_reference <- read.csv(file.path(root, "inputs", "paper_baselines.csv"), stringsAsFactors = FALSE)
stopifnot(nrow(paper_reference) == 16, !anyNA(paper_reference))

paper_read_pivot <- function(filename) {
  table <- if (filename == "April29.csv") pivot else read.csv(file.path(root, "inputs", filename), header = FALSE,
                    stringsAsFactors = FALSE, check.names = FALSE, na.strings = "")
  body <- 5:nrow(table)
  rows <- lapply(2:ncol(table), function(column) {
    values <- suppressWarnings(as.numeric(table[body, column]))
    keep <- is.finite(values) & table[body, 1] %in% c("ctgan", "tvae") & table[2, column] == "synthetic"
    if (!any(keep)) return(NULL)
    data.frame(source = filename, source_row = body[keep], source_col = column,
               dataset_label = table[3, column], method = table[body[keep], 1],
               metric = table[1, column], value = values[keep], stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}
paper_history <- do.call(rbind, lapply(c("Mar23.csv", "April02.csv", "April29.csv"), paper_read_pivot))
paper_workbook <- read.csv(file.path(root, "inputs", "older_workbook_baselines.csv"), stringsAsFactors = FALSE)
workbook_rows <- lapply(which(grepl("^test_", names(paper_workbook))), function(column) {
  keep <- is.finite(paper_workbook[[column]])
  if (!any(keep)) return(NULL)
  data.frame(source = "final_results.xlsx (older workbook)", source_row = NA_integer_, source_col = NA_integer_,
             dataset_label = paper_workbook$Dataset[keep], method = tolower(paper_workbook$Augment_Type[keep]),
             metric = names(paper_workbook)[column], value = paper_workbook[[column]][keep])
})
paper_history <- rbind(paper_history, do.call(rbind, workbook_rows))

# Adult aliases are reported individually for provenance, not counted as tasks.
dataset_lookup <- c(adult = "Adult", census = "Adult", census_kdd = "Adult",
                    covertype = "Covertype", credit = "Credit", intrusion = "Intrusion",
                    mnist12 = "MNIST12", mnist28 = "MNIST28", news = "News")
paper_history$dataset <- unname(dataset_lookup[tolower(paper_history$dataset_label)])
stopifnot(!anyNA(paper_history$dataset))
paper_history$endpoint <- ifelse(grepl("_max$", paper_history$metric), "max", "end")
paper_history$provenance_note <- ifelse(paper_history$source == "final_results.xlsx (older workbook)",
  "Older workbook: mixed/synthetic aggregation defect; not the final April record",
  ifelse(paper_history$dataset_label == "Census_Kdd", "Confirmed final routing to Adult/Census, not Census-KDD",
  ifelse(paper_history$dataset_label == "Census", "Adult alias; not an independent dataset", "Historical synthetic-only pivot")))

# Match the paper's metric exactly. Micro-F1 equals accuracy for single-label MNIST.
key <- paste(paper_reference$dataset, paper_reference$method)
match_row <- match(paste(paper_history$dataset, paper_history$method), key)
paper_history$paper_metric <- paper_reference$metric[match_row]
paper_history$matches_paper_metric <-
  (paper_history$paper_metric == "Binary F1" & grepl("^test_f1_binary_", paper_history$metric)) |
  (paper_history$paper_metric == "Macro-F1" & grepl("^test_f1_macro_", paper_history$metric)) |
  (paper_history$paper_metric == "Accuracy" & grepl("^test_(accuracy|f1_micro)_", paper_history$metric))
paper_history$paper_score <- ifelse(paper_history$matches_paper_metric, paper_reference$paper_score[match_row], NA_real_)
paper_history$difference_from_paper <- paper_history$value - paper_history$paper_score
write.csv(paper_history, file.path(paper_dir, "all_baseline_metrics_and_versions.csv"), row.names = FALSE)
paper_compatible <- paper_history[paper_history$matches_paper_metric, ]
write.csv(paper_compatible, file.path(paper_dir, "all_paper_metric_comparisons.csv"), row.names = FALSE)
paper_older_workbook_display <- paper_compatible[
  paper_compatible$source == "final_results.xlsx (older workbook)" &
    (!grepl("^MNIST", paper_compatible$dataset) | grepl("^test_accuracy_", paper_compatible$metric)),
  c("dataset_label", "method", "metric", "value", "paper_score", "difference_from_paper")]

# Check whether older pivot copies repeat April values, rather than add new trials.
april <- paper_history[paper_history$source == "April29.csv", ]
old_pivots <- paper_history[paper_history$source %in% c("Mar23.csv", "April02.csv"), ]
history_check <- merge(old_pivots, april,
                       by = c("dataset_label", "method", "metric"), suffixes = c("_older", "_April29"))
history_check$absolute_difference <- abs(history_check$value_older - history_check$value_April29)
write.csv(history_check, file.path(paper_dir, "historical_repetition_check.csv"), row.names = FALSE)

# 2. The final comparison includes all eight paper datasets, including gaps --
final_rows <- paper_compatible[paper_compatible$source == "April29.csv", ]
final_means <- aggregate(value ~ dataset + method + endpoint, final_rows, mean)
final_counts <- aggregate(value ~ dataset + method + endpoint, final_rows, length)
names(final_counts)[4] <- "aliases_averaged"
final_means <- merge(final_means, final_counts)
final_comparison <- merge(paper_reference, final_means, by = c("dataset", "method"), all.x = TRUE)
final_comparison$difference_from_paper <- final_comparison$value - final_comparison$paper_score
final_comparison$status <- ifelse(is.na(final_comparison$value),
  "No credible matching final dataset/result", "Matching metric; different evaluation protocol")
write.csv(final_comparison, file.path(paper_dir, "final_paper_metric_comparison.csv"), row.names = FALSE)

# Keep macro-F1 baselines separately: paper Adult F1 and MNIST accuracy differ.
write.csv(scores[scores$approach == "old", ], file.path(paper_dir, "final_macro_f1_baselines.csv"), row.names = FALSE)

# 3. Substitute only TVAE Covertype, leaving the actual results intact --------
paper_tvae_covertype <- paper_reference$paper_score[
  paper_reference$dataset == "Covertype" & paper_reference$method == "tvae"]
stopifnot(length(paper_tvae_covertype) == 1, paper_tvae_covertype == .433)
paper_borrowed <- scores
replace_row <- paper_borrowed$dataset == "Covertype" & paper_borrowed$method == "tvae"
stopifnot(sum(replace_row) == 1)
overrides <- data.frame(metric = c("macro_max", "macro_end"),
                       recorded = c(scores$macro_max[replace_row], scores$macro_end[replace_row]),
                       borrowed = paper_tvae_covertype,
                       assumption = "Paper classifier-average score substituted into local DNN summary; hypothetical")
paper_borrowed$macro_max[replace_row] <- paper_tvae_covertype
paper_borrowed$macro_end[replace_row] <- paper_tvae_covertype
write.csv(overrides, file.path(paper_dir, "hypothetical_overrides.csv"), row.names = FALSE)
write.csv(transform(paper_borrowed, substituted = replace_row),
          file.path(paper_dir, "hypothetical_scores.csv"), row.names = FALSE)

# Compare all methods, the old no-Gaussian family, and the requested RF/XGB family.
families <- list(All_methods = scores,
                 No_Gaussian = scores[!grepl("gauss", scores$method), ],
                 RF_XGB = scores[scores$y_synth %in% c("ctgan", "tvae", "rf", "xgb"), ])
paper_cases <- list()
paper_gains <- list()
paper_mixed <- list()
for (family in names(families)) {
  for (scenario in c("Recorded", "Paper TVAE Covertype: hypothetical")) {
    data <- families[[family]]
    if (scenario != "Recorded") {
      row <- data$dataset == "Covertype" & data$method == "tvae"
      data$macro_max[row] <- data$macro_end[row] <- paper_tvae_covertype
    }
    for (metric in c("macro_max", "macro_end")) {
      result <- focus_summary(data, metric, metric)
      names(result)[names(result) == "analysis"] <- "metric"
      names(result)[names(result) == "rf_xgb_mean"] <- "proposed_mean"
      paper_cases[[length(paper_cases) + 1]] <- cbind(family, scenario, result)
      gains <- dataset_comparison(data, metric)
      paper_gains[[length(paper_gains) + 1]] <- cbind(family, scenario, metric, gains)
    }
    # For continuity only: the previous mixed models applied to the altered table.
    for (labeler_random in c(FALSE, TRUE)) {
      model <- fit_model(data, include_labeler = labeler_random)
      test <- as.data.frame(summary(contrast(emmeans(model, ~ group),
        list(proposed_minus_joint = c(-1, .5, .5))), infer = TRUE))
      paper_mixed[[length(paper_mixed) + 1]] <- cbind(family, scenario, labeler_random,
                                                    singular = isSingular(model), test)
    }
  }
}
paper_sensitivity <- do.call(rbind, paper_cases)
write.csv(paper_sensitivity, file.path(paper_dir, "substitution_sensitivity.csv"), row.names = FALSE)
write.csv(do.call(rbind, paper_gains), file.path(paper_dir, "substitution_dataset_gains.csv"), row.names = FALSE)
write.csv(do.call(rbind, paper_mixed), file.path(paper_dir, "substitution_mixed_model_sensitivity.csv"), row.names = FALSE)

# RF/XGB comparisons matched to their own feature generator, before/after borrowing.
paper_generator_cases <- list()
for (scenario in c("Recorded", "Paper TVAE Covertype: hypothetical")) {
  data <- if (scenario == "Recorded") scores else paper_borrowed
  data <- data[data$y_synth %in% c("ctgan", "tvae", "rf", "xgb"), ]
  results <- do.call(rbind, lapply(c(0, 1), function(generator) {
    focus_summary(data[data$tvae == generator, ], "macro_max",
                  if (generator == 0) "CTGAN features" else "TVAE features")
  }))
  results$p_holm <- p.adjust(results$p_value, "holm")
  paper_generator_cases[[length(paper_generator_cases) + 1]] <- cbind(scenario, results)
}
paper_generators <- do.call(rbind, paper_generator_cases)
write.csv(paper_generators, file.path(paper_dir, "substitution_generator_comparisons.csv"), row.names = FALSE)

# 4. Figures: do not average different paper metrics together ----------------
plot_final <- rbind(
  data.frame(dataset = paper_reference$dataset, method = paper_reference$method,
             metric = paper_reference$metric, score = paper_reference$paper_score, source = "Paper"),
  data.frame(dataset = final_comparison$dataset, method = final_comparison$method,
             metric = final_comparison$metric, score = final_comparison$value,
             source = paste("Replication", final_comparison$endpoint)))
plot_final <- plot_final[is.finite(plot_final$score), ]
plot_final$panel <- paste(plot_final$dataset, plot_final$metric, sep = " / ")
p <- ggplot(plot_final, aes(method, score, color = source)) +
  geom_point(position = position_dodge(width = .4), size = 2.8) + facet_wrap(~ panel, scales = "free_y", ncol = 4) +
  labs(title = "Final CTGAN/TVAE baselines versus paper Table 6", x = NULL, y = "Paper-specific metric",
       caption = "Paper averages classifiers. Adult aliases averaged once; absent final results show paper points only. Axes vary.", color = NULL) +
  theme(legend.position = "bottom")
save_plot(p, "22_paper_baseline_comparison", 14, 8)
p <- ggplot(paper_sensitivity, aes(scenario, gain, color = scenario)) + geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower_95, ymax = upper_95), width = .15) +
  geom_hline(yintercept = 0, color = "grey60") + facet_grid(metric ~ family) +
  scale_x_discrete(labels = c("Recorded" = "Recorded", "Paper TVAE Covertype: hypothetical" = "Paper substitution")) +
  labs(title = "Borrowing the paper's TVAE Covertype score: sensitivity only", x = NULL,
       y = "Proposed-minus-joint macro-F1 gain", color = NULL,
       caption = "Four-dataset paired t intervals. Borrowed score comes from a different protocol; intervals are hypothetical.") +
  theme(legend.position = "none")
save_plot(p, "23_paper_substitution_sensitivity", 13, 8)
rf_cases <- paper_sensitivity[paper_sensitivity$family == "RF_XGB", ]
rf_cases$label <- paste(ifelse(rf_cases$scenario == "Recorded", "Recorded", "Paper substitution"), rf_cases$metric, sep = " / ")
p <- ggplot(rf_cases, aes(gain, label)) + geom_point(size = 3) +
  geom_segment(aes(x = lower_95, xend = upper_95, yend = label)) + geom_vline(xintercept = 0, color = "grey60") +
  labs(title = "RF/XGBoost versus joint baselines: borrowed-score check", x = "Macro-F1 gain with 95% interval", y = NULL,
       caption = "Recorded RF/XGB scores stay fixed; only the TVAE Covertype joint baseline changes to 0.433.")
save_plot(p, "24_rf_xgb_paper_substitution", 12, 5)

source(file.path(root, "plot_paper_comparison.R"), local = TRUE)
save_paper_replication_bars(root)

source(file.path(root, "plot_paper_substitution_boxplots.R"), local = TRUE)
save_paper_substitution_boxplots(root)

source(file.path(root, "plot_generator_comparison.R"), local = TRUE)
save_generator_comparison(root)

# 5. Concise Markdown/HTML report, plus complete CSV evidence ----------------
paper_format <- function(value) ifelse(is.na(value), "-", sprintf("%.4f", value))
paper_md_table <- function(table) {
  text <- as.data.frame(lapply(table, function(column) {
    if (is.numeric(column)) paper_format(column) else ifelse(is.na(column), "-", as.character(column))
  }), stringsAsFactors = FALSE)
  c(paste0("| ", paste(names(text), collapse = " | "), " |"),
    paste0("| ", paste(rep("---", ncol(text)), collapse = " | "), " |"),
    apply(text, 1, function(row) paste0("| ", paste(row, collapse = " | "), " |")), "")
}
paper_display <- final_comparison[c("dataset", "method", "metric", "paper_score", "endpoint", "value", "difference_from_paper")]
paper_display <- paper_display[order(paper_display$dataset, paper_display$method, paper_display$endpoint), ]
sensitivity_display <- paper_sensitivity[c("family", "scenario", "metric", "joint_mean", "proposed_mean", "gain", "relative_gain", "p_value")]
sensitivity_display$relative_gain <- sprintf("%.2f%%", 100 * sensitivity_display$relative_gain)
report_lines <- c("# Paper comparison and TVAE Covertype substitution", "",
  "All final joint CTGAN/TVAE baselines are compared using the paper's metric: binary F1 for Adult/Census-KDD/Credit, macro-F1 for Covertype/Intrusion, accuracy for MNIST, and R-squared for News. April MNIST micro-F1 equals accuracy in single-label classification. Adult, Census and the misrouted Census_Kdd entries are averaged within each method and counted once. No independent final Census-KDD result is available.", "",
  "[Paper Table 6 and Section 5.2](https://arxiv.org/pdf/1907.00503). The paper averages downstream classifiers; the local record uses DNNs and reports test maxima/final epochs. Matching a metric does not match the full protocol. No combined paper/local average is calculated across unlike metrics.", "",
  "## Final baseline comparison", "", paper_md_table(paper_display),
  "Missing rows are not zero. Credit is present only in the older workbook. Intrusion and News have no final baseline in these records. Adult binary F1 and MNIST accuracy must not be confused with the macro-F1 scores used in the synthesis-family analysis.", "",
  "## What if TVAE Covertype were 0.433?", "",
  "Only the TVAE Covertype baseline is replaced: 0.2972 (maximum) or 0.2672 (final epoch) becomes the paper's 0.433. The paper does not provide separate maximum/final-epoch values; applying the same number to both is an explicit assumption. All decoupled configurations, all other datasets, and CTGAN remain unchanged. Original recorded tables are not overwritten.", "",
  paper_md_table(sensitivity_display),
  "P values here are two-sided paired t calculations across four task differences. In the substituted scenario they are mechanical, hypothetical outputs of a hybrid table, not valid evidence from a new experiment or a leakage correction. Primary planned definitions and original broad-family results remain separate.", "",
  "## Same-feature-generator RF/XGBoost comparisons", "", paper_md_table(paper_generators),
  "Holm adjustment applies to the two generator comparisons within each scenario. CTGAN comparisons are unchanged by the TVAE substitution. Excluding Covertype leaves the recorded and hypothetical scenarios identical. Target-inclusion effects within decoupled methods are also unchanged.", "",
  "## All versions and named aliases", "",
  "The accompanying all_baseline_metrics_and_versions.csv retains every available CTGAN/TVAE metric in March 23, April 2, April 29 and the older final_results.xlsx extraction. all_paper_metric_comparisons.csv includes only matching paper metrics, with source, alias and provenance labels. Nonmatching macro/weighted/micro/loss values are retained as evidence but not compared to another metric.", "",
  sprintf("Historical repetition check: %d matched March/April 2 baseline metric cells; %d agree with April 29 within 1e-12. These repeated records do not count as independent trials.",
          nrow(history_check), sum(history_check$absolute_difference < 1e-12)), "",
  "The older workbook has no TVAE rows and includes a mixed/synthetic aggregation defect found in the audit. Its CTGAN values are historical evidence only and are not used in the hypothetical analysis.", "",
  "### Older-workbook comparisons: audit-flagged historical evidence", "",
  paper_md_table(paper_older_workbook_display),
  "Mixed-model substitutions are also saved in substitution_mixed_model_sensitivity.csv, with singularity flags; they do not replace the four-task sensitivity results. Known inconsistent Covertype test versions, test-fitted preprocessing, test-max selection, incomplete split/checkpoint provenance, and related MNIST tasks remain unresolved.", "",
  "## Figures", "",
  "Figures 22-28 are in ../figures as PNG and vector PDF. They are also included in ../legacy_artifacts_report.html. Input snapshots and hashes are in inputs/paper_comparison_manifest.json at the analysis-folder root.")
writeLines(report_lines, file.path(paper_dir, "PAPER_COMPARISON_AND_SENSITIVITY.md"))

# A small browsable HTML version, with readable tables rather than code output.
paper_escape <- function(text) {
  text <- gsub("&", "&amp;", text, fixed = TRUE)
  text <- gsub("<", "&lt;", text, fixed = TRUE)
  gsub(">", "&gt;", text, fixed = TRUE)
}
paper_html_table <- function(table) {
  values <- lapply(table, function(column) if (is.numeric(column)) paper_format(column) else ifelse(is.na(column), "-", as.character(column)))
  text <- as.data.frame(values, stringsAsFactors = FALSE)
  c(paste0("<table><tr><th>", paste(paper_escape(names(text)), collapse = "</th><th>"), "</th></tr>"),
    apply(text, 1, function(row) paste0("<tr><td>", paste(paper_escape(row), collapse = "</td><td>"), "</td></tr>")), "</table>")
}
html_lines <- c('<!doctype html><html><head><meta charset="utf-8"><title>Paper comparison and hypothetical sensitivity</title>',
  '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}h2{margin-top:35px}</style></head><body>',
  '<h1>CTGAN/TVAE replication and paper-score sensitivity</h1>',
  '<p>Adult counted once. Matching metrics, differing protocols. Missing final results remain missing. Paper-score substitution is hypothetical and is not a leakage-adjusted or corrected replication.</p>',
  '<p><a href="https://arxiv.org/pdf/1907.00503">Paper Table 6</a> | <a href="PAPER_COMPARISON_AND_SENSITIVITY.md">Full analysis and limits</a></p>',
  '<h2>Joint versus RF/XGBoost labeling</h2><img src="../figures/28_generator_matched_comparison.png" alt="CTGAN-based and TVAE-based comparison panels"><p><a href="../../GENERATOR_COMPARISON_NOTES.md">Figure notes and statistical details</a></p>',
  '<h2>Boxplot comparison: CTGAN/TVAE versus RF/XGBoost</h2><img src="../figures/26_rf_xgb_substitution_boxplots.png" alt="Recorded and hypothetical restricted-method boxplots"><h2>Six-method layout matching the original figure</h2><img src="../figures/27_six_method_substitution_boxplots.png" alt="Recorded and hypothetical six-method boxplots">',
  '<h2>Paper versus replication: labeled comparison</h2><img src="../figures/25_paper_replication_bars.png" alt="Paper, final epoch and test maximum scores for CTGAN and TVAE">',
  '<h2>Final paper-metric comparison</h2>', paper_html_table(paper_display),
  '<h2>Only TVAE Covertype replaced by 0.433</h2>', paper_html_table(sensitivity_display),
  '<h2>Generator-matched RF/XGBoost comparisons</h2>', paper_html_table(paper_generators))
html_lines <- c(html_lines,
  '<h2>Older workbook: audit-flagged historical comparisons</h2>',
  '<p>These CTGAN scores come from the older workbook with a mixed/synthetic aggregation defect. It has no TVAE rows. These values do not enter the final-record or borrowed-score analysis. Earlier March/April 2 pivot baselines repeat April 29 scores; all raw versions and aliases remain in the CSV evidence.</p>',
  paper_html_table(paper_older_workbook_display))
for (name in c("22_paper_baseline_comparison", "23_paper_substitution_sensitivity", "24_rf_xgb_paper_substitution")) {
  html_lines <- c(html_lines, paste0('<h2>', name, '</h2><img src="../figures/', name, '.png">'))
}
writeLines(c(html_lines, '</body></html>'), file.path(paper_dir, "paper_comparison_report.html"))
