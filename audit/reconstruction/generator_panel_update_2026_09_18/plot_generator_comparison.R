# Compare joint generation with RF/XGBoost labeling for each feature generator.
# All explanatory details are written to Markdown instead of the figure.
save_generator_comparison <- function(root) {
  library(ggplot2)
  source_file <- file.path(root, "outputs", "method_scores.csv")
  scores <- read.csv(source_file)
  scores <- scores[scores$y_synth %in% c("ctgan", "tvae", "rf", "xgb"), ]
  changed <- scores$dataset == "Covertype" & scores$method == "tvae"
  stopifnot(nrow(scores) == 40, sum(changed) == 1)
  recorded_covertype <- scores$macro_max[changed]
  scores$macro_max[changed] <- 0.49

  # Separate feature generators; pool only the two target-inclusion settings.
  scores$generator <- factor(ifelse(scores$tvae == 0, "CTGAN-based", "TVAE-based"),
                             levels = c("CTGAN-based", "TVAE-based"))
  scores$method_label <- factor(c(ctgan = "CTGAN", tvae = "TVAE", rf = "RF", xgb = "XGBoost")[scores$y_synth],
                               levels = c("CTGAN", "TVAE", "RF", "XGBoost"))
  scores$approach <- factor(scores$approach, levels = c("old", "new"))
  counts <- with(scores, table(generator, method_label))
  stopifnot(counts["CTGAN-based", "CTGAN"] == 4,
            counts["TVAE-based", "TVAE"] == 4,
            all(counts[, c("RF", "XGBoost")] == 8))

  # Four task pairs determine the means and statistical summary, not 20 replicates.
  per_task <- aggregate(macro_max ~ generator + dataset + approach, scores, mean)
  means <- aggregate(macro_max ~ generator + approach, per_task, mean)
  mean_points <- merge(unique(scores[c("generator", "method_label", "approach")]),
                       means, by = c("generator", "approach"))
  summary <- do.call(rbind, lapply(levels(scores$generator), function(generator) {
    task <- per_task[per_task$generator == generator, ]
    joint <- task[task$approach == "old", ]
    decoupled <- task[task$approach == "new", ]
    differences <- decoupled$macro_max[match(joint$dataset, decoupled$dataset)] - joint$macro_max
    test <- t.test(differences)
    data.frame(generator = generator, tasks = length(differences),
               joint_mean = mean(joint$macro_max), decoupled_mean = mean(decoupled$macro_max),
               gain = mean(differences), relative_gain = mean(differences) / mean(joint$macro_max),
               lower_95 = test$conf.int[1], upper_95 = test$conf.int[2], p_value = test$p.value)
  }))
  summary$p_holm <- p.adjust(summary$p_value, method = "holm")
  colors <- c(old = "#9575B1", new = "#4DB6AC")
  figure <- ggplot(scores, aes(method_label, macro_max)) +
    geom_boxplot(aes(fill = approach), alpha = .72, outlier.shape = NA,
                 width = .7, linewidth = .5) +
    geom_point(aes(color = approach), size = 2, alpha = .65,
                 position = position_jitter(width = .10, height = 0, seed = 328)) +
    geom_hline(data = means, aes(yintercept = macro_max), color = "#D7191C",
                 linetype = "dashed", linewidth = .65) +
    geom_point(data = mean_points, aes(fill = approach), shape = 21,
                 color = "#222222", size = 3.2, stroke = .7) +
    geom_segment(data = summary, aes(x = 1, xend = 2, y = joint_mean, yend = decoupled_mean),
                   inherit.aes = FALSE, color = "#D7191C", linewidth = .65,
                   arrow = grid::arrow(length = grid::unit(.12, "inches"))) +
    facet_wrap(~ generator, nrow = 1, scales = "free_x") +
    scale_fill_manual(values = colors, labels = c(old = "Joint", new = "Decoupled")) +
    scale_color_manual(values = colors, labels = c(old = "Joint", new = "Decoupled")) +
    scale_y_continuous(limits = c(.3, 1), breaks = seq(.3, 1, .1)) +
    labs(title = "Joint versus RF/XGBoost labeling", x = NULL, y = "Maximum macro-F1",
         fill = NULL, color = NULL) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank(),
          strip.text = element_text(size = 15, face = "bold", margin = margin(12, 0, 12, 0)),
          plot.title = element_text(size = 22), plot.margin = margin(16, 20, 12, 16),
          panel.spacing = grid::unit(1.5, "lines"))
  figure_dir <- file.path(root, "outputs", "figures")
  evidence_dir <- file.path(root, "outputs", "paper_comparison")
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  name <- "28_generator_matched_comparison"
  ggsave(file.path(figure_dir, paste0(name, ".png")), figure,
         width = 11, height = 6, dpi = 200, bg = "white")
  ggsave(file.path(figure_dir, paste0(name, ".pdf")), figure,
         width = 11, height = 6, device = cairo_pdf, bg = "white")
  write.csv(scores, file.path(evidence_dir, "generator_matched_049_figure_data.csv"), row.names = FALSE)
  write.csv(per_task, file.path(evidence_dir, "generator_matched_049_task_means.csv"), row.names = FALSE)
  write.csv(summary, file.path(evidence_dir, "generator_matched_049_summary.csv"), row.names = FALSE)

  # Keep interpretation and provenance beside the reproducible script.
  notes <- c("# Generator-matched figure notes", "",
    "Figure: outputs/figures/28_generator_matched_comparison.png (vector PDF alongside).", "",
    "## Layout and observations", "",
    "The left panel compares joint CTGAN with RF/XGBoost targets assigned to CTGAN-generated features. The right panel compares joint TVAE with RF/XGBoost targets assigned to TVAE-generated features. RF and XGBoost are target predictors, not feature generators. Categorical, PCA-GMM and Gaussian methods are excluded.", "",
    "Four tasks are included: Adult, Covertype, MNIST12 and MNIST28. The three Adult aliases are averaged within configuration and counted as one task. Each joint baseline has four points. Each RF/XGBoost box has eight points: four tasks times two generator-training settings (target included versus omitted). These are task/configuration summaries, not independent random-seed runs.", "",
    "Purple denotes joint synthesis; teal denotes decoupled labeling. Red dashed lines and outlined circles represent panel-specific approach means. Circles for RF and XGBoost repeat the pooled decoupled mean, rather than each labeler's individual mean. The red arrow shows the difference between approach means. Boxplot center lines are medians, not those means.", "",
    "## User-specified Covertype sensitivity", "",
    sprintf("Only the joint TVAE Covertype maximum macro-F1 changes from %.10f to **0.49**, as requested. Joint CTGAN Covertype and all RF/XGBoost scores remain recorded values. The final-epoch column is unchanged and is not plotted.", recorded_covertype), "",
    "The 0.49 value is a user-specified hypothetical assumption. It is not the CTGAN paper's reported TVAE Covertype value (0.433), not a measured replication, and not a leakage adjustment. Earlier 0.433 analyses and original recorded results are preserved.", "",
    "## Statistical summary", "",
    "Configurations are averaged within each approach and task before averaging four task differences. Two-sided paired t tests use four differences (three degrees of freedom). Holm adjustment covers the two feature-generator comparisons. The TVAE statistics are mechanical sensitivity outputs from an assumed baseline, not results of a new experiment.", "",
    "| Features | Joint mean | RF/XGB mean | Gain | Relative gain | 95% interval | p | Holm p |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |")
  rows <- apply(summary, 1, function(row) sprintf("| %s | %.4f | %.4f | %.4f | %.2f%% | [%.4f, %.4f] | %.4f | %.4f |",
    row["generator"], as.numeric(row["joint_mean"]), as.numeric(row["decoupled_mean"]),
    as.numeric(row["gain"]), 100 * as.numeric(row["relative_gain"]),
    as.numeric(row["lower_95"]), as.numeric(row["upper_95"]),
    as.numeric(row["p_value"]), as.numeric(row["p_holm"])))
  notes <- c(notes, rows, "", "## Trust limits and reproduction", "",
    "Known test-fitted preprocessing, test-maximum selection, inconsistent Covertype test versions, incomplete checkpoint/split provenance, related MNIST tasks, and absent independent seed runs remain unresolved. Separating generators and removing Adult duplication improves the comparison structure but does not remove leakage. This figure cannot establish leakage-free superiority.", "",
    "Run `Rscript plot_generator_comparison.R` from this analysis folder to reproduce the figure, source data, task means, statistical summary and these notes. The full analyse.R workflow also generates them.", "",
    paste0("Input: outputs/method_scores.csv; MD5: `", unname(tools::md5sum(source_file)), "`."), "",
    "Figure-specific CSVs are in outputs/paper_comparison and start with generator_matched_049_.")
  writeLines(notes, file.path(root, "GENERATOR_COMPARISON_NOTES.md"))
  message("Saved generator-separated figure and Markdown notes with TVAE Covertype = 0.49.")
}

if (sys.nframe() == 0) {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  save_generator_comparison(dirname(normalizePath(script)))
}
