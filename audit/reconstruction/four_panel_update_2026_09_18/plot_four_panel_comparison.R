# Four comparisons using the same four tasks and TVAE Covertype assumption.
save_four_panel_comparison <- function(root) {
  library(ggplot2)
  source_file <- file.path(root, "outputs", "method_scores.csv")
  scores <- read.csv(source_file)
  scores <- scores[scores$y_synth %in% c("ctgan", "tvae", "rf", "xgb"), ]
  changed <- scores$dataset == "Covertype" & scores$method == "tvae"
  stopifnot(nrow(scores) == 40, sum(changed) == 1)
  recorded_covertype <- scores$macro_max[changed]
  scores$macro_max[changed] <- .49
  scores$pipeline <- ifelse(scores$approach == "old", paste0(toupper(scores$y_synth), "_e2e"),
                           paste0(ifelse(scores$tvae == 0, "CTGAN", "TVAE"), "-", toupper(scores$y_synth)))
  panels <- c("CTGAN-based", "TVAE-based", "Generator training inputs", "End-to-end vs hybrid")

  # The top panels retain individual task/configuration observations.
  top <- data.frame(panel = ifelse(scores$tvae == 0, panels[1], panels[2]),
                    dataset = scores$dataset, x = scores$pipeline, score = scores$macro_max,
                    approach = ifelse(scores$approach == "old", "e2e", "hybrid"),
                    unit = "Task/configuration")
  hybrid <- scores[scores$approach == "new", ]

  # Compare target inclusion within hybrid methods only. Joint methods always
  # contain y, so including them would confound inclusion with the approach.
  inclusion <- aggregate(macro_max ~ dataset + has_y, hybrid, mean)
  bottom_left <- data.frame(panel = panels[3], dataset = inclusion$dataset,
                           x = ifelse(inclusion$has_y == 0, "Target omitted from generator training", "Target included in generator training"),
                           score = inclusion$macro_max, approach = "hybrid", unit = "Task mean")
  families <- aggregate(macro_max ~ dataset + approach, scores, mean)
  bottom_right <- data.frame(panel = panels[4], dataset = families$dataset,
                            x = ifelse(families$approach == "old", "e2e", "hybrid"),
                            score = families$macro_max,
                            approach = ifelse(families$approach == "old", "e2e", "hybrid"),
                            unit = "Task mean")
  observations <- rbind(top, bottom_left, bottom_right)
  observations$panel <- factor(observations$panel, levels = panels)
  observations$x <- factor(observations$x, levels = c("CTGAN_e2e", "CTGAN-RF", "CTGAN-XGB",
                              "TVAE_e2e", "TVAE-RF", "TVAE-XGB", "Target omitted from generator training", "Target included in generator training", "e2e", "hybrid"))
  observations$approach <- factor(observations$approach, levels = c("e2e", "hybrid"))
  stopifnot(nrow(observations) == 56, all(table(observations$panel) == c(20, 20, 8, 8)))

  top_means <- aggregate(score ~ panel + approach, top, mean)
  circles_top <- merge(unique(top[c("panel", "x", "approach")]), top_means,
                       by = c("panel", "approach"))
  bottom <- rbind(bottom_left, bottom_right)
  circles_bottom <- aggregate(score ~ panel + x + approach, bottom, mean)
  circles <- rbind(circles_top[c("panel", "x", "approach", "score")], circles_bottom)
  lines <- rbind(top_means[c("panel", "score")], circles_bottom[c("panel", "score")])

  # Calculate four paired task comparisons; configurations are not replicates.
  summary <- do.call(rbind, lapply(panels, function(panel) {
    data <- observations[observations$panel == panel, ]
    data$condition <- if (panel == panels[3]) as.character(data$x) else as.character(data$approach)
    task <- aggregate(score ~ dataset + condition, data, mean)
    reference <- if (panel == panels[3]) "Target omitted from generator training" else "e2e"
    comparison <- if (panel == panels[3]) "Target included in generator training" else "hybrid"
    left <- task[task$condition == reference, ]
    right <- task[task$condition == comparison, ]
    differences <- right$score[match(left$dataset, right$dataset)] - left$score
    stopifnot(length(differences) == 4, !anyNA(differences))
    test <- t.test(differences)
    data.frame(panel = panel, reference = reference, comparison = comparison,
               reference_mean = mean(left$score), comparison_mean = mean(right$score),
               gain = mean(differences), relative_gain = mean(differences) / mean(left$score),
               lower_95 = test$conf.int[1], upper_95 = test$conf.int[2], p_value = test$p.value)
  }))
  summary$p_holm <- p.adjust(summary$p_value, method = "holm")
  for (data_name in c("circles", "lines", "summary")) {
    data <- get(data_name)
    data$panel <- factor(data$panel, levels = panels)
    assign(data_name, data)
  }
  circles$x <- factor(circles$x, levels = levels(observations$x))
  colors <- c(e2e = "#9575B1", hybrid = "#4DB6AC")
  figure <- ggplot(observations, aes(x, score)) +
    geom_boxplot(aes(fill = approach), width = .7, alpha = .72,
                 outlier.shape = NA, linewidth = .5) +
    geom_point(aes(color = approach), alpha = .65, size = 2,
                 position = position_jitter(width = .1, height = 0, seed = 328)) +
    geom_hline(data = lines, aes(yintercept = score), color = "#D7191C",
                 linetype = "dashed", linewidth = .6) +
    geom_point(data = circles, aes(fill = approach), color = "#222222",
                 shape = 21, size = 3.1, stroke = .7) +
    geom_segment(data = summary, aes(x = 1, xend = 2, y = reference_mean, yend = comparison_mean),
                   inherit.aes = FALSE, color = "#D7191C", linewidth = .65,
                   arrow = grid::arrow(length = grid::unit(.12, "inches"))) +
    facet_wrap(~ panel, ncol = 2, scales = "free_x") +
    scale_x_discrete(labels = function(labels) {
      labels[labels == "Target omitted from generator training"] <- "X only"
      labels[labels == "Target included in generator training"] <- "X + y"
      labels
    }) +
    scale_fill_manual(values = colors, labels = c(e2e = "End-to-end", hybrid = "Hybrid")) +
    scale_color_manual(values = colors, labels = c(e2e = "End-to-end", hybrid = "Hybrid")) +
    scale_y_continuous(limits = c(.3, 1), breaks = seq(.3, 1, .1)) +
    labs(title = "Synthetic data utility", x = NULL, y = "Maximum macro-F1", fill = NULL, color = NULL) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold", size = 14, margin = margin(10, 0, 10, 0)),
          axis.text.x = element_text(size = 11), plot.title = element_text(size = 23),
          panel.spacing = grid::unit(1.4, "lines"), plot.margin = margin(16, 20, 12, 16))
  figure_dir <- file.path(root, "outputs", "figures")
  evidence_dir <- file.path(root, "outputs", "paper_comparison")
  name <- "29_four_panel_comparison"
  ggsave(file.path(figure_dir, paste0(name, ".png")), figure, width = 12, height = 9, dpi = 200, bg = "white")
  ggsave(file.path(figure_dir, paste0(name, ".pdf")), figure, width = 12, height = 9, device = cairo_pdf, bg = "white")
  write.csv(observations, file.path(evidence_dir, "four_panel_049_figure_data.csv"), row.names = FALSE)
  write.csv(summary, file.path(evidence_dir, "four_panel_049_summary.csv"), row.names = FALSE)
  notes <- c("# Four-panel comparison notes", "",
    "Figure: outputs/figures/29_four_panel_comparison.png (vector PDF alongside). Reproduce with `Rscript plot_four_panel_comparison.R`.", "",
    "## Labels and comparisons", "",
    "CTGAN_e2e and TVAE_e2e generate features and targets jointly. CTGAN-RF, CTGAN-XGB, TVAE-RF and TVAE-XGB first generate features with the named generator, then assign targets with RF or XGBoost trained on original data. RF and XGBoost do not generate the entire synthetic dataset.", "",
    "Top left: CTGAN-based pipelines. Top right: TVAE-based pipelines. Each joint box contains four task scores; each hybrid box contains eight task/configuration scores (four tasks times target included/omitted during feature-generator training).", "",
    "Bottom left (Generator training inputs): X only means target omitted from generator training; X + y means target included in generator training. Only hybrid pipelines are compared. Both final synthetic datasets contain targets assigned by RF/XGBoost. Each point averages CTGAN/TVAE feature generators and RF/XGBoost target predictors within one task and inclusion setting. Each box has four task means. Joint pipelines are excluded here because they always contain y and would confound the comparison. Both groups still use targets to train the RF/XGBoost predictor.", "",
    "Bottom right: e2e versus hybrid. Each point is a task mean. The e2e mean averages joint CTGAN and TVAE; the hybrid mean averages both feature generators, both target predictors and both inclusion settings. Each box has four task means, avoiding unequal configuration counts in this pooled comparison.", "",
    "Adult aliases are averaged and counted once. The four tasks are Adult, Covertype, MNIST12 and MNIST28. Categorical, PCA-GMM and Gaussian methods are excluded. Points are not independent experiment seeds.", "",
    "Purple: end-to-end; teal: hybrid. Red dashed lines and outlined circles show comparison-group means. Top-panel RF/XGB circles repeat their pooled hybrid mean. Bottom-panel circles show their respective condition means. Red arrows show comparison minus reference; boxplot center lines show medians.", "",
    "## Covertype assumption", "",
    sprintf("Only joint TVAE Covertype maximum macro-F1 is changed from %.10f to 0.49, following the user's requested sensitivity. The CTGAN paper reports 0.433 for TVAE Covertype; 0.49 is not a paper value or measured replication. All other scores and recorded source files remain unchanged. Final-epoch scores are not plotted.", recorded_covertype), "",
    "## Statistical summary", "",
    "Two-sided paired t tests use four task differences, not the number of plotted configurations. Holm correction covers the four comparisons in this figure. TVAE and pooled e2e results use the assumed 0.49 baseline; target-inclusion results are unaffected by it.", "",
    "| Panel | Reference mean | Comparison mean | Gain | Relative gain | p | Holm p |",
    "| --- | --- | --- | --- | --- | --- | --- |")
  rows <- apply(summary, 1, function(row) sprintf("| %s | %.4f | %.4f | %.4f | %.2f%% | %.4f | %.4f |",
    row["panel"], as.numeric(row["reference_mean"]), as.numeric(row["comparison_mean"]),
    as.numeric(row["gain"]), 100 * as.numeric(row["relative_gain"]), as.numeric(row["p_value"]), as.numeric(row["p_holm"])))
  notes <- c(notes, rows, "", "Confidence intervals and complete figure observations are in outputs/paper_comparison/four_panel_049_summary.csv and four_panel_049_figure_data.csv.", "",
    "## Trust limits", "",
    "Test-fitted preprocessing, test-maximum selection, inconsistent Covertype test versions, incomplete split/checkpoint provenance, related MNIST tasks, and absent independent seed runs remain unresolved. The assumed score is not a leakage correction. These comparisons do not establish leakage-free superiority.", "",
    paste0("Source: outputs/method_scores.csv; MD5: `", unname(tools::md5sum(source_file)), "`."))
  writeLines(notes, file.path(root, "FOUR_PANEL_COMPARISON_NOTES.md"))
  message("Saved four-panel figure, pipeline labels and comparison notes.")
}

if (sys.nframe() == 0) {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  save_four_panel_comparison(dirname(normalizePath(script)))
}
