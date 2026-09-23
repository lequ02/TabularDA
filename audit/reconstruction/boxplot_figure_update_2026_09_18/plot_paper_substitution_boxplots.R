# Boxplots in the original figure's style, with Adult aliases counted once.
# The second panel borrows only the paper's joint TVAE Covertype macro-F1.
save_paper_substitution_boxplots <- function(root) {
  library(ggplot2)
  recorded <- read.csv(file.path(root, "outputs", "method_scores.csv"))
  borrowed <- read.csv(file.path(root, "outputs", "paper_comparison", "hypothetical_scores.csv"))
  summaries <- read.csv(file.path(root, "outputs", "paper_comparison", "substitution_sensitivity.csv"))
  match_rows <- match(paste(recorded$dataset, recorded$method), paste(borrowed$dataset, borrowed$method))
  stopifnot(!anyNA(match_rows), sum(abs(recorded$macro_max - borrowed$macro_max[match_rows]) > 1e-12) == 1)

  draw <- function(methods, family, name, width) {
    original <- recorded[recorded$y_synth %in% methods, ]
    hypothetical <- borrowed[borrowed$y_synth %in% methods, names(recorded)]
    original$scenario <- "Recorded"
    hypothetical$scenario <- "Paper TVAE Covertype: hypothetical"
    data <- rbind(original, hypothetical)
    data$y_synth <- factor(data$y_synth, levels = methods)
    data$approach <- factor(data$approach, levels = c("new", "old"))
    selected <- summaries[summaries$family == family & summaries$metric == "macro_max", ]

    # Dashed lines and circles are approach means, as in the original figure.
    # Configurations have equal weight within dataset; the four tasks have equal weight.
    means <- aggregate(macro_max ~ scenario + approach, data, mean)
    points <- merge(unique(data[c("scenario", "y_synth", "approach")]), means,
                    by = c("scenario", "approach"))
    labels <- setNames(sprintf("%s\nOld %.4f; new %.4f | gain %.2f%% | p = %.4f",
                      c("Recorded results", "Paper-score substitution (hypothetical)"),
                      selected$joint_mean, selected$proposed_mean,
                      100 * selected$relative_gain, selected$p_value), selected$scenario)
    data$scenario <- factor(data$scenario, levels = selected$scenario)
    old_means <- means[means$approach == "old", ]
    new_means <- means[means$approach == "new", ]
    stopifnot(max(abs(old_means$macro_max - selected$joint_mean[match(old_means$scenario, selected$scenario)])) < 1e-10,
              max(abs(new_means$macro_max - selected$proposed_mean[match(new_means$scenario, selected$scenario)])) < 1e-10)
    low_points <- data[data$dataset == "Covertype" & data$y_synth == "tvae", ]
    scenario_order <- levels(data$scenario)
    means$scenario <- factor(means$scenario, levels = scenario_order)
    points$scenario <- factor(points$scenario, levels = scenario_order)
    selected$scenario <- factor(selected$scenario, levels = scenario_order)
    colors <- c(new = "#4DB6AC", old = "#9575B1")
    figure <- ggplot(data, aes(y_synth, macro_max)) +
      geom_boxplot(aes(fill = approach), outlier.shape = NA, alpha = .72,
                   width = .72, linewidth = .5) +
      geom_point(aes(color = approach), alpha = .65, size = 1.8,
                 position = position_jitter(width = .10, height = 0, seed = 328)) +
      geom_hline(data = means, aes(yintercept = macro_max),
                 color = "#D7191C", linetype = "dashed", linewidth = .7) +
      geom_point(data = points, aes(fill = approach), shape = 21,
                 size = 3.1, color = "#222222", stroke = .7) +
      geom_segment(data = selected, aes(x = 2, xend = 3, y = joint_mean, yend = proposed_mean),
                   inherit.aes = FALSE, color = "#D7191C", linewidth = .65,
                   arrow = grid::arrow(length = grid::unit(.12, "inches"))) +
      geom_text(data = low_points, aes(label = sprintf("Covertype: %.3f", macro_max)),
                vjust = 1.8, size = 3.4, color = "#544066") +
      facet_wrap(~ scenario, nrow = 1, labeller = as_labeller(labels)) +
      scale_fill_manual(values = colors) + scale_color_manual(values = colors) +
      scale_y_continuous(limits = c(.22, 1), breaks = seq(.3, 1, .1)) +
      labs(title = "Distribution of maximum macro-F1 by target synthesis method",
           subtitle = "Grouped by approach, showing overall means and difference",
           x = "Synthesis method (y_synth)", y = "Maximum test macro-F1",
           fill = "Approach", color = "Approach",
           caption = paste("Adult counted once. Purple: joint CTGAN/TVAE; teal: decoupled labeling. Circles and red dashed lines show approach means.",
                           "Only joint TVAE Covertype changes from 0.2972 to the paper's 0.433; all other scores remain fixed.",
                           "Points are dataset/configuration summaries, not independent seeds. P values: paired t tests across four tasks.",
                           "Paper protocols differ; substitution is hypothetical. Test selection and known leakage concerns remain.", sep = "\n")) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom", panel.grid.minor = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1),
            strip.text = element_text(size = 11, face = "bold", margin = margin(10, 0, 12, 0)),
            plot.title = element_text(size = 18), plot.caption = element_text(hjust = 0, size = 9),
            plot.margin = margin(14, 18, 12, 14), panel.spacing = grid::unit(1.3, "lines"))
    output <- file.path(root, "outputs", "figures", name)
    ggsave(paste0(output, ".png"), figure, width = width, height = 7, dpi = 180, bg = "white")
    ggsave(paste0(output, ".pdf"), figure, width = width, height = 7, device = cairo_pdf, bg = "white")
    write.csv(data, file.path(root, "outputs", "paper_comparison", paste0(name, "_data.csv")), row.names = FALSE)
  }

  # Restricted comparison requested earlier, plus the six-method example layout.
  draw(c("ctgan", "tvae", "rf", "xgb"), "RF_XGB", "26_rf_xgb_substitution_boxplots", 13)
  draw(c("ctgan", "tvae", "categorical", "pca_gmm", "rf", "xgb"),
       "No_Gaussian", "27_six_method_substitution_boxplots", 15)
  message("Saved both boxplot comparisons, using recorded and hypothetical scores.")
}

if (sys.nframe() == 0) {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  save_paper_substitution_boxplots(dirname(normalizePath(script)))
}
