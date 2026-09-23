# Compare paper scores with the recorded final-epoch and test-maximum scores.
# Run independently with Rscript, or source from paper_comparison.R.
save_paper_replication_bars <- function(root) {
  library(ggplot2)
  comparison <- read.csv(file.path(root, "outputs", "paper_comparison",
                                  "final_paper_metric_comparison.csv"))
  observed <- comparison[is.finite(comparison$value), ]
  paper <- unique(observed[c("dataset", "method", "metric", "paper_score")])
  stopifnot(nrow(observed) == 16, nrow(paper) == 8,
            !anyDuplicated(observed[c("dataset", "method", "endpoint")]))

  # Include each paper baseline once; missing final datasets are not zeros.
  bars <- rbind(
    data.frame(dataset = paper$dataset, method = paper$method,
               metric = paper$metric, score = paper$paper_score, series = "Original paper"),
    data.frame(dataset = observed$dataset, method = observed$method,
               metric = observed$metric, score = observed$value,
               series = ifelse(observed$endpoint == "end", "Our final epoch", "Our test maximum")))
  bars$series <- factor(bars$series,
                       levels = c("Original paper", "Our final epoch", "Our test maximum"))
  bars$method <- factor(toupper(bars$method), levels = c("CTGAN", "TVAE"))
  bars$panel <- factor(paste(bars$dataset, bars$metric, sep = " | "),
                      levels = c("Adult | Binary F1", "Covertype | Macro-F1",
                                 "MNIST12 | Accuracy", "MNIST28 | Accuracy"))
  stopifnot(nrow(bars) == 24, all(is.finite(bars$score)), !anyNA(bars$panel))

  # Use one 0-1 scale, but retain the correct metric in each panel.
  dodge <- position_dodge(width = 0.78)
  figure <- ggplot(bars, aes(method, score, fill = series)) +
    geom_col(position = dodge, width = 0.72) +
    geom_text(aes(label = sprintf("%.3f", score)), position = dodge,
              vjust = -0.55, size = 4.1, color = "#263238") +
    facet_wrap(~ panel, ncol = 2) +
    scale_fill_manual(values = c("Original paper" = "#8C74AB",
                                 "Our final epoch" = "#279B98",
                                 "Our test maximum" = "#99CFCD")) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1.04),
                       expand = expansion(mult = c(0, 0))) +
    labs(title = "CTGAN and TVAE: paper versus our replication",
         subtitle = "Matching metrics for the four datasets with recorded final results",
         x = NULL, y = "Score", fill = NULL,
         caption = paste("Adult aliases averaged and counted once. Paper averages classifiers; our replication uses DNNs.",
                         "Test maxima were selected on the test set. Scores do not establish leakage-free superiority.",
                         "No matching final results: Census-KDD, Credit, Intrusion, News. Source: CTGAN paper, Table 6 (1907.00503v2).",
                         sep = "\n")) +
    theme_minimal(base_size = 14) +
    theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.y = element_line(color = "#E5E9EB"),
          strip.text = element_text(face = "bold", size = 14, margin = margin(10, 0, 10, 0)),
          axis.text.x = element_text(face = "bold", color = "#263238"),
          legend.position = "top", legend.justification = "left",
          plot.title = element_text(face = "bold", size = 21),
          plot.subtitle = element_text(color = "#53616A", margin = margin(b = 8)),
          plot.caption = element_text(hjust = 0, size = 10, color = "#53616A", margin = margin(t = 16)),
          plot.margin = margin(16, 22, 14, 16), panel.spacing = grid::unit(1.1, "lines"))
  figure_dir <- file.path(root, "outputs", "figures")
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  name <- "25_paper_replication_bars"
  ggsave(file.path(figure_dir, paste0(name, ".png")), figure,
         width = 12, height = 8, dpi = 180, bg = "white")
  ggsave(file.path(figure_dir, paste0(name, ".pdf")), figure,
         width = 12, height = 8, device = cairo_pdf, bg = "white")
  write.csv(bars, file.path(root, "outputs", "paper_comparison", "comparison_figure_data.csv"),
            row.names = FALSE)
  message("Saved labeled paper/replication comparison: 24 bars, four datasets.")
}

if (sys.nframe() == 0) {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
  save_paper_replication_bars(dirname(normalizePath(script)))
}
