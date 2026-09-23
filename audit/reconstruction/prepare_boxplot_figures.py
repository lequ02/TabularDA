from pathlib import Path
import shutil

source = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
stage = Path(r"D:\SummerResearch\audit\reconstruction\boxplot_figure_update_2026_09_18")
for relative in ["paper_comparison.R", "analyse.R", "README.md", "outputs/method_scores.csv",
                 "outputs/paper_comparison/hypothetical_scores.csv", "outputs/paper_comparison/substitution_sensitivity.csv",
                 "outputs/paper_comparison/paper_comparison_report.html"]:
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)
(stage / "outputs/figures").mkdir(exist_ok=True)

sections = '<h2>Boxplot comparison: CTGAN/TVAE versus RF/XGBoost</h2><img src="../figures/26_rf_xgb_substitution_boxplots.png" alt="Recorded and hypothetical restricted-method boxplots">\n<h2>Six-method layout matching the original figure</h2><img src="../figures/27_six_method_substitution_boxplots.png" alt="Recorded and hypothetical six-method boxplots">\n'
html = stage / "outputs/paper_comparison/paper_comparison_report.html"
text = html.read_text(encoding="utf-8")
anchor = '<h2>Paper versus replication: labeled comparison</h2>'
assert anchor in text
html.write_text(text.replace(anchor, sections + anchor, 1), encoding="utf-8")
builder = stage / "paper_comparison.R"
text = builder.read_text(encoding="utf-8")
anchor = '# 5. Concise Markdown/HTML report, plus complete CSV evidence'
assert anchor in text
text = text.replace(anchor, 'source(file.path(root, "plot_paper_substitution_boxplots.R"), local = TRUE)\nsave_paper_substitution_boxplots(root)\n\n' + anchor, 1)
text = text.replace('Figures 22-25', 'Figures 22-27')
anchor = "  '<h2>Paper versus replication: labeled comparison</h2>"
assert anchor in text
text = text.replace(anchor, "  '" + sections.rstrip().replace("\n", "") + "',\n" + anchor, 1)
builder.write_text(text, encoding="utf-8")
main = stage / "analyse.R"
main.write_text(main.read_text(encoding="utf-8").replace("25 PNG/PDF", "27 PNG/PDF"), encoding="utf-8")
readme = stage / "README.md"
with readme.open("a", encoding="utf-8") as stream:
    stream.write('\n## Boxplots matching the original style\n\nFigures 26 and 27 compare recorded results with the hypothetical paper TVAE Covertype substitution. Figure 26 excludes categorical and PCA-GMM; Figure 27 retains the six methods shown in the reference figure. Both count Adult once and show approach means, dashed red lines, and a red difference arrow. Regenerate with `Rscript plot_paper_substitution_boxplots.R`.\n')
print("Staged reproducible boxplot figures and report updates.")
