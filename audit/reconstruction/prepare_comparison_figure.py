from pathlib import Path
import shutil

source = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
stage = Path(r"D:\SummerResearch\audit\reconstruction\paper_figure_update_2026_09_18")
files = ["paper_comparison.R", "analyse.R", "README.md",
         "outputs/paper_comparison/final_paper_metric_comparison.csv",
         "outputs/paper_comparison/paper_comparison_report.html",
         "outputs/paper_comparison/PAPER_COMPARISON_AND_SENSITIVITY.md"]
for relative in files:
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)

builder = stage / "paper_comparison.R"
text = builder.read_text(encoding="utf-8")
anchor = '# 5. Concise Markdown/HTML report, plus complete CSV evidence'
text = text.replace(anchor, 'source(file.path(root, "plot_paper_comparison.R"), local = TRUE)\nsave_paper_replication_bars(root)\n\n' + anchor, 1)
text = text.replace('Figures 22-24', 'Figures 22-25')
html_anchor = "  '<h2>Final paper-metric comparison</h2>', paper_html_table(paper_display),"
new_section = "  '<h2>Paper versus replication: labeled comparison</h2><img src=\"../figures/25_paper_replication_bars.png\" alt=\"Paper, final epoch and test maximum scores for CTGAN and TVAE\">',\n"
assert html_anchor in text
text = text.replace(html_anchor, new_section + html_anchor, 1)
builder.write_text(text, encoding="utf-8")

html = stage / "outputs/paper_comparison/paper_comparison_report.html"
text = html.read_text(encoding="utf-8")
anchor = '<h2>Final paper-metric comparison</h2>'
assert anchor in text
text = text.replace(anchor, '<h2>Paper versus replication: labeled comparison</h2><img src="../figures/25_paper_replication_bars.png" alt="Paper, final epoch and test maximum scores for CTGAN and TVAE">\n' + anchor, 1)
html.write_text(text, encoding="utf-8")

for relative in ["outputs/paper_comparison/PAPER_COMPARISON_AND_SENSITIVITY.md", "analyse.R", "README.md"]:
    path = stage / relative
    text = path.read_text(encoding="utf-8")
    text = text.replace("Figures 22-24", "Figures 22-25").replace("24 PNG/PDF", "25 PNG/PDF")
    if relative == "README.md":
        text += '\n## Labeled paper/replication figure\n\nFigure 25 shows the paper, our final epoch, and our test maximum for CTGAN and TVAE on matching metrics. Regenerate this figure independently with `Rscript plot_paper_comparison.R`; the full analysis also creates it. Missing final datasets are noted in the caption rather than plotted as zero.\n'
    path.write_text(text, encoding="utf-8")
print("Staged the figure builder and report updates; source scores remain unchanged.")
