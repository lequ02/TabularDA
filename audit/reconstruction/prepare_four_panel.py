from pathlib import Path
import shutil

source = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
stage = Path(r"D:\SummerResearch\audit\reconstruction\four_panel_update_2026_09_18")
for relative in ["paper_comparison.R", "analyse.R", "README.md", "outputs/method_scores.csv",
                 "outputs/paper_comparison/paper_comparison_report.html"]:
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)
(stage / "outputs/figures").mkdir(exist_ok=True)
section = '<h2>Synthetic data utility</h2><img src="../figures/29_four_panel_comparison.png" alt="Four panels comparing feature generators, target inclusion and synthesis approaches"><p><a href="../../FOUR_PANEL_COMPARISON_NOTES.md">Figure notes and statistical details</a></p>'
anchor = '<h2>Joint versus RF/XGBoost labeling</h2>'
html = stage / "outputs/paper_comparison/paper_comparison_report.html"
text = html.read_text(encoding="utf-8")
assert anchor in text
html.write_text(text.replace(anchor, section + '\n' + anchor, 1), encoding="utf-8")
builder = stage / "paper_comparison.R"
text = builder.read_text(encoding="utf-8")
marker = '# 5. Concise Markdown/HTML report, plus complete CSV evidence'
assert marker in text
text = text.replace(marker, 'source(file.path(root, "plot_four_panel_comparison.R"), local = TRUE)\nsave_four_panel_comparison(root)\n\n' + marker, 1)
text = text.replace('Figures 22-28', 'Figures 22-29')
assert "  '" + anchor in text
text = text.replace("  '" + anchor, "  '" + section + "',\n  '" + anchor, 1)
builder.write_text(text, encoding="utf-8")
main = stage / "analyse.R"
main.write_text(main.read_text(encoding="utf-8").replace("28 PNG/PDF", "29 PNG/PDF"), encoding="utf-8")
with (stage / "README.md").open("a", encoding="utf-8") as stream:
    stream.write('\n## Four-panel comparison\n\nFigure 29 labels the complete CTGAN/TVAE-to-RF/XGB pipelines and adds target-inclusion and pooled e2e/hybrid panels. Details and statistics: [FOUR_PANEL_COMPARISON_NOTES.md](FOUR_PANEL_COMPARISON_NOTES.md). Reproduce with `Rscript plot_four_panel_comparison.R`.\n')
print("Staged the four-panel update; original scores and previous figures are preserved.")
