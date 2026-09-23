from pathlib import Path
import shutil

source = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
stage = Path(r"D:\SummerResearch\audit\reconstruction\generator_panel_update_2026_09_18")
for relative in ["paper_comparison.R", "analyse.R", "README.md", "outputs/method_scores.csv",
                 "outputs/paper_comparison/paper_comparison_report.html"]:
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)

section = '<h2>Joint versus RF/XGBoost labeling</h2><img src="../figures/28_generator_matched_comparison.png" alt="CTGAN-based and TVAE-based comparison panels"><p><a href="../../GENERATOR_COMPARISON_NOTES.md">Figure notes and statistical details</a></p>'
html = stage / "outputs/paper_comparison/paper_comparison_report.html"
text = html.read_text(encoding="utf-8")
anchor = '<h2>Boxplot comparison: CTGAN/TVAE versus RF/XGBoost</h2>'
assert anchor in text
html.write_text(text.replace(anchor, section + '\n' + anchor, 1), encoding="utf-8")

builder = stage / "paper_comparison.R"
text = builder.read_text(encoding="utf-8")
anchor = '# 5. Concise Markdown/HTML report, plus complete CSV evidence'
assert anchor in text
text = text.replace(anchor, 'source(file.path(root, "plot_generator_comparison.R"), local = TRUE)\nsave_generator_comparison(root)\n\n' + anchor, 1)
text = text.replace('Figures 22-27', 'Figures 22-28')
anchor = "  '<h2>Boxplot comparison: CTGAN/TVAE versus RF/XGBoost</h2>"
assert anchor in text
text = text.replace(anchor, "  '" + section + "',\n" + anchor, 1)
builder.write_text(text, encoding="utf-8")
main = stage / "analyse.R"
main.write_text(main.read_text(encoding="utf-8").replace("27 PNG/PDF", "28 PNG/PDF"), encoding="utf-8")
readme = stage / "README.md"
with readme.open("a", encoding="utf-8") as stream:
    stream.write('\n## Generator-separated comparison\n\nFigure 28 compares CTGAN-based methods on the left and TVAE-based methods on the right. It uses a user-specified joint TVAE Covertype maximum of 0.49. Figure details, provenance, statistics and interpretation limits are in [GENERATOR_COMPARISON_NOTES.md](GENERATOR_COMPARISON_NOTES.md). Reproduce with `Rscript plot_generator_comparison.R`.\n')
print("Staged the two-panel figure update without changing recorded scores or prior analyses.")
