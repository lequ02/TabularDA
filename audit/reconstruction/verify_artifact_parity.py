from pathlib import Path
import re
import hashlib
import json
import subprocess

import pandas as pd
from PIL import Image, ImageOps
from pypdf import PdfReader

root = Path(r"D:\SummerResearch\audit\reconstruction\artifact_parity_update_2026_09_18")
previous = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
figures = root / "outputs" / "figures"
pngs = sorted(figures.glob("*.png"))
pdfs = sorted(figures.glob("*.pdf"))
assert len(pngs) == len(pdfs) == 18
assert all(len(PdfReader(str(path)).pages) == 1 for path in pdfs)

sheet = Image.new("RGB", (1500, 2100), "white")
for i, path in enumerate(pngs[8:]):
    thumb = ImageOps.contain(Image.open(path).convert("RGB"), (750, 400))
    sheet.paste(thumb, ((i % 2) * 750, (i // 2) * 420))
sheet.save(root / "outputs" / "legacy_figure_contact_sheet.png")

scores = pd.read_csv(root / "outputs" / "method_scores.csv")
assert len(scores) == 88
assert set(scores.dataset) == {"Adult", "Covertype", "MNIST12", "MNIST28"}
assert not scores.duplicated(["dataset", "method"]).any()
means = pd.read_csv(root / "outputs" / "legacy_tables" / "04_approach_means_Gaussian excluded.csv")
old = means.loc[means.approach == "old", "macro_max"].iloc[0]
new = means.loc[means.approach == "new", "macro_max"].iloc[0]
assert abs((new - old) / old - .184899080583857) < 1e-12
pd.testing.assert_frame_equal(
    pd.read_csv(previous / "outputs" / "sensitivity_summary.csv"),
    pd.read_csv(root / "outputs" / "sensitivity_summary.csv"))
assert (previous / "revised_title_and_abstract.md").read_text() == (root / "revised_title_and_abstract.md").read_text()

html = (root / "outputs" / "legacy_artifacts_report.html").read_text()
links = re.findall(r'<img src="([^"]+)"', html)
assert len(links) == 18 and all((root / "outputs" / link).exists() for link in links)
conditional = pd.read_csv(root / "outputs" / "legacy_tables" / "19_conditional_labeler_means.csv")
assert conditional.unsupported_combination.sum() == 2

for entry in json.loads((root / "inputs" / "source_manifest.json").read_text()):
    assert hashlib.sha256((root / "inputs" / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"]
reference = Path(r"D:\Rprojects\research_data_synthesis\final_328_project.Rmd")
(root / "inputs" / reference.name).write_bytes(reference.read_bytes())
(root / "inputs" / "legacy_reference_manifest.json").write_text(json.dumps({
    "original_path": str(reference), "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()
}, indent=2))

print("Verified 18 PNG/PDF pairs, report links, collapsed rows, boxplot means, and unchanged statistics/abstract.")
print("Legacy table files:", len(list((root / "outputs" / "legacy_tables").glob("*"))))

# Render the actual PDF plots as a separate visual check with bundled Poppler.
poppler = Path(r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
render_dir = root.parent / "tmp" / "parity_pdf_review"
render_dir.mkdir(parents=True, exist_ok=True)
pdf_sheet = Image.new("RGB", (1500, 2100), "white")
for i, path in enumerate(pdfs[8:]):
    prefix = render_dir / path.stem
    subprocess.run([str(poppler), "-singlefile", "-scale-to", "900", "-png", str(path), str(prefix)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rendered = Image.open(prefix.with_suffix(".png")).convert("RGB")
    pdf_sheet.paste(ImageOps.contain(rendered, (750, 400)), ((i % 2) * 750, (i // 2) * 420))
pdf_sheet.save(root / "outputs" / "legacy_pdf_contact_sheet.png")
print("Rendered all ten added PDF figures for visual review.")
