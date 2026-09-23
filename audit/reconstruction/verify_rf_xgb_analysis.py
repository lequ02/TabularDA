from pathlib import Path
import json
import hashlib
import math
import re
import subprocess

import pandas as pd
from PIL import Image, ImageOps
from pypdf import PdfReader

root = Path(r"D:\SummerResearch\audit\reconstruction\rf_xgb_update_2026_09_18")
original = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
folder = root / "outputs" / "rf_xgb"
scores = pd.read_csv(folder / "selected_method_scores.csv")
assert len(scores) == 40 and set(scores.y_synth) == {"ctgan", "tvae", "rf", "xgb"}
assert len(scores.dataset.unique()) == 4 and not scores.duplicated(["dataset", "method"]).any()
assert scores.groupby("dataset").size().eq(10).all()

# Independent check against the original 132-row table and Python's paired test.
raw = pd.read_csv(root / "inputs" / "April29_macro_max_mfa.csv")
raw.dataset = raw.dataset.replace({"Census": "Adult", "Census_Kdd": "Adult"})
expected = raw.groupby(["dataset", "method"]).macro_max.mean()
for row in scores.itertuples():
    assert math.isclose(row.macro_max, expected.loc[row.dataset, row.method], abs_tol=1e-12)
summary = pd.read_csv(folder / "summary.csv")
for row in summary.itertuples():
    subset = scores if row.datasets == 4 else scores[scores.dataset != "Covertype"]
    metric = "macro_end" if row.analysis.startswith("Final") else "macro_max"
    means = subset.groupby(["dataset", "approach"])[metric].mean().unstack()
    differences = means.new - means.old
    # Closed-form two-sided Student t probabilities for df=2 or 3.
    statistic = abs(differences.mean()) / (differences.std(ddof=1) / math.sqrt(len(differences)))
    if len(differences) == 3:
        probability = 1 - statistic / math.sqrt(statistic**2 + 2)
    else:
        theta = math.atan(statistic / math.sqrt(3))
        probability = 1 - 2 * (theta + math.sin(2 * theta) / 2) / math.pi
    assert math.isclose(row.gain, differences.mean(), abs_tol=1e-12)
    assert math.isclose(row.p_value, probability, abs_tol=1e-12)
    assert math.isclose(row.relative_gain, differences.mean() / means.old.mean(), abs_tol=1e-12)

pd.testing.assert_frame_equal(pd.read_csv(original / "outputs" / "sensitivity_summary.csv"),
                              pd.read_csv(root / "outputs" / "sensitivity_summary.csv"))
assert (root / "revised_title_and_abstract.md").read_text() == (original / "revised_title_and_abstract.md").read_text()
for entry in json.loads((root / "inputs" / "source_manifest.json").read_text()):
    assert hashlib.sha256((root / "inputs" / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"]

figures = root / "outputs" / "figures"
pdfs = sorted(figures.glob("*.pdf"))
assert len(pdfs) == len(list(figures.glob("*.png"))) == 21
assert all(len(PdfReader(str(path)).pages) == 1 for path in pdfs)
links = re.findall(r'<img src="([^"]+)"', (root / "outputs" / "legacy_artifacts_report.html").read_text())
assert len(links) == 21 and all((root / "outputs" / path).exists() for path in links)

poppler = Path(r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
render_dir = root.parent / "tmp" / "rf_xgb_pdf_review"
render_dir.mkdir(parents=True, exist_ok=True)
sheet = Image.new("RGB", (1000, 1800), "white")
for i, path in enumerate(pdfs[18:]):
    prefix = render_dir / path.stem
    subprocess.run([str(poppler), "-singlefile", "-scale-to", "1000", "-png", str(path), str(prefix)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rendered = Image.open(prefix.with_suffix(".png")).convert("RGB")
    sheet.paste(ImageOps.contain(rendered, (1000, 590)), (0, i * 600))
sheet.save(folder / "pdf_review_contact_sheet.png")
print("Verified: 40 requested scores, four collapsed datasets, independent group means and paired p values,")
print("input hashes, unchanged broad statistics/abstract, 21 PNG/PDF pairs, report links, and rendered three new PDFs.")
