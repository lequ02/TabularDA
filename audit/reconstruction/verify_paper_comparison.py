from pathlib import Path
import hashlib
import json
import math
import re
import subprocess

import pandas as pd
from PIL import Image, ImageOps
from pypdf import PdfReader

root = Path(r"D:\SummerResearch\audit\reconstruction\paper_comparison_update_2026_09_18")
existing = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
folder = root / "outputs" / "paper_comparison"
actual = pd.read_csv(root / "outputs" / "method_scores.csv")
borrowed = pd.read_csv(folder / "hypothetical_scores.csv")
assert len(actual) == len(borrowed) == 88
pd.testing.assert_frame_equal(actual, pd.read_csv(existing / "outputs" / "method_scores.csv"))
assert (root / "revised_title_and_abstract.md").read_text() == (existing / "revised_title_and_abstract.md").read_text()
changed = (actual["macro_max"] != borrowed["macro_max"]) | (actual["macro_end"] != borrowed["macro_end"])
assert changed.sum() == 1
assert borrowed.loc[changed, "method"].iloc[0] == "tvae"
assert borrowed.loc[changed, "dataset"].iloc[0] == "Covertype"
assert borrowed.loc[changed, ["macro_max", "macro_end"]].eq(.433).all().all()
assert borrowed.substituted.equals(changed)

summary = pd.read_csv(folder / "substitution_sensitivity.csv")
assert len(summary) == 12
for row in summary.itertuples():
    data = actual if row.scenario == "Recorded" else borrowed
    if row.family == "No_Gaussian":
        data = data[~data.method.str.contains("gauss")]
    elif row.family == "RF_XGB":
        data = data[data.y_synth.isin(["ctgan", "tvae", "rf", "xgb"])]
    means = data.groupby(["dataset", "approach"])[row.metric].mean().unstack()
    differences = means.new - means.old
    statistic = abs(differences.mean()) / (differences.std(ddof=1) / 2)
    theta = math.atan(statistic / math.sqrt(3))
    probability = 1 - 2 * (theta + math.sin(2 * theta) / 2) / math.pi
    assert math.isclose(row.p_value, probability, abs_tol=1e-12)
    assert math.isclose(row.gain, differences.mean(), abs_tol=1e-12)
    assert math.isclose(row.joint_mean, means.old.mean(), abs_tol=1e-12)
    assert math.isclose(row.proposed_mean, means.new.mean(), abs_tol=1e-12)
    recorded = summary[(summary.family == row.family) & (summary.metric == row.metric) & (summary.scenario == "Recorded")].iloc[0]
    assert math.isclose(row.proposed_mean, recorded.proposed_mean, abs_tol=1e-12)
    if row.scenario != "Recorded":
        uplift = (.433 - actual.loc[changed, row.metric].iloc[0]) / 8
        assert math.isclose(row.joint_mean - recorded.joint_mean, uplift, abs_tol=1e-12)

final = pd.read_csv(folder / "final_paper_metric_comparison.csv")
available = final[final.value.notna()]
assert len(available) == 16
assert set(available.dataset) == {"Adult", "Covertype", "MNIST12", "MNIST28"}
assert final[final.dataset.isin(["Census_KDD", "Credit", "Intrusion", "News"])].value.isna().all()
assert available[available.dataset == "Adult"].aliases_averaged.eq(3).all()
assert available[available.dataset.str.startswith("MNIST")].metric.eq("Accuracy").all()
assert available[available.dataset == "Adult"].metric.eq("Binary F1").all()

history = pd.read_csv(folder / "historical_repetition_check.csv")
assert len(history) == 80 and history.absolute_difference.lt(1e-12).all()
for entry in json.loads((root / "inputs" / "paper_comparison_manifest.json").read_text()):
    assert hashlib.sha256((root / "inputs" / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"]

pdfs = sorted((root / "outputs" / "figures").glob("*.pdf"))
assert len(pdfs) == len(list((root / "outputs" / "figures").glob("*.png"))) == 24
assert all(len(PdfReader(str(path)).pages) == 1 for path in pdfs)
all_links = re.findall(r'<img src="([^"]+)"', (root / "outputs" / "legacy_artifacts_report.html").read_text())
assert len(all_links) == 24 and all((root / "outputs" / path).exists() for path in all_links)
new_links = re.findall(r'<img src="([^"]+)"', (folder / "paper_comparison_report.html").read_text())
assert len(new_links) == 3 and all((folder / path).exists() for path in new_links)

render_dir = root.parent / "tmp" / "paper_comparison_pdf_review"
render_dir.mkdir(parents=True, exist_ok=True)
poppler = Path(r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
sheet = Image.new("RGB", (1400, 2400), "white")
for i, path in enumerate(pdfs[21:]):
    prefix = render_dir / path.stem
    subprocess.run([str(poppler), "-singlefile", "-scale-to", "1400", "-png", str(path), str(prefix)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image = Image.open(prefix.with_suffix(".png")).convert("RGB")
    sheet.paste(ImageOps.contain(image, (1400, 790)), (0, i * 800))
sheet.save(folder / "pdf_review_contact_sheet.png")
print("Verified exact single-row substitution, 12 independently checked paired comparisons, matching paper metrics,")
print("80 repeated historical cells, input hashes, unchanged actual results/abstract, 24 figure pairs and report links.")
print("Rendered three new PDFs for visual review.")
