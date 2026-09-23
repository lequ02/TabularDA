from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

root = Path(r"D:\SummerResearch\audit\reconstruction\paper_comparison_update_2026_09_18")
existing = Path(r"D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18")
if root.exists():
    raise RuntimeError("New paper-comparison staging directory already exists")
shutil.copytree(existing, root)
manifest = []
for name in ("Mar23.csv", "April02.csv"):
    original = existing.parent / name
    snapshot = root / "inputs" / name
    shutil.copyfile(original, snapshot)
    manifest.append(dict(original_path=str(original), snapshot=name,
                         sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()))

# Extract the older workbook without modifying it. Preserve its metadata and metrics.
original = Path(r"D:\SummerResearch\final_results.xlsx")
book = pd.read_excel(original, sheet_name="final_results")
book["Dataset"] = book.Dataset.ffill()
book["Train_Augment_Option"] = book.Train_Augment_Option.ffill()
baselines = book[book.Augment_Type.str.lower().isin(["ctgan", "tvae"])].copy()
baselines.to_csv(root / "inputs" / "older_workbook_baselines.csv", index=False)
manifest.append(dict(original_path=str(original), sheet="final_results",
                     snapshot="older_workbook_baselines.csv",
                     original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     sha256=hashlib.sha256((root / "inputs" / "older_workbook_baselines.csv").read_bytes()).hexdigest()))

# Table 6 of arXiv 1907.00503v2, verified against the user-provided screenshot.
datasets = ["Adult", "Census_KDD", "Credit", "Covertype", "Intrusion", "MNIST12", "MNIST28", "News"]
metrics = ["Binary F1", "Binary F1", "Binary F1", "Macro-F1", "Macro-F1", "Accuracy", "Accuracy", "R-squared"]
ctgan = [.601, .391, .672, .324, .528, .394, .371, -.43]
tvae = [.626, .377, .098, .433, .511, .793, .794, -.20]
records = [dict(dataset=dataset, metric=metric, method=method, paper_score=score,
                source="https://arxiv.org/pdf/1907.00503", table="6", version="1907.00503v2")
           for method, values in (("ctgan", ctgan), ("tvae", tvae))
           for dataset, metric, score in zip(datasets, metrics, values)]
pd.DataFrame(records).to_csv(root / "inputs" / "paper_baselines.csv", index=False)
manifest.append(dict(original_path="https://arxiv.org/pdf/1907.00503", table="6",
                     snapshot="paper_baselines.csv",
                     sha256=hashlib.sha256((root / "inputs" / "paper_baselines.csv").read_bytes()).hexdigest()))
(root / "inputs" / "paper_comparison_manifest.json").write_text(json.dumps(manifest, indent=2))
print("Prepared", root)
print("Older workbook baseline rows:", len(baselines))
