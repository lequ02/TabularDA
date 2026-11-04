"""Before/after comparison for all 52 Adult TabDDPM arms (pre-fix hold vs posterior-fix rerun).

Writes:
- adult/tabddpm_fix_20261030/scores.csv: the 52 new results plus the 2 reused baselines,
  in the same columns as adult/scores.csv.
- before_after/adult_tabddpm_before_after.{csv,md,png} next to this script.
"Before" is read from adult/scores.csv (held pre-fix records); "after" from the archived new run
records, whose hashes are checked against report_evidence_20261031.json.
Run from the workspace root: .venv/Scripts/python.exe comparisons/20261031_adult_tabddpm_fix/before_after.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "adult/tabddpm_fix_20261030"
OUT = HERE / "before_after"
NEW_NS = "tabddpm_adult_500ep_fix_20261030_v1"
LABELERS = {"generated": "Generated target", "rf": "RF", "xgb": "XGB", "dnn": "DNN",
            "gaussian": "Gaussian NB", "categorical": "Categorical NB", "pca_gmm": "PCA/GMM"}
PRIMARY = {"generated", "rf", "xgb", "dnn"}
METRICS = [("test_f1_binary", "Binary F1"), ("test_f1_macro", "Macro F1"), ("test_accuracy", "Accuracy")]
BEFORE, AFTER, UP, DOWN, INK, MUTED, GRID = "#898781", "#2a78d6", "#2a78d6", "#d64545", "#0b0b0b", "#52514e", "#e1e0d9"

with (ROOT / "adult/scores.csv").open(encoding="utf-8-sig", newline="") as stream:
    old_rows = list(csv.DictReader(stream))
columns = list(old_rows[0])
before = {(int(r["seed"]), r["construction"], r["labeler"], r["mode"]): r for r in old_rows
          if r["generator"] == "tabddpm" and r["mode"] in {"synthetic", "mix"}}
assert len(before) == 52 and all(r["retained_status"] == "completed_with_source_bug_caveat" for r in before.values())
baselines = [r for r in old_rows if r["generator"] == "tabddpm_reference"]
assert len(baselines) == 2

evidence = json.loads((ARCHIVE / "report_evidence_20261031.json").read_text())
after, new_rows = {}, []
for item in evidence["new_records"]:
    path = ARCHIVE / "records" / Path(item["path"]).name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    record = json.loads(path.read_text())
    stem = Path(item["path"]).name.removesuffix(".run.json")
    _, _, _, construction, labeler_and_mode = stem.split("_", 4)
    labeler = labeler_and_mode.rsplit("_", 1)[0]
    key = (item["seed"], construction, labeler, item["mode"])
    assert key in before and record["augment_option"] == item["method"]
    row = dict.fromkeys(columns, "")
    row.update(dataset="adult", seed=item["seed"], generator="tabddpm", construction=construction,
               labeler=labeler, mode=item["mode"], matrix_cell=True,
               retained_status="passes_runner_completion_check_posterior_fix",
               record_relative_path=item["path"], selected_dev_epoch=record["selected_dev_epoch"],
               selection_metric=record["selection_metric"], test_loss=record.get("test_loss", ""),
               source_sha256=record["source_sha256"], caveats_json="[]")
    for name, value in record["test_scores"].items():
        if f"test_{name}" in row:
            row[f"test_{name}"] = value
    after[key] = row
    new_rows.append(row)
assert len(after) == 52 and set(after) == set(before)
for baseline in baselines:
    row = dict(baseline)
    row["record_relative_path"] = f"output/{NEW_NS}/adult/acc/adult_seed{baseline['seed']}_real_original.run.json"
    row["caveats_json"] = json.dumps(["Linked reuse of " + baseline["record_relative_path"]])
    new_rows.insert(0, row)
with (ARCHIVE / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(new_rows)

OUT.mkdir(exist_ok=True)
order = [(c, l) for c in ("full", "xonly") for l in LABELERS if not (c == "xonly" and l == "generated")]
table = []
for mode in ("synthetic", "mix"):
    for construction, labeler in order:
        for seed in (42, 43):
            b, a = before[(seed, construction, labeler, mode)], after[(seed, construction, labeler, mode)]
            entry = dict(mode=mode, construction=construction, labeler=labeler, seed=seed,
                         in_report_panels=labeler in PRIMARY,
                         before_record=b["record_relative_path"], after_record=a["record_relative_path"],
                         before_epoch=b["selected_dev_epoch"], after_epoch=a["selected_dev_epoch"])
            for metric, _ in METRICS:
                old, new = float(b[metric]) * 100, float(a[metric]) * 100
                entry.update({f"{metric}_before": round(old, 4), f"{metric}_after": round(new, 4),
                              f"{metric}_change": round(new - old, 4)})
            table.append(entry)
with (OUT / "adult_tabddpm_before_after.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

real = {int(r["seed"]): float(r["test_f1_binary"]) * 100 for r in baselines}
real_macro = {int(r["seed"]): float(r["test_f1_macro"]) * 100 for r in baselines}
labels = [("Full" if c == "full" else "X-only") + " · " + LABELERS[l] for c, l in order]
fig, axes = plt.subplots(2, 4, figsize=(19, 12.5), sharey=True)
for row_index, mode in enumerate(("synthetic", "mix")):
    for col, (metric, metric_label, seed) in enumerate(
            [(m, lab, s) for m, lab in METRICS[:2] for s in (42, 43)]):
        ax = axes[row_index, col]
        entries = {(e["construction"], e["labeler"]): e for e in table if e["mode"] == mode and e["seed"] == seed}
        for y, key in enumerate(order):
            e = entries[key]
            old, new = e[f"{metric}_before"], e[f"{metric}_after"]
            delta = new - old
            ax.plot([old, new], [y, y], color=UP if delta >= 0 else DOWN, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(old, y, s=60, facecolor="white", edgecolor=BEFORE, linewidth=1.8, zorder=3)
            ax.scatter(new, y, s=60, color=AFTER, edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(f"{delta:+.1f}", (max(old, new), y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK if abs(delta) >= 1 else MUTED,
                        fontweight="bold" if abs(delta) >= 1 else "normal")
        reference = (real if metric == "test_f1_binary" else real_macro)[seed]
        ax.axvline(reference, color=MUTED, ls="-.", lw=1, zorder=1)
        ax.text(reference, -0.9, "real only", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.axhline(6.5, color=GRID, lw=1)
        ax.set_title(f"{mode.replace('synthetic', 'Synthetic-only').replace('mix', 'Mix')} · {metric_label} · seed {seed}",
                     fontsize=11, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel(f"{metric_label} (%)", fontsize=9, color=MUTED)
for ax in axes[:, 0]:
    ax.set_yticks(range(len(order)), labels, fontsize=9.5, color=INK)
    for tick, (_, labeler) in zip(ax.get_yticklabels(), order):
        if labeler not in PRIMARY:
            tick.set_color(MUTED)
axes[0, 0].set_ylim(len(order) - 0.5, -1.2)
handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=BEFORE, markersize=8,
                  label="Before: pre-fix held records (tabddpm_500ep_20261010_v1)"),
           Line2D([], [], marker="o", ls="none", color=AFTER, markersize=8,
                  label=f"After: posterior-fix rerun ({NEW_NS})"),
           Line2D([], [], color=UP, lw=2, label="Increase"), Line2D([], [], color=DOWN, lw=2, label="Decrease"),
           Line2D([], [], color=MUTED, ls="-.", lw=1, label="Real-only baseline (same record before and after)")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.955), ncol=3, frameon=False, fontsize=9.5)
fig.suptitle("Adult TabDDPM: before vs after the categorical-posterior fix (all 52 arms)", x=0.012, y=0.99,
             ha="left", fontsize=16, fontweight="bold", color=INK)
fig.text(0.012, 0.012, "Labels give after − before in points (bold when |change| ≥ 1). Gray row labels "
         "(NB, PCA/GMM) are outside the comparison report panels. Same real held-out test set (9,521 rows), "
         "development-loss checkpoint selection, threshold 0.5. Seeds share one prepared split.",
         fontsize=9, color=MUTED, ha="left")
fig.subplots_adjust(left=0.12, right=0.985, top=0.885, bottom=0.07, wspace=0.12, hspace=0.22)
fig.savefig(OUT / "adult_tabddpm_before_after.png", dpi=180, facecolor="#fcfcfb")

lines = ["# Adult TabDDPM before vs after the posterior fix", "",
         "All 52 arms: seeds 42/43, synthetic-only and mix. Scores are percentages on the same Adult held-out "
         "test set. Before = held pre-fix records (`tabddpm_500ep_20261010_v1`); after = posterior-fix rerun "
         f"(`{NEW_NS}`). Real-only baselines are the same records in both. Rows marked * are in the "
         "comparison report panels.", "",
         "![before/after](adult_tabddpm_before_after.png)", ""]
for mode in ("synthetic", "mix"):
    lines += [f"## {'Synthetic-only' if mode == 'synthetic' else 'Mix'}", "",
              "| Arm | Seed | Binary F1 before | after | Δ | Macro F1 before | after | Δ | Accuracy before | after | Δ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in (e for e in table if e["mode"] == mode):
        name = ("Full" if e["construction"] == "full" else "X-only") + " · " + LABELERS[e["labeler"]]
        cells = [f"{e[f'{m}_before']:.2f} | {e[f'{m}_after']:.2f} | {e[f'{m}_change']:+.2f}" for m, _ in METRICS]
        lines.append(f"| {name}{' *' if e['in_report_panels'] else ''} | {e['seed']} | " + " | ".join(cells) + " |")
    lines.append("")
changes = {m: [e[f"{m}_change"] for e in table] for m, _ in METRICS}
lines += ["## Summary", ""] + [
    f"- {label}: mean change {sum(v) / len(v):+.2f} points over 52 arms; {sum(x > 0 for x in v)} up, "
    f"{sum(x < 0 for x in v)} down; largest {max(v, key=abs):+.2f}."
    for (m, label), v in ((x, changes[x[0]]) for x in METRICS)]
(OUT / "adult_tabddpm_before_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT, "and", ARCHIVE / "scores.csv")
