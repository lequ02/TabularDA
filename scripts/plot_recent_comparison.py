"""Compare completed synthetic-only runs with CTGAN paper references."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "comparisons"
ROWS = [
    ("Original data only", "real_original"),
    ("CTGAN, generated target", "ctgan_full_generated_synthetic"),
    ("TVAE, generated target", "tvae_full_generated_synthetic"),
    ("CTGAN full features + RF", "ctgan_full_rf_synthetic"),
    ("CTGAN full features + XGB", "ctgan_full_xgb_synthetic"),
    ("CTGAN X-only features + RF", "ctgan_xonly_rf_synthetic"),
    ("CTGAN X-only features + XGB", "ctgan_xonly_xgb_synthetic"),
    ("CTGAN full features + DNN", "ctgan_full_dnn_synthetic"),
    ("CTGAN X-only features + DNN", "ctgan_xonly_dnn_synthetic"),
    ("TVAE full features + DNN", "tvae_full_dnn_synthetic"),
    ("TVAE X-only features + DNN", "tvae_xonly_dnn_synthetic"),
]
PANELS = {
    "adult_pilot": {
        "dataset": "adult", "scope": "pilot_ctgan_v1", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Adult · pilot, seed 42",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (55, 73), "coverage": "pilot",
    },
    "adult_corrected": {
        "dataset": "adult", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Adult · corrected, seed 42",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (55, 73), "coverage": "full",
    },
    "adult_corrected_seed43": {
        "dataset": "adult", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Adult · corrected, seed 43",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (50, 73), "coverage": "full",
    },
    "covertype_pilot": {
        "dataset": "covertype", "scope": "pilot_ctgan_v1", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Covertype · pilot",
        "paper": {"CTGAN": 32.4, "TVAE": 43.3, "Real": 65.2},
        "limits": (25, 80), "coverage": "pilot",
    },
    "covertype_corrected": {
        "dataset": "covertype", "scope": "corrected_v2", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Covertype · corrected",
        "paper": {"CTGAN": 32.4, "TVAE": 43.3, "Real": 65.2},
        "limits": (25, 80), "coverage": "full",
    },
    "mnist28_pilot": {
        "dataset": "mnist28", "scope": "pilot_ctgan_v1", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST28 · pilot",
        "paper": {"CTGAN": 37.1, "TVAE": 79.4, "Real": 91.6},
        "limits": (30, 101), "coverage": "pilot",
    },
    "mnist12_corrected": {
        "dataset": "mnist12", "scope": "corrected_v2", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST12 · corrected",
        "paper": {"CTGAN": 39.4, "TVAE": 79.3, "Real": 88.6},
        "limits": (30, 101), "coverage": "full",
    },
    "credit_corrected": {
        "dataset": "credit", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Credit · corrected, seed 42",
        "paper": {"CTGAN": 67.2, "TVAE": 9.8, "Real": 72.0},
        "limits": (-4, 90), "coverage": "full",
    },
    "credit_corrected_seed43": {
        "dataset": "credit", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Credit · corrected, seed 43",
        "paper": {"CTGAN": 67.2, "TVAE": 9.8, "Real": 72.0},
        "limits": (-4, 90), "coverage": "partial",
    },
    "census_kdd_corrected": {
        "dataset": "census_kdd", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Census KDD · corrected, seed 42",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (25, 58), "coverage": "baseline_only",
    },
    "census_kdd_corrected_seed43": {
        "dataset": "census_kdd", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Census KDD · corrected, seed 43",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (25, 58), "coverage": "baseline_only",
    },
}
PAPER_STYLES = {
    "CTGAN": {"color": "#8190a5", "linestyle": "--"},
    "TVAE": {"color": "#a46dbe", "linestyle": ":"},
    "Real": {"color": "#4b5563", "linestyle": "-."},
}


def run_path(info, suffix):
    dataset = info["dataset"]
    seed = info.get("seed", 42)
    return (ROOT / "output" / info["scope"] / dataset / "acc"
            / f"{dataset}_seed{seed}_{suffix}.run.json")


def read_run(info, suffix):
    dataset = info["dataset"]
    seed = info.get("seed", 42)
    path = run_path(info, suffix)
    with path.open(encoding="utf-8") as source:
        record = json.load(source)
    expected_mode = "original" if suffix == "real_original" else "synthetic"
    assert (record["dataset"], record["seed"], record["train_option"],
            record["selection_metric"]) == (dataset, seed, expected_mode, "loss")
    manifest = record["split_manifest"]
    if expected_mode == "synthetic":
        provenance = record["generator_provenance"]
        assert provenance["seed"] == seed
        train_file = manifest["files"]["train"]["raw"]
        assert provenance["training_rows"] == train_file["rows"]
        if "_full_" in suffix:
            assert provenance["training_columns"] == train_file["columns"]
            assert provenance["fit_table_sha256"] == train_file["sha256"]
        else:
            assert "_xonly_" in suffix
            assert provenance["training_columns"] == train_file["columns"][:-1]
    return round(record["test_scores"][info["metric"]] * 100, 6), path, manifest


def check_manifest(info, manifest, first):
    assert manifest["dataset"] == info["dataset"]
    assert manifest["seed"] == info.get("seed", 42)
    assert all(count == 0 for overlap in manifest["train_overlap"].values()
               for count in overlap.values())
    if first is not None:
        assert manifest["files"] == first["files"]
        assert manifest["splits"] == first["splits"]
    else:
        ids = manifest["splits"]
        assert sum(map(len, ids.values())) == len(set().union(*(set(x) for x in ids.values())))
    return manifest if first is None else first


def is_available(info, suffix):
    coverage = info["coverage"]
    if coverage == "full":
        return True
    if coverage == "pilot":
        return not suffix.startswith("tvae_")
    if coverage == "baseline_only":
        return suffix == "real_original"
    if coverage == "partial":
        return run_path(info, suffix).is_file()
    raise ValueError(f"Unknown coverage: {coverage}")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    values = {label: {} for label, _ in ROWS}
    record_paths = {label: {} for label, _ in ROWS}
    manifests = {}
    for panel, info in PANELS.items():
        first = None
        count = 0
        for label, suffix in ROWS:
            if not is_available(info, suffix):
                values[label][panel] = None
                continue
            value, path, manifest = read_run(info, suffix)
            values[label][panel] = value
            record_paths[label][panel] = path.relative_to(ROOT).as_posix()
            first = check_manifest(info, manifest, first)
            count += 1
        manifests[panel] = first
        print(f"{panel}: {count} matching records; split IDs disjoint; "
              "recorded exact train/holdout overlaps zero")

    for dataset in ("adult", "covertype"):
        pilot = manifests[f"{dataset}_pilot"]
        corrected = manifests[f"{dataset}_corrected"]
        assert pilot["splits"] == corrected["splits"]
        assert pilot["files"] == corrected["files"]
        print(f"{dataset}: pilot and corrected share split IDs and prepared-file hashes")
    assert manifests["mnist28_pilot"]["splits"] == manifests["mnist12_corrected"]["splits"]
    print("MNIST12 corrected and MNIST28 pilot share split IDs; feature representations differ")
    for dataset in ("adult", "credit", "census_kdd"):
        first = manifests[f"{dataset}_corrected"]
        second = manifests[f"{dataset}_corrected_seed43"]
        assert first["splits"] == second["splits"]
        assert first["files"] == second["files"]
    print("Adult, Credit, and Census KDD seeds 42 and 43 share the same prepared split")
    assert {panel: sum(values[label][panel] is not None for label, _ in ROWS)
            for panel in PANELS} == {
                "adult_pilot": 8, "adult_corrected": 11,
                "adult_corrected_seed43": 11,
                "covertype_pilot": 8, "covertype_corrected": 11,
                "mnist28_pilot": 8, "mnist12_corrected": 11,
                "credit_corrected": 11,
                "credit_corrected_seed43": sum(is_available(PANELS["credit_corrected_seed43"], suffix)
                                               for _, suffix in ROWS),
                "census_kdd_corrected": 1,
                "census_kdd_corrected_seed43": 1,
            }
    credit_test = manifests["credit_corrected"]["files"]["test"]["raw"]
    credit_positive_count = credit_test["target_counts"]["1"]
    credit_test_rows = credit_test["rows"]
    assert credit_positive_count > 0

    all_columns = list(PANELS)
    target_csv = DEST / "dataset_pipeline_comparison.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("configuration", *(f"{p}_{PANELS[p]['metric']}_percent" for p in PANELS),
                         *(f"{p}_record" for p in all_columns)))
        for label, _ in ROWS:
            writer.writerow((label, *(values[label].get(p, "") if values[label].get(p) is not None else ""
                                     for p in all_columns),
                             *(record_paths[label].get(p, "") for p in all_columns)))
        for name in ("CTGAN", "TVAE", "Real"):
            writer.writerow((f"Paper {name} (reference)",
                             *(info["paper"][name] for info in PANELS.values()),
                             *("" for _ in all_columns)))

    target_table = DEST / "dataset_pipeline_comparison.md"
    table_lines = [
        "# Synthetic-data pipeline comparison",
        "",
        "All scores are percentages. Adult, Credit, and Census KDD use binary F1; "
        "Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. "
        "Seed 42 is used unless the column says seed 43. Within Adult, Credit, and Census KDD, "
        "the two seeds share the same prepared train, development, and test split; "
        "they are model-seed repeats, not independent holdouts.",
        "",
        "Credit seed 43 is partially complete. Missing configurations remain marked with a dash; "
        "completed zero scores are displayed as 0.0%.",
        "",
    ]
    table_groups = {
        "Pilot · seed 42": [p for p, info in PANELS.items() if info["coverage"] == "pilot"],
        "Corrected · seed 42": [p for p, info in PANELS.items()
                                if info["scope"] == "corrected_v2" and info.get("seed", 42) == 42],
        "Corrected · seed 43": [p for p, info in PANELS.items() if info.get("seed", 42) == 43],
    }
    for title, panels in table_groups.items():
        headings = ["Configuration", *(PANELS[p]["title"].split(" · ")[0] for p in panels)]
        table_lines.extend([f"## {title}", "", "| " + " | ".join(headings) + " |",
                            "|---" + "|---:" * len(panels) + "|"])
        for label, _ in ROWS:
            cells = [label, *(f"{values[label][panel]:.1f}%" if values[label][panel] is not None
                              else "—" for panel in panels)]
            table_lines.append("| " + " | ".join(cells) + " |")
        for name in ("CTGAN", "TVAE", "Real"):
            cells = [f"Paper {name} (reference)",
                     *(f"{PANELS[p]['paper'][name]:.1f}%" for p in panels)]
            table_lines.append("| " + " | ".join(cells) + " |")
        table_lines.append("")
    table_lines.extend([
        "",
        "Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, "
        "Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "
        "\"TGAN\". Its scores use different splits and average multiple downstream classifiers.",
        "",
        f"Credit's test split contains {credit_positive_count} positive cases among "
        f"{credit_test_rows:,} rows. Its binary F1 is sensitive to each positive prediction; "
        "0.0% is a measured result, not a missing run.",
        "",
        "The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. "
        "Prepared and synthetic CSVs are not present locally, so row-level leakage checks "
        "cannot be independently repeated here.",
        "",
        "The CSV alongside this table includes the source run-record path for each score.",
    ])
    target_table.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    nrows = (len(PANELS) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18.5, 4.5 * nrows), sharey="row")
    positions = list(range(len(ROWS)))
    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        info = PANELS[panel]
        ax.set_xlim(*info["limits"])
        ax.set_ylim(-1.3, len(ROWS) - 0.5)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", colors="#475569")
        ax.set_xlabel(f"{info['metric_label']} (%)", labelpad=8, fontsize=11)
        ax.set_title(info["title"], loc="left", fontsize=13.5, fontweight="bold", pad=20)
        for name, score in info["paper"].items():
            ax.axvline(score, linewidth=1.3, alpha=0.75, zorder=0, **PAPER_STYLES[name])
        for divider in (0.5, 2.5, 6.5):
            ax.axhline(divider, color="#e2e8f0", linewidth=0.8)
        offset = 0.32 if info["dataset"] == "adult" else 1.0
        for y, (label, _) in enumerate(ROWS):
            value = values[label][panel]
            if value is None:
                if info["coverage"] != "baseline_only":
                    ax.text(info["limits"][0] + 2, y, "Not run", ha="left", va="center",
                            color="#94a3b8", style="italic", fontsize=9.5)
                continue
            color = "#1f2937" if y == 0 else "#087f8c"
            ax.scatter(value, y, s=92 if y == 0 else 66, color=color,
                       edgecolor="white", linewidth=1, zorder=3)
            place_left = value > info["limits"][1] - (3 if info["dataset"] == "adult" else 6)
            ax.text(value - offset if place_left else value + offset, y,
                    f"{value:.1f}%", va="center",
                    ha="right" if place_left else "left",
                    color="#1f2937", fontsize=9.5,
                    fontweight="bold" if y == 0 else "normal")
        if info["coverage"] == "baseline_only":
            ax.text(0.5, 0.44, "No synthetic evaluations recorded", transform=ax.transAxes,
                    ha="center", va="center", color="#94a3b8", style="italic", fontsize=11)
        if index % 2 == 0:
            ax.set_yticks(positions, [row[0] for row in ROWS])
        else:
            ax.tick_params(labelleft=False)

    for ax in list(axes.flat)[len(PANELS):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=1.8, label=f"Paper {name}")
               for name, style in PAPER_STYLES.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.979, 0.99),
               frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Synthetic-data pipelines on held-out real test sets", x=0.047,
                 y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.047, 0.959, "Panels show each seed separately · synthetic-only training except the original-data row",
             ha="left", color="#64748b", fontsize=10)
    fig.text(0.047, 0.065,
             "Adult, Credit, Census KDD: binary F1 · Covertype: macro F1 · MNIST: accuracy. Paper references use different splits and averaged classifiers.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.047,
             "Paper Table 6 labels CTGAN 'TGAN'. Adult/Credit/Census seeds share a fixed split; Census KDD has baseline runs only.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.029,
             f"Credit test split: {credit_positive_count} positive cases among {credit_test_rows:,} rows; binary F1 is sensitive to individual positive predictions.",
             ha="left", color="#475569", fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.915, bottom=0.095,
                        wspace=0.19, hspace=0.30)
    target = DEST / "dataset_pipeline_comparison.png"
    fig.savefig(target, dpi=220, facecolor="white")
    plt.close(fig)
    print(target_csv)
    print(target_table)
    print(target)


if __name__ == "__main__":
    main()
