"""Compare completed synthetic-only runs with CTGAN paper references."""

import csv
import json
import shutil
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
        "metric_label": "Binary F1", "title": "Adult · pilot",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (55, 73),
    },
    "covertype_pilot": {
        "dataset": "covertype", "scope": "pilot_ctgan_v1", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Covertype · pilot",
        "paper": {"CTGAN": 32.4, "TVAE": 43.3, "Real": 65.2},
        "limits": (25, 80),
    },
    "mnist28_pilot": {
        "dataset": "mnist28", "scope": "pilot_ctgan_v1", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST28 · pilot",
        "paper": {"CTGAN": 37.1, "TVAE": 79.4, "Real": 91.6},
        "limits": (30, 101),
    },
    "adult_corrected": {
        "dataset": "adult", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Adult · corrected",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (55, 73),
    },
    "covertype_corrected": {
        "dataset": "covertype", "scope": "corrected_v2", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Covertype · corrected",
        "paper": {"CTGAN": 32.4, "TVAE": 43.3, "Real": 65.2},
        "limits": (25, 80),
    },
    "mnist12_corrected": {
        "dataset": "mnist12", "scope": "corrected_v2", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST12 · corrected",
        "paper": {"CTGAN": 39.4, "TVAE": 79.3, "Real": 88.6},
        "limits": (30, 101),
    },
}
CENSUS = {
    "dataset": "census_kdd", "scope": "corrected_v2", "metric": "f1_binary",
    "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
}
PAPER_STYLES = {
    "CTGAN": {"color": "#8190a5", "linestyle": "--"},
    "TVAE": {"color": "#a46dbe", "linestyle": ":"},
    "Real": {"color": "#4b5563", "linestyle": "-."},
}


def read_run(info, suffix):
    dataset = info["dataset"]
    path = (ROOT / "output" / info["scope"] / dataset / "acc"
            / f"{dataset}_seed42_{suffix}.run.json")
    with path.open(encoding="utf-8") as source:
        record = json.load(source)
    expected_mode = "original" if suffix == "real_original" else "synthetic"
    assert (record["dataset"], record["seed"], record["train_option"],
            record["selection_metric"]) == (dataset, 42, expected_mode, "loss")
    return round(record["test_scores"][info["metric"]] * 100, 6), path, record["split_manifest"]


def check_manifest(info, manifest, first):
    assert manifest["dataset"] == info["dataset"]
    assert all(count == 0 for overlap in manifest["train_overlap"].values()
               for count in overlap.values())
    if first is not None:
        assert manifest["files"] == first["files"]
        assert manifest["splits"] == first["splits"]
    else:
        ids = manifest["splits"]
        assert sum(map(len, ids.values())) == len(set().union(*(set(x) for x in ids.values())))
    return manifest if first is None else first


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    values = {label: {} for label, _ in ROWS}
    record_paths = {label: {} for label, _ in ROWS}
    manifests = {}
    for panel, info in PANELS.items():
        first = None
        count = 0
        for label, suffix in ROWS:
            if info["scope"] == "pilot_ctgan_v1" and suffix.startswith("tvae_"):
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

    census_value, census_path, census_manifest = read_run(CENSUS, "real_original")
    check_manifest(CENSUS, census_manifest, None)
    values[ROWS[0][0]]["census_kdd_corrected"] = census_value
    record_paths[ROWS[0][0]]["census_kdd_corrected"] = census_path.relative_to(ROOT).as_posix()
    print("census_kdd_corrected: real-data baseline only; split IDs disjoint; "
          "recorded exact train/holdout overlaps zero")

    all_columns = [*PANELS, "census_kdd_corrected"]
    target_csv = DEST / "dataset_pipeline_comparison.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("configuration", *(f"{p}_{PANELS[p]['metric']}_percent" for p in PANELS),
                         "census_kdd_corrected_f1_binary_percent",
                         *(f"{p}_record" for p in all_columns)))
        for label, _ in ROWS:
            writer.writerow((label, *(values[label].get(p, "") if values[label].get(p) is not None else ""
                                     for p in all_columns),
                             *(record_paths[label].get(p, "") for p in all_columns)))
        for name in ("CTGAN", "TVAE", "Real"):
            writer.writerow((f"Paper {name} (reference)",
                             *(info["paper"][name] for info in PANELS.values()),
                             CENSUS["paper"][name],
                             *("" for _ in all_columns)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 13.8), sharey="row")
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
        if index % 3 == 0:
            ax.set_yticks(positions, [row[0] for row in ROWS])
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=1.8, label=f"Paper {name}")
               for name, style in PAPER_STYLES.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.979, 0.99),
               frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Synthetic-data pipelines on held-out real test sets", x=0.047,
                 y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.047, 0.959, "Seed 42 · synthetic-only training except the original-data row",
             ha="left", color="#64748b", fontsize=10)
    fig.text(0.047, 0.047,
             "Adult: binary F1 · Covertype: macro F1 · MNIST: accuracy. Paper references use different splits and average downstream classifiers.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.028,
             "Paper Table 6 labels its CTGAN row 'TGAN'. Pilots have no TVAE runs; Census KDD has a real-data baseline only (see table).",
             ha="left", color="#475569", fontsize=9)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.91, bottom=0.095,
                        wspace=0.16, hspace=0.22)
    target = DEST / "dataset_pipeline_comparison.png"
    fig.savefig(target, dpi=220, facecolor="white")
    plt.close(fig)
    # Keep links to the first comparison deliverables current.
    shutil.copyfile(target_csv, DEST / "adult_mnist28_comparison.csv")
    shutil.copyfile(target, DEST / "adult_mnist28_comparison.png")
    print(target_csv)
    print(target)


if __name__ == "__main__":
    main()
