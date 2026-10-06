"""Compare completed synthetic-only or mix runs with paper references."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    ("CTGAN full features + DNN", "ctgan_full_dnn_synthetic"),
    ("CTGAN X-only features + RF", "ctgan_xonly_rf_synthetic"),
    ("CTGAN X-only features + XGB", "ctgan_xonly_xgb_synthetic"),
    ("CTGAN X-only features + DNN", "ctgan_xonly_dnn_synthetic"),
    ("TVAE full features + DNN", "tvae_full_dnn_synthetic"),
    ("TVAE X-only features + DNN", "tvae_xonly_dnn_synthetic"),
]
PANELS = {
    "adult_corrected": {
        "dataset": "adult", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Adult · corrected, seed 42",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (55, 73),
    },
    "adult_corrected_seed43": {
        "dataset": "adult", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Adult · corrected, seed 43",
        "paper": {"CTGAN": 60.1, "TVAE": 62.6, "Real": 66.9},
        "limits": (50, 73),
    },
    "covertype_corrected": {
        "dataset": "covertype", "scope": "corrected_v2", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Covertype · corrected, seed 42",
        "paper": {"CTGAN": 32.4, "TVAE": 43.3, "Real": 65.2},
        "limits": (25, 80),
    },
    "mnist12_corrected": {
        "dataset": "mnist12", "scope": "corrected_v2", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST12 · corrected, seed 42",
        "paper": {"CTGAN": 39.4, "TVAE": 79.3, "Real": 88.6},
        "limits": (30, 101),
    },
    "mnist28_corrected": {
        "dataset": "mnist28", "scope": "corrected_v2", "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST28 · corrected, seed 42",
        "paper": {"CTGAN": 37.1, "TVAE": 79.4, "Real": 91.6},
        "limits": (30, 101),
    },
    "credit_corrected": {
        "dataset": "credit", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Credit · corrected, seed 42",
        "paper": {"CTGAN": 67.2, "TVAE": 9.8, "Real": 72.0},
        "limits": (-4, 90),
    },
    "credit_corrected_seed43": {
        "dataset": "credit", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Credit · corrected, seed 43",
        "paper": {"CTGAN": 67.2, "TVAE": 9.8, "Real": 72.0},
        "limits": (-4, 90),
    },
    "census_kdd_corrected": {
        "dataset": "census_kdd", "scope": "corrected_v2", "metric": "f1_binary",
        "metric_label": "Binary F1", "title": "Census KDD · corrected, seed 42",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (-4, 60),
    },
    "census_kdd_corrected_seed43": {
        "dataset": "census_kdd", "scope": "corrected_v2", "seed": 43,
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Census KDD · corrected, seed 43",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (-4, 60),
    },
    "intrusion_corrected": {
        "dataset": "intrusion", "scope": "corrected_v2", "metric": "f1_macro",
        "metric_label": "Macro F1", "title": "Intrusion · corrected, seed 42",
        "paper": {"CTGAN": 52.8, "TVAE": 51.1, "Real": 86.2}, "limits": (-4, 100),
    },
    "news_corrected": {
        "dataset": "news", "scope": "corrected_v2", "metric": "r2",
        "metric_label": "R²", "title": "News · corrected, seed 42",
        "paper": {"CTGAN": -0.43, "TVAE": -0.20, "Real": 0.14}, "limits": (-0.5, 0.3),
    },
}
PANELS = {
    f"{info['dataset']}_corrected" + ("_seed43" if seed == 43 else ""): {
        **info, "seed": seed,
        "scope": ("corrected_v2_seed42_mnist28_news"
                  if info["dataset"] in {"mnist28", "news"} and seed == 42 else info["scope"]),
        "title": f"{info['title'].split(' · ')[0]} · corrected, seed {seed}",
    }
    for info in PANELS.values() if info.get("seed", 42) == 42
    for seed in (42, 43)
}
PAPER_STYLES = {
    "CTGAN": {"color": "#8190a5", "linestyle": "--"},
    "TVAE": {"color": "#a46dbe", "linestyle": ":"},
    "Real": {"color": "#4b5563", "linestyle": "-."},
}
for seed in (42, 43):
    source = PANELS["news_corrected" + ("_seed43" if seed == 43 else "")]
    PANELS[f"news_nmae_seed{seed}"] = {
        **source, "metric": "nmae_sigma", "metric_label": "NMAEσ (lower is better)",
        "title": f"News NMAEσ · corrected, seed {seed}", "paper": {}, "limits": (0, 0.5),
    }
UNSCALED_METRICS = {"r2", "nmae_sigma"}
METRIC_COLORS = {
    "f1_binary": {"ctgan": "#087f8c", "tvae": "#6d28d9"},
    "f1_macro": {"ctgan": "#b45309", "tvae": "#be185d"},
    "accuracy": {"ctgan": "#15803d", "tvae": "#65a30d"},
    "r2": {"ctgan": "#15803d", "tvae": "#65a30d"},
    "nmae_sigma": {"ctgan": "#0369a1", "tvae": "#7c3aed"},
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
    expected_mode = "original" if suffix == "real_original" else suffix.rsplit("_", 1)[1]
    assert (record["dataset"], record["seed"], record["train_option"],
            record["selection_metric"]) == (dataset, seed, expected_mode, "loss")
    manifest = record["split_manifest"]
    if expected_mode in {"synthetic", "mix"}:
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
    scores = dict(record["test_scores"])
    if info["metric"] == "nmae_sigma":
        normalization = record["target_normalization"]
        assert normalization["target_table_sha256"] == manifest["files"]["test"]["raw"]["sha256"]
        assert normalization["rows"] == manifest["files"]["test"]["raw"]["rows"]
        assert normalization["split"] == "test" and normalization["ddof"] == 0
        assert abs(scores["nmae_sigma"] - scores["mae"] / normalization["sigma_y"]) < 1e-12
    return scores, path, manifest


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
    return run_path(info, suffix).is_file()


def main(train_option="synthetic"):
    rows = [(label, suffix.removesuffix("_synthetic") + "_mix"
             if train_option == "mix" and suffix != "real_original" else suffix)
            for label, suffix in ROWS]
    basename = "dataset_pipeline_comparison" + ("_mix" if train_option == "mix" else "")
    DEST.mkdir(parents=True, exist_ok=True)
    values = {label: {} for label, _ in rows}
    macro_values = {label: {} for label, _ in rows}
    record_paths = {label: {} for label, _ in rows}
    manifests = {}
    for panel, info in PANELS.items():
        first = None
        count = 0
        for label, suffix in rows:
            if not is_available(info, suffix):
                values[label][panel] = None
                macro_values[label][panel] = None
                continue
            scores, path, manifest = read_run(info, suffix)
            values[label][panel] = round(scores[info["metric"]]
                                        * (1 if info["metric"] in UNSCALED_METRICS else 100), 6)
            macro_values[label][panel] = (round(scores["f1_macro"] * 100, 6)
                                          if info["metric"] == "f1_binary" else None)
            record_paths[label][panel] = path.relative_to(ROOT).as_posix()
            first = check_manifest(info, manifest, first)
            count += 1
        manifests[panel] = first
        print(f"{panel}: {count} matching records; split IDs disjoint; "
              "recorded exact train/holdout overlaps zero")

    for dataset in ("adult", "credit", "census_kdd"):
        first = manifests[f"{dataset}_corrected"]
        second = manifests[f"{dataset}_corrected_seed43"]
        assert first["splits"] == second["splits"]
        assert first["files"] == second["files"]
    print("Adult, Credit, and Census KDD seeds 42 and 43 share the same prepared split")
    credit_test = manifests["credit_corrected"]["files"]["test"]["raw"]
    credit_positive_count = credit_test["target_counts"]["1"]
    credit_test_rows = credit_test["rows"]
    assert credit_positive_count > 0

    all_columns = list(PANELS)
    score_columns = [(p, metric) for p, info in PANELS.items()
                     for metric in ([info["metric"], "f1_macro"]
                                    if info["metric"] == "f1_binary" else [info["metric"]])]
    target_csv = DEST / f"{basename}.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("configuration", *(f"{p}_{metric}" + ("" if metric in UNSCALED_METRICS else "_percent")
                                              for p, metric in score_columns),
                         *(f"{p}_record" for p in all_columns)))
        for label, _ in rows:
            writer.writerow((label, *((macro_values if metric != PANELS[p]["metric"] else values)[label][p]
                                      if values[label][p] is not None else ""
                                      for p, metric in score_columns),
                             *(record_paths[label].get(p, "") for p in all_columns)))
        for name in ("CTGAN", "TVAE", "Real"):
            writer.writerow((f"Paper {name} (reference)",
                             *(PANELS[p]["paper"].get(name, "") if metric == PANELS[p]["metric"] else ""
                               for p, metric in score_columns),
                             *("" for _ in all_columns)))

    target_table = DEST / f"{basename}.md"
    table_lines = [
        "# Mix pipeline comparison" if train_option == "mix" else "# Synthetic-data pipeline comparison",
        "",
        "Classification scores are percentages; News reports unscaled R² (which can be negative) and NMAEσ. "
        "Adult, Credit, and Census KDD show binary F1 / macro F1; "
        "Covertype and Intrusion use macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. "
        "Seed 42 is used unless the column says seed 43. Within Adult, Credit, and Census KDD, "
        "the two seeds share the same prepared train, development, and test split; "
        "they are model-seed repeats, not independent holdouts.",
        "",
        "Only corrected runs are shown. Missing configurations remain marked with a dash; "
        "completed zero scores are displayed as 0.0%.",
        "MNIST28 and News seed 42 use the separate `corrected_v2_seed42_mnist28_news` namespace. "
        "Source paths in the CSV preserve that namespace; all other displayed runs use `corrected_v2`.",
        "News NMAEσ = MAE / σ_y; lower is better. σ_y is the population standard deviation (ddof=0) "
        "of the same real held-out test targets used to compute MAE. Seed 42 uses σ_y = 9485.506480005333 "
        "over 7,929 rows, verified against the prepared test-table hash. "
        "The paper reports News R², not NMAEσ, so its reference lines appear only in the R² panels.",
        "News run records save `test_scores.nmae_sigma` alongside `r2` and `mae`, with "
        "`target_normalization` recording σ_y, split, ddof, row count, and target-table hash. "
        "These saved results supply the report; normalization is not recomputed during plotting.",
        "", r"$$\mathrm{NMAE}_\sigma = \frac{\mathrm{MAE}}{\sigma_y}.$$", "",
        "In the figure, triangles mark original data; hollow markers mark generated-target benchmarks. "
        "Original-data triangles use teal for binary F1, orange for macro F1, and green for accuracy/R². "
        "CTGAN and TVAE have distinct colors within each metric.",
        "Macro F1 averages the F1 scores of both classes. Paper references for binary datasets "
        "are shown only under binary F1; no matching paper macro F1 reference is supplied.",
        "",
    ]
    table_groups = {info["title"].split(" · ")[0]:
                    [p for p, other in PANELS.items()
                     if other["title"].split(" · ")[0] == info["title"].split(" · ")[0]]
                    for info in PANELS.values() if info["seed"] == 42}
    for title, panels in table_groups.items():
        headings = ["Configuration", *(f"Seed {PANELS[p]['seed']}"
                    + (" (binary / macro F1)" if PANELS[p]["metric"] == "f1_binary" else "")
                    for p in panels)]
        table_lines.extend([f"## {title}", "", "| " + " | ".join(headings) + " |",
                            "|---" + "|---:" * len(panels) + "|"])
        for label, _ in rows:
            cells = [label, *((f"{values[label][panel]:.3f}" if PANELS[panel]["metric"] in UNSCALED_METRICS
                              else f"{values[label][panel]:.1f}%"
                              + (f" / {macro_values[label][panel]:.1f}%"
                                 if PANELS[panel]["metric"] == "f1_binary" else ""))
                              if values[label][panel] is not None
                              else "—" for panel in panels)]
            table_lines.append("| " + " | ".join(cells) + " |")
        for name in ("CTGAN", "TVAE", "Real"):
            cells = [f"Paper {name} (reference)",
                     *((f"{PANELS[p]['paper'][name]:.3f}" if PANELS[p]["metric"] in UNSCALED_METRICS
                        else f"{PANELS[p]['paper'][name]:.1f}%"
                        + (" / —" if PANELS[p]["metric"] == "f1_binary" else ""))
                       if name in PANELS[p]["paper"] else "—" for p in panels)]
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
    if train_option == "mix":
        completed = len({path for paths in record_paths.values() for path in paths.values()})
        table_lines[2:2] = [
            "Mix training concatenates all real training rows and 100,000 synthetic rows; "
            "the ratio varies by dataset. The original-data row is the same real-only baseline. "
            "Development and test partitions remain real held-out data.", "",
            f"This report contains {completed} distinct completed run records from the selected "
            "comparison configurations; it does not establish completion of the full 816-run matrix.", "",
        ]
    target_table.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    nrows = (len(PANELS) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18.5, 4.5 * nrows), sharey="row")
    positions = list(range(len(rows)))
    pair_limits = {}
    for info in PANELS.values():
        dataset = info["title"].split(" · ")[0]
        panels = [p for p, other in PANELS.items() if other["title"].split(" · ")[0] == dataset]
        displayed = [score for p in panels for label, _ in rows
                     for score in (values[label][p], macro_values[label][p]) if score is not None]
        pair_limits[dataset] = (min([PANELS[p]["limits"][0] for p in panels]
                                    + [score - (0.04 if info["metric"] in UNSCALED_METRICS else 4) for score in displayed]),
                                max([PANELS[p]["limits"][1] for p in panels]
                                    + [score + (0.04 if info["metric"] in UNSCALED_METRICS else 4) for score in displayed]))
    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        info = PANELS[panel]
        limits = pair_limits[info["title"].split(" · ")[0]]
        ax.set_title(info["title"], loc="left", fontsize=13.5, fontweight="bold", pad=20)
        if all(values[label][panel] is None for label, _ in rows):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No completed results recorded", transform=ax.transAxes,
                    ha="center", va="center", color="#94a3b8", fontsize=13)
            continue
        ax.set_xlim(*limits)
        ax.set_ylim(-1.3, len(rows) - 0.5)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", colors="#475569")
        binary = info["metric"] == "f1_binary"
        ax.set_xlabel("Binary and macro F1 (%)" if binary else info["metric_label"]
                      + ("" if info["metric"] in UNSCALED_METRICS else " (%)"),
                      labelpad=8, fontsize=11)
        for name, score in info["paper"].items():
            ax.axvline(score, linewidth=1.3, alpha=0.75, zorder=0, **PAPER_STYLES[name])
        for divider in (0.5, 2.5, 5.5, 8.5, 9.5):
            ax.axhline(divider, color="#e2e8f0", linewidth=0.8)
        offset = 0.02 * (limits[1] - limits[0])
        for y, (label, suffix) in enumerate(rows):
            value = values[label][panel]
            if value is None:
                ax.text(limits[0] + offset, y, "Not completed", ha="left", va="center",
                        color="#94a3b8", style="italic", fontsize=9.5)
                continue
            generator = "tvae" if suffix.startswith("tvae_") else "ctgan"
            benchmark = "_full_generated_" in suffix
            points = [(value, y - 0.17 if binary else y,
                       METRIC_COLORS[info["metric"]][generator],
                       "D" if info["metric"] == "f1_macro" else ("s" if info["metric"] in {"accuracy", "r2"} else ("v" if info["metric"] == "nmae_sigma" else "o")))]
            if binary:
                points.append((macro_values[label][panel], y + 0.17, METRIC_COLORS["f1_macro"][generator], "D"))
            for score, position, color, marker in points:
                ax.scatter(score, position, s=75 if y == 0 else 50,
                           facecolors="none" if benchmark else color,
                           marker="^" if y == 0 else marker, edgecolor=color, linewidth=1.2, zorder=3)
                place_left = score > limits[1] - 4 * offset
                ax.text(score - offset if place_left else score + offset, position,
                        f"{score:.3f}" if info["metric"] in UNSCALED_METRICS else f"{score:.1f}%", va="center",
                        ha="right" if place_left else "left", color=color,
                        fontsize=8 if binary else 9.5,
                        fontweight="bold" if y == 0 else "normal")
        if index % 2 == 0 or all(values[label][list(PANELS)[index - 1]] is None
                                for label, _ in rows):
            ax.set_yticks(positions, [row[0] for row in rows])
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

    for ax in list(axes.flat)[len(PANELS):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                      linewidth=1.8, label=f"Paper {name}")
               for name, style in PAPER_STYLES.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.979, 0.99),
               frameon=False, ncol=3, fontsize=9)
    metric_handles = [Line2D([], [], marker=marker, color=METRIC_COLORS[metric][generator],
                             linestyle="none", label=f"{generator.upper()} · {label}")
                      for metric, marker, label in (("f1_binary", "o", "binary F1"),
                                                    ("f1_macro", "D", "macro F1"),
                                                    ("accuracy", "s", "accuracy / R²"),
                                                    ("nmae_sigma", "v", "NMAEσ"))
                      for generator in ("ctgan", "tvae")]
    metric_handles.extend([Line2D([], [], marker="^", color="#64748b", linestyle="none", label="Original data"),
                           Line2D([], [], marker="o", markerfacecolor="none", color="#64748b",
                                  linestyle="none", label="Generated-target benchmark")])
    fig.legend(handles=metric_handles, loc="upper right", bbox_to_anchor=(0.979, 0.955),
               frameon=False, ncol=4, fontsize=9)
    fig.suptitle("Mix pipelines on held-out real test sets" if train_option == "mix"
                 else "Synthetic-data pipelines on held-out real test sets", x=0.047,
                 y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.047, 0.965, "Each row pairs one dataset: seed 42 left, seed 43 right · empty panels have no completed results",
             ha="left", color="#64748b", fontsize=10)
    fig.text(0.047, 0.065,
             "Adult/Credit/Census: binary and macro F1 · Covertype/Intrusion: macro F1 · MNIST: accuracy · News: R² and MAE/σ_y. Paper references match the metric.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.047,
             "Paper Table 6 labels CTGAN 'TGAN'. Adult/Credit/Census seeds share a fixed split. Missing runs are marked explicitly.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.029,
             f"Credit test split: {credit_positive_count} positive cases among {credit_test_rows:,} rows; binary F1 is sensitive to individual positive predictions.",
             ha="left", color="#475569", fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.905, bottom=0.095,
                        wspace=0.19, hspace=0.40)
    target = DEST / f"{basename}.png"
    temporary = target.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=220, facecolor="white")
    temporary.replace(target)
    plt.close(fig)
    print(target_csv)
    print(target_table)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-option", choices=("synthetic", "mix"), default="synthetic")
    main(parser.parse_args().train_option)
