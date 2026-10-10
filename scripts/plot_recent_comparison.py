"""Compare completed synthetic-only or mix runs with paper references."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "comparisons"
NEWS_SCOPE = "news_log_v1"
MNIST_SCOPE = "mnist_head_fixed_20261009"
HOUSING_SCOPE = "housing_no_faker_20261009"
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
    ("TVAE full features + RF", "tvae_full_rf_synthetic"),
    ("TVAE full features + XGB", "tvae_full_xgb_synthetic"),
    ("TVAE full features + DNN", "tvae_full_dnn_synthetic"),
    ("TVAE X-only features + RF", "tvae_xonly_rf_synthetic"),
    ("TVAE X-only features + XGB", "tvae_xonly_xgb_synthetic"),
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
        "dataset": "mnist12", "scope": MNIST_SCOPE, "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST12 · corrected, seed 42",
        "paper": {"CTGAN": 39.4, "TVAE": 79.3, "Real": 88.6},
        "limits": (30, 101),
    },
    "mnist28_corrected": {
        "dataset": "mnist28", "scope": MNIST_SCOPE, "metric": "accuracy",
        "metric_label": "Accuracy", "title": "MNIST28 · corrected, seed 42",
        "paper": {"CTGAN": 37.1, "TVAE": 79.4, "Real": 91.6},
        "limits": (30, 101),
    },
    "census_kdd_corrected": {
        "dataset": "census_kdd", "scope": "census_kdd_weighted_macro_f1_20261005", "metric": "f1_binary",
        "selection_metric": "f1_macro",
        "metric_label": "Binary F1", "title": "Census KDD · corrected, seed 42",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (-4, 60),
    },
    "census_kdd_corrected_seed43": {
        "dataset": "census_kdd", "scope": "census_kdd_weighted_macro_f1_20261005", "seed": 43,
        "selection_metric": "f1_macro",
        "metric": "f1_binary", "metric_label": "Binary F1",
        "title": "Census KDD · corrected, seed 43",
        "paper": {"CTGAN": 39.1, "TVAE": 37.7, "Real": 49.4},
        "limits": (-4, 60),
    },
    "news_corrected": {
        "dataset": "news", "scope": NEWS_SCOPE, "metric": "r2", "selection_metric": "loss",
        "metric_label": "R² / D² absolute error", "title": "News R² / D² · corrected, seed 42",
        "paper": {"CTGAN": -0.43, "TVAE": -0.20, "Real": 0.14}, "limits": (-0.5, 0.3),
    },
    "california_housing_corrected": {
        "dataset": "california_housing", "scope": HOUSING_SCOPE, "metric": "r2",
        "metric_label": "R² / D² absolute error", "title": "Housing R² / D² · corrected, seed 42",
        "paper": {}, "limits": (-0.1, 1.0),
    },
}
PANELS = {
    f"{info['dataset']}_corrected" + ("_seed43" if seed == 43 else ""): {
        **info, "seed": seed,
        "scope": info["scope"],
        "title": f"{info['title'].split(' · ')[0]} · "
                 f"{'log-target rerun' if info['dataset'] == 'news' else 'ten-class rerun' if info['dataset'] in {'mnist12', 'mnist28'} else 'no-Faker rerun' if info['dataset'] == 'california_housing' else 'class-weighted' if info['dataset'] == 'census_kdd' else 'corrected'}, seed {seed}",
    }
    for info in PANELS.values() if info.get("seed", 42) == 42
    for seed in (42, 43)
}
PAPER_STYLES = {
    "CTGAN": {"color": "#8190a5", "linestyle": "--"},
    "TVAE": {"color": "#a46dbe", "linestyle": ":"},
    "Real": {"color": "#4b5563", "linestyle": "-."},
}
for dataset, title in (("news", "News"), ("california_housing", "Housing")):
    for seed in (42, 43):
        source = PANELS[f"{dataset}_corrected" + ("_seed43" if seed == 43 else "")]
        PANELS[f"{dataset}_nmae_seed{seed}"] = {
            **source, "metric": "nmae_sigma", "metric_label": "NMAEσ (lower is better)",
            "title": f"{title} NMAEσ · {'log-target rerun' if dataset == 'news' else 'no-Faker rerun'}, seed {seed}",
            "paper": {}, "limits": (0, 0.5),
        }
UNSCALED_METRICS = {"r2", "nmae_sigma", "d2_absolute_error"}
METRIC_COLORS = {
    "f1_binary": {"ctgan": "#087f8c", "tvae": "#6d28d9"},
    "f1_macro": {"ctgan": "#b45309", "tvae": "#be185d"},
    "accuracy": {"ctgan": "#15803d", "tvae": "#65a30d"},
    "r2": {"ctgan": "#15803d", "tvae": "#65a30d"},
    "nmae_sigma": {"ctgan": "#0369a1", "tvae": "#7c3aed"},
    "d2_absolute_error": {"ctgan": "#0369a1", "tvae": "#7c3aed"},
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
    assert record["selected_dev_epoch"] is not None, path
    assert (record["dataset"], record["seed"], record["train_option"],
            record["selection_metric"]) == (dataset, seed, expected_mode, info.get("selection_metric", "loss"))
    if dataset in {"mnist12", "mnist28"}:
        evidence = json.loads((ROOT / "audit/comparison_rerun_verification_latest.json").read_text())
        assert info["scope"] == MNIST_SCOPE
        assert record["source_sha256"] == evidence["mnist"]["source_snapshot"]["source_sha256"]
    if dataset == "news":
        assert record["classifier"] == "DNN_News_log_no_norm"
        transform = record["target_transform"]
        assert transform["name"] == "log" and transform["inverse"] == "exp"
        assert transform["normalization"] == "none" and transform["training_loss"] == "log_target_mse"
        assert transform["selection"] == "raw_share_mse" and transform["metric_units"] == "shares"
    if dataset == "census_kdd":
        protocol = record["evaluation_protocol"]
        assert protocol["output_namespace"] == info["scope"]
        assert protocol["objective"] == "BCEWithLogitsLoss"
        assert protocol["selection_metric"] == "f1_macro" and protocol["threshold"] == 0.5
        weighted = record["pilot_protocol"]
        counts = weighted["training_label_counts"]
        assert abs(weighted["pos_weight"] - counts["0"] / counts["1"]) < 1e-12
    manifest = record["split_manifest"]
    if expected_mode in {"synthetic", "mix"}:
        provenance = record["generator_provenance"]
        assert provenance["seed"] == seed
        train_file = manifest["files"]["train"]["raw"]
        assert provenance["training_rows"] == train_file["rows"]
        if "_full_" in suffix:
            assert provenance["training_columns"] == train_file["columns"]
            if dataset not in {"california_housing", "news"}:
                assert provenance["fit_table_sha256"] == train_file["sha256"]
        else:
            assert "_xonly_" in suffix
            assert provenance["training_columns"] == train_file["columns"][:-1]
        if dataset == "news":
            evidence = json.loads((ROOT / "audit/comparison_news_log_fit_table_hashes.json").read_text())
            assert evidence["namespace"] == NEWS_SCOPE
            table = evidence["tables"][f"news_seed{seed}"]
            assert table["raw_train_sha256"] == train_file["sha256"]
            fit = table["fits"]["full" if "_full_" in suffix else "xonly"]
            generator = "tvae" if suffix.startswith("tvae_") else "ctgan"
            assert provenance == fit["generators"][generator]["provenance"]
            assert provenance["fit_table_sha256"] == fit["sha256"]
        if dataset == "california_housing":
            evidence = json.loads((ROOT / "audit/comparison_housing_fit_table_hashes.json").read_text())
            assert evidence["namespace"] == info["scope"] == HOUSING_SCOPE
            table = evidence["tables"][f"california_housing_seed{seed}"]
            assert table["raw_train_sha256"] == train_file["sha256"]
            fit = table["fits"]["full" if "_full_" in suffix else "xonly"]
            generator = "tvae" if suffix.startswith("tvae_") else "ctgan"
            assert provenance == fit["generators"][generator]["provenance"]
            assert provenance["fit_table_sha256"] == fit["sha256"]
    scores = dict(record["test_scores"])
    if info["metric"] == "r2":
        derived_path = path.with_name(path.name.replace(".run.json", ".d2.json"))
        derived = json.loads(derived_path.read_text(encoding="utf-8"))
        normalization = derived["absolute_error_normalization"]
        assert derived["run_record"] == path.relative_to(ROOT).as_posix()
        assert derived["run_record_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert (derived["dataset"], derived["seed"], derived["train_option"],
                derived["augment_option"], derived["selected_dev_epoch"]) == (
            dataset, seed, record["train_option"], record["augment_option"], record["selected_dev_epoch"])
        assert normalization["split"] == "test" and normalization["baseline_mae"] > 0
        assert normalization["target_table_sha256"] == manifest["files"]["test"]["raw"]["sha256"]
        assert normalization["rows"] == manifest["files"]["test"]["raw"]["rows"]
        score = derived["test_scores"]["d2_absolute_error"]
        assert abs(score - (1 - normalization["prediction_mae"] / normalization["baseline_mae"])) < 1e-12
        scores["d2_absolute_error"] = score
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
    d2_values = {label: {} for label, _ in rows}
    record_paths = {label: {} for label, _ in rows}
    manifests = {}
    for panel, info in PANELS.items():
        first = None
        count = 0
        for label, suffix in rows:
            if not is_available(info, suffix):
                values[label][panel] = None
                macro_values[label][panel] = None
                d2_values[label][panel] = None
                continue
            scores, path, manifest = read_run(info, suffix)
            values[label][panel] = round(scores[info["metric"]]
                                        * (1 if info["metric"] in UNSCALED_METRICS else 100), 6)
            macro_values[label][panel] = (round(scores["f1_macro"] * 100, 6)
                                          if info["metric"] == "f1_binary" else None)
            d2_values[label][panel] = (round(scores["d2_absolute_error"], 6)
                                       if info["metric"] == "r2" else None)
            record_paths[label][panel] = path.relative_to(ROOT).as_posix()
            first = check_manifest(info, manifest, first)
            count += 1
        manifests[panel] = first
        print(f"{panel}: {count} matching records; split IDs disjoint; "
              "recorded exact train/holdout overlaps zero")

    for dataset in ("adult", "census_kdd"):
        first = manifests[f"{dataset}_corrected"]
        second = manifests[f"{dataset}_corrected_seed43"]
        if first is not None and second is not None:
            assert first["splits"] == second["splits"]
            assert first["files"] == second["files"]
            print(f"{dataset}: seeds 42 and 43 share the same prepared split")

    all_columns = list(PANELS)
    score_columns = [(p, metric) for p, info in PANELS.items()
                     for metric in ([info["metric"], "f1_macro"]
                                    if info["metric"] == "f1_binary" else
                                    ["r2", "d2_absolute_error"] if info["metric"] == "r2" else [info["metric"]])]
    d2_panels = [p for p, info in PANELS.items() if info["metric"] == "r2"]
    target_csv = DEST / f"{basename}.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("configuration", *(f"{p}_{metric}" + ("" if metric in UNSCALED_METRICS else "_percent")
                                              for p, metric in score_columns),
                         *(f"{p}_record" for p in all_columns),
                         *(f"{p}_d2_metadata" for p in d2_panels)))
        for label, _ in rows:
            writer.writerow((label, *((d2_values if metric == "d2_absolute_error" else
                                       macro_values if metric != PANELS[p]["metric"] else values)[label][p]
                                      if values[label][p] is not None else ""
                                      for p, metric in score_columns),
                             *(record_paths[label].get(p, "") for p in all_columns),
                             *(record_paths[label][p].replace(".run.json", ".d2.json")
                               if p in record_paths[label] else "" for p in d2_panels)))
        for name in ("CTGAN", "TVAE", "Real"):
            writer.writerow((f"Paper {name} (reference)",
                             *(PANELS[p]["paper"].get(name, "") if metric == PANELS[p]["metric"] else ""
                               for p, metric in score_columns),
                             *("" for _ in all_columns), *("" for _ in d2_panels)))

    target_table = DEST / f"{basename}.md"
    completed = len({path for paths in record_paths.values() for path in paths.values()})
    snapshot_path = ROOT / "audit" / ("latest_comparison_mix_snapshot.json" if train_option == "mix"
                                      else "latest_comparison_snapshot.json")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["mnist_source_scope"] == MNIST_SCOPE
    assert snapshot["housing_source_scope"] == HOUSING_SCOPE
    mnist_completed = len({path for paths in record_paths.values() for panel, path in paths.items()
                           if PANELS[panel]["dataset"] in {"mnist12", "mnist28"}})
    mnist_verification = json.loads((ROOT / "audit/comparison_rerun_verification_latest.json").read_text())["mnist"]
    checked_at = datetime.fromisoformat(snapshot["checked_at"]).astimezone(ZoneInfo("America/Chicago"))
    table_lines = [
        "# Mix pipeline comparison" if train_option == "mix" else "# Synthetic-data pipeline comparison",
        "",
        f"Verified source snapshot: {checked_at.strftime('%B %d, %Y, %I:%M %p')} Chicago. "
        f"This report contains {completed} distinct completed run records from the selected configurations; "
        "selected report coverage does not establish completion of the original 816-run matrix.",
        "",
        "Classification scores are percentages; News and Housing display paired, unscaled R² / D² absolute-error scores. "
        "Both are higher-is-better, have a maximum of 1, and can be negative. NMAEσ remains in the CSV. "
        "Adult and Census KDD show binary F1 / macro F1; "
        "Covertype uses macro F1; MNIST uses accuracy. A dash means no evaluated run is recorded. "
        "Seed 42 is used unless the column says seed 43. Within Adult and Census KDD, "
        "the two seeds share the same prepared train, development, and test split; "
        "they are model-seed repeats, not independent holdouts.",
        "",
        "Corrected runs are shown for other datasets; News uses the completed fresh log-target rerun. "
        "Missing configurations remain marked with a dash; "
        "completed zero scores are displayed as 0.0%.",
        f"MNIST12/28 use only `{MNIST_SCOPE}`; Housing uses only `{HOUSING_SCOPE}`. "
        f"News uses only `{NEWS_SCOPE}`; Census KDD uses `census_kdd_weighted_macro_f1_20261005`; "
        "remaining datasets use `corrected_v2`. "
        "Source paths in the CSV preserve each namespace.",
        f"This {train_option} report has {mnist_completed}/60 completed MNIST configuration records "
        "across both datasets and seeds, including the four real-only baselines. "
        f"Separately, {mnist_verification['verified_completed']}/{mnist_verification['planned_total']} "
        "full-matrix MNIST fits are verified, including NB/PCA-GMM configurations excluded from these reports. "
        + ("All planned MNIST fits are complete." if mnist_verification["verified_completed"] == mnist_verification["planned_total"]
           else "Remaining full-matrix configurations are pending."),
        "The News rerun has 74 completed downstream runs across seeds 42 and 43. These reports select "
        "RF/XGB/DNN and generated-target arms, with all 15 configurations available for each seed and "
        "training mode. Every News downstream model is freshly fitted; no pilot or historical raw-target "
        "News records are reused. Full-table generators are fitted on log targets; features-only generators "
        "are reused only with verified provenance, and all labelers are fitted afresh on real training log targets.",
        "The News rerun trains MSE on log(shares), uses no BatchNorm or LayerNorm, and selects checkpoints "
        "by raw-scale real-development MSE. Predictions are transformed back with exp before final metrics "
        "are computed in shares. Log training, normalization removal, and full-table generator/labeler changes "
        "are combined changes; comparisons with historical runs do not isolate their individual effects.",
        "Census KDD uses class-weighted BCEWithLogitsLoss (positive weight = actual training negatives / positives) "
        "and development macro-F1 checkpoint selection, with threshold 0.5. Both loss and checkpoint selection "
        "changed from the earlier evaluation; improvements cannot be attributed to weighting alone. "
        "Pending weighted configurations remain missing rather than using earlier unweighted results.",
        "News and Housing NMAEσ = MAE / σ_y; lower is better. σ_y is the population standard deviation (ddof=0) "
        "of the same real held-out test targets used to compute MAE. Both News seeds use σ_y = 9485.506480005333 "
        "over 7,929 rows, verified against the prepared test-table hash. "
        "The paper reports News R², not NMAEσ, so its reference lines appear only in the News R² panels. "
        "No paper reference is supplied for California Housing.",
        "D² absolute error = 1 − MAE / MAE of a constant test-median prediction. Its zero benchmark "
        "is the test median; R² uses the test mean. D² uses absolute errors and R² uses squared errors. "
        "The two scores share a plotting axis but measure different prediction errors. "
        "Paper lines in the combined regression panels refer to R² only; no D² references are supplied.",
        "Housing now uses all 58 completed corrected rerun fits, including fresh real-only baselines. "
        "Its eight fresh generators retain Latitude and Longitude as learned numerical features and use "
        "no Faker transformers. Earlier coordinate-flawed generators and scores remain historical diagnostic "
        "evidence and do not supply this report. See the "
        "[Housing rerun](../../audit/HOUSING_FAKER_RERUN_2026_10_09.md).",
        "MNIST now uses the repaired downstream evaluator: both forward methods apply their declared "
        "ten-class output layers. Every displayed rerun model is freshly trained. Pending configurations "
        "remain blank; earlier models that bypassed those layers never fill missing cells. The full matrix "
        "plans 212 fits, with 116 report configurations first and 96 NB/PCA-GMM fits last. "
        "NB/PCA-GMM remain excluded from these reports. The retained MNIST12 TVAE artifacts still carry "
        "the documented held-out feature-collision caveat; the downstream repair does not remove it. See "
        "[the MNIST rerun](../../audit/mnist_head_rerun_20261009/README.md).",
        "News uses a symmetric-log axis with a linear region from −0.001 to 0.001 to spread scores "
        "clustered near zero while retaining negative values. Other regression pairs use a symmetric-log "
        "axis with a linear region from −0.1 to 0.1 only if their range extends below −1. "
        "Tick labels and reported R²/D² values remain unscaled; both seeds share the same axis scale.",
        "News and Housing run records save `test_scores.nmae_sigma` alongside `r2` and `mae`, with "
        "`target_normalization` recording σ_y, split, ddof, row count, and target-table hash. "
        "These saved results supply the report; normalization is not recomputed during plotting.",
        "D² is computed remotely from each run's saved held-out predictions without retraining and stored "
        "in a `.d2.json` sidecar. The sidecar saves the test median, median-baseline MAE, prediction MAE, "
        "row count, split and target-table/prediction/run-record hashes. Original records and NMAEσ are "
        "preserved. The builder reads and verifies sidecars; it does not recompute D² from test data.",
        "", r"$$\mathrm{NMAE}_\sigma = \frac{\mathrm{MAE}}{\sigma_y}.$$", "",
        "In the figure, triangles mark original data; hollow markers mark generated-target benchmarks. "
        "Original-data triangles use teal for binary F1, orange for macro F1, green for accuracy/R², and blue for D². "
        "CTGAN and TVAE have distinct colors within each metric.",
        "Macro F1 averages the F1 scores of both classes. Paper references for binary datasets "
        "are shown only under binary F1; no matching paper macro F1 reference is supplied.",
        "",
    ]
    table_groups = {info["title"].split(" · ")[0]:
                    [p for p, other in PANELS.items()
                     if other["title"].split(" · ")[0] == info["title"].split(" · ")[0]]
                    for info in PANELS.values() if info["seed"] == 42 and info["metric"] != "nmae_sigma"}
    for title, panels in table_groups.items():
        headings = ["Configuration", *(f"Seed {PANELS[p]['seed']}"
                    + (" (binary / macro F1)" if PANELS[p]["metric"] == "f1_binary" else
                       " (R² / D² absolute error)" if PANELS[p]["metric"] == "r2" else "")
                    for p in panels)]
        table_lines.extend([f"## {title}", "", "| " + " | ".join(headings) + " |",
                            "|---" + "|---:" * len(panels) + "|"])
        for label, _ in rows:
            cells = [label, *((f"{values[label][panel]:.3f} / {d2_values[label][panel]:.3f}"
                              if PANELS[panel]["metric"] == "r2" else
                              f"{values[label][panel]:.3f}" if PANELS[panel]["metric"] in UNSCALED_METRICS
                              else f"{values[label][panel]:.1f}%"
                              + (f" / {macro_values[label][panel]:.1f}%"
                                 if PANELS[panel]["metric"] == "f1_binary" else ""))
                              if values[label][panel] is not None
                              else "—" for panel in panels)]
            table_lines.append("| " + " | ".join(cells) + " |")
        for name in ("CTGAN", "TVAE", "Real"):
            cells = [f"Paper {name} (reference)",
                     *((f"{PANELS[p]['paper'][name]:.3f} / —" if PANELS[p]["metric"] == "r2" else
                        f"{PANELS[p]['paper'][name]:.3f}" if PANELS[p]["metric"] in UNSCALED_METRICS
                        else f"{PANELS[p]['paper'][name]:.1f}%"
                        + (" / —" if PANELS[p]["metric"] == "f1_binary" else ""))
                       if name in PANELS[p]["paper"] else
                       ("—" if PANELS[p]["dataset"] == "california_housing" else "Not reported")
                       for p in panels)]
            table_lines.append("| " + " | ".join(cells) + " |")
        table_lines.append("")
    table_lines.extend([
        "",
        "Paper references: [Xu et al., *Modeling Tabular Data using Conditional GAN*, "
        "Table 6](https://arxiv.org/pdf/1907.00503). The appendix labels its CTGAN row "
        "\"TGAN\". Its scores use different splits and average multiple downstream classifiers.",
        "",
        "The run manifests record disjoint split IDs and zero exact train-to-holdout overlaps. "
        "Prepared and synthetic CSVs are not present locally, so row-level leakage checks "
        "cannot be independently repeated here.",
        "",
        "The CSV alongside this table includes source run-record paths, D² sidecar paths, and the retained NMAEσ scores.",
    ])
    if train_option == "mix":
        table_lines[2:2] = [
            "Mix training concatenates all real training rows and 100,000 synthetic rows; "
            "the ratio varies by dataset. The original-data row is the same real-only baseline. "
            "Development and test partitions remain real held-out data.", "",
        ]
    target_table.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plot_panels = {p: info for p, info in PANELS.items() if info["metric"] != "nmae_sigma"}
    nrows = (len(plot_panels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18.5, 5.5 * nrows), sharey="row")
    positions = list(range(len(rows)))
    pair_limits = {}
    for info in plot_panels.values():
        dataset = info["title"].split(" · ")[0]
        panels = [p for p, other in plot_panels.items() if other["title"].split(" · ")[0] == dataset]
        displayed = [score for p in panels for label, _ in rows
                     for score in (values[label][p], macro_values[label][p], d2_values[label][p]) if score is not None]
        pair_limits[dataset] = (min([PANELS[p]["limits"][0] for p in panels]
                                    + [score - (0.04 if info["metric"] in UNSCALED_METRICS else 4) for score in displayed]),
                                max([PANELS[p]["limits"][1] for p in panels]
                                    + [score + (0.04 if info["metric"] in UNSCALED_METRICS else 4) for score in displayed]))
    for index, (ax, panel) in enumerate(zip(axes.flat, plot_panels)):
        info = PANELS[panel]
        limits = pair_limits[info["title"].split(" · ")[0]]
        ax.set_title(info["title"], loc="left", fontsize=13.5, fontweight="bold", pad=20)
        if all(values[label][panel] is None for label, _ in rows):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No completed results recorded", transform=ax.transAxes,
                    ha="center", va="center", color="#94a3b8", fontsize=13)
            continue
        ax.set_xlim(*limits)
        compressed_r2 = info["metric"] == "r2" and (info["dataset"] == "news" or limits[0] < -1)
        linear_threshold = 0.001 if info["dataset"] == "news" else 0.1
        if compressed_r2:
            ax.set_xscale("symlog", linthresh=linear_threshold)
            scale = ax.xaxis.get_transform()
            lower, upper = scale.transform(limits)
            margin = 0.04 * (upper - lower)
            ax.set_xlim(scale.inverted().transform((lower - margin, upper + margin)))
            if info["dataset"] == "news":
                ticks = [-0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1]
                ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        ax.set_ylim(-1.3, len(rows) - 0.5)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", colors="#475569")
        binary = info["metric"] == "f1_binary"
        paired = binary or info["metric"] == "r2"
        pair_offset = 0 if info["metric"] == "r2" else 0.17
        ax.set_xlabel("Binary and macro F1 (%)" if binary else info["metric_label"]
                      + (f" (linear within ±{linear_threshold:g}; log beyond)" if compressed_r2 else "")
                      + ("" if info["metric"] in UNSCALED_METRICS else " (%)"),
                      labelpad=8, fontsize=11)
        for name, score in info["paper"].items():
            ax.axvline(score, linewidth=1.6, alpha=0.95, zorder=1, **PAPER_STYLES[name])
        if info["metric"] == "r2" and info["paper"]:
            ax.text(0, 1.02, "Paper references: R² only", transform=ax.transAxes,
                    color="#64748b", fontsize=9, va="bottom")
        if not info["paper"] and info["dataset"] != "california_housing":
            ax.text(0, 1.02, "Paper references not reported", transform=ax.transAxes,
                    color="#64748b", fontsize=9, va="bottom")
        for divider in (0.5, 2.5, 5.5, 8.5, 11.5):
            ax.axhline(divider, color="#e2e8f0", linewidth=0.8)
        offset = 0.02 * (limits[1] - limits[0])
        for y, (label, suffix) in enumerate(rows):
            value = values[label][panel]
            if value is None:
                ax.text(0.02, y, "Not completed", transform=ax.get_yaxis_transform(), ha="left", va="center",
                        color="#94a3b8", style="italic", fontsize=9.5)
                continue
            generator = "tvae" if suffix.startswith("tvae_") else "ctgan"
            benchmark = "_full_generated_" in suffix
            points = [(value, y - pair_offset if paired else y,
                       METRIC_COLORS[info["metric"]][generator],
                       "D" if info["metric"] == "f1_macro" else ("s" if info["metric"] in {"accuracy", "r2"} else ("v" if info["metric"] == "nmae_sigma" else "o")))]
            if binary:
                points.append((macro_values[label][panel], y + 0.17, METRIC_COLORS["f1_macro"][generator], "D"))
            if info["metric"] == "r2":
                points.append((d2_values[label][panel], y + pair_offset,
                               METRIC_COLORS["d2_absolute_error"][generator], "v"))
            for score, position, color, marker in points:
                ax.scatter(score, position, s=75 if y == 0 else 50,
                           facecolors="none" if benchmark else color,
                           marker="^" if y == 0 else marker, edgecolor=color, linewidth=1.2, zorder=3)
                if compressed_r2:
                    fraction = ax.transAxes.inverted().transform(ax.transData.transform((score, position)))[0]
                    place_left = fraction > 0.90 or score < max(point[0] for point in points)
                    ax.annotate(f"{score:.3f}", (score, position),
                                xytext=(-5 if place_left else 5, 0), textcoords="offset points",
                                va="center", ha="right" if place_left else "left", color=color,
                                fontsize=7.5, fontweight="bold" if y == 0 else "normal")
                else:
                    place_left = score > limits[1] - 4 * offset
                    ax.text(score - offset if place_left else score + offset, position,
                            f"{score:.3f}" if info["metric"] in UNSCALED_METRICS else f"{score:.1f}%", va="center",
                            ha="right" if place_left else "left", color=color,
                            fontsize=7.5 if info["metric"] == "r2" else (8 if paired else 9.5),
                            fontweight="bold" if y == 0 else "normal")
        if index % 2 == 0 or all(values[label][list(plot_panels)[index - 1]] is None
                                for label, _ in rows):
            ax.set_yticks(positions, [row[0] for row in rows])
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

    for ax in list(axes.flat)[len(plot_panels):]:
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
                                                    ("d2_absolute_error", "v", "D² absolute error"))
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
             "Adult/Census: binary and macro F1 · Covertype: macro F1 · MNIST: accuracy · News/Housing: R² and D² absolute error. Regression paper lines: R² only.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.047,
             "Census: weighted loss + dev macro-F1 selection. News: fresh log-target rerun, no normalization; raw-scale dev MSE selection. Paper CTGAN is labeled 'TGAN'.",
             ha="left", color="#475569", fontsize=9)
    fig.text(0.047, 0.029,
             "MNIST: ten-class evaluator repaired. Housing: fresh generators retain coordinates. Seeds may share the same holdout.",
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
