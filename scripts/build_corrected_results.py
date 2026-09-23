"""Build corrected tables from completed run records, never epoch maxima."""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from run_corrected_matrix import DATASETS, SEEDS, methods_for


def expected_runs(matrix='full'):
    expected = set()
    if matrix == 'pilot':
        from run_corrected_pilot import DATASETS as pilot_datasets, METHODS, SEED
        for dataset in pilot_datasets:
            expected.add((dataset, 'original', None, SEED))
            for method in METHODS:
                for mode in ('synthetic', 'mix'):
                    expected.add((dataset, mode, method, SEED))
        return expected
    for dataset in DATASETS:
        for seed in SEEDS:
            expected.add((dataset, "original", None, seed))
            for generator in ("ctgan", "tvae"):
                for method in methods_for(dataset, generator):
                    for mode in ("synthetic", "mix"):
                        expected.add((dataset, mode, method, seed))
    return expected


def build(run_root, output_root, matrix='full'):
    records = sorted(run_root.rglob("*.run.json"))
    if not records:
        raise ValueError(f"No corrected run records found under {run_root}")
    rows = []
    keys = set()
    for path in records:
        with path.open(encoding="utf-8") as record_file:
            record = json.load(record_file)
        key = (record["dataset"], record["train_option"],
               record["augment_option"], record["seed"])
        if key in keys:
            raise ValueError(f"Duplicate corrected run: {key}")
        keys.add(key)
        if record["selected_dev_epoch"] is None:
            raise ValueError(f"Missing dev-selected checkpoint: {path}")
        if record["synthetic_path"] is not None:
            quality = record["synthetic_quality"]
            if quality["rows"] != 100_000 or quality["synthetic_label_counts"] != record["synthetic_label_counts"]:
                raise ValueError(f"Synthetic counts do not match the frozen run specification: {path}")
        predictions = pd.read_csv(record["predictions_path"])
        expected_ids = record["split_manifest"]["splits"]["test"]
        if predictions["source_id"].tolist() != expected_ids:
            raise ValueError(f"Prediction source IDs do not match split manifest: {path}")
        if len(predictions) != len(expected_ids):
            raise ValueError(f"Prediction count does not match test split: {path}")
        row = {
            "run_record": str(path),
            "predictions": record["predictions_path"],
            "dataset": record["dataset"],
            "train_option": record["train_option"],
            "augment_option": record["augment_option"] or "real",
            "seed": record["seed"],
            "selected_dev_epoch": record["selected_dev_epoch"],
            "test_loss": record["test_loss"],
        }
        row.update(record["test_scores"])
        rows.append(row)

    planned = expected_runs(matrix)
    missing = planned - keys
    unexpected = keys - planned
    if missing or unexpected:
        raise ValueError(f"Corrected matrix incomplete: {len(missing)} missing, {len(unexpected)} unexpected runs")

    run_table = pd.DataFrame(rows).sort_values(
        ["dataset", "train_option", "augment_option", "seed"]
    )
    id_columns = {"run_record", "predictions", "dataset", "train_option",
                  "augment_option", "seed", "selected_dev_epoch"}
    metric_columns = [name for name in run_table if name not in id_columns]
    summary = run_table.groupby(
        ["dataset", "train_option", "augment_option"], dropna=False
    )[metric_columns].agg(["count", "mean", "std"])
    summary.columns = [f"{name}_{statistic}" for name, statistic in summary.columns]
    output_root.mkdir(parents=True, exist_ok=True)
    run_table.to_csv(output_root / "per_run.csv", index=False)
    summary_table = summary.reset_index()
    summary_table.to_csv(output_root / "summary.csv", index=False)
    primary_metric = {
        "adult": "f1_macro", "census_kdd": "f1_macro", "credit": "f1_macro",
        "covertype": "f1_macro", "intrusion": "f1_macro",
        "mnist12": "accuracy", "mnist28": "accuracy", "news": "r2",
    }
    for dataset, group in summary_table.groupby("dataset"):
        metric = primary_metric[dataset]
        labels = group["train_option"] + ": " + group["augment_option"]
        fig, axis = plt.subplots(figsize=(max(8, len(group) * 0.6), 5))
        axis.bar(labels, group[f"{metric}_mean"], yerr=group[f"{metric}_std"].fillna(0))
        axis.set_ylabel(metric)
        axis.set_title(dataset)
        axis.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        fig.savefig(output_root / f"{dataset}_{metric}.png")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("output/corrected_v2"))
    parser.add_argument("--out", type=Path, default=Path("output/corrected_v2/results"))
    parser.add_argument("--matrix", choices=('full', 'pilot'), default='full')
    arguments = parser.parse_args()
    build(arguments.runs, arguments.out, arguments.matrix)
