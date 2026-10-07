"""Three remote seed-42 Credit pilots using the verified weighted procedure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_census_weighted_pilot as weighted

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "credit_weighted_macro_f1_pilot_20261006"
OUT = ROOT / "output" / NAMESPACE
ARMS = weighted.ARMS


def preflight():
    import pandas as pd
    import torch
    from importlib.metadata import version
    assert torch.cuda.is_available(), "Credit pilots require remote CUDA"
    assert __import__("shutil").disk_usage(ROOT).free > 1024 ** 3, "Less than 1 GB free for saved pilot artifacts"
    prepared = ROOT / "data/corrected_v2/credit/seed_42"
    manifest_path = prepared / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"] == "credit" and manifest["seed"] == 42
    ids = sum((manifest["splits"][name] for name in ("train", "dev", "test")), [])
    assert len(ids) == len(set(ids))
    paths = {manifest_path}
    report = {"checked_at_chicago": weighted.now(), "dataset": "credit", "seed": 42,
              "runner_sha256": weighted.sha256(__file__),
              "weighted_helper_sha256": weighted.sha256(weighted.__file__),
              "packages": {name: version(name) for name in ("torch", "sdv", "ctgan", "pandas", "numpy", "scikit-learn")},
              "input_sha256": {}, "label_counts": {}}
    for split in ("train", "dev", "test"):
        for view in ("raw", "onehot"):
            path = prepared / f"credit_seed42_real_{split}_{view}.csv"
            paths.add(path)
            if view == "raw":
                labels = pd.read_csv(path, usecols=["Class"]).Class
                assert len(labels) == len(manifest["splits"][split])
                report["label_counts"][split] = {str(k): int(v) for k, v in labels.value_counts().items()}
    raw = pd.read_csv(prepared / "credit_seed42_real_train_raw.csv")
    for fit in ("full", "xonly"):
        model = ROOT / "sdv trained model/corrected_v2/credit/seed_42" / f"credit_seed42_ctgan_{fit}.pkl"
        provenance = model.with_suffix(".provenance.json")
        saved = json.loads(provenance.read_text())
        assert saved["seed"] == 42 and saved["sample_size"] == 100000
        params = saved["parameters"]
        assert params["epochs"] == 500 and params["batch_size"] == 500 and params["cuda"]
        fit_table = raw if fit == "full" else raw.drop(columns="Class")
        assert list(fit_table.columns) == saved["training_columns"] and len(fit_table) == saved["training_rows"]
        assert hashlib.sha256(fit_table.to_csv(index=False, lineterminator="\n").encode()).hexdigest() == saved["fit_table_sha256"]
        table = prepared / f"credit_seed42_ctgan_{fit}_dnn_100k.csv"
        quality_path = table.with_suffix(".quality.json")
        quality = json.loads(quality_path.read_text())
        counts = {str(k): int(v) for k, v in pd.read_csv(table, usecols=["Class"]).Class.value_counts().items()}
        assert counts == quality["synthetic_label_counts"] and sum(counts.values()) == quality["rows"] == 100000
        assert set(counts) == {"0", "1"}
        report["label_counts"][fit] = counts
        paths.update((model, provenance, table, quality_path, table.with_suffix(".predictor.pt"), table.with_suffix(".dnn.json")))
    for path in sorted(paths):
        report["input_sha256"][str(path)] = weighted.sha256(path)
    logits = torch.tensor([-100., 0., 100.], device="cuda", requires_grad=True)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([500.], device="cuda"))(
        logits, torch.tensor([1., 1., 0.], device="cuda"))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        path = weighted.run_arm(42, args.arm, namespace=NAMESPACE, dataset="credit")
        record = json.loads(path.read_text())
        record["evaluation_protocol"] = {
            "dataset": "credit", "seed": 42, "namespace": NAMESPACE,
            "runner_sha256": weighted.sha256(__file__),
            "input_manifest": str(OUT / "manifest.json"),
            "objective": "BCEWithLogitsLoss", "selection_metric": "f1_macro",
            "threshold": .5, "shuffle": False,
            "note": "Combined class weighting and macro-F1 checkpoint selection; unchanged Credit architecture and splits"}
        path.write_text(json.dumps(record, indent=2))
        return
    checks = preflight()
    OUT.mkdir(exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps({"created_at_chicago": weighted.now(), "planned_runs": 3,
        "arms": ARMS, "preflight": checks, "procedure": {
            "objective": "BCEWithLogitsLoss", "positive_weight": "actual training negatives / positives",
            "checkpoint_selection": "real-development macro F1", "threshold": .5, "shuffle": False,
            "batch_size": 128, "learning_rate": .001, "epoch_budget": 100, "patience": 30,
            "architecture": "existing Credit DNN_Adult [128,64,32,16] with existing dropout/BatchNorm",
            "input_namespace": "corrected_v2", "seed": 42}}, indent=2))
    (OUT / "README.md").write_text(
        "# Credit weighted macro-F1 pilot\n\n"
        "Three seed-42 runs: real-only and synthetic-only CTGAN full-table+DNN / features-only+DNN. Existing generator, labeler, and input artifacts are reused.\n\n"
        "Class-weighted BCEWithLogitsLoss (negative/positive training counts), real-development macro-F1 checkpoint selection, strict probability > 0.5. Existing Credit architecture, split, scaling, batch order, batch 128, learning rate 0.001, maximum 100 epochs, patience 30.\n\n"
        "Original results remain in corrected_v2; these outputs are separate. manifest.json records protocol and input hashes; run records contain per-arm weights and selected development diagnostics. completed.json requires all three runs. Failures surface in queue/arm logs; no retries or model substitutions.\n\n"
        "Both objective and checkpoint selection changed. Evaluate minority precision/recall and F1, not merely positive predictions. The real test set has only 10 fraud cases, so also inspect confusion counts.\n")
    logs = OUT / "logs"
    logs.mkdir()
    for arm in ARMS:
        assert __import__("shutil").disk_usage(ROOT).free > 1024 ** 3, "Less than 1 GB free for saved pilot artifacts"
        print(f"{weighted.now()} START seed=42 arm={arm}", flush=True)
        with (logs / (arm + ".log")).open("x") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--arm", arm],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{weighted.now()} COMPLETE seed=42 arm={arm}", flush=True)
    (OUT / "completed.json").write_text(json.dumps({"completed_at_chicago": weighted.now(), "runs": 3}))


if __name__ == "__main__":
    main()
