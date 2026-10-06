"""Census-only weighted/macro-F1 evaluation, separate from corrected_v2."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_census_weighted_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "census_kdd_weighted_macro_f1_20261005"
OUT = ROOT / "output" / NAMESPACE
os.environ["CORRECTED_RUN_NAMESPACE"] = "corrected_v2"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT / "src"))
from modeling import constants
from run_corrected_matrix import methods_for

PROTOCOL = {
    "dataset": "census_kdd", "input_namespace": "corrected_v2",
    "output_namespace": NAMESPACE, "seeds": [42, 43],
    "objective": "BCEWithLogitsLoss", "model_output": "logits",
    "pos_weight_rule": "actual_training_negatives / actual_training_positives",
    "selection_metric": "f1_macro", "threshold": .5, "shuffle": False,
    "architecture": "DNN_Census [256,128,64,32], BatchNorm/ReLU/dropout 0.6",
    "batch_size": 128, "learning_rate": .001, "epoch_budget": 100, "patience": 30,
    "mix_rule": "all real training rows + all 100000 synthetic rows",
    "test_loss_definition": "weighted BCE with each arm's training-derived weight; not comparable across arms",
    "interpretation": "Both objective and checkpoint selection changed; pilot does not isolate their effects",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def configurations():
    jobs = []
    for seed in (42, 43):
        for mode, method in [("original", None)] + [
                (mode, method) for generator in ("ctgan", "tvae")
                for method in methods_for("census_kdd", generator)
                for mode in ("synthetic", "mix")]:
            jobs.append({"seed": seed, "train_option": mode, "augment_option": method,
                         "run_id": constants.run_name("census_kdd", seed, mode, method)})
    assert len(jobs) == len({job["run_id"] for job in jobs}) == 106
    return jobs


def preflight():
    import pandas as pd
    import torch
    from importlib.metadata import version
    assert torch.cuda.is_available(), "Remote CUDA is required"
    report = {"checked_at_chicago": pilot.now(), "input_sha256": {}, "label_counts": {},
              "packages": {name: version(name) for name in ("torch", "sdv", "ctgan", "pandas", "numpy", "scikit-learn")},
              "runner_sha256": pilot.sha256(__file__), "weighted_trainer_sha256": pilot.sha256(pilot.__file__)}
    for seed in (42, 43):
        prepared = ROOT / "data/corrected_v2/census_kdd" / f"seed_{seed}"
        manifest_path = prepared / "split_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == seed and manifest["dataset"] == "census_kdd"
        ids = sum((manifest["splits"][name] for name in ("train", "dev", "test")), [])
        assert len(ids) == len(set(ids))
        paths = {manifest_path}
        for split in ("train", "dev", "test"):
            for view in ("raw", "onehot"):
                paths.add(prepared / constants.split_name("census_kdd", seed, split, view))
        real_raw = pd.read_csv(prepared / constants.split_name("census_kdd", seed, "train", "raw"))
        for generator in ("ctgan", "tvae"):
            for fit in ("full", "xonly"):
                model = ROOT / "sdv trained model/corrected_v2/census_kdd" / f"seed_{seed}" / constants.generator_name("census_kdd", seed, generator, fit)
                provenance = model.with_suffix(".provenance.json")
                saved = json.loads(provenance.read_text())
                assert saved["seed"] == seed and saved["sample_size"] == 100000
                params = saved["parameters"]
                assert params["epochs"] == 500 and params["batch_size"] == 500 and params["cuda"]
                fit_table = real_raw if fit == "full" else real_raw.drop(columns="income")
                assert saved["training_columns"] == list(fit_table.columns)
                assert saved["training_rows"] == len(fit_table)
                import hashlib
                assert saved["fit_table_sha256"] == hashlib.sha256(fit_table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
                paths.update((model, provenance))
            for method in methods_for("census_kdd", generator):
                table = prepared / constants.synthetic_name("census_kdd", seed, method)
                quality_path = table.with_suffix(".quality.json")
                quality = json.loads(quality_path.read_text())
                counts = {str(key): int(value) for key, value in pd.read_csv(table, usecols=["income"]).income.value_counts().items()}
                assert counts == quality["synthetic_label_counts"]
                assert set(counts) == {"0", "1"} and sum(counts.values()) == quality["rows"] == 100000
                report["label_counts"][str(table)] = counts
                paths.update((table, quality_path))
                if method not in ("ctgan", "tvae"):
                    paths.add(table.with_suffix(".predictor.pt" if method.endswith("dnn") else ".predictor.pkl"))
                if method.endswith("dnn"):
                    paths.add(table.with_suffix(".dnn.json"))
        for path in sorted(paths):
            report["input_sha256"][str(path)] = pilot.sha256(path)
        print(f"{pilot.now()} PREFLIGHT seed={seed} verified", flush=True)
    # Reuse the pilot's focused numerical and JSON-serialization checks.
    pilot.preflight()
    return report


def verify_record(path):
    import pandas as pd
    from sklearn.metrics import f1_score
    record = json.loads(path.read_text())
    protocol = record["pilot_protocol"]
    assert protocol["objective"] == "BCEWithLogitsLoss" and protocol["threshold"] == .5
    assert protocol["shuffle"] is False and record["selection_metric"] == "f1_macro"
    assert record["batch_size"] == 128 and record["learning_rate"] == .001 and record["epoch_budget"] == 100
    assert record["selected_dev_epoch"] is not None
    assert Path(record["downstream_weight_path"]).is_file()
    counts = protocol["training_label_counts"]
    assert abs(protocol["pos_weight"] - counts["0"] / counts["1"]) < 1e-12
    predictions = pd.read_csv(record["predictions_path"])
    assert predictions.source_id.tolist() == record["split_manifest"]["splits"]["test"]
    assert (predictions.y_pred == (predictions.score > .5).astype(int)).all()
    assert abs(f1_score(predictions.y_true, predictions.y_pred) - record["test_scores"]["f1_binary"]) < 1e-12
    return record


def import_pilot(checks):
    imports = []
    source = ROOT / "output/census_weighted_pilot_20261004"
    old_checks = json.loads((source / "preflight.json").read_text())
    for path, digest in old_checks["input_sha256"].items():
        assert checks["input_sha256"][path] == digest
    for path in sorted((source / "census_kdd/acc").glob("*.run.json")):
        record = verify_record(path)
        assert record["seed"] == 42
        run_id = path.name.removesuffix(".run.json")
        artifacts = []
        for folder in ("acc", "weight"):
            target = OUT / "census_kdd" / folder
            target.mkdir(parents=True, exist_ok=True)
            for artifact in (source / "census_kdd" / folder).glob(run_id + ".*"):
                if artifact.suffix == ".json":
                    continue
                destination = target / artifact.name
                shutil.copy2(artifact, destination)
                digest = pilot.sha256(artifact)
                assert pilot.sha256(destination) == digest
                artifacts.append({"source": str(artifact), "destination": str(destination), "sha256": digest})
        record["predictions_path"] = str(OUT / "census_kdd/acc" / Path(record["predictions_path"]).name)
        record["downstream_weight_path"] = str(OUT / "census_kdd/weight" / Path(record["downstream_weight_path"]).name)
        record["evaluation_protocol"] = dict(PROTOCOL, origin="verified_pilot_import", source_record=str(path), source_record_sha256=pilot.sha256(path))
        destination = OUT / "census_kdd/acc" / path.name
        write_json(destination, record)
        verify_record(destination)
        imports.append({"run_id": run_id, "source_record": str(path), "source_record_sha256": pilot.sha256(path), "artifacts": artifacts})
    assert len(imports) == 3
    return imports


def train_configuration(seed, mode, method):
    name = constants.run_name("census_kdd", seed, mode, method)
    record_path = pilot.run_arm(seed, name, namespace=NAMESPACE, configuration=(mode, method))
    record = verify_record(record_path)
    record["evaluation_protocol"] = dict(PROTOCOL, origin="new_weighted_training", runner_sha256=pilot.sha256(__file__))
    write_json(record_path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 43))
    parser.add_argument("--mode", choices=("original", "synthetic", "mix"))
    parser.add_argument("--method")
    args = parser.parse_args()
    if args.mode:
        assert args.seed is not None
        train_configuration(args.seed, args.mode, args.method)
        return
    checks = preflight()
    if args.check_only:
        print(json.dumps({"planned_runs": len(configurations()), "input_files": len(checks["input_sha256"]), "preflight": "passed"}), flush=True)
        return
    jobs = configurations()
    if args.resume:
        manifest = json.loads((OUT / "manifest.json").read_text())
        assert manifest["preflight"]["input_sha256"] == checks["input_sha256"]
    else:
        OUT.mkdir(exist_ok=False)
        imports = import_pilot(checks)
        manifest = {"created_at_chicago": pilot.now(), "protocol": PROTOCOL, "planned_runs": 106,
                    "pilot_reused_runs": 3, "new_training_runs": 103, "jobs": jobs,
                    "preflight": checks, "pilot_imports": imports}
        write_json(OUT / "manifest.json", manifest)
        (OUT / "README.md").write_text(
            "# Census KDD weighted macro-F1 evaluation\n\n"
            "106 planned configurations: seeds 42/43, real-only plus CTGAN/TVAE full-generated, full-relabelled, and features-only-relabelled arms under synthetic and mix training.\n\n"
            "Three seed-42 pilot results were copied with verified artifact hashes and explicit source-record provenance; 103 runs are new. Original corrected_v2 and pilot results remain preserved.\n\n"
            "BCEWithLogitsLoss, positive weight = actual training negatives / positives, real-development macro-F1 checkpoint selection, strict probability > 0.5, unshuffled batches. Existing architecture, splits, scaling, learning rate 0.001, batch 128, 100 epochs, patience 30. Mix uses every real training row plus 100,000 synthetic rows.\n\n"
            "manifest.json records the exact protocol, inputs, configurations, and pilot imports. status.json reports current completed/failed/running counts. completed.json exists only after all 106 records pass verification. Logs retain actual failures; there are no automatic retries.\n\n"
            "Binary F1 and macro F1 remain distinct. Weighted losses differ across arms and are not common unweighted BCE. Both weighting and checkpoint selection changed; do not attribute improvements to weighting alone.\n")
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    status = {"planned": 106, "completed": [], "failed": [], "running": None}
    for job in jobs:
        saved = OUT / "census_kdd/acc" / (job["run_id"] + ".run.json")
        if saved.exists():
            verify_record(saved)
            status["completed"].append(job["run_id"])
    for job in jobs:
        name = job["run_id"]
        record_path = OUT / "census_kdd/acc" / (name + ".run.json")
        if record_path.exists():
            verify_record(record_path)
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--seed", str(job["seed"]), "--mode", job["train_option"]]
        if job["augment_option"] is not None:
            command.extend(["--method", job["augment_option"]])
        status["running"] = name
        status["updated_at_chicago"] = pilot.now()
        write_json(OUT / "status.json", status)
        print(f"{pilot.now()} START {name}", flush=True)
        with (logs / (name + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status["failed"].append({"run_id": name, "returncode": result.returncode, "log": str(logs / (name + ".log"))})
            print(f"{pilot.now()} FAILED {name} exit={result.returncode}; see arm log", flush=True)
        else:
            verify_record(record_path)
            status["completed"].append(name)
            print(f"{pilot.now()} COMPLETE {name}", flush=True)
        status["running"] = None
        status["updated_at_chicago"] = pilot.now()
        write_json(OUT / "status.json", status)
    if status["failed"]:
        raise RuntimeError(f"{len(status['failed'])} weighted Census runs failed; see status.json and arm logs")
    assert len(status["completed"]) == 106
    write_json(OUT / "completed.json", {"completed_at_chicago": pilot.now(), "runs": 106, "pilot_reused": 3, "new_training": 103})
    # Use the existing results builder on this exact Census-only matrix.
    import build_corrected_results as builder
    builder.DATASETS = ("census_kdd",)
    builder.build(OUT, OUT / "results")


if __name__ == "__main__":
    main()
