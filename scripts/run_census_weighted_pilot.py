"""Isolated remote Census pilot; never modifies production code or inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"original": ("original", None),
        "ctgan_full_dnn": ("synthetic", "compare_dnn"),
        "ctgan_xonly_dnn": ("synthetic", "dnn")}
NAMESPACE = "census_weighted_pilot_20261004"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


def preflight():
    import torch
    from importlib.metadata import version
    if not torch.cuda.is_available():
        raise RuntimeError("The pilot requires the remote CUDA environment")
    result = {"checked_at_chicago": now(), "packages": {
        name: version(name) for name in ("torch", "sdv", "ctgan", "pandas", "numpy", "scikit-learn")},
        "pilot_source_sha256": sha256(__file__), "input_sha256": {}}
    for seed in (42,):
        prepared = ROOT / "data/corrected_v2/census_kdd" / f"seed_{seed}"
        manifest = prepared / "split_manifest.json"
        split = json.loads(manifest.read_text())
        assert split["dataset"] == "census_kdd" and split["seed"] == seed
        ids = sum((split["splits"][name] for name in ("train", "dev", "test")), [])
        assert len(ids) == len(set(ids)), "Source partitions overlap"
        paths = [manifest]
        for name in ("train", "dev", "test"):
            for view in ("onehot", "raw"):
                paths.append(prepared / f"census_kdd_seed{seed}_real_{name}_{view}.csv")
        for fit in ("full", "xonly"):
            table = prepared / f"census_kdd_seed{seed}_ctgan_{fit}_dnn_100k.csv"
            paths.extend([table, table.with_suffix(".quality.json"),
                          table.with_suffix(".dnn.json"), table.with_suffix(".predictor.pt")])
            model = ROOT / "sdv trained model/corrected_v2/census_kdd" / f"seed_{seed}" / f"census_kdd_seed{seed}_ctgan_{fit}.pkl"
            provenance = model.with_suffix(".provenance.json")
            saved = json.loads(provenance.read_text())
            assert saved["seed"] == seed and saved["sample_size"] == 100000
            params = saved["parameters"]
            assert params["epochs"] == 500 and params["batch_size"] == 500 and params["cuda"]
            real_raw = prepared / f"census_kdd_seed{seed}_real_train_raw.csv"
            import pandas as pd
            fit_table = pd.read_csv(real_raw)
            if fit == "xonly":
                fit_table = fit_table.drop(columns="income")
            assert list(fit_table.columns) == saved["training_columns"]
            assert len(fit_table) == saved["training_rows"]
            assert saved["fit_table_sha256"] == hashlib.sha256(
                fit_table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
            paths.extend([model, provenance])
        for path in paths:
            result["input_sha256"][str(path)] = sha256(path)
    # Focused objective check, including stable gradients on extreme logits.
    logits = torch.tensor([-100., 0., 100.], device="cuda", requires_grad=True)
    labels = torch.tensor([1., 1., 0.], device="cuda")
    weight = torch.tensor(3., device="cuda")
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)(logits, labels)
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none") * torch.where(labels == 1, weight, 1.)).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    import numpy as np
    assert json.loads(json.dumps({"resolved": bool(np.float64(1.) > 0)}))["resolved"] is True
    return result


def run_arm(seed, arm):
    os.environ["CORRECTED_RUN_NAMESPACE"] = "corrected_v2"
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(ROOT / "src"))
    import random
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from modeling.classification_train import train, device
    from modeling import constants

    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    class WeightedCensus(train):
        def setup_trainer(self, pre_trained_w_file):
            super().setup_trainer(pre_trained_w_file)
            labels = self.train_data.dataset.tensors[1]
            assert set(labels.unique().tolist()) == {0., 1.}
            positives = int((labels == 1).sum())
            negatives = int((labels == 0).sum())
            self.class_counts = {"0": negatives, "1": positives}
            self.pos_weight = negatives / positives
            self.trainer.model.sigmoid = nn.Identity()
            self.trainer.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=device))
            print(f"PILOT training_counts={self.class_counts} pos_weight={self.pos_weight}", flush=True)

        def validate(self, data, load_weight=False):
            if load_weight:
                self.trainer.model.load_state_dict(torch.load(
                    self.w_dir + self.w_file_name, weights_only=True))
            self.trainer.model.eval()
            loss, total = 0., 0
            labels, predictions, probabilities = [], [], []
            with torch.no_grad():
                for X, y in data:
                    X, y = X.to(device), y.to(device).unsqueeze(1)
                    logits = self.trainer.model(X)
                    probability = logits.sigmoid()
                    batch_loss = self.trainer.criterion(logits, y)
                    assert torch.isfinite(batch_loss), "Nonfinite weighted loss"
                    loss += batch_loss.item() * len(y)
                    total += len(y)
                    labels.extend(y.cpu().numpy().ravel().tolist())
                    predictions.extend((probability > .5).float().cpu().numpy().ravel().tolist())
                    probabilities.extend(probability.cpu().numpy().ravel().tolist())
            scores = self.compute_scores(labels, predictions, probabilities)
            diagnostics = {"rows": total, "predicted_positive": int(sum(predictions)),
                           "positive_rate": sum(predictions) / total,
                           "probability_min": min(probabilities),
                           "probability_max": max(probabilities), "scores": scores}
            if data is self.dev_data:
                self.dev_diagnostics = diagnostics
                print("PILOT development " + json.dumps(diagnostics), flush=True)
            if load_weight and data is self.test_data:
                self.test_diagnostics = diagnostics
                pd.DataFrame({"source_id": self.data_loader.test_source_ids,
                              "y_true": labels, "y_pred": predictions,
                              "score": probabilities}).to_csv(
                    Path(self.acc_dir) / (self.run_id + ".predictions.csv"), index=False)
            return loss / total, scores

    mode, method = ARMS[arm]
    output = ROOT / "output" / NAMESPACE / "census_kdd"
    run_id = constants.run_name("census_kdd", seed, mode, method)
    if (output / "acc" / (run_id + ".run.json")).exists():
        raise FileExistsError("Refusing to overwrite a completed pilot arm")
    metrics = {"accuracy": None, "balanced_accuracy": None, "pr_auc": None,
               "roc_auc": None, "f1": ["macro", "binary"],
               "precision": ["binary"], "recall": ["binary"]}
    started = now()
    pilot = WeightedCensus(
        dataset_name="census_kdd", train_option=mode, augment_option=method,
        test_option="original", validation=.2, batch_size=128,
        learning_rate=.001, num_epochs=100, patience=30,
        early_stop_criterion="f1_macro", seed=seed,
        eval_metrics=metrics, metric_to_plot="f1_macro",
        w_dir=str(output / "weight") + "/", acc_dir=str(output / "acc") + "/")
    if method is not None:
        assert len(pilot.train_data.dataset) == 100000
    pilot.training()
    pilot.validate(pilot.dev_data, load_weight=True)
    record_path = output / "acc" / (run_id + ".run.json")
    record = json.loads(record_path.read_text())
    record["pilot_protocol"] = {
        "namespace": NAMESPACE, "input_namespace": "corrected_v2",
        "started_at_chicago": started, "completed_at_chicago": now(),
        "pilot_source_sha256": sha256(__file__),
        "objective": "BCEWithLogitsLoss", "model_output": "logits",
        "pos_weight_rule": "actual_training_negatives / actual_training_positives",
        "pos_weight": pilot.pos_weight, "training_label_counts": pilot.class_counts,
        "shuffle": False, "threshold": .5,
        "test_loss_definition": "weighted BCE using this arm's training pos_weight",
        "selected_development": pilot.dev_diagnostics,
        "test_diagnostics": pilot.test_diagnostics,
        "development_collapse_resolved": bool(
            0 < pilot.dev_diagnostics["predicted_positive"] < pilot.dev_diagnostics["rows"]
            and pilot.dev_diagnostics["scores"]["recall_binary"] > 0)}
    record_path.write_text(json.dumps(record, indent=2))
    print("PILOT COMPLETED " + json.dumps(record["pilot_protocol"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42,))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(preflight(), indent=2))
    elif args.arm:
        if args.seed is None:
            parser.error("--arm requires --seed")
        run_arm(args.seed, args.arm)
    else:
        checks = preflight()
        output = ROOT / "output" / NAMESPACE
        if args.resume:
            previous = json.loads((output / "preflight.json").read_text())
            assert previous["input_sha256"] == checks["input_sha256"]
            (output / "resume_preflight.json").write_text(json.dumps(checks, indent=2))
        else:
            output.mkdir(exist_ok=False)
            (output / "preflight.json").write_text(json.dumps(checks, indent=2))
        logs = output / "logs"
        logs.mkdir(exist_ok=args.resume)
        for seed in (42,):
            for arm in ARMS:
                if args.resume:
                    mode, method = ARMS[arm]
                    name = (f"census_kdd_seed{seed}_real_original" if method is None
                            else f"census_kdd_seed{seed}_ctgan_{'full' if method == 'compare_dnn' else 'xonly'}_dnn_synthetic")
                    saved = output / "census_kdd/acc" / (name + ".run.json")
                    if saved.exists():
                        record = json.loads(saved.read_text())
                        assert record["pilot_protocol"]["objective"] == "BCEWithLogitsLoss"
                        assert record["selection_metric"] == "f1_macro"
                        assert Path(record["downstream_weight_path"]).is_file()
                        assert Path(record["predictions_path"]).is_file()
                        print(f"{now()} PRESERVED seed={seed} arm={arm}", flush=True)
                        continue
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--seed", str(seed), "--arm", arm]
                print(f"{now()} START seed={seed} arm={arm}", flush=True)
                with (logs / f"seed{seed}_{arm}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                print(f"{now()} COMPLETE seed={seed} arm={arm}", flush=True)
        (output / "completed.json").write_text(json.dumps({"completed_at_chicago": now(), "runs": 3}))


if __name__ == "__main__":
    main()
