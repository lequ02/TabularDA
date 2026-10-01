"""Run the paper's seven simulated oracles and a separate labeled-method extension.

From SDGym-research: python -m synthetic_data_benchmark.run_simulated_methods
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .simulated_benchmark import (
    DATASETS, MIXTURES, bn_model, evaluate, generate, mixture_oracle,
    prepare, read_table, validate_table,
)


LABELERS = ("gaussian", "categorical", "pca_gmm", "rf", "xgb", "dnn")
PAPER_METHODS = ("identity", "ctgan", "tvae")
GENERATORS = ("ctgan", "tvae")
TARGETS = {"asia": "dysp", "alarm": "BP", "child": "Disease",
           "insurance": "Accident"}
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "synthesize_data"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_device(requested, gpu_index):
    import torch

    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu-index {gpu_index} is outside the {torch.cuda.device_count()} available GPUs")
        torch.cuda.set_device(gpu_index)
        print(f"Using CUDA GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}",
              flush=True)
    return device


def requested_methods(names):
    if names == ["all"]:
        return list(PAPER_METHODS) + [
            f"{generator}-{suffix}"
            for generator in GENERATORS
            for suffix in ("full", *(LABELERS),
                           *(f"full-{labeler}" for labeler in LABELERS))
        ]
    allowed = set(PAPER_METHODS) | {
        f"{generator}-{suffix}" for generator in GENERATORS
        for suffix in ("full", "nb", *(LABELERS),
                       *(f"full-{labeler}" for labeler in LABELERS), "full-nb")
    }
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return list(dict.fromkeys(names))


def verified_split(root, dataset, seed, train_rows, test_rows):
    folder = root / f"seed_{seed}" / dataset
    if not folder.exists():
        raise FileNotFoundError(
            f"Prepared data is missing: {folder}. Run the prepare command first.")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["dataset"], manifest["seed"], manifest["train_rows"],
            manifest["test_rows"]) != (dataset, seed, train_rows, test_rows):
        raise ValueError(f"Prepared split configuration differs: {folder}")
    if sha256(Path(manifest["oracle_path"])) != manifest["oracle_sha256"]:
        raise ValueError(f"Oracle changed: {folder}")
    for split in ("train", "test"):
        if sha256(folder / f"{split}.csv") != manifest[f"{split}_sha256"]:
            raise ValueError(f"Prepared {split} changed: {folder}")
    train = read_table(folder / "train.csv", dataset)
    test = read_table(folder / "test.csv", dataset)
    if len(train) != train_rows or len(test) != test_rows:
        raise ValueError(f"Prepared split row count differs: {folder}")
    oracle = (mixture_oracle(dataset, seed) if dataset in MIXTURES
              else manifest["categories"])
    for table in (train, test):
        validate_table(table, manifest["columns"], oracle, dataset)
    return folder, manifest, train, test


def labeled(table, dataset):
    if dataset in MIXTURES:
        result = table.copy()
        result["label"] = (result.feature_1 > 1.5 * result.feature_0 + 0.8).astype(int)
        return result
    return table.copy()


def target_name(dataset):
    return "label" if dataset in MIXTURES else TARGETS[dataset]


def sample_and_save(folder, name, train, dataset, seed, rows, epochs, device):
    path = folder / f"{name}_sample.csv"
    meta_path = path.with_suffix(".json")
    settings = {
        "name": name, "dataset": dataset, "seed": seed, "rows": rows,
        "epochs": epochs, "device": device,
        "train_sha256": hashlib.sha256(train.to_csv(index=False).encode("utf-8")).hexdigest(),
    }
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists():
            raise ValueError(f"Saved sample provenance is incomplete: {path}. Use a new --root.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in settings.items()):
            raise ValueError(f"Saved sample settings differ: {path}. Use a new --root.")
        if metadata.get("sample_sha256") != sha256(path):
            raise ValueError(f"Saved sample changed: {path}")
        sample = read_table(path, dataset)
    else:
        method = name.split("_")[0]
        if name.endswith("_full") and dataset in MIXTURES:
            import torch
            from ctgan import CTGAN, TVAE

            torch.manual_seed(seed)
            np.random.seed(seed)
            cls = {"ctgan": CTGAN, "tvae": TVAE}[method]
            model = cls(epochs=epochs, batch_size=500, cuda=device == "cuda")
            model.fit(train, discrete_columns=["label"])
            sample = model.sample(rows)
        else:
            sample = generate(method, train, dataset, seed, rows, epochs, device)
        sample.to_csv(path, index=False)
    if list(sample.columns) != list(train.columns) or len(sample) != rows:
        raise ValueError(f"Saved sample schema or row count differs: {path}")
    if not meta_path.exists():
        meta_path.write_text(json.dumps({**settings, "sample_sha256": sha256(path)}, indent=2),
                             encoding="utf-8")
    return sample, path


def encode_features(train, dev, synthetic, dataset):
    if dataset in MIXTURES:
        return train.astype(float), dev.astype(float), synthetic.astype(float)
    categories = {column: sorted(train[column].unique()) for column in train}
    encoded = []
    for frame in (train, dev, synthetic):
        parts = []
        for column in train:
            for state in categories[column]:
                parts.append(pd.Series((frame[column] == state).astype(int).to_numpy(),
                                       name=f"{column}={state}"))
        encoded.append(pd.concat(parts, axis=1))
    return tuple(encoded)


def predict_labels(labeler, x_train, y_train, x_dev, y_dev, x_synthetic,
                   dataset, seed, report_path, device):
    if labeler == "nb":
        labeler = "gaussian" if dataset in MIXTURES else "categorical"
    if labeler in ("gaussian", "categorical"):
        from synthesize_data.naive_bayes import (
            create_label_categoricalNB, create_label_gaussianNB)

        function = (create_label_gaussianNB if labeler == "gaussian"
                    else create_label_categoricalNB)
        return function(x_train, y_train, x_synthetic, "label")["label"]
    if labeler in ("rf", "xgb"):
        from synthesize_data.ensemble import Ensemble

        _, result = Ensemble(x_train, y_train, x_synthetic, "label", labeler,
                             str(report_path.with_suffix(".encoded.csv")),
                             verbose=False, random_state=seed).fit()
        report_path.with_suffix(".encoded.csv").unlink()
        return result["label"]
    if labeler == "pca_gmm":
        from synthesize_data.pca_gmm import PCA_GMM

        numerical = list(x_train.columns) if dataset in MIXTURES else []
        _, result = PCA_GMM(x_train.copy(), y_train, x_synthetic.copy(),
                            numerical, "label", verbose=False).fit()
        return result["label"]
    if labeler == "dnn":
        from synthesize_data.dnn_labeler import fit_predict_dnn

        result = fit_predict_dnn(
            x_train, y_train, x_dev, y_dev, x_synthetic,
            target_name="label", is_classification=True, seed=seed,
            report_path=report_path, dataset_name=dataset,
            device_name=device, enforce_quality_gate=False)
        return result["label"]
    raise ValueError(f"Unknown labeler: {labeler}")


def checked_score(folder, root, dataset, seed, name, sample_path, rows,
                  epochs, device):
    path = folder / f"{name}_result.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (result["synthetic_sha256"] != sha256(sample_path) or
                result["synthetic_rows"] != rows or
                result["dataset"] != dataset or result["seed"] != seed or
                result["method"] != name or
                any(result.get(key) != manifest[key] for key in
                    ("train_sha256", "test_sha256", "oracle_sha256"))):
            raise ValueError(f"Existing score inputs differ: {path}. Use a new --root.")
        return result
    result = evaluate(dataset, seed, name, root, sample_path, rows, epochs, device)
    if name in GENERATORS:
        result["model_epochs"] = epochs
        result["batch_size"] = 500
        result["device"] = device
        result["package_versions"]["ctgan"] = version("ctgan")
        result["package_versions"]["torch"] = version("torch")
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score_labeled(folder, root, dataset, seed, name, synthetic, test,
                  source_path, manifest, rows, epochs, device):
    target = target_name(dataset)
    sample_path = folder / f"{name}_labeled.csv"
    result_path = folder / f"{name}_labeled.json"
    if sample_path.exists():
        if not result_path.exists():
            raise ValueError(f"Labeled result is incomplete or changed: {sample_path}")
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (sha256(sample_path) != previous["synthetic_sha256"] or
                sha256(source_path) != previous["source_sha256"] or
                previous["train_sha256"] != manifest["train_sha256"] or
                previous["test_sha256"] != manifest["test_sha256"] or
                previous["synthetic_rows"] != rows or
                previous["epochs"] != epochs or previous["device"] != device or
                (previous.get("dataset"), previous.get("seed"), previous.get("method")) !=
                (dataset, seed, name) or
                previous.get("oracle_sha256") != manifest["oracle_sha256"]):
            raise ValueError(f"Labeled result inputs differ: {sample_path}")
        return previous
    if result_path.exists():
        raise ValueError(f"Labeled sample is missing: {sample_path}")
    if synthetic is None:
        raise ValueError(f"Labeled sample is missing: {sample_path}")
    if list(synthetic.columns) != list(test.columns) or len(synthetic) != rows:
        raise ValueError(f"Labeled sample has incorrect schema or rows: {name}")
    allowed_labels = (0, 1) if dataset in MIXTURES else manifest["categories"][target]
    if synthetic[target].isna().any() or not synthetic[target].isin(allowed_labels).all():
        raise ValueError(f"Invalid synthetic labels: {name}")
    synthetic.to_csv(sample_path, index=False)
    x_syn = synthetic.drop(columns=target)
    x_test = test.drop(columns=target)
    if dataset in MIXTURES:
        feature_path = folder / f"{name}_features.csv"
        x_syn.to_csv(feature_path, index=False)
        likelihood = checked_score(folder, root, dataset, seed, name, feature_path,
                                   rows, epochs, device)
        oracle_labels = labeled(x_syn, dataset)[target]
        consistency = float(accuracy_score(oracle_labels, synthetic[target]))
    else:
        likelihood = checked_score(folder, root, dataset, seed, name,
                                   sample_path, rows, epochs, device)
        consistency = None
    x_fit, x_holdout, _ = encode_features(x_syn, x_test, x_test, dataset)
    classifier = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    classifier.fit(x_fit, synthetic[target])
    predictions = classifier.predict(x_holdout)
    result = {
        "oracle_sha256": manifest["oracle_sha256"],
        "dataset": dataset, "seed": seed, "method": name,
        "benchmark": "labeled_extension", "target": target,
        "test_accuracy": float(accuracy_score(test[target], predictions)),
        "test_macro_f1": float(f1_score(test[target], predictions,
                                         average="macro", zero_division=0)),
        "oracle_label_consistency": consistency,
        "l_syn": likelihood["l_syn"], "l_test": likelihood["l_test"],
        "likelihood_scope": "features" if dataset in MIXTURES else "joint",
        "synthetic_path": str(sample_path.resolve()),
        "synthetic_sha256": sha256(sample_path),
        "source_path": str(source_path.resolve()), "source_sha256": sha256(source_path),
        "train_sha256": manifest["train_sha256"],
        "test_sha256": manifest["test_sha256"],
        "synthetic_rows": rows, "epochs": epochs, "device": device,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def prepared_dev(folder, dataset, seed, rows):
    path = folder / "labeled_dev.csv"
    meta_path = folder / "labeled_dev.json"
    settings = {"dataset": dataset, "seed": seed, "rows": rows}
    if path.exists() or meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(meta[key] != value for key, value in settings.items()):
            raise ValueError(f"Dev split settings differ: {path}")
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Dev split changed: {path}")
        return read_table(path, dataset)
    if dataset in MIXTURES:
        from .simulated_benchmark import sample_mixture

        dev = labeled(sample_mixture(mixture_oracle(dataset, seed),
                                     rows, seed + 3), dataset)
    else:
        from pgmpy.sampling import BayesianModelSampling

        dev = BayesianModelSampling(bn_model(dataset)).forward_sample(
            size=rows, seed=seed + 3, show_progress=False).astype(str)
    dev.to_csv(path, index=False)
    meta_path.write_text(json.dumps({**settings, "sha256": sha256(path)}, indent=2),
                         encoding="utf-8")
    return dev


def run_one(root, dataset, seed, methods, train_rows, test_rows, rows,
            epochs, device):
    folder, manifest, paper_train, paper_test = verified_split(
        root, dataset, seed, train_rows, test_rows)
    target = target_name(dataset)
    train = labeled(paper_train, dataset)
    test = labeled(paper_test, dataset)
    dev = None
    if any(method.endswith("dnn") for method in methods):
        dev = prepared_dev(folder, dataset, seed, test_rows)[list(test.columns)]
    for name in methods:
        suffix = "result" if name in PAPER_METHODS else "labeled"
        completed = (folder / f"{name}_{suffix}.json").exists()
        print(f"{'skip' if completed else 'run'} {dataset} seed={seed} method={name}",
              flush=True)
        if name == "identity":
            if rows != train_rows:
                raise ValueError("identity requires synthetic_rows = train_rows")
            source = folder / "train.csv"
            checked_score(folder, root, dataset, seed, name, source, rows, epochs, device)
            continue
        generator = name.split("-")[0]
        if name in ("ctgan", "tvae"):
            sample, source = sample_and_save(folder, f"{name}_paper", paper_train,
                                             dataset, seed, rows, epochs, device)
            checked_score(folder, root, dataset, seed, name, source, rows,
                          epochs, device)
            continue
        suffix = name[len(generator) + 1:]
        full = suffix == "full" or suffix.startswith("full-")
        labeler = suffix[5:] if suffix.startswith("full-") else suffix
        if full:
            if dataset in MIXTURES:
                source_train = train
                source_name = f"{generator}_full"
            else:
                source_train = paper_train
                source_name = f"{generator}_paper"
        else:
            source_train = train.drop(columns=target)
            source_name = (f"{generator}_paper" if dataset in MIXTURES else
                           f"{generator}_xonly")
        source_sample, source_path = sample_and_save(
            folder, source_name, source_train, dataset, seed, rows, epochs, device)
        if (folder / f"{name}_labeled.json").exists():
            score_labeled(folder, root, dataset, seed, name, None, test,
                          source_path, manifest, rows, epochs, device)
            continue
        if suffix == "full":
            synthetic = source_sample.copy()
        else:
            x_raw = source_sample.drop(columns=target) if full else source_sample
            x_train, x_dev, x_synthetic = encode_features(
                train.drop(columns=target),
                (dev if labeler == "dnn" else train).drop(columns=target),
                x_raw, dataset)
            predictions = predict_labels(
                labeler, x_train, train[target], x_dev,
                (dev if labeler == "dnn" else train)[target], x_synthetic,
                dataset, seed, folder / f"{name}_labeler.json", device)
            synthetic = x_raw.copy()
            synthetic[target] = np.asarray(predictions)
            synthetic = synthetic[list(train.columns)]
        score_labeled(folder, root, dataset, seed, name, synthetic, test,
                      source_path, manifest, rows, epochs, device)


def summarize_runs(root, datasets, seeds, methods):
    records = []
    for seed in seeds:
        for dataset in datasets:
            folder = root / f"seed_{seed}" / dataset
            for method in methods:
                suffix = "result" if method in PAPER_METHODS else "labeled"
                path = folder / f"{method}_{suffix}.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record.get("method"), record.get("seed"), record.get("dataset")) != (
                        method, seed, dataset):
                    raise ValueError(f"Result identity does not match its path: {path}")
                record["benchmark"] = ("paper" if method in PAPER_METHODS
                                       else "labeled_extension")
                record["family"] = "GM" if dataset in MIXTURES else "BN"
                records.append(record)
    table = pd.DataFrame(records)
    columns = ["l_syn", "l_test", "test_accuracy", "test_macro_f1",
               "oracle_label_consistency"]
    for column in columns:
        if column not in table:
            table[column] = np.nan
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "simulated_methods_per_run.csv", index=False)
    summary = table.groupby(["benchmark", "method", "family"], as_index=False)[
        columns].mean()
    summary.to_csv(root / "simulated_methods_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"),
                        help="write shared simulated tables, or run methods")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--root", type=Path, default=Path("data/simulated_paper"))
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--synthetic-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="GPU by default when available; otherwise CPU")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="CUDA device used for CTGAN, TVAE, and DNN fits")
    args = parser.parse_args()
    methods = requested_methods(args.methods)
    if args.action == "prepare":
        for seed in args.seeds:
            for dataset in args.datasets:
                folder = args.root / f"seed_{seed}" / dataset
                if folder.exists():
                    raise FileExistsError(f"Prepared data already exists: {folder}")
                print(prepare(dataset, seed, args.train_rows, args.test_rows,
                              args.root), flush=True)
        return
    device = resolve_device(args.device, args.gpu_index)
    for seed in args.seeds:
        for dataset in args.datasets:
            run_one(args.root, dataset, seed, methods, args.train_rows,
                    args.test_rows, args.synthetic_rows, args.epochs,
                    device)
    print(summarize_runs(args.root, args.datasets, args.seeds, methods).to_string(index=False))


if __name__ == "__main__":
    main()
