"""Paper-style CTGAN simulated-data benchmark.

Prepare fixed oracle train/test tables, then evaluate every synthesizer against
the same tables with L_syn and L_test from Xu et al. (NeurIPS 2019, section 5).
Run from the SDGym-research directory with ``python -m
synthetic_data_benchmark.simulated_benchmark``.
"""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


DATASETS = ("grid", "gridr", "ring", "asia", "alarm", "child", "insurance")
MIXTURES = ("grid", "gridr", "ring")
BIF_DIR = Path(__file__).resolve().parent / "oracles"


def mixture_oracle(name, seed):
    if name == "ring":
        angles = np.arange(8) * math.pi / 4
        means = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        means = np.array([(x, y) for x in range(-4, 5, 2)
                          for y in range(-4, 5, 2)], dtype=float)
        if name == "gridr":
            means += np.random.default_rng(seed).uniform(-0.5, 0.5, means.shape)
    return {"means": means.tolist(), "variance": 0.05,
            "weights": [1 / len(means)] * len(means)}


def sample_mixture(oracle, size, seed):
    means = np.asarray(oracle["means"])
    rng = np.random.default_rng(seed)
    components = np.arange(size) % len(means)
    rng.shuffle(components)
    points = means[components] + rng.normal(
        scale=math.sqrt(oracle["variance"]), size=(size, 2))
    return pd.DataFrame(points, columns=["feature_0", "feature_1"])


def mixture_log_prob(data, oracle):
    points = data.to_numpy(dtype=float)
    means = np.asarray(oracle["means"])
    variance = oracle["variance"]
    squared = ((points[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    log_components = -math.log(2 * math.pi * variance) - squared / (2 * variance)
    return logsumexp(log_components + np.log(oracle["weights"]), axis=1)


def bn_model(name):
    from pgmpy.readwrite import BIFReader

    return BIFReader(str(BIF_DIR / f"{name}.bif")).get_model()


def bn_log_prob(data, model):
    score = np.zeros(len(data), dtype=float)
    for cpd in model.get_cpds():
        indices = []
        for variable in cpd.variables:
            states = {value: index for index, value in
                      enumerate(cpd.state_names[variable])}
            indices.append(data[variable].map(states).to_numpy(dtype=int))
        probabilities = cpd.values[tuple(indices)]
        score += np.log(np.maximum(probabilities, 1e-8))
    return score


def validate_table(data, columns, oracle, dataset):
    if list(data.columns) != columns or len(data) == 0 or data.isna().any().any():
        raise ValueError(f"{dataset}: invalid columns, empty table, or missing values")
    if dataset in MIXTURES:
        values = data.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{dataset}: non-finite feature value")
    else:
        for column in columns:
            if not data[column].isin(oracle[column]).all():
                raise ValueError(f"{dataset}: unknown state in {column}")


def prepare(dataset, seed, train_rows, test_rows, root):
    if train_rows < 1 or test_rows < 1:
        raise ValueError("train_rows and test_rows must be positive")
    folder = root / f"seed_{seed}" / dataset
    if folder.exists():
        raise FileExistsError(f"Prepared dataset already exists: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if dataset in MIXTURES:
        oracle = mixture_oracle(dataset, seed)
        train = sample_mixture(oracle, train_rows, seed + 1)
        test = sample_mixture(oracle, test_rows, seed + 2)
        oracle_path = folder / "oracle.json"
        oracle_path.write_text(json.dumps(oracle, indent=2), encoding="utf-8")
        categories = None
    else:
        from pgmpy.sampling import BayesianModelSampling

        model = bn_model(dataset)
        train = BayesianModelSampling(model).forward_sample(
            size=train_rows, seed=seed + 1, show_progress=False)
        test = BayesianModelSampling(model).forward_sample(
            size=test_rows, seed=seed + 2, show_progress=False)
        oracle_path = BIF_DIR / f"{dataset}.bif"
        categories = {column: model.get_cpds(column).state_names[column]
                      for column in model.nodes()}
        train = train[list(categories)].astype(str)
        test = test[list(categories)].astype(str)
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    manifest = {
        "dataset": dataset, "seed": seed, "train_rows": train_rows,
        "test_rows": test_rows, "columns": list(train.columns),
        "categories": categories, "oracle_path": str(oracle_path.resolve()),
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((folder / "train.csv").read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256((folder / "test.csv").read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    return folder


def read_table(path, dataset):
    if dataset in MIXTURES:
        return pd.read_csv(path, dtype=float)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def generate(method, train, dataset, seed, rows, epochs, device):
    if method == "identity":
        if rows != len(train):
            raise ValueError("Identity requires synthetic_rows = train_rows")
        return train.copy()

    import torch
    from ctgan import CTGAN, TVAE

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_class = {"ctgan": CTGAN, "tvae": TVAE}[method]
    model = model_class(epochs=epochs, batch_size=500, cuda=device == "cuda")
    discrete = list(train.columns) if dataset not in MIXTURES else []
    model.fit(train, discrete_columns=discrete)
    return model.sample(rows)


def evaluate(dataset, seed, method, root, synthetic_path, synthetic_rows,
             epochs, device):
    folder = root / f"seed_{seed}" / dataset
    result_path = folder / f"{method}_result.json"
    if result_path.exists():
        raise FileExistsError(f"Evaluation already exists: {result_path}")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["seed"] != seed:
        raise ValueError("Prepared split does not match dataset and seed")
    oracle_path = Path(manifest["oracle_path"])
    if hashlib.sha256(oracle_path.read_bytes()).hexdigest() != manifest["oracle_sha256"]:
        raise ValueError("Oracle changed after dataset preparation")
    for split in ("train", "test"):
        path = folder / f"{split}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[f"{split}_sha256"]:
            raise ValueError(f"Prepared {split} table changed after preparation")
    train = read_table(folder / "train.csv", dataset)
    test = read_table(folder / "test.csv", dataset)
    if len(train) != manifest["train_rows"] or len(test) != manifest["test_rows"]:
        raise ValueError("Prepared split row counts differ from the manifest")
    oracle = (json.loads(oracle_path.read_text(encoding="utf-8"))
              if dataset in MIXTURES else manifest["categories"])
    for table in (train, test):
        validate_table(table, manifest["columns"], oracle, dataset)

    generated_in_run = synthetic_path is None
    if generated_in_run:
        synthetic = generate(method, train, dataset, seed, synthetic_rows,
                             epochs, device)
        synthetic_path = folder / f"{method}_synthetic.csv"
        synthetic.to_csv(synthetic_path, index=False)
    else:
        synthetic_path = Path(synthetic_path)
        synthetic = read_table(synthetic_path, dataset)
    validate_table(synthetic, manifest["columns"], oracle, dataset)
    if len(synthetic) != synthetic_rows:
        raise ValueError("Synthetic row count differs from --synthetic-rows")

    if dataset in MIXTURES:
        l_syn = mixture_log_prob(synthetic, oracle).mean()
        refitted = GaussianMixture(
            n_components=len(oracle["means"]), covariance_type="diag",
            random_state=seed, n_init=5)
        refitted.fit(synthetic)
        l_test = refitted.score(test)
    else:
        from pgmpy.estimators import MaximumLikelihoodEstimator

        original = bn_model(dataset)
        l_syn = bn_log_prob(synthetic, original).mean()
        refitted = bn_model(dataset)
        refitted.fit(synthetic, estimator=MaximumLikelihoodEstimator,
                     state_names=oracle)
        l_test = bn_log_prob(test, refitted).mean()

    result = {
        "dataset": dataset, "seed": seed, "method": method,
        "model_epochs": epochs if generated_in_run and method != "identity" else None,
        "batch_size": 500 if generated_in_run and method != "identity" else None,
        "device": device if generated_in_run and method != "identity" else None,
        "train_rows": len(train), "test_rows": len(test),
        "synthetic_rows": len(synthetic), "l_syn": float(l_syn),
        "l_test": float(l_test), "synthetic_path": str(synthetic_path.resolve()),
        "synthetic_sha256": hashlib.sha256(synthetic_path.read_bytes()).hexdigest(),
        "oracle_sha256": manifest["oracle_sha256"],
        "train_sha256": manifest["train_sha256"],
        "test_sha256": manifest["test_sha256"],
        "package_versions": {
            name: version(name) for name in
            (("numpy", "pandas", "scikit-learn", "pgmpy")
             if dataset not in MIXTURES else ("numpy", "pandas", "scikit-learn"))
        },
    }
    if generated_in_run and method in ("ctgan", "tvae"):
        result["package_versions"]["ctgan"] = version("ctgan")
        result["package_versions"]["torch"] = version("torch")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(root, methods, seeds):
    records = []
    for method in methods:
        for seed in seeds:
            for dataset in DATASETS:
                path = root / f"seed_{seed}" / dataset / f"{method}_result.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                if (record["method"], record["seed"], record["dataset"]) != (
                        method, seed, dataset):
                    raise ValueError(f"Result identity does not match its path: {path}")
                records.append(record)
    table = pd.DataFrame(records)
    table["family"] = np.where(table["dataset"].isin(MIXTURES), "GM", "BN")
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "per_run.csv", index=False)
    summary = table.groupby(["method", "family"], as_index=False)[
        ["l_syn", "l_test"]].mean()
    summary.to_csv(root / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "summarize"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--root", type=Path, default=Path("data/simulated_paper"))
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--synthetic-rows", type=int, default=10000)
    parser.add_argument("--method", default="ctgan")
    parser.add_argument("--methods", nargs="+", default=["identity", "ctgan", "tvae"])
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.action == "summarize":
        print(summarize(args.root, args.methods, args.seeds).to_string(index=False))
        return
    if args.dataset is None:
        parser.error("--dataset is required for prepare and evaluate")
    if args.action == "prepare":
        print(prepare(args.dataset, args.seed, args.train_rows,
                      args.test_rows, args.root))
    else:
        if args.synthetic is None and args.method not in ("ctgan", "tvae", "identity"):
            parser.error("Custom methods require --synthetic CSV")
        print(json.dumps(evaluate(args.dataset, args.seed, args.method,
                                  args.root, args.synthetic, args.synthetic_rows,
                                  args.epochs, args.device), indent=2))


if __name__ == "__main__":
    main()
