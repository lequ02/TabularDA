"""Run complete-table generators and optional supervised relabeling experiments."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .labelers import LABELERS, fit_labeler, make_dnn
from .oracle import FEATURES, Oracle, validate
from .plot import plot_results

METRICS = ["joint_log_likelihood", "accuracy", "macro_f1", "h_star_agreement"]


def generate(method, train, rows, seed, epochs, batch_size, device):
    import torch
    from ctgan import CTGAN, TVAE

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model_class = {"ctgan": CTGAN, "tvae": TVAE}[method]
    model = model_class(epochs=epochs, batch_size=batch_size, cuda=device == "cuda")
    model.set_random_state(seed)
    model.fit(train, discrete_columns=[c for c in ("feature_3", "target") if c in train])
    sample = model.sample(rows)
    validate(sample, with_target="target" in train)
    return sample


def evaluate(sample, test, oracle, seed):
    likelihood = float(oracle.log_prob(sample).mean())
    validate(test)
    evaluator = make_dnn(seed)
    evaluator.fit(sample[FEATURES], sample["target"])
    predictions = evaluator.predict(test[FEATURES])
    return {
        "joint_log_likelihood": likelihood,
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "macro_f1": float(f1_score(test["target"], predictions, labels=[0, 1],
                                    average="macro", zero_division=0)),
        "h_star_agreement": float(accuracy_score(oracle.clean_target(test), predictions)),
    }


def run(args):
    oracle = Oracle(kappa=args.kappa, beta=args.beta, noise=args.noise)
    if min(args.train_rows, args.test_rows, args.synthetic_rows, args.epochs) <= 0:
        raise ValueError("Row counts and epochs must be positive")
    if args.batch_size <= 0 or args.batch_size % 10:
        raise ValueError("batch-size must be positive and divisible by 10 (CTGAN pac=10)")
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        raise ValueError("Seeds must be distinct and nonnegative")
    if len(args.generators) != len(set(args.generators)) or len(args.labelers) != len(set(args.labelers)):
        raise ValueError("Generators and labelers must not be repeated")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output": str(args.output), "oracle": asdict(oracle),
              "evaluation_classifier": "dnn",
              "packages": {p: version(p) for p in (
                  "numpy", "pandas", "scipy", "scikit-learn", "ctgan", "torch", "matplotlib")}}
    if "xgb" in args.labelers:
        config["packages"]["xgboost"] = version("xgboost")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    for seed in args.seeds:
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        train = oracle.sample(args.train_rows, train_rng)
        test = oracle.sample(args.test_rows, test_rng)
        train.to_csv(folder / "train.csv", index=False)
        test.to_csv(folder / "test.csv", index=False)
        labelers = {name: fit_labeler(name, train, seed) for name in args.labelers}

        def record(method, sample):
            sample.to_csv(folder / f"{method}.csv", index=False)
            scores = evaluate(sample, test, oracle, seed)
            records.append({"seed": seed, "method": method, "synthetic_rows": len(sample), **scores})
            pd.DataFrame(records).to_csv(args.output / "per_run.csv", index=False)
            print(f"seed={seed} {method}: " + ", ".join(f"{m}={scores[m]:.4f}" for m in METRICS), flush=True)

        record("original", train)
        for generator in args.generators:
            print(f"seed={seed} training {generator} joint ({args.epochs} epochs)", flush=True)
            sample = generate(generator, train, args.synthetic_rows, seed,
                              args.epochs, args.batch_size, args.device)
            record(generator, sample)
            if labelers:
                print(f"seed={seed} training {generator} features ({args.epochs} epochs)", flush=True)
                features = generate(generator, train[FEATURES], args.synthetic_rows, seed,
                                    args.epochs, args.batch_size, args.device)
                features.to_csv(folder / f"{generator}-features.csv", index=False)
                for source_name, source in (("joint", sample), ("features", features)):
                    for name, labeler in labelers.items():
                        relabeled = source.copy()
                        relabeled["target"] = labeler.predict(source[FEATURES])
                        record(f"{generator}-{source_name}-{name}", relabeled)
    results = pd.DataFrame(records)
    summary = results.groupby("method", sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(args.output / "summary.csv")
    plot_results(results, oracle.noise, args.output, args.epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/simulated_eval"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--generators", choices=["ctgan", "tvae"], nargs="+", default=["ctgan", "tvae"])
    parser.add_argument("--labelers", choices=LABELERS, nargs="*", default=LABELERS)
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--test-rows", type=int, default=10000)
    parser.add_argument("--synthetic-rows", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--noise", type=float, default=0.10)
    run(parser.parse_args())
