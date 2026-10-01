"""Run complete-table generators and optional supervised relabeling experiments."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .labelers import LABELERS, fit_labeler, make_dnn
from .oracle import Oracle
from .bn import BNOracle, TARGETS
from .plot import plot_results

DATASETS = ["gaussian", "grid", "ring", *TARGETS]
METRICS = ["l_syn", "l_test", "accuracy", "macro_f1", "h_star_agreement"]
BN_LIKELIHOOD_EPSILON = 1e-8


def make_oracle(dataset, args):
    return (BNOracle(dataset) if dataset in TARGETS else
            Oracle(kappa=args.kappa, beta=args.beta, noise=args.noise,
                   dataset=dataset, spread=args.spread))


def likelihood_scores(sample, test, oracle, seed):
    log_prob = oracle.log_prob(sample)
    oracle.validate(test)
    test_log_prob = oracle.test_log_prob(sample, test, seed)
    violations = float(np.isneginf(log_prob).mean())
    if oracle.dataset in TARGETS:
        log_prob = np.logaddexp(log_prob, np.log(BN_LIKELIHOOD_EPSILON))
        test_log_prob = np.logaddexp(test_log_prob, np.log(BN_LIKELIHOOD_EPSILON))
    return {"l_syn": float(log_prob.mean()), "l_test": float(test_log_prob.mean()),
            "support_violation_rate": violations}


def format_record(record):
    return (f"{record['dataset']} seed={record['seed']} {record['method']}: " +
            ", ".join(f"{m}={record[m]:.4f}" for m in METRICS))


def write_results(results, args):
    results.to_csv(args.output / "per_run.csv", index=False)
    results.groupby(["dataset", "method"], sort=False)[METRICS + ["support_violation_rate", "accuracy_ceiling"]].mean().to_csv(
        args.output / "summary.csv")
    for dataset in args.datasets:
        plot_results(results[results.dataset == dataset], args.noise, args.output / dataset, args.epochs)


def generate(method, train, rows, seed, epochs, batch_size, device, oracle):
    import torch
    from ctgan import CTGAN, TVAE

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model_class = {"ctgan": CTGAN, "tvae": TVAE}[method]
    model = model_class(epochs=epochs, batch_size=batch_size, cuda=device == "cuda")
    model.set_random_state(seed)
    model.fit(train, discrete_columns=[c for c in oracle.discrete if c in train])
    sample = model.sample(rows)
    oracle.validate(sample, with_target="target" in train)
    return sample


def evaluate(sample, test, oracle, seed):
    scores = likelihood_scores(sample, test, oracle, seed)
    evaluator = make_dnn(seed, oracle)
    evaluator.fit(sample[oracle.features], sample["target"])
    predictions = evaluator.predict(test[oracle.features])
    posterior = oracle.posterior(test)
    return {
        **scores,
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "macro_f1": float(f1_score(test["target"], predictions, labels=oracle.classes,
                                    average="macro", zero_division=0)),
        "h_star_agreement": float(accuracy_score(posterior.argmax(axis=1), predictions)),
        "accuracy_ceiling": float(posterior.max(axis=1).mean()),
    }


def run(args):
    if min(args.train_rows, args.test_rows, args.synthetic_rows, args.epochs) <= 0:
        raise ValueError("Row counts and epochs must be positive")
    if args.batch_size <= 0 or args.batch_size % 10:
        raise ValueError("batch-size must be positive and divisible by 10 (CTGAN pac=10)")
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        raise ValueError("Seeds must be distinct and nonnegative")
    if any(len(values) != len(set(values)) for values in (args.datasets, args.generators, args.labelers)):
        raise ValueError("Datasets, generators and labelers must not be repeated")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "output": str(args.output), "bn_refit_alpha": .5,
              "bn_likelihood_epsilon": BN_LIKELIHOOD_EPSILON,
              "evaluation_classifier": "dnn",
              "packages": {p: version(p) for p in (
                  "numpy", "pandas", "scipy", "scikit-learn", "ctgan", "torch", "matplotlib", "pgmpy", "networkx")}}
    if "xgb" in args.labelers:
        config["packages"]["xgboost"] = version("xgboost")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    for dataset in args.datasets:
        oracle = make_oracle(dataset, args)
        run_dataset(args, oracle, records)
    write_results(pd.DataFrame(records), args)


def run_dataset(args, oracle, records):
    for seed in args.seeds:
        folder = args.output / oracle.dataset / f"seed_{seed}"
        folder.mkdir(parents=True)
        train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        train = oracle.sample(args.train_rows, train_rng)
        test = oracle.sample(args.test_rows, test_rng)
        train.to_csv(folder / "train.csv", index=False)
        test.to_csv(folder / "test.csv", index=False)
        labelers = {name: fit_labeler(name, train, seed, oracle) for name in args.labelers}

        def record(method, sample):
            sample.to_csv(folder / f"{method}.csv", index=False)
            scores = evaluate(sample, test, oracle, seed)
            records.append({"dataset": oracle.dataset, "seed": seed, "method": method,
                            "synthetic_rows": len(sample), **scores})
            pd.DataFrame(records).to_csv(args.output / "per_run.csv", index=False)
            print(format_record(records[-1]), flush=True)

        record("original", train)
        for generator in args.generators:
            print(f"{oracle.dataset} seed={seed} training {generator} joint ({args.epochs} epochs)", flush=True)
            sample = generate(generator, train, args.synthetic_rows, seed,
                              args.epochs, args.batch_size, args.device, oracle)
            record(generator, sample)
            if labelers:
                print(f"{oracle.dataset} seed={seed} training {generator} features ({args.epochs} epochs)", flush=True)
                features = generate(generator, train[oracle.features], args.synthetic_rows, seed,
                                    args.epochs, args.batch_size, args.device, oracle)
                features.to_csv(folder / f"{generator}-features.csv", index=False)
                for source_name, source in (("joint", sample), ("features", features)):
                    for name, (labeler, classes) in labelers.items():
                        relabeled = source.copy()
                        relabeled["target"] = classes[labeler.predict(source[oracle.features]).astype(int)]
                        record(f"{generator}-{source_name}-{name}", relabeled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/simulated_eval"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
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
    parser.add_argument("--spread", type=float, default=0.20)
    run(parser.parse_args())
