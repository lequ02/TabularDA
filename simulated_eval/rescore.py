"""Refresh BN likelihoods and reports from a completed run's saved tables."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np

from .benchmark import BN_LIKELIHOOD_EPSILON, format_record, likelihood_scores, make_oracle, write_results
from .bn import TARGETS


def rescore(output):
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(**{**config, "output": output})
    results = pd.read_csv(output / "per_run.csv")
    expected = len(args.datasets) * len(args.seeds) * (1 + len(args.generators) * (1 + 2*len(args.labelers)))
    if len(results) != expected or results.duplicated(["dataset", "seed", "method"]).any():
        raise ValueError("Rescoring requires a complete run with unique method records")
    log_path = output.with_suffix(".log")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for dataset in args.datasets:
        if dataset not in TARGETS:
            continue
        oracle = make_oracle(dataset, args)
        for seed in args.seeds:
            folder = output / dataset / f"seed_{seed}"
            test = pd.read_csv(folder / "test.csv")
            subset = results[(results.dataset == dataset) & (results.seed == seed)]
            for index, record in subset.iterrows():
                name = "train" if record.method == "original" else record.method
                sample = pd.read_csv(folder / f"{name}.csv")
                for metric, value in likelihood_scores(sample, test, oracle, seed).items():
                    results.loc[index, metric] = value
        print(f"Rescored {dataset}", flush=True)
    if not np.isfinite(results[["l_syn", "l_test"]].to_numpy()).all():
        raise ValueError("Rescored likelihoods must be finite")
    records = {format_record(r).split(":", 1)[0]: format_record(r) for r in results.to_dict("records")}
    keys = [line.split(":", 1)[0] for line in lines if line.split(":", 1)[0] in records]
    if len(keys) != expected or len(set(keys)) != expected:
        raise ValueError("Log must contain one metric line for every saved method record")
    lines = [records[line.split(":", 1)[0]] if line.split(":", 1)[0] in records else line for line in lines]
    stamp = datetime.now(timezone.utc).isoformat()
    lines.append(f"Likelihood reports rescored at {stamp}: BN L_syn/L_test use log(p + 1e-8); saved samples and prediction metrics retained.")
    config.update(bn_likelihood_epsilon=BN_LIKELIHOOD_EPSILON, likelihood_rescored_at=stamp)
    write_results(results, args)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {len(results)} result records, summaries, plots, configuration and log.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    rescore(parser.parse_args().output)
