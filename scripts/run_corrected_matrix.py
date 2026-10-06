"""Run the frozen corrected matrix on a GPU host; never run this on the laptop."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CORRECTED_RUN_NAMESPACE', 'corrected_v2')
sys.path.insert(0, str(ROOT / 'src'))
from modeling import constants
DATASETS = (
    "adult", "census_kdd", "credit", "covertype", "intrusion",
    "mnist12", "mnist28", "news", "california_housing",
)
SEEDS = (42, 43)
CLASSIFICATION_LABELERS = ("gaussian", "categorical", "pca_gmm", "rf", "xgb", "dnn")
REGRESSION_LABELERS = ("pca_gmm", "rf", "xgb", "dnn")


def methods_for(dataset, generator):
    labelers = REGRESSION_LABELERS if dataset in {"news", "california_housing"} else CLASSIFICATION_LABELERS
    prefix = "" if generator == "ctgan" else "tvae_"
    return ((generator,)
            + tuple(f"{prefix}{labeler}" for labeler in labelers)
            + tuple(f"{prefix}compare_{labeler}" for labeler in labelers))


def run_command(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode == 0


def classifier_command(dataset, seed, mode, method):
    command = [
        sys.executable, "-m", "modeling",
        "--dataset-name", dataset,
        "--train-option", mode,
        "--test-option", "original",
        "--validation", "0.2",
        "--batchsize", "128",
        "--lr", "0.001",
        "--global-round", "100",
        "--patience", "30",
        "--early-stop-crit", "loss",
        "--seed", str(seed),
    ]
    if method is not None:
        command.extend(("--augment-option", method))
    return command


def completed_run(output, dataset, seed, mode, method):
    path = output / dataset / 'acc' / (constants.run_name(dataset, seed, mode, method) + '.run.json')
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == (dataset, seed, mode, method)
    assert record['selected_dev_epoch'] is not None
    assert Path(record['predictions_path']).is_file() and Path(record['downstream_weight_path']).is_file()
    return True


def run_matrix(dataset_names, seeds, stage, resume=False):
    if resume and stage != 'classifiers':
        raise ValueError('Resume requires the classifiers stage; generation has separate artifact dependencies')
    output = ROOT / "output" / constants.RUN_NAMESPACE
    log_root = output / "logs"
    failures = []
    src_dir = ROOT / "src"
    if stage in {"all", "generators"}:
        subprocess.run([sys.executable, "audit/sdv_api_smoke.py"], cwd=ROOT, check=True)

    for seed in seeds:
        for dataset in dataset_names:
            for generator in ("ctgan", "tvae"):
                generated = True
                if stage in {"all", "generators"}:
                    log_path = log_root / dataset / f"seed_{seed}" / f"{dataset}_seed{seed}_{generator}_generate.log"
                    command = [
                        sys.executable, "synthesize_data/main.py",
                        "--dataset", dataset, "--seed", str(seed),
                        "--generator", generator.upper(),
                    ]
                    generated = run_command(command, ROOT / "src", log_path)
                    if not generated:
                        failures.append({"dataset": dataset, "seed": seed,
                                         "arm": f"{generator}_generation", "log": str(log_path)})
                if stage not in {"all", "classifiers"}:
                    continue
                if not generated:
                    continue
                for method in methods_for(dataset, generator):
                    for mode in ("synthetic", "mix"):
                        if resume and completed_run(output, dataset, seed, mode, method):
                            continue
                        arm = constants.run_name(dataset, seed, mode, method)
                        log_path = log_root / dataset / f"seed_{seed}" / f"{arm}.log"
                        if not run_command(classifier_command(dataset, seed, mode, method), src_dir, log_path):
                            failures.append({"dataset": dataset, "seed": seed,
                                             "arm": arm, "log": str(log_path)})
            if stage in {"all", "classifiers"}:
                if resume and completed_run(output, dataset, seed, 'original', None):
                    continue
                log_path = log_root / dataset / f"seed_{seed}" / f"{constants.run_name(dataset, seed, 'original', None)}.log"
                if not run_command(classifier_command(dataset, seed, "original", None), src_dir, log_path):
                    failures.append({"dataset": dataset, "seed": seed, "arm": "real", "log": str(log_path)})

    output.mkdir(parents=True, exist_ok=True)
    full_matrix = tuple(dataset_names) == DATASETS and tuple(seeds) == SEEDS
    failure_name = ("failures.json" if stage == "all" and full_matrix else
                    f"failures_{stage}_{'-'.join(dataset_names)}_{'-'.join(map(str, seeds))}.json")
    failure_path = output / failure_name
    with failure_path.open("w", encoding="utf-8") as file:
        json.dump(failures, file, indent=2)
    if failures:
        raise SystemExit(f"{len(failures)} corrected runs failed; see {failure_path}")
    if stage in {"all", "classifiers"} and full_matrix:
        command = [sys.executable, "scripts/build_corrected_results.py"]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--stage", choices=("all", "generators", "classifiers"), default="all")
    parser.add_argument("--resume", action="store_true", help="Preserve completed classifier runs")
    args = parser.parse_args()
    run_matrix((args.dataset,) if args.dataset else DATASETS,
               (args.seed,) if args.seed is not None else SEEDS, args.stage, args.resume)
