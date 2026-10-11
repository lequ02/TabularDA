"""Run an explicit corrected dataset/seed matrix on the research GPU host."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CORRECTED_RUN_NAMESPACE', 'corrected_v2')
sys.path.insert(0, str(ROOT / 'src'))
DATASETS = (
    "adult", "census_kdd", "credit", "covertype",
    "mnist12", "mnist28", "news", "california_housing",
)
SEEDS = (42, 43)
CLASSIFICATION_LABELERS = ("gaussian", "categorical", "pca_gmm", "rf", "xgb", "dnn")
REGRESSION_LABELERS = ("pca_gmm", "rf", "xgb", "dnn")


def methods_for(dataset, generator):
    labelers = REGRESSION_LABELERS if dataset in {"news", "california_housing"} else CLASSIFICATION_LABELERS
    if generator not in {'ctgan', 'tvae', 'tabddpm'}:
        raise ValueError(f'Unsupported generator: {generator}')
    prefix = "" if generator == "ctgan" else generator + '_'
    if generator == 'tabddpm':
        primary = ('rf', 'xgb', 'dnn')
        return ((generator,)
                + tuple(f'{prefix}compare_{labeler}' for labeler in primary)
                + tuple(f'{prefix}{labeler}' for labeler in primary)
                + tuple(f'{prefix}{labeler}' for labeler in labelers if labeler not in primary)
                + tuple(f'{prefix}compare_{labeler}' for labeler in labelers if labeler not in primary))
    return ((generator,)
            + tuple(f"{prefix}{labeler}" for labeler in labelers)
            + tuple(f"{prefix}compare_{labeler}" for labeler in labelers))


def run_command(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{datetime.now(ZoneInfo('America/Chicago')).isoformat()} START {' '.join(command)} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return True


def classifier_command(dataset, seed, mode, method, protocol='legacy'):
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
        "--early-stop-crit", "f1_macro" if protocol == 'corrected' and dataset == 'census_kdd' else "loss",
        "--seed", str(seed),
    ]
    if method is not None:
        command.extend(("--augment-option", method))
    if protocol == 'corrected' and dataset == 'census_kdd':
        command.extend(('--evaluation-protocol', 'census-weighted'))
    if protocol == 'corrected' and dataset == 'news':
        command.extend(('--target-transform', 'log'))
    return command


def completed_run(output, dataset, seed, mode, method, protocol='legacy'):
    from modeling import constants
    path = output / dataset / 'acc' / (constants.run_name(dataset, seed, mode, method) + '.run.json')
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == (dataset, seed, mode, method)
    assert record['selected_dev_epoch'] is not None
    assert Path(record['predictions_path']).is_file() and Path(record['downstream_weight_path']).is_file()
    if protocol == 'corrected':
        from modeling.run_record import file_sha256, source_sha256
        expected_protocol = 'census-weighted' if dataset == 'census_kdd' else 'standard'
        if dataset not in {'news', 'california_housing'} and record.get('evaluation_protocol', {}).get('name') != expected_protocol:
            raise ValueError(f'Completed run has the wrong classification protocol: {path}')
        if dataset == 'news' and record.get('target_transform', {}).get('name') != 'log':
            raise ValueError(f'Completed News run has the wrong target transform: {path}')
        if dataset in {'news', 'california_housing'}:
            if 'target_normalization' not in record or not {'mae', 'r2', 'nmae_sigma'} <= record['test_scores'].keys():
                raise ValueError(f'Completed regression run is missing normalization or metrics: {path}')
        if record['source_sha256'] != source_sha256():
            raise ValueError(f'Completed run used different main source: {path}')
        for artifact, expected_hash in record['artifact_sha256'].items():
            if file_sha256(artifact) != expected_hash:
                raise ValueError(f'Completed run artifact changed: {artifact}')
    return True


def run_matrix(dataset_names, seeds, stage, resume=False, generators=('ctgan', 'tvae'), tabddpm_steps=None,
               *, namespace=None, protocol='legacy', tabddpm_epochs=None):
    dataset_names, seeds = tuple(dataset_names), tuple(seeds)
    if not dataset_names or len(set(dataset_names)) != len(dataset_names) or set(dataset_names) - set(DATASETS):
        raise ValueError('Choose a nonempty list of distinct supported datasets')
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Choose distinct integer seeds in [0, 2**32)')
    if stage not in {'all', 'generators', 'classifiers'} or protocol not in {'legacy', 'corrected'}:
        raise ValueError('Unknown experiment stage or protocol')
    if not generators or len(set(generators)) != len(generators) or set(generators) - {'ctgan', 'tvae', 'tabddpm'}:
        raise ValueError('Choose distinct supported generators')
    if tabddpm_steps is not None and tabddpm_epochs is not None:
        raise ValueError('Choose either Tab-DDPM steps or epochs, not both')
    if any(value is not None and (not isinstance(value, int) or value <= 0)
           for value in (tabddpm_steps, tabddpm_epochs)):
        raise ValueError('Tab-DDPM budgets must be positive integers')
    if resume and stage != 'classifiers':
        raise ValueError('Resume requires the classifiers stage; generation has separate artifact dependencies')
    namespace = namespace or os.environ.get('CORRECTED_RUN_NAMESPACE', 'corrected_v2')
    if not namespace or not namespace.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Namespace must contain only letters, numbers, underscores, and hyphens')
    if ROOT != Path('/home/thuy/Research/minh_data_synth/TabularDA'):
        raise RuntimeError('Execute experiments only in the research-server repository')
    os.environ['CORRECTED_RUN_NAMESPACE'] = namespace
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    import torch
    from modeling import constants
    if constants.RUN_NAMESPACE != namespace:
        raise RuntimeError('Set the namespace before importing modeling modules; use a fresh Python process')
    if not torch.cuda.is_available():
        raise RuntimeError('Production experiments require remote CUDA')
    torch.set_num_threads(2)
    output = ROOT / "output" / constants.RUN_NAMESPACE
    log_root = output / "logs"
    src_dir = ROOT / "src"
    if stage in {"all", "generators"} and any(g != 'tabddpm' for g in generators):
        subprocess.run([sys.executable, "audit/sdv_api_smoke.py"], cwd=ROOT, check=True)

    for seed in seeds:
        for dataset in dataset_names:
            if stage in {'all', 'generators'}:
                command = [sys.executable, 'synthesize_data/main.py', '--dataset', dataset,
                           '--seed', str(seed), '--generator', generators[0].upper(), '--prepare-only']
                run_command(command, src_dir, log_root / dataset / f'seed_{seed}' / 'prepare.log')
            if stage in {'all', 'classifiers'}:
                run_classifier(output, dataset, seed, 'original', None, resume, protocol)
            for generator in generators:
                if stage in {"all", "generators"}:
                    log_path = log_root / dataset / f"seed_{seed}" / f"{dataset}_seed{seed}_{generator}_generate.log"
                    command = [
                        sys.executable, "synthesize_data/main.py",
                        "--dataset", dataset, "--seed", str(seed),
                        "--generator", generator.upper(),
                    ]
                    if generator == 'tabddpm':
                        if tabddpm_epochs is not None:
                            command.extend(('--tabddpm-epochs', str(tabddpm_epochs)))
                        elif tabddpm_steps is not None:
                            command.extend(('--tabddpm-steps', str(tabddpm_steps)))
                    if protocol == 'corrected' and dataset == 'news':
                        command.extend(('--target-transform', 'log'))
                    run_command(command, src_dir, log_path)
                if stage not in {"all", "classifiers"}:
                    continue
                for method in methods_for(dataset, generator):
                    for mode in ("synthetic", "mix"):
                        run_classifier(output, dataset, seed, mode, method, resume, protocol)

    full_matrix = tuple(dataset_names) == DATASETS and tuple(seeds) == SEEDS
    if stage in {"all", "classifiers"} and full_matrix:
        command = [sys.executable, "scripts/build_corrected_results.py",
                   '--runs', str(output), '--out', str(output / 'results'),
                   '--generators', *generators]
        subprocess.run(command, cwd=ROOT, check=True)


def run_classifier(output, dataset, seed, mode, method, resume, protocol):
    from modeling import constants
    arm = constants.run_name(dataset, seed, mode, method)
    record = output / dataset / 'acc' / (arm + '.run.json')
    if record.exists():
        if resume and completed_run(output, dataset, seed, mode, method, protocol):
            return
        raise FileExistsError(f'Refusing to overwrite a completed run: {record}')
    run_command(classifier_command(dataset, seed, mode, method, protocol), ROOT / 'src',
                output / 'logs' / dataset / f'seed_{seed}' / (arm + '.log'))


def run_experiment(seed, datasets, namespace=None, stage='all', resume=False,
                   *, generators=('ctgan', 'tvae'), generator_epochs=500):
    """Run selected generators and applicable corrected arms for an explicit seed/list."""
    run_matrix(datasets, (seed,), stage, resume, generators=generators,
               tabddpm_epochs=generator_epochs if 'tabddpm' in generators else None,
               namespace=namespace or f'seed{seed}_corrected', protocol='corrected')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", choices=DATASETS)
    selection.add_argument("--datasets", nargs='+', choices=DATASETS)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--namespace')
    parser.add_argument('--protocol', choices=('legacy', 'corrected'),
                        help='Dataset lists use corrected News/Census protocols; older calls retain legacy defaults')
    parser.add_argument("--stage", choices=("all", "generators", "classifiers"), default="all")
    parser.add_argument("--resume", action="store_true", help="Preserve completed classifier runs")
    parser.add_argument('--generators', nargs='+', choices=('ctgan', 'tvae', 'tabddpm'),
                        default=('ctgan', 'tvae'))
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--tabddpm-steps', type=int)
    budget.add_argument('--tabddpm-epochs', type=int)
    args = parser.parse_args()
    run_matrix(args.datasets or ((args.dataset,) if args.dataset else DATASETS),
               (args.seed,) if args.seed is not None else SEEDS, args.stage, args.resume,
               tuple(args.generators), args.tabddpm_steps,
               tabddpm_epochs=args.tabddpm_epochs,
               namespace=args.namespace or (f'seed{args.seed}_corrected' if args.seed not in (*SEEDS, None) else None),
               protocol=args.protocol if args.protocol is not None else ('corrected' if args.datasets else 'legacy'))
