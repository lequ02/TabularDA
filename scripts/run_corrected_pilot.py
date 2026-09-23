"""Run the three-dataset CTGAN pilot on a GPU host."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_corrected_matrix import classifier_command, run_command
from modeling import constants


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('adult', 'mnist28', 'covertype')
METHODS = ('ctgan', 'rf', 'xgb', 'dnn', 'compare_rf', 'compare_xgb', 'compare_dnn')
SEED = 42


def dataset_commands(dataset, stage):
    if stage in ('all', 'generators'):
        yield (f'{dataset}_seed{SEED}_ctgan_generate', ROOT / 'src',
               [sys.executable, 'synthesize_data/main.py', '--dataset', dataset,
                '--seed', str(SEED), '--generator', 'CTGAN', '--profile', 'pilot'])
    if stage in ('all', 'classifiers'):
        for method in METHODS:
            for mode in ('synthetic', 'mix'):
                yield (constants.run_name(dataset, SEED, mode, method), ROOT / 'src',
                       classifier_command(dataset, SEED, mode, method))
        yield (constants.run_name(dataset, SEED, 'original', None), ROOT / 'src',
               classifier_command(dataset, SEED, 'original', None))


def planned_commands(datasets, stage):
    if stage in ('all', 'generators'):
        yield 'sdv_smoke', ROOT, [sys.executable, 'audit/sdv_api_smoke.py']
    for dataset in datasets:
        yield from dataset_commands(dataset, stage)


def run_one(command_spec, output):
    name, cwd, command = command_spec
    log = output / 'logs' / f'{name}.log'
    print(f'Running {name}; log: {log}', flush=True)
    if not run_command(command, cwd, log):
        raise RuntimeError(f'Pilot run failed; see {log}')


def run(datasets, stage, dry_run, jobs):
    os.environ['CORRECTED_RUN_NAMESPACE'] = 'pilot_ctgan_v1'
    output = ROOT / 'output' / 'pilot_ctgan_v1'
    if dry_run:
        for name, _, command in planned_commands(datasets, stage):
            print(name, ':', ' '.join(command))
        return

    if stage in ('all', 'generators'):
        run_one(('sdv_smoke', ROOT, [sys.executable, 'audit/sdv_api_smoke.py']), output)

    def run_dataset(dataset):
        for command_spec in dataset_commands(dataset, stage):
            run_one(command_spec, output)

    if jobs == 1:
        for dataset in datasets:
            run_dataset(dataset)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            list(executor.map(run_dataset, datasets))

    if stage in ('all', 'classifiers') and tuple(datasets) == DATASETS:
        subprocess.run([
            sys.executable, 'scripts/build_corrected_results.py',
            '--matrix', 'pilot', '--runs', str(output),
            '--out', str(output / 'results'),
        ], cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--stage', choices=('all', 'generators', 'classifiers'), default='all')
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run((args.dataset,) if args.dataset else DATASETS, args.stage, args.dry_run, args.jobs)
