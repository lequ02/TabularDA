"""Run the three-dataset CTGAN pilot on a GPU host."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_corrected_matrix import classifier_command, run_command
from modeling import constants


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('adult', 'mnist28', 'covertype')
METHODS = ('ctgan', 'rf', 'xgb', 'dnn', 'compare_rf', 'compare_xgb', 'compare_dnn')
SEED = 42


def planned_commands(datasets, stage):
    if stage in ('all', 'generators'):
        yield 'sdv_smoke', ROOT, [sys.executable, 'audit/sdv_api_smoke.py']
    for dataset in datasets:
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


def run(datasets, stage, dry_run):
    os.environ['CORRECTED_RUN_NAMESPACE'] = 'pilot_ctgan_v1'
    output = ROOT / 'output' / 'pilot_ctgan_v1'
    for name, cwd, command in planned_commands(datasets, stage):
        if dry_run:
            print(name, ':', ' '.join(command))
            continue
        log = output / 'logs' / f'{name}.log'
        print(f'Running {name}; log: {log}', flush=True)
        if not run_command(command, cwd, log):
            raise SystemExit(f'Pilot run failed; see {log}')

    if not dry_run and stage in ('all', 'classifiers') and tuple(datasets) == DATASETS:
        subprocess.run([
            sys.executable, 'scripts/build_corrected_results.py',
            '--matrix', 'pilot', '--runs', str(output),
            '--out', str(output / 'results'),
        ], cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--stage', choices=('all', 'generators', 'classifiers'), default='all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run((args.dataset,) if args.dataset else DATASETS, args.stage, args.dry_run)
