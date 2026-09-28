"""Retry the failed seed-42 corrected runs with available GPUs visible."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('census_kdd', 'credit', 'intrusion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, help='Restrict visibility to one GPU')
    args = parser.parse_args()

    os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cache = ROOT / '.cache' / 'corrected_v2' / 'seed_42'
    for name, folder in (('SCIKIT_LEARN_DATA', 'sklearn'),
                         ('XDG_CACHE_HOME', 'xdg'), ('MPLCONFIGDIR', 'matplotlib')):
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    sys.path.insert(0, str(ROOT / 'src'))
    from modeling import constants
    from run_corrected_matrix import classifier_command, methods_for, run_command

    subprocess.run([sys.executable, 'audit/sdv_api_smoke.py'], cwd=ROOT, check=True)
    failures = []
    for dataset in DATASETS:
        logs = ROOT / 'output' / 'corrected_v2' / 'logs' / dataset / 'seed_42'
        generated_any = False
        for generator in ('ctgan', 'tvae'):
            command = [sys.executable, 'synthesize_data/main.py', '--dataset', dataset,
                       '--seed', '42', '--generator', generator.upper()]
            if dataset == 'census_kdd':
                command.append('--resume-from-ensemble')
            log = logs / f'{dataset}_seed42_{generator}_generate.retry.log'
            print(f'Running {dataset} {generator} generation', flush=True)
            if not run_command(command, ROOT / 'src', log):
                failures.append(str(log))
                continue
            generated_any = True
            for method in methods_for(dataset, generator):
                for mode in ('synthetic', 'mix'):
                    arm = constants.run_name(dataset, 42, mode, method)
                    log = logs / f'{arm}.retry.log'
                    print(f'Running {arm}', flush=True)
                    if not run_command(classifier_command(dataset, 42, mode, method), ROOT / 'src', log):
                        failures.append(str(log))
        if dataset != 'census_kdd' and generated_any:
            arm = constants.run_name(dataset, 42, 'original', None)
            log = logs / f'{arm}.retry.log'
            print(f'Running {arm}', flush=True)
            if not run_command(classifier_command(dataset, 42, 'original', None), ROOT / 'src', log):
                failures.append(str(log))

    path = ROOT / 'output' / 'corrected_v2' / 'failures_retry_seed42.json'
    path.write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} retries failed; see {path}')


if __name__ == '__main__':
    main()
