"""Fresh Housing runs using the existing verified task helper and modeling CLI."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'housing_no_faker_20261009'
os.environ['CORRECTED_RUN_NAMESPACE'] = NAMESPACE
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from modeling import constants
from run_corrected_matrix import classifier_command, completed_run


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def main(seed):
    output = ROOT / 'output' / NAMESPACE
    logs = output / 'logs' / f'seed_{seed}'
    logs.mkdir(parents=True, exist_ok=True)
    status_path = output / f'seed_{seed}.status.json'
    status = {'seed': seed, 'state': 'running', 'started_at_chicago': now(),
              'planned_downstream_runs': 29, 'completed_downstream_runs': 0}

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    def execute(name, command, cwd):
        status.update(current_task=name, updated_at_chicago=now())
        save()
        print(f'{now()} START {name}', flush=True)
        with (logs / f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'{now()} COMPLETE {name}', flush=True)

    def evaluate(mode, method):
        name = constants.run_name('california_housing', seed, mode, method)
        execute(name, classifier_command('california_housing', seed, mode, method), ROOT / 'src')
        if not completed_run(output, 'california_housing', seed, mode, method):
            raise RuntimeError(f'Missing completed record: {name}')
        record = json.loads((output / 'california_housing' / 'acc' / f'{name}.run.json').read_text())
        assert (record['batch_size'], record['learning_rate'], record['epoch_budget']) == (128, .001, 100)
        assert record['selected_dev_epoch'] is not None
        status['completed_downstream_runs'] += 1
        save()

    def task(action, generator=None, fit=None, labeler=None):
        command = [sys.executable, '-u', str(ROOT / '.cache/housing_tasks/experiment_task.py'),
                   action, '--dataset', 'california_housing', '--seed', str(seed)]
        for option, value in (('--generator', generator), ('--fit', fit), ('--labeler', labeler)):
            if value is not None:
                command.extend([option, value])
        execute('_'.join(str(v) for v in (action, seed, generator, fit, labeler) if v is not None), command, ROOT)

    with (output / f'seed_{seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise FileExistsError('Seed already launched; inspect its status and logs before resuming')
        save()
        try:
            task('prepare')
            evaluate('original', None)
            for generator in ('ctgan', 'tvae'):
                for fit in ('full', 'xonly'):
                    task('fit', generator, fit)
                    prefix = '' if generator == 'ctgan' else 'tvae_'
                    methods = [generator] if fit == 'full' else []
                    for labeler in ('rf', 'xgb', 'dnn'):
                        task('label', generator, fit, labeler)
                        methods.append(prefix + ('compare_' if fit == 'full' else '') + labeler)
                    for method in methods:
                        for mode in ('synthetic', 'mix'):
                            evaluate(mode, method)
            assert status['completed_downstream_runs'] == 29
            status.update(state='complete', finished_at_chicago=now(), current_task=None)
        except Exception as error:
            status.update(state='failed', failed_at_chicago=now(), error=repr(error))
            raise
        finally:
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(42, 43), required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not ROOT.as_posix().startswith('/home/thuy/Research/'):
        parser.error('Run this only in the remote Housing source snapshot')
    main(args.seed)
