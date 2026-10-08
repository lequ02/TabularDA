"""Three-worker, dependency-aware queue for the explicitly approved retained matrix."""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
CACHE = ROOT / '.cache/full_completion_20261007'
STATE = CACHE / 'state.json'
HELPER = CACHE / 'experiment_task.py'
PYTHON = '/home/thuy/miniconda3/envs/env/bin/python'
ENVIRONMENT = {'CORRECTED_RUN_NAMESPACE': 'corrected_v2', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '0'}


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


@contextmanager
def locked_state():
    with (CACHE / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text())
        try:
            yield state
        finally:
            state['updated_at_chicago'] = now()
            write_json(STATE, state)
            fcntl.flock(lock, fcntl.LOCK_UN)


def record_path(job):
    return ROOT / 'output' / job['output_namespace'] / job['dataset'] / 'acc' / (job['run_id'] + '.run.json')


def verify_record(job):
    path = record_path(job)
    record = json.loads(path.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == (job['dataset'], job['seed'], job['train_option'], job['augment_option'])
    assert record['selected_dev_epoch'] is not None
    assert record['batch_size'] == 128 and record['learning_rate'] == .001 and record['epoch_budget'] == 100
    assert Path(record['predictions_path']).is_file() and Path(record['downstream_weight_path']).is_file()
    with Path(record['predictions_path']).open(newline='') as source:
        ids = [int(row['source_id']) for row in csv.DictReader(source)]
    assert ids == record['split_manifest']['splits']['test'], 'Prediction source IDs differ from reserved test partition'
    if record['synthetic_path'] is not None:
        assert record['synthetic_quality']['rows'] == 100000
        assert record['synthetic_quality']['synthetic_label_counts'] == record['synthetic_label_counts']
        assert record['generator_provenance']['parameters']['epochs'] == 500
        assert record['generator_provenance']['parameters']['batch_size'] == 500
    if job['dataset'] == 'census_kdd':
        sys.path.insert(0, str(ROOT / 'scripts'))
        from run_census_weighted_matrix import verify_record as verify_weighted
        verify_weighted(path)
    elif job['dataset'] == 'news':
        assert record['target_normalization']['split'] == 'test' and record['target_normalization']['ddof'] == 0
        assert abs(record['test_scores']['nmae_sigma'] - record['test_scores']['mae'] / record['target_normalization']['sigma_y']) < 1e-12
    print(f'VERIFIED {job["run_id"]}', flush=True)


def initialize():
    assert not STATE.exists(), 'Queue state already exists; use existing workers/state, not a duplicate launch'
    plan = json.loads((CACHE / 'parallel_queues.json').read_text())
    tasks = []
    block_rank = {('census_kdd', 42): 1, ('mnist12', 43): 1, ('census_kdd', 43): 2, ('mnist28', 43): 2.5, ('news', 43): 3, ('california_housing', 42): 4, ('california_housing', 43): 4}
    prep_ids = {}
    table_ids = {}

    def add(identifier, owner, rank, kind, dependencies, argv, **fields):
        tasks.append({'id': identifier, 'owner': owner, 'rank': rank, 'kind': kind, 'dependencies': dependencies, 'argv': argv, 'cwd': str(ROOT), 'status': 'pending', **fields})

    for block in plan['blocks']:
        dataset, seed, owner = block['dataset'], block['seed'], block['default_session']
        prefix = f'{dataset}_seed{seed}'
        rank = block_rank[(dataset, seed)]
        prep = prefix + '_prepare'
        prep_ids[(dataset, seed)] = prep
        add(prep, owner, rank, 'prepare', [], [PYTHON, '-u', str(HELPER), 'prepare', '--dataset', dataset, '--seed', str(seed)])
        for item in block['new_generator_fits']:
            generator, fit = item['generator'], item['fit']
            fit_id = f'{prefix}_{generator}_{fit}_fit_sample'
            # Full fits precede X-only fits so useful paired comparisons release early.
            fit_order = (0 if fit == 'full' else .1) + (0 if generator == 'ctgan' else .01)
            add(fit_id, owner, rank + .2 + fit_order, 'heavy_fit' if dataset == 'mnist28' else 'fit', [prep], [PYTHON, '-u', str(HELPER), 'fit', '--dataset', dataset, '--seed', str(seed), '--generator', generator, '--fit', fit])
            if fit == 'full':
                table_ids[(dataset, seed, generator, fit, 'generated')] = fit_id
            for label in ('rf', 'xgb', 'dnn'):
                identifier = f'{prefix}_{generator}_{fit}_{label}_label'
                add(identifier, owner, rank + .12, 'label', [fit_id], [PYTHON, '-u', str(HELPER), 'label', '--dataset', dataset, '--seed', str(seed), '--generator', generator, '--fit', fit, '--labeler', label])
                table_ids[(dataset, seed, generator, fit, label)] = identifier
        if dataset == 'mnist12':
            for label in ('rf', 'dnn'):
                identifier = f'{prefix}_tvae_full_{label}_label'
                add(identifier, owner, rank + .05, 'label', [prep], [PYTHON, '-u', str(HELPER), 'label', '--dataset', dataset, '--seed', str(seed), '--generator', 'tvae', '--fit', 'full', '--labeler', label])
                table_ids[(dataset, seed, 'tvae', 'full', label)] = identifier

    for lane in plan['queues']:
        for job in lane['jobs']:
            prep = prep_ids[(job['dataset'], job['seed'])]
            key = (job['dataset'], job['seed'], job['generator'], job['fit'], job['labeler'])
            dependency = prep if job['augment_option'] is None else table_ids.get(key, prep)
            rank = block_rank[(job['dataset'], job['seed'])] + .1
            if job['train_option'] == 'original':
                rank -= .08
            if job['labeler'] == 'generated':
                rank -= .04
            add(job['run_id'], lane['session'], rank, 'evaluate', [dependency], [PYTHON, '-u', str(Path(__file__).resolve()), 'evaluate', '--run-id', job['run_id']], job=job)
    identifiers = {t['id'] for t in tasks}
    assert len(identifiers) == len(tasks) == 242
    assert sum(t['kind'] == 'evaluate' for t in tasks) == 169
    assert sum(t['kind'] in ('fit', 'heavy_fit') for t in tasks) == 16
    assert all(set(t['dependencies']) <= identifiers for t in tasks)
    finished = set()
    while len(finished) < len(tasks):
        available = {t['id'] for t in tasks if t['id'] not in finished and set(t['dependencies']) <= finished}
        assert available, 'Dependency cycle'
        finished.update(available)
    write_json(STATE, {'created_at_chicago': now(), 'updated_at_chicago': now(), 'plan_sha256': hashlib.sha256((CACHE / 'parallel_queues.json').read_bytes()).hexdigest(), 'scope': plan['scope'], 'tasks': tasks, 'workers': {}})
    print(f'INITIALIZED {len(tasks)} tasks: 169 evaluations, 16 fits, 50 labels, 7 preparations', flush=True)


def evaluate(run_id):
    state = json.loads(STATE.read_text())
    job = next(t['job'] for t in state['tasks'] if t['id'] == run_id)
    saved = record_path(job)
    if saved.exists():
        verify_record(job)
        print('PRESERVED COMPLETED RECORD', run_id, flush=True)
        return
    # Preserve interrupted classifier artifacts before the original trainer opens them.
    for folder in ('acc', 'weight'):
        source = ROOT / 'output' / job['output_namespace'] / job['dataset'] / folder
        for path in source.glob(run_id + '.*'):
            destination = CACHE / 'preserved_partial' / 'downstream' / job['output_namespace'] / job['dataset'] / folder / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists(), destination
            shutil.copy2(path, destination)
            assert digest(path) == digest(destination)
    env = dict(os.environ, **job['environment'])
    subprocess.run(job['argv'], cwd=job['cwd'], env=env, check=True)
    verify_record(job)


def worker(name):
    (CACHE / 'logs').mkdir(exist_ok=True)
    with locked_state() as state:
        assert name not in state['workers'], 'Worker name already registered; explicit recovery required'
        state['workers'][name] = {'pid': os.getpid(), 'status': 'active', 'started_at_chicago': now()}
    while True:
        chosen = None
        with locked_state() as state:
            statuses = {t['id']: t['status'] for t in state['tasks']}
            available = [t for t in state['tasks'] if t['status'] == 'pending' and all(statuses[d] == 'complete' for d in t['dependencies']) and (t['kind'] != 'heavy_fit' or name == 'research-slow')]
            # The slow worker starts/continues long fits; fast workers prioritize closed blocks.
            if name == 'research-slow':
                preferred = [t for t in available if t['owner'] == name and t['kind'] in ('prepare', 'heavy_fit')]
                if preferred:
                    available = preferred
                else:
                    available = [t for t in available if t['kind'] in ('evaluate', 'label')]
            if available:
                chosen = min(available, key=lambda t: (t['rank'], t['owner'] != name, t['id']))
                chosen['status'] = 'running'
                chosen['claimed_by'] = name
                chosen['started_at_chicago'] = now()
                chosen['log'] = str(CACHE / 'logs' / (chosen['id'] + '.log'))
                state['workers'][name]['status'] = 'active'
                state['workers'][name]['current_task'] = chosen['id']
            elif not any(t['status'] == 'running' for t in state['tasks']):
                pending = [t for t in state['tasks'] if t['status'] != 'complete']
                state['workers'][name]['status'] = 'waiting' if pending else 'complete'
                if not pending:
                    write_json(CACHE / 'completed.json', {'completed_at_chicago': now(), 'downstream_evaluations': 169, 'tasks': len(state['tasks'])})
                    print('ALL QUEUE TASKS COMPLETE', flush=True)
                    return
                if any(t['status'] == 'failed' for t in state['tasks']):
                    state['workers'][name]['status'] = 'blocked_by_failure'
                    print('QUEUE INCOMPLETE; failed task/dependencies require explicit review', flush=True)
                    return
        if chosen is None:
            time.sleep(5)
            continue
        print(f'{now()} START {chosen["id"]} ({chosen["kind"]})', flush=True)
        with Path(chosen['log']).open('x') as log:
            result = subprocess.run(chosen['argv'], cwd=chosen['cwd'], env=dict(os.environ, **ENVIRONMENT), stdout=log, stderr=subprocess.STDOUT)
        with locked_state() as state:
            saved = next(t for t in state['tasks'] if t['id'] == chosen['id'])
            saved['finished_at_chicago'] = now()
            saved['returncode'] = result.returncode
            saved['status'] = 'complete' if result.returncode == 0 else 'failed'
            state['workers'][name]['current_task'] = None
            if result.returncode:
                state['workers'][name]['status'] = 'failed'
        print(f'{now()} {"COMPLETE" if result.returncode == 0 else "FAILED"} {chosen["id"]} exit={result.returncode}', flush=True)
        if result.returncode:
            raise SystemExit(f'Task failed; see {chosen["log"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('init', 'worker', 'evaluate'))
    parser.add_argument('--name', choices=('research-slow', 'research-fast-1', 'research-fast-2'))
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'init':
        initialize()
    elif args.action == 'evaluate':
        evaluate(args.run_id)
    else:
        worker(args.name)
