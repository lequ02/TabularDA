"""Run both real-only seeds for one frozen dataset/dropout configuration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = json.loads(Path(sys.argv[1]).read_text())
runtime = Path(config['runtime'])
out = Path(config['output'])
status = out / 'worker_status.json'
state = {'state': 'running', 'pid': os.getpid(), 'started_at_chicago': stamp(),
         'dataset': config['dataset'], 'dropout': config['dropout'],
         'completed': [], 'running_seed': None}


def save():
    status.write_text(json.dumps(state, indent=2) + '\n')


save()
environment = os.environ.copy()
environment.update(config['environment'])
for task in config['tasks']:
    for name, expected in config['source_hashes'].items():
        assert sha(runtime / name) == expected, name
    for name, expected in task['input_hashes'].items():
        assert sha(Path(task['input_directory']) / name) == expected, name
    seed = task['seed']
    record_path = Path(task['record_path'])
    if record_path.exists():
        raise FileExistsError(record_path)
    state.update(running_seed=seed, command=task['command'], log_path=task['log_path'])
    save()
    print(f"Starting {config['dataset']} dropout={config['dropout']} seed={seed} at {stamp()}", flush=True)
    with Path(task['log_path']).open('x') as log:
        result = subprocess.run(task['command'], cwd=runtime, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(state='failed', exit_code=result.returncode, finished_at_chicago=stamp())
        save()
        raise SystemExit(result.returncode)
    record = json.loads(record_path.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == (
        config['dataset'], seed, 'original', None)
    assert (record['batch_size'], record['learning_rate'], record['epoch_budget'],
            record['selection_metric']) == (128, 0.001, 100, 'loss')
    assert record['source_sha256'] == config['source_tree_sha256']
    assert record['target_normalization']['split'] == 'test'
    assert record['target_normalization']['ddof'] == 0
    assert 'nmae_sigma' in record['test_scores']
    if config['target_transform'] == 'log':
        assert record['target_transform']['name'] == 'log'
    else:
        assert 'target_transform' not in record
    state['completed'].append({'seed': seed, 'record_path': str(record_path),
                               'record_sha256': sha(record_path),
                               'selected_dev_epoch': record['selected_dev_epoch'],
                               'test_scores': record['test_scores'],
                               'finished_at_chicago': stamp()})
    save()
state.update(state='complete', running_seed=None, finished_at_chicago=stamp())
save()
print('Both real-only seeds completed at ' + stamp(), flush=True)
