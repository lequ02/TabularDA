"""Read-only small-file inventory, executed on the research server via SSH stdin."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
DATASETS = ('adult', 'census_kdd', 'credit', 'covertype', 'intrusion', 'mnist12', 'mnist28', 'news', 'california_housing')
LABELERS = ('rf', 'xgb', 'dnn')


def tail(path, size=2500):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode('utf-8', errors='replace')


def resolved(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


result = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(), 'groups': [], 'records': [], 'queues': {}, 'generator_logs': [], 'simulated': [], 'namespaces': []}
for output in sorted((ROOT / 'output').iterdir()):
    if output.is_dir():
        result['namespaces'].append(output.name)

for dataset in DATASETS:
    for seed in (42, 43):
        namespace = ('census_kdd_weighted_macro_f1_20261005' if dataset == 'census_kdd' else
                     'corrected_v2_seed42_mnist28_news' if dataset in ('mnist28', 'news') and seed == 42 else 'corrected_v2')
        acc = ROOT / 'output' / namespace / dataset / 'acc'
        prefix = f'{dataset}_seed{seed}_'
        expected = [prefix + 'real_original']
        for generator in ('ctgan', 'tvae'):
            expected += [prefix + f'{generator}_full_generated_{mode}' for mode in ('synthetic', 'mix')]
            expected += [prefix + f'{generator}_{fit}_{label}_{mode}' for fit in ('full', 'xonly') for label in LABELERS for mode in ('synthetic', 'mix')]
        records = {}
        for path in sorted(acc.glob(prefix + '*.run.json')):
            saved = json.loads(path.read_text())
            missing_artifacts = [saved[key] for key in ('predictions_path', 'downstream_weight_path') if key not in saved or not resolved(saved[key]).is_file()]
            assert saved['dataset'] == dataset and saved['seed'] == seed, str(path)
            name = path.name.removesuffix('.run.json')
            row = {'dataset': dataset, 'seed': seed, 'namespace': namespace, 'run_id': name, 'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'selected_dev_epoch': saved.get('selected_dev_epoch'), 'train_option': saved['train_option'], 'augment_option': saved['augment_option'], 'missing_artifacts': missing_artifacts, 'test_scores': saved['test_scores']}
            result['records'].append(row)
            records[name] = row
        complete = [name for name in expected if name in records and records[name]['selected_dev_epoch'] is not None and not records[name]['missing_artifacts']]
        missing = []
        for name in expected:
            if name in complete:
                continue
            candidates = [ROOT / 'output' / namespace / 'logs' / dataset / f'seed_{seed}' / (name + '.log'), ROOT / 'output' / namespace / 'logs' / (name + '.log')]
            log = next((p for p in candidates if p.is_file()), None)
            missing.append({'run_id': name, 'state': 'record_incomplete' if name in records else 'attempted_no_completed_record' if log else 'no_attempt_log_in_selected_namespace', 'log': str(log.relative_to(ROOT)) if log else None, 'log_tail': tail(log) if log else None})
        artifact_ns = 'corrected_v2_seed42_mnist28_news' if dataset in ('mnist28', 'news') and seed == 42 else 'corrected_v2'
        prepared = ROOT / 'data' / artifact_ns / dataset / f'seed_{seed}'
        fits = []
        for generator in ('ctgan', 'tvae'):
            for fit in ('full', 'xonly'):
                model = ROOT / 'sdv trained model' / artifact_ns / dataset / f'seed_{seed}' / f'{prefix}{generator}_{fit}.pkl'
                prov = model.with_suffix('.provenance.json')
                fits.append({'generator': generator, 'fit': fit, 'model_present': model.is_file(), 'provenance_present': prov.is_file(), 'parameters': json.loads(prov.read_text()).get('parameters') if prov.is_file() else None})
        tables = {name: (prepared / (name.rsplit('_', 1)[0] + '_100k.csv')).is_file() for name in expected if not name.endswith('real_original')}
        result['groups'].append({'dataset': dataset, 'seed': seed, 'namespace': namespace, 'planned_selected': len(expected), 'completed_selected': len(complete), 'complete': complete, 'missing': missing, 'total_record_files': len(records), 'fits': fits, 'tables_present_for_run': tables, 'split_manifest_present': (prepared / 'split_manifest.json').is_file()})
        for output_ns in {'corrected_v2', namespace}:
            logdir = ROOT / 'output' / output_ns / 'logs' / dataset / f'seed_{seed}'
            for log in sorted(logdir.glob('*generate*.log')):
                result['generator_logs'].append({'path': str(log.relative_to(ROOT)), 'tail': tail(log)})

for output_ns in ('corrected_v2', 'corrected_v2_seed42_mnist28_news', 'census_kdd_weighted_macro_f1_20261005', 'credit_weighted_macro_f1_pilot_20261006', 'census_weighted_pilot_20261004'):
    folder = ROOT / 'output' / output_ns
    for filename in ('status.json', 'completed.json', 'manifest.json'):
        path = folder / filename
        if path.is_file():
            saved = json.loads(path.read_text())
            if filename == 'manifest.json':
                saved = {k: saved[k] for k in ('created_at_chicago', 'planned_runs', 'new_training_runs', 'jobs', 'protocol') if k in saved}
            result['queues'][str(path.relative_to(ROOT))] = saved
    for path in folder.glob('failures*.json'):
        result['queues'][str(path.relative_to(ROOT))] = json.loads(path.read_text())
    if 'pilot' in output_ns:
        result['queues'][output_ns + '/record_names'] = [str(p.relative_to(ROOT)) for p in folder.rglob('*.run.json')]
for folder in sorted((ROOT / '.cache').iterdir()):
    if not folder.is_dir() or not any(x in folder.name for x in ('repair', 'weighted', 'seed', 'intrusion')):
        continue
    for path in folder.glob('*.json'):
        if any(x in path.name for x in ('status', 'failure', 'completed')):
            result['queues'][str(path.relative_to(ROOT))] = json.loads(path.read_text())
    for path in folder.glob('*queue*.log'):
        result['queues'][str(path.relative_to(ROOT))] = tail(path)
for folder in sorted((ROOT / 'output').glob('simulated*')):
    per_run = folder / 'per_run.csv'
    if not per_run.is_file():
        continue
    rows = list(csv.DictReader(per_run.open()))
    config_path = folder / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    result['simulated'].append({'namespace': folder.name, 'rows': len(rows), 'sha256': hashlib.sha256(per_run.read_bytes()).hexdigest(), 'columns': list(rows[0]) if rows else [], 'config': config})
result['processes'] = subprocess.check_output(['ps', '-u', 'thuy', '-o', 'pid,ppid,etimes,pcpu,args'], text=True)
result['tmux_panes'] = subprocess.check_output(['tmux', 'list-panes', '-a', '-F', '#S:#I.#P pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'], text=True)
result['disk'] = subprocess.check_output(['df', '-h', str(ROOT)], text=True)
print(json.dumps(result))
