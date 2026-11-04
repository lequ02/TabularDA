"""Read-only matrix and artifact metadata inventory; execute on the research host."""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.run_corrected_matrix import DATASETS, SEEDS, methods_for
from modeling.constants import method_parts, run_name

SOURCES = {
    'adult': 'corrected_v2', 'credit': 'corrected_v2',
    'covertype': 'corrected_v2',
    'census_kdd': 'census_kdd_weighted_macro_f1_20261005',
    'mnist12': 'mnist_head_fixed_20261009',
    'mnist28': 'mnist_head_fixed_20261009',
    'news': 'news_log_v1', 'california_housing': 'housing_no_faker_20261009',
}
TAB_SOURCES = {
    'census_kdd': 'tabddpm_census_kdd_500ep_fix_20261011_v1',
    'covertype': 'tabddpm_covertype_500ep_fix_20261011_v1',
}
assets = {}


def asset(path, role):
    path = Path(path)
    key = str(path)
    if key not in assets:
        exists = path.is_file()
        resolved = path.resolve()
        relative = str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else ''
        assets[key] = {
            'remote_path': key, 'resolved_remote_path': str(resolved),
            'relative_path': relative, 'remote_present': exists,
            'bytes': path.stat().st_size if exists else None, 'roles': [],
        }
    if role not in assets[key]['roles']:
        assets[key]['roles'].append(role)
    return key


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


rows = []
for dataset in DATASETS:
    for seed in SEEDS:
        expected = [('original', None, 'real', SOURCES[dataset])]
        reference_scope = TAB_SOURCES.get(dataset, 'tabddpm_500ep_20261010_v1')
        reference = ROOT / 'output' / reference_scope / dataset / 'acc' / (run_name(dataset, seed, 'original', None) + '.run.json')
        if reference.is_file():
            expected.append(('original', None, 'tabddpm_reference', reference_scope))
        for generator in ('ctgan', 'tvae', 'tabddpm'):
            scope = TAB_SOURCES.get(dataset, 'tabddpm_500ep_20261010_v1') if generator == 'tabddpm' else SOURCES[dataset]
            expected += [(mode, method, generator, scope)
                         for method in methods_for(dataset, generator)
                         for mode in ('synthetic', 'mix')]
        for mode, method, generator, scope in expected:
            name = run_name(dataset, seed, mode, method)
            path = ROOT / 'output' / scope / dataset / 'acc' / (name + '.run.json')
            _, construction, labeler = method_parts(method) if method else ('real', 'real', 'observed')
            row = {
                'dataset': dataset, 'seed': seed, 'generator': generator,
                'construction': construction, 'labeler': labeler, 'mode': mode,
                'method': method, 'scope': scope, 'run_name': name,
                'record_relative_path': str(path.relative_to(ROOT)),
                'status': 'absent_record', 'errors': [], 'assets': {},
                'matrix_cell': generator != 'tabddpm_reference',
            }
            if not path.is_file():
                rows.append(row)
                continue
            record_bytes = path.read_bytes()
            record = json.loads(record_bytes)
            row['record_sha256'] = hashlib.sha256(record_bytes).hexdigest()
            identity = (record['dataset'], int(record['seed']), record['train_option'], record['augment_option'])
            if identity != (dataset, seed, mode, method):
                row['errors'].append('record_identity_mismatch')
            if record.get('selected_dev_epoch') is None:
                row['errors'].append('missing_selected_dev_epoch')
            row['selected_dev_epoch'] = record.get('selected_dev_epoch')
            row['test_scores'] = record.get('test_scores', {})
            row['source_sha256'] = record.get('source_sha256')
            row['target_normalization'] = record.get('target_normalization')
            provenance = record['generator_provenance'] if mode != 'original' else {}
            row['generator_parameters'] = provenance.get('parameters', {})
            row['generator_sample_size'] = provenance.get('sample_size')
            refs = {'run_record': [asset(path, 'run_record')]}
            for key, value in record.items():
                if key.endswith('_path') and value:
                    refs[key] = [asset(value, key)]
            for split, views in record.get('prepared_real_paths', {}).items():
                for view, value in views.items():
                    refs[f'real_{split}_{view}'] = [asset(value, f'real_{split}_{view}')]
            refs['saved_hash_dependency'] = [asset(p, 'saved_hash_dependency') for p in record.get('artifact_sha256', {})]
            for required in ('predictions_path', 'downstream_weight_path', 'split_manifest_path'):
                if not refs.get(required):
                    row['errors'].append('missing_field:' + required)
            if mode != 'original':
                for required in ('synthetic_path', 'generator_model_path', 'generator_provenance_path'):
                    if not refs.get(required):
                        row['errors'].append('missing_field:' + required)
                if record.get('synthetic_quality', {}).get('rows') != 100000:
                    row['errors'].append('synthetic_rows_not_100000')
            if dataset in ('news', 'california_housing'):
                normal = record.get('target_normalization', {})
                if not {'mae', 'r2', 'nmae_sigma'} <= row['test_scores'].keys():
                    row['errors'].append('missing_regression_metrics')
                if normal.get('split') != 'test' or normal.get('ddof') != 0 or not normal.get('target_table_sha256') or not normal.get('rows'):
                    row['errors'].append('invalid_saved_target_normalization')
            for role, paths in refs.items():
                for ref in paths:
                    if not assets[ref]['remote_present']:
                        row['errors'].append('missing_artifact:' + role + ':' + ref)
            # Curves, history, reports and metric sidecars are indexed separately
            # from completion dependencies. Their absence is visible, not hidden.
            for sibling in sorted(path.parent.glob(name + '.*')):
                suffix = sibling.name[len(name) + 1:]
                role = ('training_plot' if suffix.endswith('_curve.png') else
                        'epoch_history' if 'history' in suffix else
                        'metric_sidecar' if suffix == 'd2.json' else 'run_output')
                refs.setdefault(role, []).append(asset(sibling, role))
            refs['arm_log'] = [asset(p, 'arm_log') for p in sorted((ROOT / 'output' / scope / 'logs').glob('**/' + name + '.log'))]
            model_path = record.get('generator_model_path')
            if model_path:
                for p in sorted(Path(model_path).parent.glob(Path(model_path).stem + '*')):
                    if p.is_file():
                        refs.setdefault('generator_output', []).append(asset(p, 'generator_output'))
            row['assets'] = refs
            row['status'] = 'invalid_present' if row['errors'] else 'complete_presence_checked'
            rows.append(row)

scopes = sorted(set(SOURCES.values()) | {'tabddpm_500ep_20261010_v1'} | set(TAB_SOURCES.values()))
logs = []
for scope in scopes:
    log_root = ROOT / 'output' / scope / 'logs'
    # Worker/queue logs provide state evidence, not experiment completion proof.
    for p in sorted(log_root.rglob('*.log')) if log_root.exists() else []:
        if 'workers' in p.parts or 'queue' in p.name:
            with p.open('rb') as handle:
                handle.seek(max(0, p.stat().st_size - 2400))
                tail = handle.read().decode('utf-8', errors='replace')
            logs.append({'path': asset(p, 'worker_queue_log'), 'tail': tail})
queue_logs = []
for rel in ('.cache/full_completion_20261007', '.cache/experiment_repairs_20261001'):
    p = ROOT / rel
    if p.exists():
        for log in sorted(p.glob('*.log')):
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 1500))
                tail = handle.read().decode('utf-8', errors='replace')
            queue_logs.append({'path': asset(log, 'historical_queue_log'), 'tail': tail})
summary = Counter(row['status'] for row in rows if row['matrix_cell'])
ps = command(['ps', '-eo', 'pid,etimes,args'])
active = [line for line in ps['stdout'].splitlines()
          if any(token in line for token in ('-m modeling', '-m synthesize_data', 'queue_runner.py', 'worker.py', 'run_corrected_matrix.py'))
          and not ('ps -eo' in line or 'sshd:' in line)]
print(json.dumps({
    'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'remote_root': str(ROOT), 'datasets': list(DATASETS), 'seeds': list(SEEDS),
    'source_namespaces': SOURCES, 'tabddpm_source_overrides': TAB_SOURCES,
    'summary': dict(summary), 'rows': rows, 'assets': assets,
    'active_research_processes': active,
    'tmux': command(['tmux', 'list-panes', '-a', '-F', '#S:#I.#P dead=#{pane_dead} command=#{pane_current_command}']),
    'environment': command(['/home/thuy/miniconda3/envs/env/bin/python', '-c',
        'import sys,importlib.metadata as m; print(sys.executable); print({p:m.version(p) for p in ("torch","sdv","ctgan","numpy","pandas","scikit-learn")})']),
    'resources': {'ram': command(['free', '-h']), 'gpu': command(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'])},
    'worker_logs': logs, 'historical_queue_logs': queue_logs,
    'checks': {'record_identity': True, 'selected_dev_epoch': True,
               'referenced_paths_present': True, 'saved_regression_normalization': True,
               'synthetic_row_count_metadata': True, 'all_artifact_hashes_recomputed': False,
               'models_loaded': False, 'experiments_executed': False},
}, indent=2))
