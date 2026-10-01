"""Read-only inventory; supplied to the research host through SSH stdin."""
import json
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
datasets = ('adult', 'census_kdd', 'credit', 'covertype', 'intrusion', 'mnist12', 'mnist28', 'news')
result = {'checked_at': datetime.now().astimezone().isoformat(), 'root': str(ROOT), 'groups': []}
for dataset in datasets:
    labels = ('pca_gmm', 'rf', 'xgb', 'dnn') if dataset == 'news' else ('gaussian', 'categorical', 'pca_gmm', 'rf', 'xgb', 'dnn')
    for seed in (42, 43):
        prefix = f'{dataset}_seed{seed}_'
        expected = [prefix + 'real_original']
        for generator in ('ctgan', 'tvae'):
            expected.extend(prefix + f'{generator}_full_generated_{mode}' for mode in ('synthetic', 'mix'))
            expected.extend(prefix + f'{generator}_{fit}_{label}_{mode}'
                            for fit in ('full', 'xonly') for label in labels for mode in ('synthetic', 'mix'))
        complete = {p.name[:-len('.run.json')] for p in (ROOT / 'output/corrected_v2' / dataset / 'acc').glob('*.run.json')}
        missing = [name for name in expected if name not in complete]
        logs = ROOT / 'output/corrected_v2/logs' / dataset / f'seed_{seed}'
        missing_logs = []
        for name in missing:
            path = logs / (name + '.log')
            if path.exists():
                with path.open('rb') as f:
                    f.seek(max(0, path.stat().st_size - 3500))
                    tail = f.read().decode('utf-8', errors='replace')
                missing_logs.append({'run': name, 'modified': datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(), 'tail': tail})
        provenance = [p.name for p in (ROOT / 'sdv trained model/corrected_v2' / dataset / f'seed_{seed}').glob('*.provenance.json')]
        result['groups'].append({'dataset': dataset, 'seed': seed, 'expected': len(expected), 'completed': len(set(expected) & complete), 'completed_names': sorted(set(expected) & complete), 'missing': missing, 'missing_logs': missing_logs, 'generator_provenance': provenance})
result['processes'] = subprocess.check_output(['ps', '-u', 'thuy', '-o', 'pid,ppid,etimes,pcpu,args'], text=True)
result['recent_logs'] = []
paths = sorted((ROOT / 'output/corrected_v2/logs').rglob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
for path in paths:
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 2000))
        tail = f.read().decode('utf-8', errors='replace')
    result['recent_logs'].append({'path': str(path.relative_to(ROOT)), 'modified': datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(), 'tail': tail})
result['failure_files'] = {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / 'output/corrected_v2').glob('failures*.json')}
print(json.dumps(result))
