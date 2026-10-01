"""Read-only checks of predictions, checkpoint selection, and simulated provenance."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(kind, **values):
    print(json.dumps(dict(kind=kind, **values), default=str), flush=True)


issues = []
cache = {}
counts = Counter()
source_hash = hashlib.sha256()
for path in sorted((ROOT / 'src').rglob('*.py')):
    source_hash.update(str(path.relative_to(ROOT)).encode())
    source_hash.update(path.read_bytes())
current_hash = source_hash.hexdigest()
record_hashes = Counter()
for scope in ('corrected_v2', 'pilot_ctgan_v1'):
    for path in sorted((ROOT / 'output' / scope).rglob('*.run.json')):
        r = json.loads(path.read_text())
        key = (scope, r['dataset'], r['seed'])
        record_hashes[r.get('source_sha256')] += 1
        if key not in cache:
            raw = pd.read_csv(r['prepared_real_paths']['train']['raw'])
            target = r['split_manifest']['files']['train']['raw']['columns'][-1]
            test = pd.read_csv(r['prepared_real_paths']['test']['raw'])
            labels = np.sort(raw[target].unique())
            y = pd.Index(labels).get_indexer(test[target])
            cache[key] = y
        pred = pd.read_csv(r['predictions_path'])
        true_col = 'y_true' if 'y_true' in pred else 'target'
        if true_col not in pred:
            issues.append(path.name + ': unexpected prediction schema ' + str(list(pred)))
        elif not np.array_equal(pred[true_col].to_numpy(), cache[key]):
            issues.append(path.name + ': true labels differ from real test')
        else:
            counts['prediction_labels_checked'] += 1
        epochs_path = path.with_name(path.name.replace('.run.json', '.epochs.csv'))
        epochs = pd.read_csv(epochs_path)
        selected = int(r['selected_dev_epoch'])
        # Replay the trainer's 1e-5 improvement threshold.
        best = float('inf')
        min_epoch = None
        for _, epoch_row in epochs.iterrows():
            if best > float(epoch_row['dev_loss']) + 1e-5:
                best = float(epoch_row['dev_loss'])
                min_epoch = int(epoch_row['global_round'])
        if selected != min_epoch:
            issues.append(path.name + f': selected epoch {selected} differs from min dev epoch {min_epoch}')
        else:
            counts['dev_checkpoint_selections_checked'] += 1
        for field in ('downstream_weight_path', 'generator_model_path', 'predictor_model_path'):
            if r.get(field) and not Path(r[field]).is_file():
                issues.append(path.name + ': missing ' + field)
        manifest = json.loads(Path(r['split_manifest_path']).read_text())
        if manifest != r['split_manifest']:
            issues.append(path.name + ': embedded manifest differs from disk')
emit('real_provenance', counts=counts, current_source_hash=current_hash,
     recorded_source_hashes=record_hashes, issues=issues)

run = ROOT / 'SDGym-research/data/simulated_paper'
rows = pd.read_csv(run / 'simulated_methods_per_run.csv')
hashes = {}
sim_issues = []
for (dataset, seed), group in rows.groupby(['dataset', 'seed']):
    folder = run / f'seed_{seed}' / dataset
    manifest = json.loads((folder / 'manifest.json').read_text())
    train = pd.read_csv(folder / 'train.csv', dtype=str)
    test = pd.read_csv(folder / 'test.csv', dtype=str)
    train_hash = pd.util.hash_pandas_object(train, index=False)
    test_hash = pd.util.hash_pandas_object(test, index=False)
    emit('simulated_split', dataset=dataset, seed=int(seed),
         train_rows=len(train), test_rows=len(test),
         exact_test_matches=int(test_hash.isin(set(train_hash)).sum()),
         train_hash_match=sha(folder / 'train.csv') == manifest['train_sha256'],
         test_hash_match=sha(folder / 'test.csv') == manifest['test_sha256'])
    for _, row in group.iterrows():
        suffix = 'result' if row['benchmark'] == 'paper' else 'labeled'
        record_path = folder / f"{row['method']}_{suffix}.json"
        r = json.loads(record_path.read_text())
        for field in ('synthetic', 'source'):
            if r.get(field + '_path'):
                value = r[field + '_path']
                if value not in hashes:
                    hashes[value] = sha(value)
                if hashes[value] != r[field + '_sha256']:
                    sim_issues.append(str(record_path) + ': ' + field + ' hash mismatch')
        if any(r.get(k) != manifest[k] for k in ('train_sha256', 'test_sha256')):
            sim_issues.append(str(record_path) + ': split/oracle hash mismatch')
        if 'oracle_sha256' in r and r['oracle_sha256'] != manifest['oracle_sha256']:
            sim_issues.append(str(record_path) + ': oracle hash mismatch')
    sample_metadata = list(folder.glob('*_sample.json'))
    emit('simulated_generation_metadata', dataset=dataset, seed=int(seed),
         records=len(group), sample_metadata_files=len(sample_metadata))
emit('simulated_complete', runs=len(rows), unique_data_hashes_checked=len(hashes), issues=sim_issues)
