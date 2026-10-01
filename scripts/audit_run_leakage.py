"""Read-only leakage audit. Emits aggregate JSON lines; never loads model pickles."""
import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def emit(kind, **values):
    print(json.dumps(dict(kind=kind, **values), default=str), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mapped(root, value):
    value = str(value)
    marker = '/TabularDA/'
    return root / value.split(marker, 1)[1] if marker in value else Path(value)


def fingerprints(path, target, numeric):
    before = path.stat()
    features, full = [], []
    rows = 0
    for frame in pd.read_csv(path, chunksize=4096, dtype='float64' if numeric else str,
                             keep_default_na=False):
        columns = sorted(c for c in frame if c != target)
        features.append(pd.util.hash_pandas_object(frame[columns], index=False).to_numpy())
        full.append(pd.util.hash_pandas_object(frame[columns + [target]], index=False).to_numpy())
        rows += len(frame)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during audit: ' + str(path))
    return np.concatenate(features), np.concatenate(full), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--data', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    groups = defaultdict(list)
    failures = []
    for scope in ('corrected_v2', 'pilot_ctgan_v1'):
        for path in sorted((root / 'output' / scope).rglob('*.run.json')):
            record = json.loads(path.read_text())
            groups[(scope, record['dataset'], record['seed'])].append((path, record))
    emit('inventory', groups=len(groups), runs=sum(map(len, groups.values())))
    for (scope, dataset, seed), records in sorted(groups.items()):
        issues = []
        first = records[0][1]['split_manifest']
        ids = first['splits']
        pairs = list(itertools.combinations(('train', 'dev', 'test'), 2))
        id_overlaps = {a + '_vs_' + b: len(set(ids[a]) & set(ids[b])) for a, b in pairs}
        if any(id_overlaps.values()) or any(len(v) != len(set(v)) for v in ids.values()):
            issues.append('source IDs overlap or repeat')
        if any(v < 0 or v >= first['source_row_count'] for values in ids.values() for v in values):
            issues.append('source IDs outside source row count')
        if any(count for overlap in first['train_overlap'].values() for count in overlap.values()):
            issues.append('manifest records train/holdout overlap')
        splits = records[0][1]['prepared_real_paths']
        target = first['files']['train']['raw']['columns'][-1]
        synthetic = {}
        prediction_checks = 0
        source_versions = Counter()
        for path, r in records:
            source_versions[r.get('source_sha256', 'missing')] += 1
            if r['split_manifest']['files'] != first['files'] or r['split_manifest']['splits'] != ids:
                issues.append(path.name + ': inconsistent split manifest')
            if r.get('selection_metric') != 'loss' or r.get('selected_dev_epoch') is None:
                issues.append(path.name + ': missing dev checkpoint selection')
            if r.get('synthetic_path'):
                synthetic[r['synthetic_path']] = r
                p = r.get('generator_provenance') or {}
                train = first['files']['train']['raw']
                if p.get('training_rows') != train['rows'] or p.get('seed') != seed:
                    issues.append(path.name + ': generator fit identity mismatch')
                if p.get('training_columns') == train['columns']:
                    if p.get('fit_table_sha256') != train['sha256']:
                        issues.append(path.name + ': full generator fit hash mismatch')
                elif p.get('training_columns') != train['columns'][:-1]:
                    issues.append(path.name + ': unexpected generator fit columns')
            prediction_path = mapped(root, r['predictions_path'])
            if prediction_path.exists():
                pred = pd.read_csv(prediction_path)
                if pred['source_id'].tolist() != ids['test']:
                    issues.append(path.name + ': test prediction IDs mismatch')
                prediction_checks += 1
            else:
                issues.append(path.name + ': missing test predictions')
        emit('records', scope=scope, dataset=dataset, seed=seed, runs=len(records),
             synthetic_tables=len(synthetic), id_overlaps=id_overlaps,
             prediction_id_checks=prediction_checks, source_versions=source_versions, issues=issues)
        failures.extend(issues)
        if not args.data:
            continue
        arrays = {}
        for view in ('raw', 'onehot'):
            for split in ('train', 'dev', 'test'):
                path = mapped(root, splits[split][view])
                if not path.exists():
                    emit('unavailable', path=str(path))
                    continue
                expected = first['files'][split][view]
                digest = sha(path)
                array = fingerprints(path, target, view == 'onehot')
                arrays[(split, view)] = array
                valid = digest == expected['sha256'] and array[2] == expected['rows'] == len(ids[split])
                emit('split_file', scope=scope, dataset=dataset, seed=seed, split=split,
                     view=view, hash_match=digest == expected['sha256'], rows=array[2],
                     expected_rows=expected['rows'], ids=len(ids[split]))
                if not valid:
                    failures.append(str(path) + ': split hash/row mismatch')
            for a, b in pairs:
                if (a, view) in arrays and (b, view) in arrays:
                    aa, bb = arrays[(a, view)], arrays[(b, view)]
                    overlap = int(np.isin(bb[0], aa[0]).sum())
                    emit('real_overlap', scope=scope, dataset=dataset, seed=seed, view=view,
                         pair=a + '_vs_' + b, feature_rows=overlap,
                         full_rows=int(np.isin(bb[1], aa[1]).sum()))
                    if overlap:
                        failures.append(f'{scope}/{dataset}/{seed}/{view}/{a}-{b}: {overlap} feature matches')
        raw_train = mapped(root, splits['train']['raw'])
        checked_fit = set()
        if raw_train.exists():
            # Reproduce precisely the serialization used when fitting each generator.
            train_frame = pd.read_csv(raw_train)
            for _, r in records:
                p = r.get('generator_provenance')
                if not p or p['fit_table_sha256'] in checked_fit:
                    continue
                checked_fit.add(p['fit_table_sha256'])
                digest = hashlib.sha256(train_frame[p['training_columns']].to_csv(
                    index=False, lineterminator='\n').encode()).hexdigest()
                valid = digest == p['fit_table_sha256']
                emit('generator_fit', scope=scope, dataset=dataset, seed=seed,
                     columns=len(p['training_columns']), match=valid)
                if not valid:
                    failures.append(f'{dataset}/{seed}: generator fit hash mismatch')
            del train_frame
        for value, record in sorted(synthetic.items()):
            path = mapped(root, value)
            if not path.exists():
                emit('unavailable', path=str(path))
                continue
            syn = fingerprints(path, target, True)
            overlaps = {}
            for split in ('train', 'dev', 'test'):
                if (split, 'onehot') not in arrays:
                    continue
                reference = arrays[(split, 'onehot')]
                overlaps[split] = dict(
                    synthetic_feature_rows=int(np.isin(syn[0], reference[0]).sum()),
                    holdout_feature_rows=int(np.isin(reference[0], syn[0]).sum()),
                    synthetic_full_rows=int(np.isin(syn[1], reference[1]).sum()))
                if split != 'train' and overlaps[split]['holdout_feature_rows']:
                    failures.append(str(path) + f': {split} feature overlap')
            emit('synthetic_overlap', scope=scope, dataset=dataset, seed=seed,
                 table=path.name, rows=syn[2], overlaps=overlaps)
        emit('group_complete', scope=scope, dataset=dataset, seed=seed)
    emit('complete', issues=failures, issue_count=len(failures), data_checked=args.data)


if __name__ == '__main__':
    main()
