"""Save D² absolute-error sidecars for verified completed regression runs.

Run remotely. Original run records, predictions, and checkpoints are unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import d2_absolute_error_score, mean_absolute_error, r2_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(root, requests):
    target_cache = {}
    results = []
    for request in requests:
        path = root / request['path']
        assert path.resolve().is_relative_to(root / 'output'), path
        assert sha(path) == request['sha256'], path
        record = json.loads(path.read_text())
        assert record['dataset'] in {'news', 'california_housing'}
        assert record['selected_dev_epoch'] is not None
        assert Path(record['downstream_weight_path']).is_file()
        manifest = record['split_manifest']
        ids = manifest['splits']['test']
        source = Path(record['prepared_real_paths']['test']['raw'])
        source_hash = manifest['files']['test']['raw']['sha256']
        if source not in target_cache:
            assert sha(source) == source_hash, source
            target = ' shares' if record['dataset'] == 'news' else 'MedHouseVal'
            with source.open(newline='') as handle:
                targets = np.array([float(row[target]) for row in csv.DictReader(handle)])
            target_cache[source] = (source_hash, targets)
        expected_hash, raw_targets = target_cache[source]
        assert expected_hash == source_hash
        predictions = Path(record['predictions_path'])
        predictions_hash = sha(predictions)
        with predictions.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row['source_id']) for row in rows] == ids
        assert len(rows) == manifest['files']['test']['raw']['rows']
        y = np.array([float(row['y_true']) for row in rows])
        predicted = np.array([float(row['y_pred']) for row in rows])
        assert np.isfinite(y).all() and np.isfinite(predicted).all()
        np.testing.assert_allclose(y, raw_targets, rtol=1e-7, atol=1e-6)
        median = float(np.median(y))
        baseline_mae = float(np.mean(np.abs(y - median)))
        prediction_mae = float(mean_absolute_error(y, predicted))
        assert math.isfinite(baseline_mae) and baseline_mae > 0
        score = float(d2_absolute_error_score(y, predicted))
        assert math.isfinite(score)
        assert abs(score - (1 - prediction_mae / baseline_mae)) < 1e-12
        assert math.isclose(prediction_mae, record['test_scores']['mae'], rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(float(r2_score(y, predicted)), record['test_scores']['r2'], rel_tol=1e-6, abs_tol=1e-6)
        metadata = {
            'metric': 'd2_absolute_error', 'test_scores': {'d2_absolute_error': score},
            'dataset': record['dataset'], 'seed': record['seed'],
            'train_option': record['train_option'], 'augment_option': record['augment_option'],
            'selected_dev_epoch': record['selected_dev_epoch'],
            'run_record': request['path'], 'run_record_sha256': request['sha256'],
            'predictions_path': str(predictions), 'predictions_sha256': predictions_hash,
            'scikit_learn_version': version('scikit-learn'),
            'absolute_error_normalization': {
                'median_y': median, 'baseline_mae': baseline_mae,
                'prediction_mae': prediction_mae, 'split': 'test', 'rows': len(rows),
                'target_table_sha256': source_hash,
                'definition': 'd2_absolute_error = 1 - mean(abs(y_true - y_pred)) / mean(abs(y_true - median(y_true))); higher is better',
                'target_values': 'saved held-out prediction y_true; checked against prepared real test targets',
            },
        }
        destination = path.with_name(path.name.replace('.run.json', '.d2.json'))
        if destination.exists():
            assert json.loads(destination.read_text()) == metadata, destination
        results.append((destination, metadata, path, predictions))
    # Complete all input/metric checks before writing any new sidecar.
    for destination, metadata, path, predictions in results:
        assert sha(path) == metadata['run_record_sha256']
        assert sha(predictions) == metadata['predictions_sha256']
        if not destination.exists():
            destination.write_text(json.dumps(metadata, indent=2) + '\n')
    assert all(sha(path) == metadata['run_record_sha256'] for _, metadata, path, _ in results)
    return [{'path': str(destination.relative_to(root)), 'sha256': sha(destination),
             'record': metadata['run_record'], 'record_sha256': metadata['run_record_sha256'],
             'dataset': metadata['dataset'], 'seed': metadata['seed'],
             'd2_absolute_error': metadata['test_scores']['d2_absolute_error'],
             'normalization': metadata['absolute_error_normalization']}
            for destination, metadata, _, _ in results]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
    if not root.is_dir():
        raise RuntimeError('Compute regression scores on the research server only')
    records = compute(root, json.loads(args.requests.read_text()))
    evidence = {'computed_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                'script_sha256': sha(Path(__file__)), 'records': records,
                'original_records_preserved': True, 'predictions_preserved': True,
                'method': 'sklearn.metrics.d2_absolute_error_score on saved held-out predictions'}
    args.evidence.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'records': len(records), 'computed_at_chicago': evidence['computed_at_chicago']}))
