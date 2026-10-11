"""Run Table 5 paired loss tests on the research server, using frozen fits."""
import hashlib
import itertools
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import d2_absolute_error_score, r2_score
import sklearn

HERE = Path(__file__).resolve().parent
REPO = Path('/home/thuy/Research/minh_data_synth/TabularDA')
assert HERE.is_relative_to(REPO / '.cache'), 'Execute statistical calculations remotely.'
B = 99_999
BATCH = 128
RNG_SEED = 20261011
LABELERS = ('rf', 'xgb', 'dnn')
MODES = ('synthetic', 'mix')
CONTRASTS = ('relabel_minus_generated', 'has_y1_minus_has_y0')
METRICS = ('r2', 'd2_absolute_error')
requests = json.loads((HERE / 'requests.json').read_text())
assert len(requests['records']) == 112
OUT = HERE / 'results'
OUT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


verified = []
tests = []
datasets = []
vectors = []
for dataset, target in [('news', ' shares'), ('california_housing', 'MedHouseVal')]:
    selected = [r for r in requests['records'] if r['dataset'] == dataset]
    assert len(selected) == 56
    losses = {}
    canonical = None
    for request in selected:
        record_path = REPO / request['record']
        sidecar_path = REPO / request['sidecar']
        assert sha(record_path) == request['record_sha256'], record_path
        assert sha(sidecar_path) == request['sidecar_sha256'], sidecar_path
        record = json.loads(record_path.read_text())
        sidecar = json.loads(sidecar_path.read_text())
        assert record['selected_dev_epoch'] is not None
        assert Path(record['downstream_weight_path']).is_file()
        assert record['dataset'] == dataset and record['seed'] == request['seed']
        assert record['train_option'] == request['mode']
        manifest = record['split_manifest']
        raw = Path(record['prepared_real_paths']['test']['raw'])
        raw_hash = manifest['files']['test']['raw']['sha256']
        assert sha(raw) == raw_hash
        frame = pd.read_csv(raw)
        assert list(frame.columns) == manifest['files']['test']['raw']['columns']
        prediction_path = Path(record['predictions_path'])
        prediction_hash = sha(prediction_path)
        assert prediction_hash == sidecar['predictions_sha256']
        prediction = pd.read_csv(prediction_path)
        ids = prediction['source_id'].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(ids, manifest['splits']['test'])
        y = prediction['y_true'].to_numpy(dtype=float)
        pred = prediction['y_pred'].to_numpy(dtype=float)
        assert len(y) == len(frame) == manifest['files']['test']['raw']['rows']
        assert len(np.unique(ids)) == len(ids)
        assert np.isfinite(y).all() and np.isfinite(pred).all()
        np.testing.assert_allclose(y, frame[target].to_numpy(), rtol=1e-7, atol=1e-6)
        feature_hash = pd.util.hash_pandas_object(frame.drop(columns=[target]), index=False).to_numpy()
        if canonical is None:
            canonical = (ids, y, raw_hash, feature_hash)
        else:
            np.testing.assert_array_equal(ids, canonical[0])
            np.testing.assert_array_equal(y, canonical[1])
            assert raw_hash == canonical[2]
            np.testing.assert_array_equal(feature_hash, canonical[3])
        error = np.column_stack(((y - pred)**2, np.abs(y - pred)))
        mse, mae = error.mean(axis=0)
        r2 = float(r2_score(y, pred))
        d2 = float(d2_absolute_error_score(y, pred))
        np.testing.assert_allclose([mse, mae, r2],
                                   [record['test_scores'][k] for k in ('mse', 'mae', 'r2')],
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose([record['test_scores'][k] for k in ('mse', 'mae', 'r2')],
                                   [request[k] for k in ('mse', 'mae', 'r2')], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d2, request['d2_absolute_error'], rtol=1e-12, atol=1e-12)
        norm = sidecar['absolute_error_normalization']
        assert norm['rows'] == len(y) and norm['target_table_sha256'] == raw_hash
        np.testing.assert_allclose(norm['prediction_mae'], mae, rtol=1e-12)
        key = (request['seed'], request['generator'], request['source'], request['labeler'], request['mode'])
        assert key not in losses
        losses[key] = error
        assert sha(record_path) == request['record_sha256']
        assert sha(sidecar_path) == request['sidecar_sha256']
        verified.append({**request, 'predictions_path': str(prediction_path),
                         'predictions_sha256': prediction_hash, 'test_table_sha256': raw_hash,
                         'rows': len(y), 'r2_reproduced': r2, 'd2_reproduced': d2})
    ids, y, raw_hash, feature_hash = canonical
    _, groups = np.unique(feature_hash, return_inverse=True)
    counts = np.bincount(groups)
    # Verify that grouping uses identical feature rows, not accidental hash collisions.
    assert len(counts) == len(frame.drop(columns=[target]).drop_duplicates())
    normalizers = np.array([np.mean((y-y.mean())**2), np.mean(np.abs(y-np.median(y)))])
    assert np.all(normalizers > 0)
    columns, labels = [], []
    for mode in MODES:
        relabeled = [losses[(s, g, source, label, mode)] for s in (42, 43)
                     for g in ('ctgan', 'tvae') for source in ('full', 'xonly') for label in LABELERS]
        generated = [losses[(s, g, 'full', 'generated', mode)] for s in (42, 43) for g in ('ctgan', 'tvae')]
        joint = [losses[(s, g, 'full', label, mode)] for s in (42, 43) for g in ('ctgan', 'tvae') for label in LABELERS]
        xonly = [losses[(s, g, 'xonly', label, mode)] for s in (42, 43) for g in ('ctgan', 'tvae') for label in LABELERS]
        for contrast, left, right in [(CONTRASTS[0], relabeled, generated), (CONTRASTS[1], joint, xonly)]:
            # Average losses of fixed models, rather than errors of averaged predictions.
            benefit = (np.mean(right, axis=0) - np.mean(left, axis=0)) / normalizers
            for j, metric in enumerate(METRICS):
                columns.append(benefit[:, j])
                labels.append((mode, contrast, metric))
    matrix = np.column_stack(columns)
    grouped = np.column_stack([np.bincount(groups, weights=matrix[:, j]) for j in range(8)])
    observed = grouped.sum(axis=0) / len(y)
    rng = np.random.default_rng(np.random.SeedSequence([RNG_SEED, 0 if dataset == 'news' else 1]))
    exceed = np.zeros(8, dtype=np.int64)
    # A group gets one sign shared across all fits, seeds, regimes, and metrics.
    # This preserves the dependence induced by their common test cases.
    threshold = np.abs(observed) - np.maximum(1e-15, np.abs(observed)*1e-12)
    for start in range(0, B, BATCH):
        n = min(BATCH, B-start)
        signs = 2*rng.integers(0, 2, size=(n, len(counts)), dtype=np.int8)-1
        simulated = signs.astype(float) @ grouped / len(y)
        exceed += np.sum(np.abs(simulated) >= threshold, axis=0)
    for j, (mode, contrast, metric) in enumerate(labels):
        expected = next(r for r in requests['task_contrasts'] if
                        (r['dataset'], r['training'], r['contrast'], r['metric']) == (dataset, mode, contrast, metric))
        delta = float(expected['native_difference'])
        assert abs(observed[j] - delta) < 1e-6, (dataset, mode, contrast, metric, observed[j], delta)
        tests.append({'dataset': dataset, 'training': mode, 'contrast': contrast, 'metric': metric,
                      'native_difference': delta, 'prediction_difference': float(observed[j]),
                      'p_raw': float((exceed[j]+1)/(B+1)), 'extreme_resamples': int(exceed[j]),
                      'n_rows': len(y), 'n_feature_groups': len(counts), 'resamples': B})
        vectors.append(pd.DataFrame({'dataset': dataset, 'training': mode, 'contrast': contrast,
                                     'metric': metric, 'feature_group': np.arange(len(counts)),
                                     'group_rows': counts, 'normalized_benefit_sum': grouped[:, j]}))
    datasets.append({'dataset': dataset, 'rows': len(y), 'feature_groups': len(counts),
                     'largest_group': int(counts.max()), 'duplicate_feature_rows': int(len(y)-len(counts)),
                     'source_id_order_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                     'target_table_sha256': raw_hash, 'r2_baseline_mse': float(normalizers[0]),
                     'd2_baseline_mae': float(normalizers[1])})
    print(dataset, 'verified; tests complete', flush=True)
assert len(verified) == 112 and len(tests) == 16
pd.concat(vectors, ignore_index=True).to_csv(OUT / 'group_contrasts.csv', index=False)
result = {'completed_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'method': 'Two-sided paired group sign-flip permutation test of the equally weighted mean loss contrast',
          'statistic': 'Mean normalized loss reduction; R2 uses squared error, D2 uses absolute error',
          'p_definition': '(1 + count(abs(permuted statistic) >= abs(observed statistic))) / (1 + resamples)',
          'null_assumption': 'Independent feature groups with sign-exchangeable aggregate paired loss contrasts under the null',
          'scope': 'Exploratory held-out-case inference conditional on the fixed fitted models, not variation from retraining or across datasets',
          'multiplicity_adjustment': None, 'resamples': B, 'batch': BATCH, 'rng_seed': RNG_SEED,
          'experiment_seeds': [42, 43], 'datasets': datasets, 'tests': tests, 'verified_records': verified,
          'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                       'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
          'source_hashes': requests['source_hashes'], 'requests_sha256': sha(HERE / 'requests.json'),
          'script_sha256': sha(Path(__file__)), 'group_contrasts_sha256': sha(OUT / 'group_contrasts.csv'),
          'training_or_generation_performed': False, 'original_records_modified': False}
(OUT / 'regression_p_values.json').write_text(json.dumps(result, indent=2)+'\n')
(OUT / 'regression_p_values.csv').write_text(pd.DataFrame(tests).to_csv(index=False))
print('COMPLETE: 112 records verified, 16 raw p-values', flush=True)
