"""Isolated remote News target-transform pilot; never train on the laptop."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = {43: 'corrected_v2'}
TRANSFORMS = ('raw', 'log', 'yeo_johnson')
TABLES = (('original', None), ('synthetic', 'ctgan'), ('synthetic', 'tvae'))


def timestamp():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def configure(seed):
    os.environ['CORRECTED_RUN_NAMESPACE'] = NAMESPACES[seed]
    sys.path.insert(0, str(ROOT / 'src'))


def finite(values, context):
    import numpy as np
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values in {context}; no clipping or substitution is applied')
    return values


def target_functions(name, real_targets):
    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    real_targets = np.asarray(real_targets, dtype=np.float64).reshape(-1, 1)
    if name == 'raw':
        return lambda y: np.asarray(y, dtype=np.float64), lambda y: np.asarray(y, dtype=np.float64), None
    if name == 'log':
        def forward(y):
            y = np.asarray(y, dtype=np.float64)
            if np.any(y <= 0):
                raise ValueError('Log pilot requires strictly positive targets')
            return finite(np.log(y), 'log targets')
        return forward, lambda z: finite(np.exp(z), 'inverse-log predictions'), None
    transform = PowerTransformer(method='yeo-johnson', standardize=False).fit(real_targets)
    def forward(y):
        return finite(transform.transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)).ravel(), 'Yeo-Johnson targets')
    def inverse(z):
        return finite(transform.inverse_transform(np.asarray(z, dtype=np.float64).reshape(-1, 1)).ravel(), 'inverse Yeo-Johnson predictions')
    return forward, inverse, float(transform.lambdas_[0])


def preflight(output):
    import numpy as np
    import pandas as pd
    checks = {'checked_at_chicago': timestamp(), 'seeds': {}, 'pilot_script_sha256': digest(__file__)}
    for seed, namespace in NAMESPACES.items():
        prepared = ROOT / 'data' / namespace / 'news' / f'seed_{seed}'
        manifest_path = prepared / 'split_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['seed'] != seed or manifest['dataset'] != 'news':
            raise ValueError('Incorrect split manifest')
        all_ids = sum((manifest['splits'][split] for split in ('train', 'dev', 'test')), [])
        if len(all_ids) != len(set(all_ids)):
            raise ValueError('Overlapping source IDs')
        sources = {}
        for split in ('train', 'dev', 'test'):
            for view in ('raw', 'onehot'):
                path = prepared / f'news_seed{seed}_real_{split}_{view}.csv'
                sha = digest(path)
                if sha != manifest['files'][split][view]['sha256']:
                    raise ValueError(f'Prepared table hash mismatch: {path}')
                sources[f'{split}_{view}'] = {'path': str(path), 'sha256': sha}
        real_targets = pd.read_csv(sources['train_onehot']['path'], usecols=[' shares'])[' shares'].to_numpy()
        dev_targets = pd.read_csv(sources['dev_onehot']['path'], usecols=[' shares'])[' shares'].to_numpy()
        tables = {'real': real_targets}
        for generator in ('ctgan', 'tvae'):
            path = prepared / f'news_seed{seed}_{generator}_full_generated_100k.csv'
            quality = json.loads(path.with_suffix('.quality.json').read_text())
            model = ROOT / 'sdv trained model' / namespace / 'news' / f'seed_{seed}' / f'news_seed{seed}_{generator}_full.pkl'
            provenance_path = model.with_suffix('.provenance.json')
            provenance = json.loads(provenance_path.read_text())
            if not model.is_file():
                raise FileNotFoundError(model)
            if (provenance['seed'] != seed or provenance['sample_size'] != 100000 or
                    provenance['fit_table_sha256'] != sources['train_raw']['sha256']):
                raise ValueError(f'Generator provenance mismatch: {model}')
            params = provenance['parameters']
            if (params['epochs'], params['batch_size'], params['cuda']) != (500, 500, True):
                raise ValueError(f'Non-production generator settings: {model}')
            targets = pd.read_csv(path, usecols=[' shares'])[' shares'].to_numpy()
            if len(targets) != 100000 or quality['rows'] != 100000:
                raise ValueError(f'Incorrect synthetic row count: {path}')
            finite(targets, str(path))
            tables[generator] = targets
            sources[generator] = {'path': str(path), 'sha256': digest(path),
                                  'generator_sha256': digest(model),
                                  'provenance_sha256': digest(provenance_path)}
        lambdas = {}
        for name in TRANSFORMS:
            forward, inverse, fitted_lambda = target_functions(name, real_targets)
            for label, targets in {**tables, 'dev': dev_targets}.items():
                restored = inverse(forward(targets))
                if not np.allclose(targets, restored, rtol=1e-9, atol=1e-7):
                    raise ValueError(f'{name} round-trip mismatch for {label}')
            lambdas[name] = fitted_lambda
        checks['seeds'][str(seed)] = {'namespace': namespace, 'sources': sources,
                                    'manifest_sha256': digest(manifest_path), 'lambdas': lambdas,
                                    'rows': {split: len(manifest['splits'][split]) for split in ('train', 'dev', 'test')}}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'preflight.json', checks)
    print(json.dumps({'preflight': 'passed', 'seeds': {k: {'rows': v['rows'], 'lambdas': v['lambdas']} for k, v in checks['seeds'].items()}}), flush=True)


def run_one(args):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    configure(args.seed)
    from modeling import constants
    from modeling.data_loader import data_loader
    from modeling.run_record import write_run_record
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    method = None if args.table == 'real' else args.table
    run_id = constants.run_name('news', args.seed, args.mode, method)
    root = args.output / args.transform / 'news'
    acc, weights = root / 'acc', root / 'model'
    acc.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    record_path = acc / f'{run_id}.run.json'
    weight_path = weights / f'{run_id}.weights.pth'
    if record_path.exists() or weight_path.exists():
        raise FileExistsError(f'Pilot output already exists: {run_id}')
    loader = data_loader('news', 128, multi_y=False, problem_type='regression', seed=args.seed)
    train_data, dev_data = loader.load_train_augment_data(args.mode, method)
    expected = json.loads((args.output / 'preflight.json').read_text())['seeds'][str(args.seed)]
    for key in ('train_onehot', 'dev_onehot'):
        if digest(expected['sources'][key]['path']) != expected['sources'][key]['sha256']:
            raise ValueError(f'Input changed since preflight: {key}')
    if method and digest(loader.synthetic_path) != expected['sources'][method]['sha256']:
        raise ValueError('Synthetic input changed since preflight')
    real_y = pd.read_csv(loader.paths['train_original'], usecols=[' shares'])[' shares'].to_numpy()
    forward, inverse, fitted_lambda = target_functions(args.transform, real_y)
    if fitted_lambda != expected['lambdas'][args.transform]:
        raise ValueError('Target-transform parameter changed since preflight')
    x, y = train_data.dataset.tensors
    transformed_y = torch.tensor(forward(y.numpy()), dtype=torch.float32)
    # Preserve the corrected loader's training order to isolate target treatment.
    train_data = DataLoader(TensorDataset(x, transformed_y), batch_size=128, shuffle=False)
    blocks = []
    width = x.shape[1]
    for hidden in (512, 512, 256, 128):
        blocks.extend((nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.4)))
        width = hidden
    model = nn.Sequential(*blocks, nn.Linear(width, 1))
    if any(isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)) for layer in model.modules()):
        raise ValueError('Pilot model unexpectedly contains normalization')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    def predict(data):
        model.eval()
        with torch.no_grad():
            z = np.concatenate([model(batch_x).squeeze(-1).numpy() for batch_x, _ in data])
        finite(z, 'network outputs')
        predictions = inverse(z.astype(np.float64))
        targets = data.dataset.tensors[1].numpy().astype(np.float64)
        scores = {'mse': float(mean_squared_error(targets, predictions)),
                  'mae': float(mean_absolute_error(targets, predictions)),
                  'r2': float(r2_score(targets, predictions))}
        finite(list(scores.values()), 'raw-scale metrics')
        return predictions, targets, scores
    started = timestamp()
    best, selected_epoch, stale = float('inf'), None, 0
    epochs_path = acc / f'{run_id}.epochs.csv'
    with epochs_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_objective_mse', 'dev_mse', 'dev_mae', 'dev_r2', 'seconds'])
        writer.writeheader()
        for epoch in range(1, 101):
            tick = time.monotonic()
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_data:
                optimizer.zero_grad()
                loss = criterion(model(batch_x).squeeze(-1), batch_y)
                if not torch.isfinite(loss):
                    raise ValueError(f'Non-finite training loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_y)
            _, _, scores = predict(dev_data)
            row = {'epoch': epoch, 'train_objective_mse': total_loss / len(y),
                   'dev_mse': scores['mse'], 'dev_mae': scores['mae'], 'dev_r2': scores['r2'],
                   'seconds': time.monotonic() - tick}
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if best > scores['mse'] + 1e-5:
                best, selected_epoch, stale = scores['mse'], epoch, 0
                torch.save(model.state_dict(), weight_path)
            else:
                stale += 1
                if stale > 30:
                    break
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    _, _, dev_scores = predict(dev_data)
    # Test is evaluated only once, after development checkpoint selection.
    for view in ('raw', 'onehot'):
        source = expected['sources']['test_' + view]
        if digest(source['path']) != source['sha256']:
            raise ValueError('Held-out input changed since preflight')
    test_data = loader.load_test_data()
    predictions, targets, scores = predict(test_data)
    predictions_path = acc / f'{run_id}.predictions.csv'
    pd.DataFrame({'source_id': loader.test_source_ids, 'y_true': targets, 'y_pred': predictions}).to_csv(predictions_path, index=False)
    write_run_record(str(record_path), dataset='news', seed=args.seed, train_option=args.mode,
                     augment_option=method, synthetic_path=loader.synthetic_path,
                     synthetic_label_counts=loader.synthetic_label_counts,
                     split_manifest_path=loader.manifest_path, classifier='DNN_News_no_norm',
                     batch_size=128, learning_rate=0.001, epoch_budget=100,
                     selected_epoch=selected_epoch, selection_metric='mse', test_loss=scores['mse'],
                     test_scores=scores, predictions_path=str(predictions_path), weight_path=str(weight_path))
    record = json.loads(record_path.read_text())
    record.update({'experiment_namespace': args.output.name, 'started_at_chicago': started,
                   'completed_at_chicago': timestamp(), 'device': 'cpu', 'cpu_threads': 2,
                   'epochs_run': epoch, 'patience': 30, 'training_shuffle': False,
                   'hidden_sizes': [512, 512, 256, 128], 'dropout': 0.4,
                   'normalization': 'none', 'dev_scores': dev_scores,
                   'target_transform': {'name': args.transform, 'lambda': fitted_lambda,
                                        'standardize': False, 'fit_split': 'real_train' if fitted_lambda is not None else None,
                                        'fit_table_sha256': expected['sources']['train_onehot']['sha256'],
                                        'selection': 'raw-scale development MSE', 'metric_units': 'shares'},
                   'pilot_script_sha256': digest(__file__), 'preflight_sha256': digest(args.output / 'preflight.json'),
                   'input_namespace': NAMESPACES[args.seed], 'input_provenance': expected,
                   'scipy_version': version('scipy')})
    save_json(record_path, record)
    print('COMPLETE', run_id, args.transform, json.dumps(record['test_scores']), flush=True)


def worker(output, transform):
    manifest = {'started_at_chicago': timestamp(), 'status': 'running', 'planned_runs': 3,
                'target_transform': transform,
                'session': output.name, 'tasks': []}
    manifest_path = output / 'pilot_status.json'
    save_json(manifest_path, manifest)
    for mode, method in TABLES:
        for seed in NAMESPACES:
            for transform in (transform,):
                label = method or 'real'
                command = [sys.executable, str(Path(__file__).resolve()), 'run', '--output', str(output),
                           '--seed', str(seed), '--mode', mode, '--table', label, '--transform', transform]
                log_path = output / 'logs' / f'seed{seed}_{label}_{mode}_{transform}.log'
                log_path.parent.mkdir(exist_ok=True)
                task = {'seed': seed, 'table': label, 'mode': mode, 'transform': transform,
                        'command': command, 'log_path': str(log_path), 'status': 'running', 'started_at_chicago': timestamp()}
                manifest['tasks'].append(task)
                save_json(manifest_path, manifest)
                print('START', json.dumps(task), flush=True)
                with log_path.open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                task.update({'status': 'complete' if result.returncode == 0 else 'failed',
                             'exit_code': result.returncode, 'finished_at_chicago': timestamp()})
                save_json(manifest_path, manifest)
                print(task['status'].upper(), str(log_path), flush=True)
    # Aggregate each transformation separately to avoid duplicate configuration keys.
    for transform in (transform,):
        run_root = output / transform
        if list(run_root.rglob('*.run.json')):
            subprocess.run([sys.executable, str(ROOT / 'scripts/build_corrected_results.py'),
                            '--runs', str(run_root), '--out', str(run_root / 'results'),
                            '--generators', 'ctgan', 'tvae'], cwd=ROOT, check=True)
    records = [json.loads(path.read_text()) for path in output.rglob('*.run.json')]
    with (output / 'pilot_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seed', 'table', 'mode', 'transform', 'selected_epoch', 'dev_r2', 'test_r2', 'test_mae', 'test_nmae_sigma'])
        writer.writeheader()
        for record in records:
            writer.writerow({'seed': record['seed'], 'table': record['augment_option'] or 'real',
                             'mode': record['train_option'], 'transform': record['target_transform']['name'],
                             'selected_epoch': record['selected_dev_epoch'], 'dev_r2': record['dev_scores']['r2'],
                             'test_r2': record['test_scores']['r2'], 'test_mae': record['test_scores']['mae'],
                             'test_nmae_sigma': record['test_scores']['nmae_sigma']})
    manifest.update({'status': 'completed_with_failures' if any(t['status'] == 'failed' for t in manifest['tasks']) else 'complete',
                     'completed_records': len(records), 'finished_at_chicago': timestamp()})
    save_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)
    if manifest['status'] != 'complete':
        raise SystemExit('Pilot finished with recorded failures; inspect task logs')


if __name__ == '__main__':
    if os.name == 'nt':
        raise SystemExit('This pilot must run in the remote experiment environment')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'worker', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(43,))
    parser.add_argument('--table', choices=('real', 'ctgan', 'tvae'))
    parser.add_argument('--mode', choices=('original', 'synthetic', 'mix'))
    parser.add_argument('--transform', choices=TRANSFORMS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == 'preflight':
        if (args.output / 'preflight.json').exists():
            raise FileExistsError('Pilot preflight already exists')
        preflight(args.output)
    elif args.action == 'worker':
        if (args.output / 'pilot_status.json').exists():
            raise FileExistsError('Pilot worker already launched')
        if args.transform not in ('log', 'yeo_johnson'):
            raise ValueError('Choose one target transform for the three-run pilot')
        worker(args.output, args.transform)
    else:
        run_one(args)
