"""Explicit phased MNIST reruns; existing modeling CLI and labeler functions only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
RUNTIME = ROOT / '.cache/mnist_head_fixed_runtime_20261009'
NAMESPACE = 'mnist_head_fixed_20261009'
PLAN_PATH = Path(__file__).with_name('plan.json')
OUTPUT = ROOT / 'output' / NAMESPACE


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def source_digest():
    result = hashlib.sha256()
    for source in sorted((RUNTIME / 'src').rglob('*.py')):
        result.update(str(source.relative_to(RUNTIME)).encode('utf-8'))
        result.update(source.read_bytes())
    return result.hexdigest()


def configure():
    os.environ['CORRECTED_RUN_NAMESPACE'] = NAMESPACE
    sys.path[:0] = [str(RUNTIME / 'src'), str(RUNTIME / 'scripts'),
                    str(RUNTIME / 'src/synthesize_data')]


def prepare(plan):
    assert not RUNTIME.exists(), 'Runtime already staged; inspect it instead of overwriting'
    assert not OUTPUT.exists(), 'Output namespace already exists; explicit recovery required'
    for ds, checksum in plan['model_source_sha256'].items():
        assert digest(ROOT / f'src/modeling/models_folder/model_{ds}.py') == checksum
    RUNTIME.mkdir()
    for source in (ROOT / 'src').rglob('*.py'):
        destination = RUNTIME / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert digest(source) == digest(destination)
    (RUNTIME / 'scripts').mkdir()
    for name in ('run_corrected_matrix.py', 'build_corrected_results.py'):
        shutil.copy2(ROOT / 'scripts' / name, RUNTIME / 'scripts' / name)
    for name in ('data', 'output', 'sdv trained model'):
        (RUNTIME / name).symlink_to(ROOT / name, target_is_directory=True)
    OUTPUT.mkdir()
    for block in plan['blocks']:
        ds, seed = block['dataset'], block['seed']
        destination = ROOT / 'data' / NAMESPACE / ds / f'seed_{seed}'
        destination.mkdir(parents=True)
        for item in block['prepared']:
            source = Path(item['path'])
            assert digest(source) == item['sha256']
            (destination / source.name).symlink_to(source)
        manifest = Path(block['source_data']) / 'split_manifest.json'
        assert digest(manifest) == block['split_manifest_sha256']
        (destination / manifest.name).symlink_to(manifest)
        models = ROOT / 'sdv trained model' / NAMESPACE / ds / f'seed_{seed}'
        models.mkdir(parents=True)
        for item in block['generators']:
            source = Path(item['path'])
            assert digest(source) == item['sha256']
            provenance = source.with_suffix('.provenance.json')
            assert digest(provenance) == item['provenance_sha256']
            (models / source.name).symlink_to(source)
            (models / provenance.name).symlink_to(provenance)
        for item in block['tables']:
            if item['action'] != 'reuse':
                continue
            source = Path(item['path'])
            assert digest(source) == item['sha256']
            quality = source.with_suffix('.quality.json')
            assert digest(quality) == item['quality_sha256']
            (destination / source.name).symlink_to(source)
            (destination / quality.name).symlink_to(quality)
            if item['labeler'] != 'generated':
                predictor = Path(item['predictor_path'])
                assert digest(predictor) == item['predictor_sha256']
                (destination / predictor.name).symlink_to(predictor)
            if item['labeler'] == 'dnn':
                report = source.with_suffix('.dnn.json')
                assert report.is_file()
                (destination / report.name).symlink_to(report)
    (OUTPUT / 'source_snapshot.json').write_text(json.dumps({
        'source_sha256': source_digest(), 'runtime': str(RUNTIME),
        'model_source_sha256': plan['model_source_sha256'],
        'plan_sha256': digest(PLAN_PATH)}, indent=2) + '\n')
    print('STAGED; no training launched', flush=True)


def verify(job):
    import torch
    path = OUTPUT / job['dataset'] / 'acc' / (job['run_id'] + '.run.json')
    record = json.loads(path.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == (
        job['dataset'], job['seed'], job['train_option'], job['augment_option'])
    assert (record['batch_size'], record['learning_rate'], record['epoch_budget'], record['selection_metric']) == (128, .001, 100, 'loss')
    assert record['selected_dev_epoch'] is not None
    snapshot = json.loads((OUTPUT / 'source_snapshot.json').read_text())
    assert record['source_sha256'] == snapshot['source_sha256'] == source_digest()
    state = torch.load(record['downstream_weight_path'], map_location='cpu', weights_only=True)
    assert state['output.weight'].shape[0] == 10 and state['output.bias'].shape[0] == 10
    with Path(record['predictions_path']).open(newline='') as source:
        predictions = list(csv.DictReader(source))
    assert [int(r['source_id']) for r in predictions] == record['split_manifest']['splits']['test']
    assert all(0 <= int(r['y_pred']) <= 9 for r in predictions)
    return record


def require_primary(plan):
    for job in plan['jobs']:
        if job['phase'] == 'primary':
            verify(job)


def labels(plan):
    require_primary(plan)
    import numpy as np
    import pandas as pd
    import torch
    from synthesizer import synthesize_comparison_from_trained_model
    for block in plan['blocks']:
        for item in block['tables']:
            if item['action'] == 'reuse':
                continue
            destination = ROOT / 'data' / NAMESPACE / block['dataset'] / f"seed_{block['seed']}"
            output = destination / Path(item['path']).name
            assert not output.exists(), 'Label artifact already present; inspect before recovery'
            train = pd.read_csv(destination / f"{block['dataset']}_seed{block['seed']}_real_train_raw.csv")
            source = next(t for t in block['tables'] if t['generator'] == item['generator'] and
                          t['fit'] == item['fit'] and t['labeler'] == ('generated' if item['fit'] == 'full' else 'rf'))
            assert digest(source['path']) == source['sha256']
            features = pd.read_csv(source['path']).drop(columns='label')
            seed = block['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            labeler = {'gaussian': 'gaussianNB', 'categorical': 'categoricalNB', 'pca_gmm': 'pca_gmm'}[item['labeler']]
            result = synthesize_comparison_from_trained_model(
                train.drop(columns='label'), train['label'], [], 'label', sample_size=100000,
                numerical_columns_pca_gmm=[], target_synthesizer=labeler,
                csv_file_name=str(output), is_classification=True, seed=seed,
                dataset_name=block['dataset'], synthetic_features=features)
            quality = json.loads(output.with_suffix('.quality.json').read_text())
            assert quality['rows'] == 100000
            assert set(result.columns) == set(features.columns) | {'label'}
            assert np.array_equal(result[features.columns].to_numpy(), features.to_numpy())
            assert result['label'].isin(range(10)).all()
            with output.with_suffix('.predictor.pkl').open('rb') as source:
                predictor = pickle.load(source)
            if item['labeler'] == 'categorical':
                for index, column in enumerate(predictor['feature_columns']):
                    if set(train[column].unique()) == {0, 1}:
                        assert np.array_equal(predictor['quantile_bin_edges'][index], [.5])
                        assert (predictor['estimator'].category_count_[index][:, :2].sum(axis=0) > 0).all()
            del result, features, train, predictor
            print('LABEL COMPLETE', output.name, flush=True)


def train_lane(plan, phase, lane):
    import fcntl
    from run_corrected_matrix import classifier_command
    if phase == 'supplementary':
        require_primary(plan)
        for block in plan['blocks']:
            for item in block['tables']:
                if item['action'] != 'reuse':
                    assert (ROOT / 'data' / NAMESPACE / block['dataset'] / f"seed_{block['seed']}" / Path(item['path']).name).is_file()
    logs = OUTPUT / 'logs' / phase
    logs.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / f'{phase}_{lane}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in plan['jobs']:
            if job['phase'] != phase or job['lane'] != lane:
                continue
            record = OUTPUT / job['dataset'] / 'acc' / (job['run_id'] + '.run.json')
            if record.exists():
                verify(job)
                print('PRESERVED COMPLETE', job['run_id'], flush=True)
                continue
            with (logs / (job['run_id'] + '.log')).open('x') as log:
                print('START', job['run_id'], flush=True)
                subprocess.run(classifier_command(job['dataset'], job['seed'], job['train_option'], job['augment_option']),
                               cwd=RUNTIME / 'src', stdout=log, stderr=subprocess.STDOUT, check=True)
            verify(job)
            print('COMPLETE', job['run_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'labels', 'train', 'verify'))
    parser.add_argument('--phase', choices=('primary', 'supplementary'))
    parser.add_argument('--lane', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if sys.platform != 'linux' or not ROOT.is_dir():
        parser.error('Run only on the research server')
    if args.action == 'train' and (args.phase is None or args.lane is None):
        parser.error('Training requires --phase and --lane')
    plan = json.loads(PLAN_PATH.read_text())
    if args.action == 'prepare':
        prepare(plan)
    else:
        configure()
        if args.action == 'labels':
            labels(plan)
        elif args.action == 'train':
            train_lane(plan, args.phase, args.lane)
        else:
            for job in plan['jobs']:
                if args.phase is None or job['phase'] == args.phase:
                    verify(job)
            print('VERIFIED', args.phase or 'all', flush=True)
