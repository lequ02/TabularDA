"""News log-target matrix, using frozen splits and existing X-only samples.

Run remotely. The default preflight action does not train any models.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {42: 'corrected_v2_seed42_mnist28_news', 43: 'corrected_v2'}
TARGET = ' shares'
ROWS = 100_000
GENERATORS = ('ctgan', 'tvae')
LABELERS = ('pca_gmm', 'rf', 'xgb', 'dnn')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and digest(destination) != digest(source):
        raise ValueError(f'Refusing to replace a different artifact: {destination}')
    if not destination.exists():
        shutil.copy2(source, destination)
    if digest(destination) != digest(source):
        raise ValueError(f'Artifact copy failed verification: {destination}')


def preflight(namespace):
    import numpy as np
    import pandas as pd
    import torch
    from importlib.metadata import version
    from commons.log_target import log_target
    from modeling import constants

    report = {'namespace': namespace, 'target_transform': 'log', 'planned_runs': 74,
              'versions': {p: version(p) for p in ('sdv', 'ctgan', 'rdt', 'torch', 'numpy', 'pandas', 'scikit-learn', 'xgboost')},
              'cuda_available': torch.cuda.is_available(), 'seeds': {}}
    for seed, source_namespace in SOURCES.items():
        source = ROOT / 'data' / source_namespace / 'news' / f'seed_{seed}'
        manifest = json.loads((source / 'split_manifest.json').read_text())
        if (manifest['dataset'], manifest['seed']) != ('news', seed):
            raise ValueError('Wrong source split manifest')
        ids = sum((manifest['splits'][s] for s in ('train', 'dev', 'test')), [])
        if len(ids) != len(set(ids)):
            raise ValueError('Source partitions overlap')
        for split in ('train', 'dev', 'test'):
            for view in ('raw', 'onehot'):
                path = source / constants.split_name('news', seed, split, view)
                if digest(path) != manifest['files'][split][view]['sha256']:
                    raise ValueError(f'Source split hash mismatch: {path}')
            frame = pd.read_csv(source / constants.split_name('news', seed, split, 'raw'))
            log_target(frame[TARGET])
            if len(frame) != len(manifest['splits'][split]):
                raise ValueError('Source row counts do not match split manifest')
        train_x = pd.read_csv(source / constants.split_name('news', seed, 'train', 'raw')).drop(columns=TARGET)
        expected_fit_hash = hashlib.sha256(train_x.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
        reused = {}
        for generator in GENERATORS:
            model = ROOT / 'sdv trained model' / source_namespace / 'news' / f'seed_{seed}' / constants.generator_name('news', seed, generator, 'xonly')
            provenance = json.loads(model.with_suffix('.provenance.json').read_text())
            parameters = provenance['parameters']
            if (provenance['seed'] != seed or provenance['fit_table_sha256'] != expected_fit_hash or
                    provenance['sample_size'] != ROWS or
                    (parameters['epochs'], parameters['batch_size'], parameters['cuda']) != (500, 500, True)):
                raise ValueError(f'X-only generator does not match the frozen design: {model}')
            # The RF target is discarded; its saved X-only features are reused.
            method = 'rf' if generator == 'ctgan' else 'tvae_rf'
            table = source / constants.synthetic_name('news', seed, method)
            quality = json.loads(table.with_suffix('.quality.json').read_text())
            features = pd.read_csv(table).drop(columns=TARGET)
            if quality['rows'] != ROWS or len(features) != ROWS or set(features) != set(train_x) or not np.isfinite(features.to_numpy()).all():
                raise ValueError(f'Invalid saved X-only features: {table}')
            reused[generator] = {str(p): digest(p) for p in (model, model.with_suffix('.provenance.json'), table, table.with_suffix('.quality.json'))}
        report['seeds'][str(seed)] = {'input_namespace': source_namespace,
            'split_manifest_sha256': digest(source / 'split_manifest.json'), 'reused_artifacts': reused}
    return report


def prepare(namespace, report):
    from modeling import constants
    for seed, source_namespace in SOURCES.items():
        source = ROOT / 'data' / source_namespace / 'news' / f'seed_{seed}'
        destination = ROOT / 'data' / namespace / 'news' / f'seed_{seed}'
        for name in ['split_manifest.json'] + [constants.split_name('news', seed, s, v)
                for s in ('train', 'dev', 'test') for v in ('raw', 'onehot')]:
            copy_verified(source / name, destination / name)
        for generator in GENERATORS:
            name = constants.generator_name('news', seed, generator, 'xonly')
            model_root = ROOT / 'sdv trained model'
            for filename in (name, Path(name).with_suffix('.provenance.json').name):
                copy_verified(model_root / source_namespace / 'news' / f'seed_{seed}' / filename,
                              model_root / namespace / 'news' / f'seed_{seed}' / filename)
    output = ROOT / 'output' / namespace
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'preflight.json'
    if path.exists() and json.loads(path.read_text())['seeds'] != report['seeds']:
        raise ValueError('Frozen source artifacts changed since preflight')
    save_json(path, report)


def generate(namespace):
    import numpy as np
    import pandas as pd
    import torch
    from commons.log_target import log_target, inverse_log_target
    from modeling import constants
    from create_synthetic_data.news import CreateSyntheticDataNews
    from dnn_labeler import fit_predict_dnn
    from ensemble import Ensemble
    from pca_gmm import PCA_GMM
    from synthesizer import (train_synthesizer_ctgan, train_tvae_synthesizer,
                            load_synthesizer, _save_synthesis_provenance, _save_synthetic_quality)

    if not torch.cuda.is_available():
        raise RuntimeError('Production News generator fits require CUDA')
    for seed, source_namespace in SOURCES.items():
        data = ROOT / 'data' / namespace / 'news' / f'seed_{seed}'
        models = ROOT / 'sdv trained model' / namespace / 'news' / f'seed_{seed}'
        train = pd.read_csv(data / constants.split_name('news', seed, 'train', 'raw'))
        dev = pd.read_csv(data / constants.split_name('news', seed, 'dev', 'onehot'))
        columns = sorted(c for c in train if c != TARGET)
        x, y = train[columns], train[TARGET]
        log_train = train.copy()
        log_train[TARGET] = log_target(y)
        groups = []
        feature_sources = {}
        for generator in GENERATORS:
            model_path = models / constants.generator_name('news', seed, generator, 'full')
            table_path = data / constants.synthetic_name('news', seed, generator)
            if model_path.exists():
                provenance = json.loads(model_path.with_suffix('.provenance.json').read_text())
                parameters = provenance['parameters']
                if (provenance.get('target_transform') != 'log' or provenance['sample_size'] != ROWS or
                        (parameters['epochs'], parameters['batch_size'], parameters['cuda']) != (500, 500, True)):
                    raise ValueError('Full-table checkpoint was not fitted on log targets')
                model = load_synthesizer(str(model_path), expected_data=log_train, expected_seed=seed)
            else:
                fit = train_synthesizer_ctgan if generator == 'ctgan' else train_tvae_synthesizer
                model = fit(log_train, categorical_columns=[], seed=seed)
                model.save(str(model_path))
                _save_synthesis_provenance(model, log_train, ROWS, str(model_path), seed)
                provenance_path = model_path.with_suffix('.provenance.json')
                provenance = json.loads(provenance_path.read_text())
                provenance.update({'target_transform': 'log', 'target_inverse': 'exp',
                                   'real_train_sha256': digest(data / constants.split_name('news', seed, 'train', 'raw'))})
                save_json(provenance_path, provenance)
            if table_path.exists():
                generated = pd.read_csv(table_path)
                quality = json.loads(table_path.with_suffix('.quality.json').read_text())
                if quality.get('table_sha256') != digest(table_path) or quality.get('generator_sha256') != digest(model_path):
                    raise ValueError('Generated table does not match its saved checkpoint')
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                generated = model.sample(num_rows=ROWS)
                generated[TARGET] = inverse_log_target(generated[TARGET])
                generated = generated[columns + [TARGET]]
                generated.to_csv(table_path, index=False)
                _save_synthetic_quality(generated, train, y, TARGET, ROWS, str(table_path), False)
                quality_path = table_path.with_suffix('.quality.json')
                quality = json.loads(quality_path.read_text())
                quality.update({'target_transform': 'log', 'table_sha256': digest(table_path), 'generator_sha256': digest(model_path)})
                save_json(quality_path, quality)
            if len(generated) != ROWS or list(generated.columns) != columns + [TARGET]:
                raise ValueError('Full-table sample has an invalid schema or row count')
            log_target(generated[TARGET])
            feature_sources[f'{generator}_full'] = digest(table_path)
            groups.append((generator, 'full', generated[columns]))
            source = ROOT / 'data' / source_namespace / 'news' / f'seed_{seed}'
            method = 'rf' if generator == 'ctgan' else 'tvae_rf'
            source_path = source / constants.synthetic_name('news', seed, method)
            feature_sources[f'{generator}_xonly'] = digest(source_path)
            groups.append((generator, 'xonly', pd.read_csv(source_path)[columns]))
            del model
        combined_x = pd.concat([features for _, _, features in groups], ignore_index=True)
        cache = ROOT / '.cache' / namespace / f'seed_{seed}'
        cache.mkdir(parents=True, exist_ok=True)
        numerical = CreateSyntheticDataNews(seed=seed).numerical_cols_pca_gmm
        for labeler in LABELERS:
            combined_path = cache / f'{labeler}.csv'
            artifact = combined_path.with_suffix('.predictor.pt' if labeler == 'dnn' else '.predictor.pkl')
            marker = combined_path.with_suffix('.complete.json')
            recipe = {'seed': seed, 'target_transform': 'log', 'feature_sources': feature_sources,
                      'real_train_sha256': digest(data / constants.split_name('news', seed, 'train', 'raw')),
                      'real_dev_sha256': digest(data / constants.split_name('news', seed, 'dev', 'onehot')),
                      'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
                        ROOT / 'src/commons/log_target.py', ROOT / 'src/synthesize_data' / ('dnn_labeler.py' if labeler == 'dnn' else 'ensemble.py' if labeler in ('rf', 'xgb') else 'pca_gmm.py')]}}
            if marker.exists():
                completed = json.loads(marker.read_text())
                if completed['recipe'] != recipe or completed['table_sha256'] != digest(combined_path) or completed['predictor_sha256'] != digest(artifact):
                    raise ValueError('Completed labeler cache no longer matches its inputs')
                labeled = pd.read_csv(combined_path)
            else:
                if artifact.exists() or combined_path.exists():
                    raise FileExistsError(f'Incomplete labeler artifacts require inspection: {cache / labeler}')
                np.random.seed(seed)
                torch.manual_seed(seed)
                if labeler == 'dnn':
                    labeled = fit_predict_dnn(x, y, dev[columns], dev[TARGET], combined_x,
                        target_name=TARGET, is_classification=False, seed=seed,
                        report_path=combined_path.with_suffix('.dnn.json'), dataset_name='news',
                        artifact_path=artifact, target_transform='log')
                    labeled.to_csv(combined_path, index=False)
                elif labeler in ('rf', 'xgb'):
                    _, labeled = Ensemble(x, y, combined_x, TARGET, labeler, str(combined_path),
                        is_classification=False, artifact_path=artifact, target_transform='log').fit()
                else:
                    _, labeled = PCA_GMM(x.copy(), y, combined_x.copy(), numerical, TARGET,
                        filename=str(combined_path), is_classification=False,
                        artifact_path=artifact, target_transform='log').fit()
                log_target(labeled[TARGET])
                save_json(marker, {'recipe': recipe, 'table_sha256': digest(combined_path), 'predictor_sha256': digest(artifact)})
            if len(labeled) != ROWS * len(groups):
                raise ValueError('Combined labeler predictions have the wrong row count')
            log_target(labeled[TARGET])
            for index, (generator, fit, features) in enumerate(groups):
                method = ('' if generator == 'ctgan' else 'tvae_') + ('compare_' if fit == 'full' else '') + labeler
                table_path = data / constants.synthetic_name('news', seed, method)
                table = labeled.iloc[index * ROWS:(index + 1) * ROWS].reset_index(drop=True)
                if not np.allclose(table[columns].to_numpy(), features.to_numpy(), rtol=1e-12, atol=1e-12):
                    raise ValueError('Labeler changed or rearranged synthetic features')
                if table_path.exists():
                    if not pd.read_csv(table_path).equals(table):
                        raise ValueError(f'Refusing to replace a different synthetic table: {table_path}')
                else:
                    table.to_csv(table_path, index=False)
                copy_verified(artifact, table_path.with_suffix('.predictor.pt' if labeler == 'dnn' else '.predictor.pkl'))
                if labeler == 'dnn':
                    copy_verified(combined_path.with_suffix('.dnn.json'), table_path.with_suffix('.dnn.json'))
                _save_synthetic_quality(table, train, y, TARGET, ROWS, str(table_path), False)
                quality_path = table_path.with_suffix('.quality.json')
                quality = json.loads(quality_path.read_text())
                quality.update({'target_transform': 'log', 'table_sha256': digest(table_path),
                    'predictor_sha256': digest(artifact), 'labeler_provenance': json.loads(marker.read_text())})
                save_json(quality_path, quality)


def downstream(namespace):
    from modeling import constants
    from run_corrected_matrix import classifier_command, completed_run, methods_for
    output = ROOT / 'output' / namespace
    # Verify all generated inputs before starting even the real-only baseline.
    for seed in SOURCES:
        for generator in GENERATORS:
            model = ROOT / 'sdv trained model' / namespace / 'news' / f'seed_{seed}' / constants.generator_name('news', seed, generator, 'full')
            baseline = ROOT / 'data' / namespace / 'news' / f'seed_{seed}' / constants.synthetic_name('news', seed, generator)
            if json.loads(baseline.with_suffix('.quality.json').read_text())['generator_sha256'] != digest(model):
                raise ValueError('Full-table checkpoint changed since sampling')
            for method in methods_for('news', generator):
                path = ROOT / 'data' / namespace / 'news' / f'seed_{seed}' / constants.synthetic_name('news', seed, method)
                quality = json.loads(path.with_suffix('.quality.json').read_text())
                if quality.get('target_transform') != 'log' or quality['rows'] != ROWS or quality['table_sha256'] != digest(path):
                    raise ValueError(f'Invalid log-experiment table: {path}')
                if method not in GENERATORS:
                    artifact = path.with_suffix('.predictor.pt' if method.endswith('dnn') else '.predictor.pkl')
                    if quality['predictor_sha256'] != digest(artifact):
                        raise ValueError('Labeler checkpoint changed since generation')
    for seed in SOURCES:
        tasks = [('original', None)] + [(mode, method) for generator in GENERATORS
                 for method in methods_for('news', generator) for mode in ('synthetic', 'mix')]
        for mode, method in tasks:
            record_path = output / 'news' / 'acc' / (constants.run_name('news', seed, mode, method) + '.run.json')
            if completed_run(output, 'news', seed, mode, method):
                record = json.loads(record_path.read_text())
                if record.get('target_transform', {}).get('name') != 'log' or record['classifier'] != 'DNN_News_log_no_norm':
                    raise ValueError('Completed downstream record has the wrong protocol')
                if 'target_normalization' not in record or not {'mae', 'r2', 'nmae_sigma'} <= record['test_scores'].keys():
                    raise ValueError('Completed News record is missing required metrics or target normalization')
                if method and record['synthetic_quality']['table_sha256'] != digest(record['synthetic_path']):
                    raise ValueError('Completed downstream input table changed')
                continue
            command = classifier_command('news', seed, mode, method) + ['--target-transform', 'log']
            log = output / 'logs' / f'seed_{seed}' / f'{constants.run_name("news", seed, mode, method)}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as handle:
                subprocess.run(command, cwd=ROOT / 'src', stdout=handle, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_corrected_results.py'),
                    '--runs', str(output / 'news'), '--out', str(output / 'results'), '--matrix', 'news-log'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='news_log_v1')
    parser.add_argument('--stage', choices=('preflight', 'generators', 'downstream', 'all'), default='preflight')
    args = parser.parse_args()
    server_root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
    if ROOT not in (server_root, server_root / '.cache/news_log_runtime_20261008'):
        parser.error('Run this experiment only in the research-server repository')
    if (not args.namespace.startswith('news_log_') or 'pilot' in args.namespace.lower() or
            Path(args.namespace).name != args.namespace or not args.namespace.replace('_', '').isalnum()):
        parser.error('Use a separate news_log_ namespace, never a corrected or pilot namespace')
    os.environ['CORRECTED_RUN_NAMESPACE'] = args.namespace
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/synthesize_data')]
    report = preflight(args.namespace)
    if args.stage == 'preflight':
        print(json.dumps(report, indent=2))
        return
    import fcntl
    output = ROOT / 'output' / args.namespace
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.namespace, report)
        if args.stage in ('generators', 'all'):
            generate(args.namespace)
        if args.stage in ('downstream', 'all'):
            downstream(args.namespace)


if __name__ == '__main__':
    main()
