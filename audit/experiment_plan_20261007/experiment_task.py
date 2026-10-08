"""Execute one scoped preparation/generation/relabeling task on the research host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
CACHE = ROOT / '.cache/full_completion_20261007'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/synthesize_data')]
os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
os.environ['MPLBACKEND'] = 'Agg'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def job_for(dataset, seed, generator='ctgan'):
    from create_synthetic_data.mnist12 import CreateSyntheticDataMnist12
    from create_synthetic_data.mnist28 import CreateSyntheticDataMnist28
    from create_synthetic_data.news import CreateSyntheticDataNews
    from create_synthetic_data.census_kdd import CreateSyntheticDataCensusKdd
    from create_synthetic_data.california_housing import CreateSyntheticDataCaliforniaHousing
    factories = {'mnist12': CreateSyntheticDataMnist12, 'mnist28': CreateSyntheticDataMnist28,
                 'news': CreateSyntheticDataNews, 'census_kdd': CreateSyntheticDataCensusKdd,
                 'california_housing': CreateSyntheticDataCaliforniaHousing}
    return factories[dataset](feature_synthesizer=generator.upper(), seed=seed)


def method_for(generator, fit, label):
    if label == 'generated':
        return generator
    return ('tvae_' if generator == 'tvae' else '') + ('compare_' if fit == 'full' else '') + label


def paths_for(job, generator, fit, label=None):
    from modeling import constants
    model = ROOT / 'sdv trained model/corrected_v2' / job.ds_name / f'seed_{job.seed}' / constants.generator_name(job.ds_name, job.seed, generator, fit)
    table = (ROOT / 'data/corrected_v2' / job.ds_name / f'seed_{job.seed}' / constants.synthetic_name(job.ds_name, job.seed, method_for(generator, fit, label))) if label else None
    return model, table


def validate_manifest(job):
    folder = Path(job.paths['data_dir'])
    path = folder / 'split_manifest.json'
    saved = json.loads(path.read_text())
    assert (saved['dataset'], saved['seed']) == (job.ds_name, job.seed)
    all_ids = [i for part in ('train', 'dev', 'test') for i in saved['splits'][part]]
    assert len(all_ids) == len(set(all_ids)), 'Overlapping source partitions'
    for split in ('train', 'dev', 'test'):
        for view, key in (('raw', f'{split}_csv'), ('onehot', f'{split}_csv_onehot')):
            assert digest(folder / job.paths[key]) == saved['files'][split][view]['sha256'], (split, view)
    return saved


def validate_model(job, generator, fit, *, compare_training=True):
    import pandas as pd
    model, _ = paths_for(job, generator, fit)
    assert model.is_file(), model
    saved = json.loads(model.with_suffix('.provenance.json').read_text())
    assert saved['seed'] == job.seed and saved['sample_size'] == 100000
    assert saved['parameters']['epochs'] == 500 and saved['parameters']['batch_size'] == 500 and saved['parameters']['cuda']
    if compare_training:
        real = pd.read_csv(Path(job.paths['data_dir']) / job.paths['train_csv'])
        if fit == 'xonly':
            real = real.drop(columns=job.target_name)
        assert list(real.columns) == saved['training_columns'] and len(real) == saved['training_rows']
        assert hashlib.sha256(real.to_csv(index=False, lineterminator='\n').encode()).hexdigest() == saved['fit_table_sha256']
    return saved


def validate_table(job, generator, fit, label, expected_hashes=None):
    _, table = paths_for(job, generator, fit, label)
    saved = json.loads(table.with_suffix('.quality.json').read_text())
    assert saved['rows'] == 100000 and job.target_name in saved['columns']
    if job.is_classification:
        assert sum(saved['synthetic_label_counts'].values()) == 100000
    dependencies = [table, table.with_suffix('.quality.json')]
    if label != 'generated':
        dependencies.append(table.with_suffix('.predictor.pt' if label == 'dnn' else '.predictor.pkl'))
    if label == 'dnn':
        dependencies.append(table.with_suffix('.dnn.json'))
    for path in dependencies:
        assert path.is_file(), path
        if expected_hashes is not None:
            assert digest(path) == expected_hashes[str(path)], path


def prepare(dataset, seed):
    job = job_for(dataset, seed)
    manifest = Path(job.paths['data_dir']) / 'split_manifest.json'
    if not manifest.exists():
        assert not any(Path(job.paths['data_dir']).glob('*real*.csv')), 'Partial preparation requires explicit recovery'
        job.prepare_train_test()
    validate_manifest(job)
    if dataset == 'census_kdd':
        authority = json.loads((ROOT / 'output/census_kdd_weighted_macro_f1_20261005/manifest.json').read_text())['preflight']['input_sha256']
        for generator in ('ctgan', 'tvae'):
            for fit in ('full', 'xonly'):
                model, _ = paths_for(job, generator, fit)
                validate_model(job, generator, fit)
                for path in (model, model.with_suffix('.provenance.json')):
                    assert digest(path) == authority[str(path)], path
                labels = ('generated', 'rf', 'xgb', 'dnn') if fit == 'full' else ('rf', 'xgb', 'dnn')
                for label in labels:
                    validate_table(job, generator, fit, label, authority)
    elif dataset == 'mnist12':
        for generator in ('ctgan', 'tvae'):
            for fit in ('full', 'xonly'):
                validate_model(job, generator, fit)
                labels = ('generated', 'rf', 'xgb', 'dnn') if fit == 'full' else ('rf', 'xgb', 'dnn')
                for label in labels:
                    if generator == 'tvae' and fit == 'full' and label in ('rf', 'dnn'):
                        continue
                    validate_table(job, generator, fit, label)
    print(f'PREPARED AND VERIFIED {dataset} seed {seed}', flush=True)


def preserve_partial(table):
    # Never overwrite a completed run's inputs. These specific stems are unfinished.
    folder = CACHE / 'preserved_partial' / table.parent.parent.name / table.parent.name
    for artifact in table.parent.glob(table.stem + '.*'):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / artifact.name
        assert not target.exists(), target
        shutil.copy2(artifact, target)
        assert digest(target) == digest(artifact)


def fit_sample(dataset, seed, generator, fit):
    import numpy as np
    import pandas as pd
    import torch
    from commons.onehot import onehot
    from synthesizer import train_synthesizer_ctgan, train_tvae_synthesizer, _save_synthesis_provenance, _save_synthetic_quality
    job = job_for(dataset, seed, generator)
    model_path, full_table = paths_for(job, generator, fit, 'generated' if fit == 'full' else None)
    assert not model_path.exists() and not model_path.with_suffix('.provenance.json').exists(), 'Existing generator requires explicit validated reuse'
    x, y, _, categories = job.read_train_data()
    training = pd.concat([x, y], axis=1) if fit == 'full' else x
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = train_synthesizer_ctgan if generator == 'ctgan' else train_tvae_synthesizer
    target = job.target_name if fit == 'full' and job.is_classification else None
    model = train(training, verbose=False, categorical_columns=categories, target_name=target, seed=seed)
    # Save the completed fit before sampling, then persist its post-sampling state.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    _save_synthesis_provenance(model, training, 100000, str(model_path), seed)
    sample = model.sample(num_rows=100000)
    _, sample = onehot(training, sample, categories, verbose=False)
    assert len(sample) == 100000 and np.isfinite(sample.to_numpy()).all()
    if fit == 'full':
        ordered = sorted(c for c in sample.columns if c != job.target_name) + [job.target_name]
        sample = sample.loc[:, ordered]
        assert not full_table.exists(), full_table
        sample.to_csv(full_table, index=False)
        _save_synthetic_quality(sample, training, y, job.target_name, 100000, str(full_table), job.is_classification)
    else:
        sample = sample.reindex(sorted(sample.columns), axis=1)
        feature_path = CACHE / 'features' / f'{dataset}_seed{seed}_{generator}_xonly.csv'
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        assert not feature_path.exists(), feature_path
        sample.to_csv(feature_path, index=False)
        feature_path.with_suffix('.json').write_text(json.dumps({'rows': len(sample), 'columns': list(sample.columns), 'sha256': digest(feature_path), 'generator': str(model_path), 'seed': seed}, indent=2))
    model.save(str(model_path))
    validate_model(job, generator, fit, compare_training=False)
    print(f'FIT AND SAMPLE COMPLETE {dataset} {seed} {generator} {fit}', flush=True)


def label(dataset, seed, generator, fit, labeler):
    import numpy as np
    import pandas as pd
    import torch
    from synthesizer import synthesize_comparison_from_trained_model
    job = job_for(dataset, seed, generator)
    _, output = paths_for(job, generator, fit, labeler)
    preserve_partial(output)
    x, y, _, categories = job.read_train_data()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kwargs = {}
    if fit == 'full':
        _, baseline = paths_for(job, generator, fit, 'generated')
        kwargs['full_table_csv'] = str(baseline)
    else:
        path = CACHE / 'features' / f'{dataset}_seed{seed}_{generator}_xonly.csv'
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == metadata['sha256']
        kwargs['synthetic_features'] = pd.read_csv(path)
    synthesize_comparison_from_trained_model(x, y, categories, job.target_name,
        sample_size=100000, target_synthesizer=labeler, csv_file_name=str(output),
        is_classification=job.is_classification, seed=seed, dnn_dev_data=job.read_dev_data(),
        dataset_name=dataset, verbose=True, **kwargs)
    validate_table(job, generator, fit, labeler)
    print(f'RELABEL COMPLETE {dataset} {seed} {generator} {fit} {labeler}', flush=True)


def check():
    import torch
    from importlib.metadata import version
    from synthesizer import create_synthesizer_ctgan, create_synthesizer_tvae
    from sdv.metadata import Metadata
    import pandas as pd
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    metadata = Metadata.detect_from_dataframe(pd.DataFrame({'x': [0., 1.], 'y': [0., 1.]}))
    for factory in (create_synthesizer_ctgan, create_synthesizer_tvae):
        params = factory(metadata).get_parameters()
        assert params['epochs'] == 500 and params['batch_size'] == 500 and params['cuda']
    for dataset in ('mnist12', 'mnist28', 'news', 'census_kdd', 'california_housing'):
        job = job_for(dataset, 43)
        assert job.seed == 43 and job.sample_size_to_synthesize == 100000
    print(json.dumps({'cuda': True, 'versions': {name: version(name) for name in ('torch','sdv','ctgan','numpy','pandas','scikit-learn','xgboost')}, 'production_parameters_verified': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'prepare', 'fit', 'label'))
    parser.add_argument('--dataset', choices=('mnist12','mnist28','news','census_kdd','california_housing'))
    parser.add_argument('--seed', type=int, choices=(42,43))
    parser.add_argument('--generator', choices=('ctgan','tvae'))
    parser.add_argument('--fit', choices=('full','xonly'))
    parser.add_argument('--labeler', choices=('rf','xgb','dnn'))
    args = parser.parse_args()
    if args.action == 'check':
        check()
    else:
        import torch
        assert torch.cuda.is_available(), 'Remote CUDA is required'
        torch.set_num_threads(2)
        if args.action == 'prepare':
            prepare(args.dataset, args.seed)
        elif args.action == 'fit':
            fit_sample(args.dataset, args.seed, args.generator, args.fit)
        else:
            label(args.dataset, args.seed, args.generator, args.fit, args.labeler)
