"""Paths and short artifact names for corrected runs and the CTGAN pilot."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAMESPACE = os.environ.get('CORRECTED_RUN_NAMESPACE', 'corrected_v2')


def method_parts(method):
    if method in ('ctgan', 'tvae'):
        return method, 'full', 'generated'
    generator = 'tvae' if method.startswith('tvae_') else 'ctgan'
    label = method.removeprefix('tvae_')
    if label.startswith('compare_'):
        return generator, 'full', label.removeprefix('compare_')
    return generator, 'xonly', label


def split_name(dataset, seed, split, view):
    return f'{dataset}_seed{seed}_real_{split}_{view}.csv'


def generator_name(dataset, seed, generator, fit):
    return f'{dataset}_seed{seed}_{generator}_{fit}.pkl'


def synthetic_name(dataset, seed, method):
    generator, fit, label = method_parts(method)
    return f'{dataset}_seed{seed}_{generator}_{fit}_{label}_100k.csv'


def run_name(dataset, seed, mode, method):
    if method is None:
        return f'{dataset}_seed{seed}_real_original'
    return f'{synthetic_name(dataset, seed, method).removesuffix("_100k.csv")}_{mode}'


def create_path_dict(dataset_name, target_name):
    root = PROJECT_ROOT / 'data' / RUN_NAMESPACE / dataset_name / 'seed_{seed}'
    methods = {'ctgan', 'tvae'}
    for generator in ('ctgan', 'tvae'):
        prefix = '' if generator == 'ctgan' else 'tvae_'
        for label in ('gaussian', 'categorical', 'pca_gmm', 'rf', 'xgb', 'dnn'):
            methods.add(prefix + label)
            methods.add(prefix + 'compare_' + label)
    return {
        'train_original': f'{root}/{split_name(dataset_name, "{seed}", "train", "onehot")}',
        'dev': f'{root}/{split_name(dataset_name, "{seed}", "dev", "onehot")}',
        'test': f'{root}/{split_name(dataset_name, "{seed}", "test", "onehot")}',
        'split_manifest': f'{root}/split_manifest.json',
        'synthetic': {
            method: f'{root}/{synthetic_name(dataset_name, "{seed}", method)}'
            for method in methods
        },
        'target_name': target_name,
    }


IN_DATA_PATHS = {
    'adult': create_path_dict('adult', 'income'),
    'census': create_path_dict('census', 'income'),
    'census_kdd': create_path_dict('census_kdd', 'income'),
    'news': create_path_dict('news', ' shares'),
    'covertype': create_path_dict('covertype', 'Cover_Type'),
    'intrusion': create_path_dict('intrusion', 'target'),
    'credit': create_path_dict('credit', 'Class'),
    'mnist12': create_path_dict('mnist12', 'label'),
    'mnist28': create_path_dict('mnist28', 'label'),
}

for _dataset in ('mnist12', 'mnist28'):
    _root = PROJECT_ROOT / 'data' / RUN_NAMESPACE / _dataset / 'seed_{seed}'
    for _variant in ('num', 'cat'):
        _method = f'pca_gmm_{_variant}'
        IN_DATA_PATHS[_dataset]['synthetic'][_method] = (
            f'{_root}/{synthetic_name(_dataset, "{seed}", _method)}'
        )

OUT_DATA_PATHS = '../../output/'
