"""Read-only, isolated checks of G:/DA. Does not load saved models or train on data."""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path('G:/DA')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(name, **values):
    print(json.dumps({'check': name, **values}, default=str), flush=True)


def code_checks():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    splitter = load('da_splitter', 'src/commons/create_train_test.py').create_train_test
    for branch in ('train_only', 'test_only'):
        frame = pd.DataFrame({'id': range(40), 'cat': ['common'] * 40, 'y': [0] * 40})
        train, test = train_test_split(frame, test_size=.25, random_state=42)
        ids = (train if branch == 'train_only' else test).index[:4]
        frame.loc[ids, 'cat'] = 'rare'
        with contextlib.redirect_stdout(io.StringIO()):
            xtrain, xtest, _, _ = splitter(frame, 'y', ['cat'], test_size=.25)
        emit('split_' + branch, input_rows=len(frame), output_rows=len(xtrain) + len(xtest),
             overlap_ids=sorted(set(xtrain.id) & set(xtest.id)),
             duplicated_train_ids=int(xtrain.id.duplicated().sum()))

    constants = load('da_constants', 'src/modeling_thuy/constants.py')
    emit('census_kdd_path', census=constants.IN_DATA_PATHS['census']['train_original'],
         census_kdd=constants.IN_DATA_PATHS['census_kdd']['train_original'])
    sys.modules['constants'] = constants
    loader_module = load('da_loader', 'src/modeling_thuy/data_loader.py')
    loader = loader_module.data_loader('adult', 4, multi_y=False)
    emit('independent_scaler', train=loader._standardize(pd.DataFrame({'x': [0., 2.]})).x.tolist(),
         test=loader._standardize(pd.DataFrame({'x': [100., 102.]})).x.tolist())
    for n in (3, 5):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                batches = loader._distribute_in_batches(np.arange(n).reshape(-1, 1), np.zeros(n))
            emit('batch_rows', input=n, output=len(batches.dataset))
        except Exception as exc:
            emit('batch_rows', input=n, error=type(exc).__name__)

    for name, size in (('mnist12', 144), ('mnist28', 784)):
        model_mod = load('da_' + name, 'src/modeling_thuy/models_folder/model_' + name + '.py')
        model = getattr(model_mod, 'DNN_' + name.upper())(input_size=size).eval()
        with torch.no_grad():
            result = model(torch.zeros(2, size))
        emit('model_' + name, input_features=size, output_shape=list(result.shape),
             declared_classes=model.output.out_features)

    try:
        f1_score([0, 1, 2], [0, 1, 2], average='binary', zero_division=0)
    except Exception as exc:
        emit('multiclass_binary_f1', error=type(exc).__name__, message=str(exc))
    try:
        if (pd.Index(['a']) != []) and (pd.Index(['a']) != pd.Index([])):
            pass
    except Exception as exc:
        emit('pca_gmm_index_condition', error=type(exc).__name__, message=str(exc))


def data_checks():
    for dataset in ('adult', 'census', 'census_kdd', 'credit', 'mnist12', 'mnist28', 'covertype', 'news'):
        train = ROOT / 'data' / dataset / f'{dataset}_train.csv'
        test = ROOT / 'data' / dataset / f'{dataset}_test.csv'
        if not (train.exists() and test.exists()):
            continue
        def records(path):
            with path.open('rb') as handle:
                header = next(handle).rstrip(b'\r\n')
                for line in handle:
                    row = line.rstrip(b'\r\n')
                    if row:
                        yield row
            return header
        with train.open('rb') as handle:
            train_header = next(handle).rstrip(b'\r\n')
        with test.open('rb') as handle:
            test_header = next(handle).rstrip(b'\r\n')
        if train_header != test_header:
            emit('data_' + dataset, error='header mismatch')
            continue
        features = set()
        full = set()
        train_n = 0
        for row in records(train):
            train_n += 1
            features.add(hashlib.sha256(row.rsplit(b',', 1)[0]).digest())
            full.add(hashlib.sha256(row).digest())
        test_n = overlap_features = overlap_full = 0
        for row in records(test):
            test_n += 1
            overlap_features += hashlib.sha256(row.rsplit(b',', 1)[0]).digest() in features
            overlap_full += hashlib.sha256(row).digest() in full
        emit('data_' + dataset, train=train_n, test=test_n,
             feature_overlap=overlap_features, full_overlap=overlap_full)


def covertype_provenance_check():
    import pandas as pd
    from pandas.util import hash_pandas_object

    names = ('G:/DA/data/covertype/onehot_covertype_train.csv',
             'G:/DA/data/covertype/onehot_covertype_test.csv')
    train, test = [pd.read_csv(name) for name in names]
    columns = sorted(set(train.columns) - {'Cover_Type'})
    if set(train.columns) != set(test.columns):
        emit('covertype_onehot', error='column mismatch')
        return
    train_hashes = set(hash_pandas_object(train[columns + ['Cover_Type']], index=False).to_numpy())
    test_hashes = hash_pandas_object(test[columns + ['Cover_Type']], index=False).to_numpy()
    emit('covertype_onehot', train=len(train), test=len(test),
         exact_overlap=sum(item in train_hashes for item in test_hashes))

    old_test = Path('D:/SummerResearch/data/covertype/covertype_test.csv')
    new_test = ROOT / 'data/covertype/covertype_test.csv'
    with old_test.open('rb') as handle:
        next(handle)
        old_rows = {hashlib.sha256(row.rstrip(b'\r\n')).digest() for row in handle}
    with new_test.open('rb') as handle:
        next(handle)
        old_overlap = sum(hashlib.sha256(row.rstrip(b'\r\n')).digest() in old_rows for row in handle)
    emit('covertype_old_test_overlap', new_test=len(test),
         rows_also_in_old_test=old_overlap)


if __name__ == '__main__':
    code_checks()
    if '--data' in sys.argv:
        data_checks()
    if '--covertype' in sys.argv:
        covertype_provenance_check()
