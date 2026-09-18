"""Read-only audit probes. Run: python -B audit/reproduce.py [--data].

Exercises existing code on artificial data; never trains saved models or rewrites data.
Outputs observations, not a passing regression suite.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(name, **values):
    print(json.dumps({'probe': name, **values}, default=str), flush=True)


def probes():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from commons.create_train_test import create_train_test

    for branch in ('train_only', 'test_only'):
        df = pd.DataFrame({'id': range(40), 'cat': ['common'] * 40, 'y': [0] * 40})
        tr, te = train_test_split(df, test_size=0.25, random_state=42)
        ids = (tr if branch == 'train_only' else te).index[:4]
        df.loc[ids, 'cat'] = 'rare'
        with contextlib.redirect_stdout(io.StringIO()):
            xtr, xte, _, _ = create_train_test(df, 'y', ['cat'], test_size=0.25)
        emit('split_' + branch, input_rows=len(df), output_rows=len(xtr)+len(xte),
             overlap_ids=sorted(set(xtr.id) & set(xte.id)), train_duplicate_ids=int(xtr.id.duplicated().sum()))

    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            create_train_test(df, 'y', [], stratify=df.y)
    except Exception as exc:
        emit('regression_stratification', error=type(exc).__name__, message=str(exc))

    from commons.handle_missing_values import handle_missing_values
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            handle_missing_values(pd.DataFrame({'x': [1., np.nan, 3.]}), pd.Series([0, 1, 0], name='y'), 'y', 'mean')
    except Exception as exc:
        emit('imputation_mean', error=type(exc).__name__, message=str(exc))

    # Import only the loader and constants; no training entrypoints are executed.
    sys.modules['constants'] = load('audit_constants', 'src/modeling_thuy/constants.py')
    module = load('audit_loader', 'src/modeling_thuy/data_loader.py')
    loader = module.data_loader('adult', 4, multi_y=False)
    a = loader._standardize(pd.DataFrame({'x': [0., 2.]}))
    b = loader._standardize(pd.DataFrame({'x': [100., 102.]}))
    emit('independent_scalers', train=a.x.tolist(), test=b.x.tolist(), expected_test_using_train=[99., 101.])
    for count in (3, 5):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                batches = loader._distribute_in_batches(np.arange(count).reshape(-1, 1), np.zeros(count))
            emit('batch_rows', input=count, output=len(batches.dataset))
        except Exception as exc:
            emit('batch_rows', input=count, error=type(exc).__name__, message=str(exc))

    mnist = module.DataLoaderMNIST('mnist12', 4)
    try:
        mnist._load_data_in_batches(np.zeros((5, 144)), np.zeros(5))
    except Exception as exc:
        emit('mnist12_dimensions', error=type(exc).__name__, message=str(exc))

    import torch
    model_module = load('audit_mnist_model', 'src/modeling_thuy/models_folder/model_mnist28.py')
    batch, _ = next(iter(module.DataLoaderMNIST('mnist28', 4)._load_data_in_batches(np.zeros((4, 784)), np.zeros(4))))
    model = model_module.DNN_MNIST28(input_size=batch.shape[1])
    try:
        with torch.no_grad():
            model(batch)
    except Exception as exc:
        emit('mnist28_model_shape', input_shape=list(batch.shape), error=type(exc).__name__, message=str(exc))

    from unittest.mock import patch
    huy_module = load('audit_huy_loader', 'src/modeling_huy/data_loader.py')
    huy = huy_module.data_loader('adult', 4)
    huy.train_columns = pd.Index(['feature_a', 'feature_b'])
    example = pd.DataFrame({'feature_a': [1., 2.], 'feature_b': [10., 20.], 'income': [0, 1]})
    with patch.object(pd, 'read_csv', return_value=example), contextlib.redirect_stdout(io.StringIO()):
        result = huy.load_test_data()
    emit('huy_test_target', expected=[0, 1], actual=result.dataset.tensors[1].tolist())

    loader.problem_type = 'regression'
    first = pd.DataFrame({'x': range(100)})
    second = pd.DataFrame({'x': range(1000, 1040)})
    mixed = loader.concat(first, second, concat_ratio=.8, n_sample=100)
    emit('mix_sampling', requested_total=100, requested_synthetic=20, actual_total=len(mixed), actual_synthetic=int((mixed.x >= 1000).sum()))

    gmm = load('audit_gmm', 'src/synthesize_data/GaussMix_nb.py').GMMNaiveBayes(n_components=1)
    rng = np.random.default_rng(42)
    x = pd.DataFrame({'x': np.r_[rng.normal(-5, .3, 30), rng.normal(5, .3, 30)]})
    y = pd.Series([0] * 30 + [1] * 30)
    with contextlib.redirect_stdout(io.StringIO()):
        gmm.fit(x, y, ['x'])
    query = pd.DataFrame({'x': [-5., 5.]})
    emit('gmm_batch_dependence', together=gmm.predict(query).tolist(),
         individually=[int(gmm.predict(query.iloc[[i]])[0]) for i in range(2)],
         singleton_probabilities=gmm.predict_proba(query.iloc[[1]]).tolist())


def inventory():
    sources = [p for folder in ('src', 'SDGym-research', 'GLRM', 'pyglrm', 'plots')
               for p in (ROOT / folder).rglob('*.py') if '.ipynb_checkpoints' not in p.parts]
    errors = []
    for p in sources:
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        except SyntaxError as exc:
            errors.append({'file': str(p.relative_to(ROOT)), 'line': exc.lineno, 'error': exc.msg})
    emit('syntax_inventory', python_files=len(sources), errors=errors)
    notebooks = [p for folder in ('src', 'plots', 'pyglrm') for p in (ROOT / folder).rglob('*.ipynb')
                 if '.ipynb_checkpoints' not in p.parts]
    emit('notebook_inventory', notebooks=len(notebooks),
         code_cells=sum(sum(c['cell_type'] == 'code' for c in json.loads(p.read_text(encoding='utf-8'))['cells']) for p in notebooks))


def data_checks():
    # Exact CSV row comparison (SHA-256, excluding header/newlines), streamed to
    # limit memory. Equal values do not prove equal original record identities.
    for folder in sorted((ROOT / 'data').iterdir()):
        if not folder.is_dir():
            continue
        train = folder / (folder.name + '_train.csv')
        test = folder / (folder.name + '_test.csv')
        if not train.exists() or not test.exists():
            continue
        def rows(path):
            with path.open('rb') as f:
                next(f)
                for row in f:
                    yield hashlib.sha256(row.rstrip(b'\r\n')).digest()
        test_hashes = list(rows(test))
        wanted = set(test_hashes)
        matched = set()
        train_count = 0
        for row_hash in rows(train):
            train_count += 1
            if row_hash in wanted:
                matched.add(row_hash)
        emit('saved_split_exact_row_overlap', dataset=folder.name, train_rows=train_count,
             test_rows=len(test_hashes), test_rows_matching_train=sum(h in matched for h in test_hashes),
             distinct_matching_rows=len(matched))
    for suffix in ('train', 'test'):
        paths = [ROOT / 'data' / ds / f'{ds}_{suffix}.csv' for ds in ('adult', 'census')]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        emit('adult_census_identical', split=suffix, identical=hashes[0] == hashes[1], sha256=hashes)


if __name__ == '__main__':
    inventory()
    probes()
    if '--data' in sys.argv:
        data_checks()
