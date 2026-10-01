"""Validate candidate repairs using real training/development data only."""
import ast
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
SCRATCH = ROOT / '.cache/experiment_repairs_20261001'
CANDIDATE = SCRATCH / 'candidate'
sys.path.insert(0, str(ROOT / 'src'))
torch.set_num_threads(2)
step = sys.argv[1]
start = time.time()

if step == 'census':
    spec = importlib.util.spec_from_file_location('candidate_dnn', CANDIDATE / 'src/synthesize_data/dnn_labeler.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for seed in (42, 43):
        directory = ROOT / f'data/corrected_v2/census_kdd/seed_{seed}'
        train = pd.read_csv(directory / f'census_kdd_seed{seed}_real_train_onehot.csv')
        dev = pd.read_csv(directory / f'census_kdd_seed{seed}_real_dev_onehot.csv')
        target = train.columns[-1]
        print(f'Validating Census seed {seed}: train={len(train)}, dev={len(dev)}; test excluded', flush=True)
        xtrain, xdev = train.drop(columns=target), dev.drop(columns=target)
        module.fit_predict_dnn(xtrain, train[target], xdev, dev[target], xtrain.iloc[:100],
                               target_name=target, is_classification=True, seed=seed,
                               report_path=SCRATCH / f'census_seed{seed}_dev_validation.json',
                               dataset_name='census_kdd', device_name='cuda')
        print((SCRATCH / f'census_seed{seed}_dev_validation.json').read_text(), flush=True)
elif step == 'loader':
    spec = importlib.util.spec_from_file_location('modeling.data_loader', CANDIDATE / 'src/modeling/data_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.data_loader('intrusion', 128)
    train, dev = loader.load_train_augment_data('original', None)
    print(json.dumps({'training_rows': len(train.dataset), 'development_rows': len(dev.dataset),
                      'training_features': train.dataset.tensors[0].shape[1],
                      'tensor_dtype': str(train.dataset.tensors[0].dtype)}), flush=True)
elif step == 'transform':
    from ctgan.data_transformer import DataTransformer
    source = ast.parse((CANDIDATE / 'src/synthesize_data/synthesizer.py').read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_transform_float32')
    namespace = {'pd': pd, 'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<transform>', 'exec'), namespace)
    path = ROOT / 'data/corrected_v2/intrusion/seed_42/intrusion_seed42_real_train_raw.csv'
    data = pd.read_csv(path)
    categorical = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login', 'target']
    transformer = DataTransformer()
    print('Fitting a diagnostic transformer on 5,000 TRAINING rows; production models remain unchanged', flush=True)
    transformer.fit(data.iloc[:5000], categorical)
    print(f'Transforming all {len(data)} training rows into {transformer.output_dimensions} columns', flush=True)
    output = namespace['_transform_float32'](transformer, data)
    print(json.dumps({'rows': output.shape[0], 'columns': output.shape[1],
                      'bytes': output.nbytes, 'finite': bool(np.isfinite(output).all())}), flush=True)
else:
    raise ValueError(step)

print(json.dumps({'step': step, 'status': 'passed', 'elapsed_seconds': time.time() - start,
                  'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}), flush=True)
