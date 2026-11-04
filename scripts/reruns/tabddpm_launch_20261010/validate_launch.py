import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
os.environ['CORRECTED_RUN_NAMESPACE'] = 'tabddpm_launch_validation_20261010'
sys.path[:0] = [str(root / 'src'), str(root / 'src/synthesize_data'), str(root / 'scripts')]
import numpy as np
import pandas as pd
import torch
import tabddpm
import run_corrected_matrix as matrix
from build_corrected_results import expected_runs
from modeling import constants
from modeling.run_record import file_sha256, source_sha256
from commons.onehot import onehot

assert torch.cuda.is_available()
torch.set_num_threads(2)
datasets = ('adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28', 'news', 'california_housing')
commands = []
with patch.object(matrix, 'run_command', side_effect=lambda command, cwd, log: commands.append(command)):
    for dataset in datasets:
        for seed in (42, 43):
            matrix.run_experiment(seed, [dataset], generators=('tabddpm',), generator_epochs=500,
                                  namespace=constants.RUN_NAMESPACE)
downstream = [command for command in commands if '-m' in command]
generation = [command for command in commands if 'synthesize_data/main.py' in command and '--prepare-only' not in command]
keys = set()
for command in downstream:
    read = lambda name: command[command.index(name) + 1]
    dataset, mode = read('--dataset-name'), read('--train-option')
    method = read('--augment-option') if '--augment-option' in command else None
    keys.add((dataset, mode, method, int(read('--seed'))))
    assert read('--global-round') == '100' and read('--patience') == '30'
    assert read('--batchsize') == '128' and read('--lr') == '0.001'
    if dataset == 'census_kdd':
        assert read('--evaluation-protocol') == 'census-weighted' and read('--early-stop-crit') == 'f1_macro'
    if dataset == 'news':
        assert read('--target-transform') == 'log'
assert keys == expected_runs(datasets=datasets, seeds=(42, 43), generators=('tabddpm',))
assert len(keys) == 346 and len(generation) == 14
assert all(command[command.index('--tabddpm-epochs') + 1] == '500' for command in generation)
assert all('--tabddpm-steps' not in command and command[command.index('--generator') + 1] == 'TABDDPM' for command in generation)
assert matrix.methods_for('adult', 'tabddpm')[:7] == (
    'tabddpm', 'tabddpm_compare_rf', 'tabddpm_compare_xgb', 'tabddpm_compare_dnn',
    'tabddpm_rf', 'tabddpm_xgb', 'tabddpm_dnn')
commands.clear()
with patch.object(matrix, 'run_command', side_effect=lambda command, cwd, log: commands.append(command)), patch.object(matrix.subprocess, 'run'):
    matrix.run_experiment(5, ['adult', 'news'], namespace=constants.RUN_NAMESPACE)
assert len([command for command in commands if '-m' in command]) == 90
assert {command[command.index('--generator') + 1] for command in commands if '--generator' in command} == {'CTGAN', 'TVAE'}

real_model = tabddpm.TabDDPM
real_init = real_model.__init__
def tiny_init(model, **kwargs):
    real_init(model, **kwargs, batch_size=32, num_timesteps=4, sample_batch_size=7, device='cuda')

rng = np.random.default_rng(7)
x = pd.DataFrame({'value': rng.normal(size=512), 'category': np.tile(['a', 'b'], 256)})
tables = 0
with tempfile.TemporaryDirectory(prefix='tabddpm-launch-check-') as directory:
    temporary = Path(directory)
    for dataset, classification, transform in (('adult', True, 'raw'), ('news', False, 'log')):
        features = x if classification else x[['value']]
        noise = rng.normal(size=len(features))
        y = pd.Series((features['value'] + .35 * noise > 0).astype(int) if classification
                      else np.exp(3 + .4 * features['value'] + .1 * noise), name='target')
        categories = ['category'] if classification else []
        _, dev_x = onehot(features.iloc[:400], features.iloc[400:], categories)
        folder = temporary / dataset
        data, models = folder / 'data', folder / 'models'
        data.mkdir(parents=True)
        models.mkdir()
        job = SimpleNamespace(ds_name=dataset, seed=42, target_transform=transform,
            paths={'data_dir': str(data), 'synthesizer_dir': str(models)}, is_classification=classification,
            sample_size_to_synthesize=15, numerical_cols_pca_gmm=['value'], ensure_prepared=lambda: None,
            read_train_data=lambda: (features.iloc[:400].reset_index(drop=True), y.iloc[:400].reset_index(drop=True), 'target', categories),
            read_dev_data=lambda: (dev_x.reset_index(drop=True), y.iloc[400:].reset_index(drop=True)))
        with patch.object(real_model, '__init__', tiny_init):
            tabddpm.create_tabddpm_data(job, epochs=2)
        paths = list(data.glob('*.csv'))
        assert len(paths) == (13 if classification else 9)
        tables += len(paths)
        for source in ('full', 'xonly'):
            model_path = models / constants.generator_name(dataset, 42, 'tabddpm', source)
            provenance = json.loads(model_path.with_suffix('.provenance.json').read_text())
            assert provenance['parameters']['epochs'] == 2 and provenance['parameters']['steps'] == 26
            assert ('target' in provenance['training_columns']) == (source == 'full')
            expected = pd.concat([features.iloc[:400], y.iloc[:400]], axis=1) if source == 'full' else features.iloc[:400]
            if source == 'full' and transform == 'log':
                expected = expected.copy()
                expected['target'] = np.log(expected['target'])
            assert provenance['fit_table_sha256'] == hashlib.sha256(expected.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
            shared = pd.read_csv(model_path.with_suffix('.sample.csv'))
            for method in matrix.methods_for(dataset, 'tabddpm'):
                _, fit, label = constants.method_parts(method)
                if fit != source:
                    continue
                table_path = data / constants.synthetic_name(dataset, 42, method)
                table = pd.read_csv(table_path)
                pd.testing.assert_frame_equal(table.drop(columns='target'), shared)
                quality = json.loads(table_path.with_suffix('.quality.json').read_text())
                assert quality['target_transform'] == transform and quality['rows'] == 15
                assert quality['table_sha256'] == file_sha256(table_path)
                if label != 'generated':
                    assert quality['feature_source']['sha256'] == file_sha256(model_path.with_suffix('.sample.csv'))
                if transform == 'log':
                    assert (table['target'] > 0).all()
        import unittest
        with unittest.TestCase().assertRaises(FileExistsError):
            tabddpm.create_tabddpm_data(job, epochs=2)
result = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'cuda': True, 'planned_downstream': 346, 'smoke_tables': tables,
          'news_log_targets_and_raw_share_exports': True, 'same_features_all_labelers': True,
          'preserves_ctgan_tvae_defaults': True, 'main_source_sha256': source_sha256()}
(root / '.cache/tabddpm_launch_20261010/validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
