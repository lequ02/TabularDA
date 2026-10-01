"""Regression checks for generator and evaluation memory repairs."""
import ast
import copy
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modeling.data_loader import data_loader
from modeling import constants
from sklearn.model_selection import GroupShuffleSplit


def repaired_transform():
    source = ast.parse((ROOT / 'src/synthesize_data/synthesizer.py').read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_transform_float32')
    namespace = {'pd': pd, 'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<transform>', 'exec'), namespace)
    return namespace['_transform_float32']


class ExperimentRepairTests(unittest.TestCase):
    def test_full_feature_pca_excludes_onehot_category_columns(self):
        sys.path.insert(0, str(constants.PROJECT_ROOT / 'src/synthesize_data'))
        import synthesizer
        source = ast.parse((ROOT / 'src/synthesize_data/synthesizer.py').read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'synthesize_comparison_from_trained_model')
        namespace = vars(synthesizer).copy()
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<comparison>', 'exec'), namespace)
        train = pd.DataFrame({'value': np.linspace(0, 1, 200), 'kind': ['a', 'b'] * 100})
        labels = pd.Series([0, 1] * 100, name='target')
        _, sample = synthesizer.onehot(train, train.iloc[:20], ['kind'])
        with tempfile.TemporaryDirectory() as directory:
            sample['target'] = labels.iloc[:20]
            full = Path(directory) / 'full.csv'
            full.write_text(sample.to_csv(index=False))
            output = Path(directory) / 'pca.csv'
            namespace[function.name](train, labels, ['kind'], 'target', sample_size=20,
                                     target_synthesizer='pca_gmm', csv_file_name=str(output),
                                     full_table_csv=str(full))
            with output.with_suffix('.predictor.pkl').open('rb') as artifact:
                predictor = pickle.load(artifact)
            self.assertEqual(predictor['numerical_columns'], ['value'])
            self.assertEqual(predictor['pca'].n_features_in_, 1)
            self.assertEqual(len(pd.read_csv(output)), 20)

    def test_group_split_retains_a_rare_training_class_without_overlap(self):
        source = ast.parse((ROOT / 'src/synthesize_data/create_synthetic_data/CreateSyntheticData.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'split_source_data')
        namespace = {'pd': pd, 'np': np, 'GroupShuffleSplit': GroupShuffleSplit}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<split>', 'exec'), namespace)
        split = namespace['split_source_data']
        job = SimpleNamespace(target_name='target', test_size=0.2, is_classification=True)
        data = pd.DataFrame({'value': np.repeat(np.arange(50), 2), 'target': ['common'] * 100})
        split(job, data)
        rare_groups = pd.unique(data.iloc[job._source_ids['test']]['value'])[:2]
        data.loc[data['value'].isin(rare_groups), 'target'] = 'rare'
        train, dev, test = split(job, data)
        self.assertIn('rare', set(train['target']))
        self.assertIn('rare', set(test['target']))
        self.assertEqual(len(train) + len(dev) + len(test), len(data))
        for first, second in ((train, dev), (train, test), (dev, test)):
            self.assertFalse(set(first['value']) & set(second['value']))

    def test_generator_transform_preserves_encoding(self):
        from ctgan.data_transformer import DataTransformer
        frame = pd.DataFrame({'value': np.linspace(-2, 2, 120), 'kind': ['a', 'b', 'c'] * 40})
        transformer = DataTransformer(max_clusters=3)
        transformer.fit(frame, ['kind'])
        reference = copy.deepcopy(transformer)
        expected = np.concatenate(reference._synchronous_transform(
            frame, reference._column_transform_info_list), axis=1).astype(np.float32)
        actual = repaired_transform()(transformer, frame)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)
        recovered = transformer.inverse_transform(actual)
        self.assertEqual(recovered['kind'].tolist(), frame['kind'].tolist())
        np.testing.assert_allclose(recovered['value'], frame['value'], atol=1e-6)

    def test_loader_keeps_train_fitted_scaling_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            train = pd.DataFrame({'x': [1., 3., 5., 7.], 'target': ['a', 'b', 'a', 'b']})
            dev = pd.DataFrame({'x': [101., 103.], 'target': ['a', 'b']})
            for name, frame in [('train', train), ('dev', dev)]:
                frame.to_csv(Path(directory) / (name + '.csv'), index=False)
            loader = data_loader('intrusion', 2)
            loader.paths = {'target_name': 'target', 'train_original': str(Path(directory) / 'train.csv'),
                            'dev': str(Path(directory) / 'dev.csv')}
            parsed = loader._read_table(loader.paths['train_original'])
            self.assertEqual(parsed['x'].dtype, np.float32)
            train_batches, dev_batches = loader.load_train_augment_data('original', None)
            self.assertEqual(loader.label_encoder.classes_.tolist(), ['a', 'b'])
            np.testing.assert_allclose(loader.scaler.mean_, [4.])
            np.testing.assert_allclose(dev_batches.dataset.tensors[0].numpy().ravel(),
                                       (dev['x'] - 4.) / np.sqrt(5.), rtol=1e-6)
            np.testing.assert_array_equal(train_batches.dataset.tensors[1].numpy(), [0, 1, 0, 1])

    def test_float32_transform_supports_both_generator_training_loops(self):
        from ctgan.data_transformer import DataTransformer
        from ctgan import CTGAN, TVAE
        frame = pd.DataFrame({'value': np.linspace(-2, 2, 120), 'kind': ['a', 'b', 'c'] * 40})
        with patch.object(DataTransformer, 'transform', repaired_transform()):
            for factory in (CTGAN, TVAE):
                model = factory(epochs=1, batch_size=100, cuda=False)
                model.fit(frame, discrete_columns=['kind'])
                sample = model.sample(10)
                self.assertEqual(sample.columns.tolist(), frame.columns.tolist())
                self.assertTrue(np.isfinite(sample['value']).all())
                self.assertTrue(sample['kind'].isin(frame['kind']).all())

    def test_resume_preserves_completed_classifier_results(self):
        spec = importlib.util.spec_from_file_location('matrix', ROOT / 'scripts/run_corrected_matrix.py')
        matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'artifact'
            artifact.touch()
            output = root / 'output/corrected_v2/adult/acc'
            output.mkdir(parents=True)
            runs = [('original', None)] + [(mode, method) for generator in ('ctgan', 'tvae')
                    for method in matrix.methods_for('adult', generator) for mode in ('synthetic', 'mix')]
            for mode, method in runs:
                record = {'dataset': 'adult', 'seed': 42, 'train_option': mode, 'augment_option': method,
                          'selected_dev_epoch': 3, 'predictions_path': str(artifact),
                          'downstream_weight_path': str(artifact)}
                path = output / (matrix.constants.run_name('adult', 42, mode, method) + '.run.json')
                path.write_text(json.dumps(record))
            with patch.object(matrix, 'ROOT', root), patch.object(matrix, 'run_command') as run:
                matrix.run_matrix(('adult',), (42,), 'classifiers', resume=True)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
