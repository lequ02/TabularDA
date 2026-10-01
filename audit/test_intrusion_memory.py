"""Checks that compact encoding and chunked manifests preserve split safeguards."""
import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from commons.onehot import onehot_many


class MemoryPreparationTests(unittest.TestCase):
    def test_compact_encoder_preserves_values_and_ignores_holdout_categories(self):
        train = pd.DataFrame({'value': [1.125, 2.25], 'kind': ['a', 'b']})
        dev = pd.DataFrame({'value': [3.5], 'kind': ['unseen']})
        test = pd.DataFrame({'value': [4.75], 'kind': ['b']})
        encoded = onehot_many(train, (dev, test), ['kind'])
        original = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train[['kind']])
        for source, result in zip((train, dev, test), encoded):
            np.testing.assert_array_equal(result[['kind_a', 'kind_b']], original.transform(source[['kind']]))
            np.testing.assert_array_equal(result['value'], source['value'])
            self.assertEqual(result['kind_a'].dtype, np.dtype('uint8'))
        self.assertEqual(train.columns.tolist(), ['value', 'kind'])

    def manifest_function(self):
        # Exercise the actual method without importing GPU synthesis dependencies.
        source = ast.parse((ROOT / 'src/synthesize_data/create_synthetic_data/CreateSyntheticData.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'save_split_manifest')
        namespace = dict(pd=pd, np=np, json=json, hashlib=hashlib,
                         check_directory=SimpleNamespace(check_directory=lambda path: None))
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<manifest>', 'exec'), namespace)
        return namespace['save_split_manifest']

    def test_chunked_manifest_counts_and_overlap_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            job = SimpleNamespace(target_name='target', is_classification=True, ds_name='intrusion',
                                  seed=42, _source_row_count=50_004, paths={'data_dir': directory + '/'},
                                  _source_ids={'train': np.arange(50_002), 'dev': np.array([50_002]), 'test': np.array([50_003])})
            for split, ids in job._source_ids.items():
                frame = pd.DataFrame({'value': ids, 'target': np.where(ids % 2, 'attack.', 'normal.')})
                for suffix, view in (('', 'raw'), ('_onehot', 'onehot')):
                    name = f'{split}_{view}.csv'
                    job.paths[f'{split}_csv{suffix}'] = name
                    frame.to_csv(Path(directory) / name, index=False)
            save = self.manifest_function()
            save(job)
            manifest = json.loads((Path(directory) / 'split_manifest.json').read_text())
            self.assertEqual(manifest['files']['train']['raw']['rows'], 50_002)
            self.assertEqual(manifest['files']['train']['raw']['target_counts'], {'normal.': 25_001, 'attack.': 25_001})
            self.assertTrue(all(item['exact_feature_rows'] == 0 for item in manifest['train_overlap'].values()))
            pd.DataFrame({'value': [0], 'target': ['different-label']}).to_csv(Path(directory) / 'dev_raw.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Exact feature overlap'):
                save(job)


if __name__ == '__main__':
    unittest.main()
