"""Small remote-only regression check; never run on the local research laptop."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/synthesize_data'))
sys.path.insert(0, str(ROOT / 'scripts'))
from tabddpm import TabDDPM
from commons.onehot import onehot
from modeling import constants
from synthesizer import synthesize_comparison_from_trained_model
from synthesizer import _save_synthesis_provenance, load_synthesizer
from build_corrected_results import expected_runs
from run_corrected_matrix import methods_for


def main(device='cpu'):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    x = pd.DataFrame({'value': rng.normal(size=96), 'category': np.tile(['a', 'b'], 48)})
    y = pd.Series(np.where(x['value'] > 0, 'yes', 'no'), name='target')
    kwargs = dict(seed=42, steps=2, batch_size=32, num_timesteps=4,
                  sample_batch_size=7, device=device)
    with tempfile.TemporaryDirectory(prefix='tabddpm-smoke-') as directory:
        output = Path(directory)
        model = TabDDPM(**kwargs)
        model.fit(x, ['category'])
        sample = model.sample(15)
        assert list(sample) == list(x) and len(sample) == 15 and 'target' not in sample
        assert set(sample['category']) <= {'a', 'b'}
        assert not model._diffusion()._denoise_fn.is_y_cond
        path = output / 'xonly.pkl'
        model.save(path)
        _save_synthesis_provenance(model, x, 15, str(path), 42,
                                  metadata={'columns': model.columns, 'categorical_columns': model.categorical})
        loaded = load_synthesizer(str(path), expected_data=x, expected_seed=42)
        pd.testing.assert_frame_equal(sample, loaded.sample(15))

        changed = pd.concat([x, y.sample(frac=1, random_state=1).reset_index(drop=True)], axis=1)
        other = TabDDPM(**kwargs)
        other.fit(changed.drop(columns=['target']), ['category'])
        assert all(torch.equal(model.weights[k], other.weights[k]) for k in model.weights)
        pd.testing.assert_frame_equal(sample, other.sample(15))
        seed43 = TabDDPM(**{**kwargs, 'seed': 43})
        seed43.fit(x, ['category'])
        assert any(not torch.equal(model.weights[k], seed43.weights[k]) for k in model.weights)

        joint = TabDDPM(**kwargs)
        joint.fit(pd.concat([x, y], axis=1), ['category', 'target'])
        joint_sample = joint.sample(15)
        assert set(joint_sample['target']) <= {'yes', 'no'}
        assert list(joint_sample) == ['value', 'category', 'target']
        all_cat = TabDDPM(**kwargs)
        all_cat.fit(x[['category']], ['category'])
        assert set(all_cat.sample(15)['category']) <= {'a', 'b'}

        regression = x[['value']].assign(target=200 + 30 * x['value'])
        continuous = TabDDPM(**kwargs)
        continuous.fit(regression, [])
        generated = continuous.sample(15)
        assert generated['target'].min() >= regression['target'].min() - 1e-4
        assert generated['target'].max() <= regression['target'].max() + 1e-4
        np.testing.assert_allclose(
            continuous.normalizer.inverse_transform(continuous.normalizer.transform(regression.to_numpy())),
            regression.to_numpy(), atol=1e-3,
        )

        _, encoded = onehot(x, sample, ['category'])
        for task, labels in ((True, y), (False, regression['target'])):
            before = encoded.copy()
            csv = output / ('classification.csv' if task else 'regression.csv')
            table = synthesize_comparison_from_trained_model(
                x, labels, ['category'], target_name='target', sample_size=15,
                target_synthesizer='rf', csv_file_name=str(csv),
                is_classification=task, seed=42, synthetic_features=encoded,
            )
            pd.testing.assert_frame_equal(table.drop(columns='target'), before)
            pd.testing.assert_frame_equal(encoded, before)
            assert csv.with_suffix('.predictor.pkl').is_file()
            assert json.loads(csv.with_suffix('.quality.json').read_text())['rows'] == 15

        for dataset in ('adult', 'california_housing'):
            for method in methods_for(dataset, 'tabddpm'):
                generator, source, label = constants.method_parts(method)
                assert generator == 'tabddpm'
                assert source in {'full', 'xonly'}
                assert label == 'generated' or label in {'gaussian', 'categorical', 'pca_gmm', 'rf', 'xgb', 'dnn'}
        assert len(methods_for('adult', 'tabddpm')) == 13
        assert len(methods_for('california_housing', 'tabddpm')) == 9
        assert len(expected_runs()) == 890
        assert len(expected_runs(generators=('tabddpm',))) == 454
        assert len(expected_runs(generators=('ctgan', 'tvae', 'tabddpm'))) == 1326
        assert constants.method_parts('compare_rf') == ('ctgan', 'full', 'rf')
        assert constants.method_parts('tvae_rf') == ('tvae', 'xonly', 'rf')
    print('PASS: X-only independence, seeds, mixed/categorical/numerical schemas, reload, labelers, matrix counts')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cpu')
