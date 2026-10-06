"""Read-only diagnosis; execute on the research server, never locally."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
METHODS = [
    'real_original', 'ctgan_full_generated_synthetic',
    'tvae_full_generated_synthetic', 'ctgan_full_rf_synthetic',
    'ctgan_full_xgb_synthetic', 'ctgan_xonly_rf_synthetic',
    'ctgan_xonly_xgb_synthetic', 'ctgan_full_dnn_synthetic',
    'ctgan_xonly_dnn_synthetic', 'tvae_full_dnn_synthetic',
    'tvae_xonly_dnn_synthetic',
]
result = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'scope': 'Existing artifacts only; no model fits or threshold selection.',
          'runs': [], 'real_partitions': [], 'feature_diagnostics': []}
for dataset in ['credit', 'census_kdd']:
    for seed in [42, 43]:
        for method in METHODS:
            path = ROOT / 'output/corrected_v2' / dataset / 'acc' / f'{dataset}_seed{seed}_{method}.run.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            pred = pd.read_csv(record['predictions_path'])
            y, yp = pred.y_true, pred.y_pred
            item = {
                'dataset': dataset, 'seed': seed, 'method': method,
                'record_path': str(path), 'synthetic_path': record['synthetic_path'],
                'synthetic_label_counts_recorded': record['synthetic_label_counts'],
                'selected_epoch': record['selected_dev_epoch'],
                'test_rows': len(pred), 'tp': int(((y == 1) & (yp == 1)).sum()),
                'fp': int(((y == 0) & (yp == 1)).sum()),
                'fn': int(((y == 1) & (yp == 0)).sum()),
                'tn': int(((y == 0) & (yp == 0)).sum()),
                'max_test_probability': float(pred.score.max()),
                'test_scores': record['test_scores'],
                'generator_parameters': (record['generator_provenance'] or {}).get('parameters'),
            }
            epochs = pd.read_csv(path.with_name(path.name.replace('.run.json', '.epochs.csv')))
            cols = ['global_round', 'train_loss', 'dev_loss', 'train_f1_binary', 'dev_f1_binary']
            item['selected_epoch_metrics'] = epochs.loc[epochs.global_round == item['selected_epoch'], cols].iloc[0].to_dict()
            item['final_epoch_metrics'] = epochs[cols].iloc[-1].to_dict()
            target = 'Class' if dataset == 'credit' else 'income'
            if record['synthetic_path']:
                counts = pd.read_csv(record['synthetic_path'], usecols=[target])[target].value_counts()
                item['synthetic_label_counts_verified'] = {str(k): int(v) for k, v in counts.items()}
                assert item['synthetic_label_counts_verified'] == item['synthetic_label_counts_recorded']
            else:
                for part, paths in record['prepared_real_paths'].items():
                    counts = pd.read_csv(paths['raw'], usecols=[target])[target].value_counts()
                    result['real_partitions'].append({'dataset': dataset, 'seed': seed,
                                                     'partition': part, 'counts': {str(k): int(v) for k, v in counts.items()}})
            result['runs'].append(item)
        # Check representative feature sources once per dataset (seed 42).
        if seed != 42:
            continue
        base = ROOT / 'data/corrected_v2' / dataset / 'seed_42'
        sources = {
            'real_train': base / f'{dataset}_seed42_real_train_onehot.csv',
            'ctgan_full': base / f'{dataset}_seed42_ctgan_full_generated_100k.csv',
            'ctgan_xonly': base / f'{dataset}_seed42_ctgan_xonly_rf_100k.csv',
            'tvae_full': base / f'{dataset}_seed42_tvae_full_generated_100k.csv',
            'tvae_xonly': base / f'{dataset}_seed42_tvae_xonly_dnn_100k.csv',
        }
        numeric = ['V10', 'V12', 'V14', 'V17', 'Amount'] if dataset == 'credit' else ['AAGE', 'CAPGAIN', 'GAPLOSS', 'DIVVAL', 'WKSWORK', 'AHRSPAY']
        for source, path in sources.items():
            hashes, values = [], []
            for chunk in pd.read_csv(path, chunksize=5000):
                hashes.append(pd.util.hash_pandas_object(chunk.drop(columns=[target]), index=False).to_numpy())
                values.append(chunk[numeric].copy())
            hashes = np.concatenate(hashes)
            values = pd.concat(values, ignore_index=True)
            result['feature_diagnostics'].append({
                'dataset': dataset, 'seed': seed, 'source': source, 'path': str(path),
                'rows': len(hashes), 'unique_feature_fingerprints': int(len(np.unique(hashes))),
                'numeric_summary': {col: {'mean': float(values[col].mean()),
                                         'std': float(values[col].std()),
                                         'zero_fraction': float(values[col].eq(0).mean())} for col in numeric},
            })
print(json.dumps(result, indent=2))
