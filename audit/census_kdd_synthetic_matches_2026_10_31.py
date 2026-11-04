"""Read-only: describe Census KDD synthetic rows that exactly match real dev/test features."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1])
def mapped(v):
    v = str(v); m = '/TabularDA/'
    return ROOT / v.split(m, 1)[1] if m in v else Path(v)
for seed, table in [(42, 'census_kdd_seed42_tvae_xonly_dnn_100k.csv'), (42, 'census_kdd_seed42_ctgan_xonly_gaussian_100k.csv'),
                    (43, 'census_kdd_seed43_tvae_xonly_gaussian_100k.csv')]:
    rec = next(json.loads(p.read_text()) for p in (ROOT / 'output/corrected_v2/census_kdd').rglob(f'*seed{seed}*.run.json')
               if json.loads(p.read_text()).get('synthetic_path', '').endswith(table))
    paths = rec['prepared_real_paths']; target = rec['split_manifest']['files']['train']['raw']['columns'][-1]
    real = {s: pd.read_csv(mapped(paths[s]['onehot'])) for s in ('train', 'dev', 'test')}
    raw = {s: pd.read_csv(mapped(paths[s]['raw'])) for s in ('dev', 'test')}
    syn = pd.read_csv(mapped(rec['synthetic_path']))
    cols = sorted(c for c in real['train'] if c != target)
    key = lambda f: pd.util.hash_pandas_object(f[cols].astype('float64'), index=False).to_numpy()
    sk = key(syn); tr = real['train'][cols].to_numpy('float64')
    for split in ('dev', 'test'):
        rk = key(real[split]); hit = np.isin(rk, sk)
        for i in np.flatnonzero(hit):
            row = real[split][cols].iloc[i].to_numpy('float64')
            d = np.abs(tr - row).sum(1)
            print(json.dumps({'seed': seed, 'table': table, 'split': split, 'synthetic_copies': int((sk == rk[i]).sum()),
                   'same_pattern_rows_in_split': int((rk == rk[i]).sum()), 'nearest_train_l1': float(d.min()),
                   'train_rows_within_l1_2': int((d <= 2).sum()),
                   'raw_row': raw[split].drop(columns=[target]).iloc[i].to_dict()}, default=str))
