"""Confirm candidates by direct numeric row joins, independently of fingerprints."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
folder = root / 'data/corrected_v2/mnist12/seed_42'
columns = [c for c in pd.read_csv(folder / 'mnist12_seed42_real_test_onehot.csv', nrows=0) if c != 'label']
columns.sort()
holdouts = {s: pd.read_csv(folder / f'mnist12_seed42_real_{s}_onehot.csv', dtype='float64')
            for s in ('dev', 'test')}
for path in sorted(folder.glob('*tvae*100k.csv')):
    synthetic = pd.read_csv(path, dtype='float64')
    unique = synthetic[columns].drop_duplicates()
    counts = {}
    for split, holdout in holdouts.items():
        matched = holdout[columns].merge(unique, on=columns, how='inner')
        synthetic_matches = synthetic[columns].merge(holdout[columns].drop_duplicates(), on=columns, how='inner')
        full_matched = synthetic.merge(holdout.drop_duplicates(), on=columns + ['label'], how='inner')
        foreground = matched.sum(axis=1)
        counts[split] = dict(holdout_rows=len(holdout), matching_holdout_rows=len(matched),
                             matching_synthetic_rows=len(synthetic_matches),
                             matching_synthetic_full_rows=len(full_matched),
                             unique_shared_features=len(matched.drop_duplicates()),
                             matched_foreground_range=([float(foreground.min()), float(foreground.max())]
                                                       if len(matched) else None))
    print(json.dumps(dict(table=path.name, direct_join_counts=counts)), flush=True)
