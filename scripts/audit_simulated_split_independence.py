"""Regenerate independent oracle draws in memory; never alter saved tables."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'SDGym-research'))
from synthetic_data_benchmark import simulated_benchmark as bench
from pgmpy.sampling import BayesianModelSampling

run = root / 'SDGym-research/data/simulated_paper'
for folder in sorted(run.glob('seed_*/*')):
    if not (folder / 'manifest.json').exists():
        continue
    manifest = json.loads((folder / 'manifest.json').read_text())
    dataset, seed = manifest['dataset'], manifest['seed']
    matches = {}
    for split, offset in [('train', 1), ('test', 2), ('labeled_dev', 3)]:
        path = folder / (split + '.csv')
        if not path.exists():
            continue
        saved = pd.read_csv(path, dtype=str, keep_default_na=False)
        if dataset in bench.MIXTURES:
            fresh = bench.sample_mixture(bench.mixture_oracle(dataset, seed), len(saved), seed + offset)
            columns = list(fresh.columns)
            matches[split] = bool(np.allclose(fresh.to_numpy(), saved[columns].to_numpy(dtype=float), rtol=0, atol=1e-14))
        else:
            fresh = BayesianModelSampling(bench.bn_model(dataset)).forward_sample(
                size=len(saved), seed=seed + offset, show_progress=False).astype(str)
            matches[split] = bool(np.array_equal(fresh[list(saved.columns)].to_numpy(), saved.astype(str).to_numpy()))
    print(json.dumps(dict(dataset=dataset, seed=seed, distinct_rng_seeds=[seed+1, seed+2, seed+3],
                          regenerated_matches=matches)), flush=True)
