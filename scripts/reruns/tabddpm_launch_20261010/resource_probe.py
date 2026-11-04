import json
from pathlib import Path
import resource
import sys
import time

root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
sys.path[:0] = [str(root / 'src'), str(root / 'src/synthesize_data')]
import pandas as pd
import torch
from tabddpm import TabDDPM
from create_synthetic_data.adult import CreateSyntheticDataAdult
from create_synthetic_data.covertype import CreateSyntheticDataCovertype
from create_synthetic_data.mnist28 import CreateSyntheticDataMnist28
from create_synthetic_data.california_housing import CreateSyntheticDataCaliforniaHousing

factories = {'adult': CreateSyntheticDataAdult, 'covertype': CreateSyntheticDataCovertype,
             'mnist28': CreateSyntheticDataMnist28, 'california_housing': CreateSyntheticDataCaliforniaHousing}
dataset = sys.argv[1]
torch.set_num_threads(2)
job = factories[dataset](feature_synthesizer='TABDDPM', seed=42)
path = root / 'data/tabddpm_500ep_20261010_v1' / dataset / 'seed_42' / f'{dataset}_seed42_real_train_raw.csv'
data = pd.read_csv(path, nrows=1000)
categories = list(job.categorical_columns) + ([job.target_name] if job.is_classification else [])
start = time.monotonic()
model = TabDDPM(seed=42, steps=2, batch_size=500, num_timesteps=1000, sample_batch_size=500, device='cuda')
model.fit(data, categories)
sample = model.sample(500)
assert sample.shape == (500, len(data.columns)) and not sample.isna().any().any()
result = {'dataset': dataset, 'purpose': 'four-process resource check, not a production result',
          'fit_rows': len(data), 'columns': len(data.columns), 'steps': 2, 'num_timesteps': 1000,
          'training_batch': 500, 'sampling_batch': 500, 'sample_rows': 500,
          'elapsed_seconds': time.monotonic() - start,
          'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
          'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
(root / '.cache/tabddpm_launch_20261010' / f'resource-{dataset}.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
