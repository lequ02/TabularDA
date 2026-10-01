"""Trace trusted MNIST checkpoints, sample holdout-blind, verify cached source IDs."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sdv.single_table import TVAESynthesizer

ROOT = Path(sys.argv[1]).resolve()
DATA = ROOT / 'data/corrected_v2/mnist12/seed_42'
MODELS = ROOT / 'sdv trained model/corrected_v2/mnist12/seed_42'
torch.set_num_threads(1)


def emit(kind, **values):
    print(json.dumps(dict(kind=kind, **values), default=str), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp(frame, columns):
    return pd.util.hash_pandas_object(frame[columns].astype('float64'), index=False).to_numpy()


# Deny reads of real CSV data throughout checkpoint loading and sampling.
# Checkpoint files are the user's own saved experiment artifacts.
sampling = True
blocked_attempts = []


def audit_open(event, args):
    if sampling and event == 'open' and isinstance(args[0], (str, bytes)):
        value = str(args[0])
        if '/data/' in value and value.endswith('.csv'):
            blocked_attempts.append(value)
            raise RuntimeError('Real/synthetic CSV access blocked during fresh sampling')


sys.addaudithook(audit_open)
samples = []
checkpoint_hashes = {}
for fit in ('full', 'xonly'):
    path = MODELS / f'mnist12_seed42_tvae_{fit}.pkl'
    before = sha(path)
    checkpoint_hashes[fit] = before
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    model = TVAESynthesizer.load(path)
    model._model.set_device(torch.device('cpu'))
    metadata = model.get_metadata().to_dict()['tables']['table']['columns']
    column_info = model._model.transformer._column_transform_info_list
    loss = model._model.loss_values
    emit('checkpoint', fit=fit, sha256=before,
         provenance_training_rows=provenance['training_rows'],
         metadata_types=Counter(v['sdtype'] for v in metadata.values()),
         model_column_types=Counter(v.column_type for v in column_info),
         loss_columns=list(loss.columns),
         loss_epochs=int(loss['Epoch'].nunique()) if 'Epoch' in loss else None)
    for seed in (20260930, 20260931):
        model.reset_sampling()
        model._set_random_state(seed)
        sample = model.sample(num_rows=100000, output_file_path='disable')
        samples.append((fit, seed, sample))
        emit('fresh_sample_generated', fit=fit, seed=seed, rows=len(sample),
             holdout_csv_access_denied=True, attempted_csv_reads=len(blocked_attempts))
    if sha(path) != before:
        raise RuntimeError('Checkpoint changed during audit')
    del model
sampling = False

manifest = json.loads((DATA / 'split_manifest.json').read_text())
real = {split: pd.read_csv(DATA / f'mnist12_seed42_real_{split}_raw.csv')
        for split in ('train', 'dev', 'test')}
columns = sorted(c for c in real['train'] if c != 'label')
hashes = {s: fp(frame, columns) for s, frame in real.items()}
for fit, seed, sample in samples:
    sample_hashes = fp(sample, columns)
    overlaps = {s: dict(matching_synthetic_rows=int(np.isin(sample_hashes, h).sum()),
                        matching_real_rows=int(np.isin(h, sample_hashes).sum()))
                for s, h in hashes.items()}
    emit('fresh_sample_overlap', fit=fit, seed=seed, overlaps=overlaps,
         binary_features=bool(np.isin(sample[columns].to_numpy(), [0, 1]).all()))
    del sample
samples.clear()

for fit in ('full', 'xonly'):
    path = MODELS / f'mnist12_seed42_tvae_{fit}.provenance.json'
    p = json.loads(path.read_text())
    fit_frame = real['train'][p['training_columns']]
    digest = hashlib.sha256(fit_frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    emit('fit_input', fit=fit, rows=len(fit_frame), training_hash_matches=digest == p['fit_table_sha256'],
         train_vs_dev_features=int(np.isin(hashes['dev'], hashes['train']).sum()),
         train_vs_test_features=int(np.isin(hashes['test'], hashes['train']).sum()))

# Compare independently against cached source images. Network access is forbidden.
import sklearn.datasets._openml as openml_module
from sklearn.datasets import fetch_openml


def forbid_network(*args, **kwargs):
    raise RuntimeError('Source data unavailable in cache; network disabled for this audit')


openml_module.urlopen = forbid_network
source = fetch_openml(data_id=554, as_frame=False, data_home='/home/thuy/scikit_learn_data')
source12 = np.empty((len(source.data), 144), dtype=np.uint8)
for i, row in enumerate(source.data):
    binary = (row.reshape(28, 28) > 0).astype(np.uint8)
    source12[i] = np.array(Image.fromarray(binary).resize((12, 12))).reshape(-1)
numeric_columns = [str(i) for i in range(144)]
for split, frame in real.items():
    ids = np.asarray(manifest['splits'][split], dtype=int)
    pixel_matches = np.all(frame[numeric_columns].to_numpy() == source12[ids], axis=1)
    label_matches = frame['label'].astype(str).to_numpy() == np.asarray(source.target)[ids].astype(str)
    emit('source_identity', split=split, rows=len(ids), pixels_match=bool(pixel_matches.all()),
         labels_match=bool(label_matches.all()), source_rows=len(source.data))

# Explain whether shared holdout patterns are close to training patterns.
saved = pd.read_csv(DATA / 'mnist12_seed42_tvae_full_generated_100k.csv')
test_shared = real['test'][columns].merge(saved[columns].drop_duplicates(), on=columns).drop_duplicates()
train = real['train'][columns].drop_duplicates().to_numpy(dtype=np.float32)
query = test_shared.to_numpy(dtype=np.float32)
best = np.full(len(query), 144, dtype=np.float32)
for start in range(0, len(train), 5000):
    block = train[start:start+5000]
    distances = query.sum(axis=1)[:, None] + block.sum(axis=1)[None, :] - 2 * (query @ block.T)
    best = np.minimum(best, distances.min(axis=1))
emit('shared_pattern_nearest_train', unique_test_patterns=len(query),
     hamming_distance_counts=Counter(str(int(x)) for x in best))
emit('complete', checkpoints_unchanged=all(
    sha(MODELS / f'mnist12_seed42_tvae_{fit}.pkl') == digest for fit, digest in checkpoint_hashes.items()),
     attempted_csv_reads_during_sampling=len(blocked_attempts))
