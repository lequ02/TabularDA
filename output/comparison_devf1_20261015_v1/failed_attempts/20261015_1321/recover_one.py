"""Re-evaluate the development-selected checkpoint in an isolated namespace."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
ITEM = json.loads((BASE / 'inventory.json').read_text())[int(sys.argv[1])]
INDEX = json.loads((BASE / 'files_index.json').read_text())
JOB = BASE / 'jobs' / ITEM['record_relative_path'].replace('/', '__').removesuffix('.run.json')
JOB.mkdir(parents=True, exist_ok=False)
STARTED = datetime.now(ZoneInfo('America/Chicago')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def guard():
    available = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if available < 10 * 1024**2 or peak > 6 * 1024**2:
        message = f'Resource guard: available RAM {available} KiB, own peak RSS {peak} KiB'
        (JOB / 'resource_failure.json').write_text(json.dumps({'reason': message}))
        print(message, flush=True)
        os._exit(70)
    return {'available_ram_kib': available, 'own_peak_rss_kib': peak}


def watch_memory():
    while True:
        guard()
        time.sleep(0.25)


guard()
threading.Thread(target=watch_memory, daemon=True).start()
relative = ITEM['record_relative_path']
record_path = ROOT / relative
history_path = ROOT / relative.replace('.run.json', '.epochs.csv')
verified = {}
for path in (record_path, history_path):
    expected = INDEX[path.relative_to(ROOT).as_posix()]['sha256']
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Retained file hash mismatch: {path}')
    verified[str(path)] = actual
record = json.loads(record_path.read_text())
history = list(csv.DictReader(history_path.open()))
metric = ITEM['selection_metric']
target_epoch = int(max(history, key=lambda x: float(x['dev_' + metric]))['global_round'])
assert target_epoch == ITEM['target_epoch']
versions = {name: importlib.metadata.version(name) for name in record['package_versions']}
for name, expected in record['package_versions'].items():
    if versions[name].split('+')[0] != expected.split('+')[0]:
        raise ValueError(f'Package mismatch: {name} {versions[name]} versus {expected}')
required = [record['downstream_weight_path'], record['predictions_path']]
required.extend(record['prepared_real_paths'][s]['onehot'] for s in ('train', 'dev', 'test'))
if record['synthetic_path']:
    required.append(record['synthetic_path'])
saved_audit = json.loads((BASE / 'input_hashes.json').read_text())
for path in required:
    expected = record.get('artifact_sha256', {}).get(path)
    if expected is None and path == record['synthetic_path']:
        expected = saved_audit[path]
    if expected is None and path in [record['prepared_real_paths'][s]['onehot'] for s in ('train', 'dev', 'test')]:
        split = next(s for s in ('train', 'dev', 'test') if record['prepared_real_paths'][s]['onehot'] == path)
        expected = record['split_manifest']['files'][split]['onehot']['sha256']
    if expected is None:
        expected = INDEX[Path(path).relative_to(ROOT).as_posix()]['sha256']
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Original input hash mismatch: {path}')
    verified[path] = actual

runtime = JOB / 'runtime'
with zipfile.ZipFile(BASE / 'runtime.zip') as archive:
    archive.extractall(runtime)
(runtime / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
sys.path[:0] = [str(runtime), str(runtime / 'src')]
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('Historical recovery requires the original CUDA backend')
free, _ = torch.cuda.mem_get_info()
if free < 3 * 1024**3:
    raise RuntimeError('Less than 3 GiB GPU headroom before recovery')
torch.cuda.set_per_process_memory_fraction(0.08)
regression = record['dataset'] in ('news', 'california_housing')
if regression:
    from src.modeling.regression_train import train
else:
    from src.modeling.classification_train import train
from src.modeling import constants
# Bind the preserved loader to the explicit original artifacts. No original file
# is edited and no prepared table is regenerated.
paths = constants.IN_DATA_PATHS[record['dataset']]
paths.update({
    'train_original': record['prepared_real_paths']['train']['onehot'],
    'dev': record['prepared_real_paths']['dev']['onehot'],
    'test': record['prepared_real_paths']['test']['onehot'],
    'split_manifest': record['split_manifest_path'],
})
if record['synthetic_path']:
    paths['synthetic'][record['augment_option']] = record['synthetic_path']
score_keys = [x.removeprefix('dev_') for x in history[0] if x.startswith('dev_') and x != 'dev_loss']
if regression:
    metrics = score_keys
    extra = {'target_transform': record.get('target_transform', {}).get('name', 'raw')}
else:
    metrics = {}
    for key in score_keys:
        if key.startswith(('f1_', 'precision_', 'recall_')):
            name, average = key.split('_', 1)
            metrics.setdefault(name, []).append(average)
        else:
            metrics[key] = None
    extra = {'evaluation_protocol': record.get('evaluation_protocol', {}).get('name', 'standard')}
out = JOB / 'recovered'
runner = train(dataset_name=record['dataset'], train_option=record['train_option'],
               augment_option=record['augment_option'], mix_ratio=-1, n_sample=-1,
               test_option='original', validation=.2, batch_size=record['batch_size'],
               learning_rate=record['learning_rate'], num_epochs=record['epoch_budget'],
               patience=30, early_stop_criterion=record['selection_metric'], seed=record['seed'],
               eval_metrics=metrics, metric_to_plot=metric, pre_trained_w_file=None,
               w_dir=str(out) + '/', acc_dir=str(out) + '/', **extra)
original = torch.load(record['downstream_weight_path'], map_location='cpu', weights_only=True)
replay = []
selected = None
exact = ITEM['same_checkpoint']
if ITEM['same_checkpoint']:
    selected = original
else:
    for epoch in range(1, ITEM['replay_epochs'] + 1):
        guard()
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise RuntimeError('GPU headroom fell below 2 GiB')
        tick = time.monotonic()
        runner.current_epoch = epoch
        runner.trainer.train(torch.device('cuda:0'), epochs=1)
        train_loss, train_scores = runner.validate(runner.train_data, load_weight=False)
        dev_loss, dev_scores = runner.validate(runner.dev_data, load_weight=False)
        observed = {'global_round': epoch, 'train_loss': train_loss, 'dev_loss': dev_loss,
                    **{'train_' + k: v for k, v in train_scores.items()},
                    **{'dev_' + k: v for k, v in dev_scores.items()}}
        diffs = {k: abs(float(v) - float(history[epoch - 1][k])) for k, v in observed.items()}
        if any(not math.isfinite(v) or v > 1e-8 for v in diffs.values()):
            (JOB / 'replay_mismatch.json').write_text(json.dumps({'epoch': epoch, 'differences': diffs,
                                                               'observed': observed, 'reference': history[epoch - 1]}, indent=2))
            raise ValueError(f'Historical replay mismatch at epoch {epoch}: {max(diffs.values())}')
        if epoch == ITEM['original_epoch']:
            state = runner.trainer.model.state_dict()
            exact = all(torch.equal(state[k].cpu(), original[k]) for k in original)
            if not exact:
                raise ValueError('Original checkpoint tensors did not reproduce exactly')
        if epoch == target_epoch:
            selected = {k: v.detach().cpu().clone() for k, v in runner.trainer.model.state_dict().items()}
        replay.append(observed)
        print(f'REPLAY epoch={epoch} history_match=True dev_{metric}={dev_scores[metric]:.12f} elapsed_s={time.monotonic()-tick:.2f}', flush=True)
assert selected is not None and exact
runner.w_file_name = runner.run_id + '.max_dev_' + metric + '.weights.pth'
torch.save(selected, out / runner.w_file_name)
runner.trainer.model.load_state_dict(selected)
selected_dev_loss, selected_dev_scores = runner.validate(runner.dev_data, load_weight=False)
if abs(selected_dev_scores[metric] - float(history[target_epoch - 1]['dev_' + metric])) > 1e-8:
    raise ValueError('Selected checkpoint development metric did not reproduce')
runner.test_data = runner.data_loader.load_test_data()
test_loss, test_scores = runner.validate(runner.test_data, load_weight=True)
import pandas as pd
from sklearn.metrics import d2_absolute_error_score, f1_score, accuracy_score, r2_score
prediction_path = out / (runner.run_id + '.predictions.csv')
pred = pd.read_csv(prediction_path)
old = pd.read_csv(record['predictions_path'])
if not pred[['source_id', 'y_true']].equals(old[['source_id', 'y_true']]):
    raise ValueError('Test source IDs or targets differ from the original evaluation')
if regression:
    test_scores['d2_absolute_error'] = d2_absolute_error_score(pred.y_true, pred.y_pred)
    normalization = record['target_normalization']
    assert normalization['split'] == 'test' and normalization['ddof'] == 0
    test_scores['nmae_sigma'] = test_scores['mae'] / normalization['sigma_y']
    independently_scored = r2_score(pred.y_true, pred.y_pred)
else:
    independently_scored = f1_score(pred.y_true, pred.y_pred, average=metric.removeprefix('f1_'), zero_division=0)
if abs(independently_scored - test_scores[metric]) > 1e-8:
    raise ValueError('Saved-prediction metric verification failed')
if ITEM['same_checkpoint']:
    for key, value in record['test_scores'].items():
        if key in test_scores and abs(test_scores[key] - value) > 1e-8:
            raise ValueError(f'Unchanged-checkpoint test metric did not reproduce: {key}')
if replay:
    with (JOB / 'replay_history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replay[0]))
        writer.writeheader()
        writer.writerows(replay)
result = {
    **ITEM, 'started_at_chicago': STARTED,
    'completed_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'selection_scope': 'maximum development metric over original completed history; earliest tied maximum',
    'test_loss': test_loss, 'test_scores': test_scores,
    'selected_dev_loss': selected_dev_loss, 'selected_dev_scores': selected_dev_scores,
    'original_test_scores': record['test_scores'],
    'original_checkpoint_tensors_exact': exact, 'replayed_epochs': len(replay),
    'history_absolute_tolerance': 1e-8, 'test_ids_and_targets_match': True,
    'versions': versions, 'verified_input_hashes': verified,
    'artifact_sha256': {str(p): digest(p) for p in (prediction_path, out / runner.w_file_name)},
    'target_normalization': record.get('target_normalization'),
    'resource_usage': {'gpu_allocator_fraction': .08, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), **guard()},
}
(JOB / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('RESULT ' + json.dumps({'dataset': record['dataset'], 'test_scores': test_scores}), flush=True)
