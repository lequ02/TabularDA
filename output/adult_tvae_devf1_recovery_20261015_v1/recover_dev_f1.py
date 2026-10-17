"""Recover a development-selected historical checkpoint without changing matrix outputs."""
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
SPEC = json.loads((HERE / 'spec.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def guard_resources():
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 8 * 1024 * 1024:
        raise RuntimeError(f'Insufficient available RAM for isolated replay: {available} KiB')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 2 * 1024 * 1024:
        raise RuntimeError(f'Replay exceeded its 2 GiB resident-memory guard: {rss} KiB')
    return {'available_ram_kib': available, 'own_peak_rss_kib': rss}


started = datetime.now(ZoneInfo('America/Chicago')).isoformat()
guard_resources()
verified_inputs = {}
for path, expected in SPEC['expected_inputs'].items():
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Input bytes changed: {path}; expected {expected}, got {actual}')
    verified_inputs[path] = actual
record_path = Path(SPEC['remote_record'])
record = json.loads(record_path.read_text())
canonical = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
if canonical != SPEC['retained_record_canonical_sha256']:
    raise ValueError('Server run record differs from the retained run record')
history = list(csv.DictReader(Path(SPEC['remote_history']).open()))
best = max(history, key=lambda row: float(row['dev_f1_binary']))
target_epoch = int(best['global_round'])
if target_epoch != SPEC['target_epoch']:
    raise ValueError('Development-selected epoch changed before replay')

versions = {name: importlib.metadata.version(name) for name in SPEC['package_versions']}
for name, expected in SPEC['package_versions'].items():
    if versions[name].split('+')[0] != expected.split('+')[0]:
        raise ValueError(f'Package version differs from the original run: {name}')
runtime = HERE / 'runtime'
runtime.mkdir(exist_ok=False)
with zipfile.ZipFile(HERE / 'runtime.zip') as archive:
    archive.extractall(runtime)
(runtime / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
sys.path.insert(0, str(runtime))
os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('Historical CUDA replay requires the original GPU backend')
free, total = torch.cuda.mem_get_info()
if free < 3 * 1024**3:
    raise RuntimeError(f'Insufficient free GPU memory before replay: {free}')
torch.cuda.set_per_process_memory_fraction(0.08)
from src.modeling.classification_train import train

out = HERE / 'recovered'
out.mkdir(exist_ok=False)
averages = ['macro', 'micro', 'weighted', 'binary']
metrics = {'accuracy': None, 'balanced_accuracy': None,
           'precision': averages.copy(), 'recall': averages.copy(), 'f1': averages.copy(),
           'pr_auc': None, 'roc_auc': None}
runner = train(dataset_name='adult', train_option='synthetic', augment_option='tvae',
               mix_ratio=-1, n_sample=-1, test_option='original', validation=0.2,
               batch_size=128, learning_rate=0.001, num_epochs=100, patience=30,
               early_stop_criterion='loss', seed=43, eval_metrics=metrics,
               metric_to_plot='f1_micro', pre_trained_w_file=None,
               w_dir=str(out) + '/', acc_dir=str(out) + '/')
original_state = torch.load(record['downstream_weight_path'], map_location='cpu', weights_only=True)
replay = []
best_loss, best_score, patience_counter = float('inf'), float('-inf'), 0
best_f1, best_f1_epoch, best_f1_state = float('-inf'), None, None
saved_checkpoint_exact = None
for epoch in range(1, target_epoch + 1):
    guard_resources()
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise RuntimeError(f'GPU headroom fell below 2 GiB: {free}')
    tick = time.monotonic()
    runner.current_epoch = epoch
    runner.trainer.train(device=torch.device('cuda:0'), epochs=1)
    train_loss, train_scores = runner.validate(runner.train_data, load_weight=False)
    dev_loss, dev_scores = runner.validate(runner.dev_data, load_weight=False)
    observed = {'global_round': epoch, 'train_loss': train_loss, 'dev_loss': dev_loss,
                **{'train_' + k: v for k, v in train_scores.items()},
                **{'dev_' + k: v for k, v in dev_scores.items()}}
    reference = history[epoch - 1]
    differences = {key: abs(float(value) - float(reference[key]))
                   for key, value in observed.items()}
    if max(differences.values()) > 1e-8:
        (HERE / 'replay_mismatch.json').write_text(json.dumps({
            'epoch': epoch, 'observed': observed, 'reference': reference,
            'absolute_differences': differences}, indent=2) + '\n')
        raise ValueError(f'Historical trajectory did not reproduce at epoch {epoch}; '
                         f'maximum difference {max(differences.values())}')
    stop, patience_counter, best_loss, best_score = runner.check_early_stop(
        dev_loss, dev_scores, best_loss, best_score, patience_counter)
    if stop:
        raise ValueError('Replay stopped before the development-selected checkpoint')
    if epoch == record['selected_dev_epoch']:
        state = runner.trainer.model.state_dict()
        saved_checkpoint_exact = all(torch.equal(state[k].cpu(), original_state[k]) for k in state)
        if not saved_checkpoint_exact:
            raise ValueError('Replayed original loss-selected tensors differ from saved weights')
    if dev_scores['f1_binary'] > best_f1:
        best_f1 = dev_scores['f1_binary']
        best_f1_epoch = epoch
        best_f1_state = {k: v.detach().cpu().clone() for k, v in runner.trainer.model.state_dict().items()}
    replay.append(observed)
    print(f'REPLAY epoch={epoch} history_match=True dev_loss={dev_loss:.12f} '
          f'dev_f1={dev_scores["f1_binary"]:.12f} elapsed_s={time.monotonic()-tick:.2f}', flush=True)

if best_f1_epoch != target_epoch or saved_checkpoint_exact is not True:
    raise ValueError('Recovered checkpoint did not pass development-selection and replay checks')
runner.w_file_name = 'adult_seed43_tvae_max_dev_f1_epoch11.weights.pth'
torch.save(best_f1_state, out / runner.w_file_name)
runner.test_data = runner.data_loader.load_test_data()
test_loss, test_scores = runner.validate(runner.test_data, load_weight=True)
result = {
    'started_at_chicago': started,
    'completed_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'dataset': 'adult', 'generator': 'tvae', 'construction': 'full_generated',
    'mode': 'synthetic', 'selection': 'maximum binary F1 over the original completed development history',
    'historical_source_sha256': record['source_sha256'],
    'original_trajectory_epochs': len(history), 'replayed_epochs': target_epoch,
    'all_replayed_history_fields_match_tolerance': 1e-8,
    'loss_selected_checkpoint_tensors_exactly_reproduced': saved_checkpoint_exact,
    'seed42': SPEC['seed42_result'],
    'seed43': {'old_selected_epoch': record['selected_dev_epoch'],
               'old_test_scores': record['test_scores'], 'selected_epoch': best_f1_epoch,
               'selected_dev_f1_binary': best_f1, 'test_loss': test_loss, 'test_scores': test_scores},
    'resource_limits': {'torch_cpu_threads': 1, 'gpu_allocator_fraction': 0.08,
                        'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                        **guard_resources()},
    'versions': versions, 'verified_input_hashes': verified_inputs,
    'artifact_hashes': {str(p): digest(p) for p in (out / runner.w_file_name,
        out / (runner.run_id + '.predictions.csv'), HERE / 'runtime.zip', HERE / 'recover_dev_f1.py')},
}
with (HERE / 'replay_history.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(replay[0]))
    writer.writeheader()
    writer.writerows(replay)
(HERE / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('RESULT ' + json.dumps(result['seed43']), flush=True)
