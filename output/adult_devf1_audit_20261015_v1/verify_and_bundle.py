"""Check all three preselected checkpoint comparisons and preserve the evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parent
completed = json.loads((root / 'completed.json').read_text())
selected = json.loads((root / 'selected_candidates.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


comparisons = []
files = [root / name for name in ('completed.json', 'selected_candidates.json', 'inventory.json',
                                  'queue.log', 'launch.json', 'payload.zip')]
for candidate in selected['jobs']:
    name = Path(candidate['record_relative_path']).name.removesuffix('.run.json')
    job = root / 'jobs' / name
    spec = json.loads((job / 'spec.json').read_text())
    result = json.loads((job / 'result.json').read_text())
    record = json.loads(Path(spec['remote_record']).read_text())
    canonical = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    assert canonical == spec['retained_record_canonical_sha256'], name
    for path, expected in result['verified_input_hashes'].items():
        assert sha(path) == expected, ('original artifact changed', path)
    for path, expected in result['artifact_hashes'].items():
        assert sha(path) == expected, ('recovered artifact changed', path)
    predictions = job / 'recovered' / (name + '.predictions.csv')
    rows = list(csv.DictReader(predictions.open()))
    original_rows = list(csv.DictReader(Path(record['predictions_path']).open()))
    identities = lambda data: [(x['source_id'], float(x['y_true'])) for x in data]
    assert identities(rows) == identities(original_rows), name
    assert [int(x['source_id']) for x in rows] == record['split_manifest']['splits']['test']
    counts = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for row in rows:
        counts[int(float(row['y_true'])), int(float(row['y_pred']))] += 1
    tn, fp, fn, tp = [counts[pair] for pair in ((0, 0), (0, 1), (1, 0), (1, 1))]
    f1 = 2 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) / len(rows)
    assert abs(f1 - result['run']['test_scores']['f1_binary']) < 1e-12
    assert abs(accuracy - result['run']['test_scores']['accuracy']) < 1e-12
    assert result['loss_selected_checkpoint_tensors_exactly_reproduced'] is True
    replay_rows = list(csv.DictReader((job / 'replay_history.csv').open()))
    assert len(replay_rows) == candidate['dev_f1_epoch']
    old = record['test_scores']['f1_binary']
    comparisons.append({
        'record_relative_path': candidate['record_relative_path'], 'seed': record['seed'],
        'generator': candidate['generator'], 'construction': candidate['construction'],
        'labeler': candidate['labeler'], 'mode': candidate['mode'],
        'loss_selected_epoch': record['selected_dev_epoch'],
        'f1_selected_epoch': result['run']['selected_epoch'],
        'old_test_f1_percent': 100 * old, 'new_test_f1_percent': 100 * f1,
        'test_f1_change_pp': 100 * (f1 - old),
        'development_f1_change_pp': candidate['dev_f1_gain_pp'],
        'test_confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'test_rows': len(rows), 'test_source_ids_and_targets_unchanged': True,
        'original_record_and_input_hashes_unchanged': True,
        'replayed_original_loss_checkpoint_tensors_exact': True,
        'result_sha256': sha(job / 'result.json'),
    })
    files.extend(job / name for name in ('spec.json', 'result.json', 'replay_history.csv', 'worker.log'))
    files.extend(sorted((job / 'recovered').iterdir()))

verification = {
    'verified_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'selection_rule': selected['selection_rule'], 'comparisons': comparisons,
    'existing_jobs_after': subprocess.check_output(
        ['ps', '-p', '2312829,2317009', '-o', 'pid,pcpu,rss,args'], text=True),
    'gpu_processes_after': subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True),
}
(root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
files.append(root / 'verification.json')
with zipfile.ZipFile(root / 'verified_results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root).as_posix())
print(json.dumps(verification, indent=2))
print('BUNDLE_SHA256 ' + sha(root / 'verified_results.zip'))
