"""Recover a completed evaluation whose record write hit Git conflict markers.

Run on the research server in its pinned environment. Never retrain or retest.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
sys.path.insert(0, str(ROOT / 'src'))
from modeling.run_record import write_run_record


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


name = 'census_kdd_seed42_ctgan_xonly_dnn_mix'
acc = ROOT / 'output/corrected_v2/census_kdd/acc'
record_path = acc / (name + '.run.json')
assert not record_path.exists(), 'Refusing to replace an existing record'
existing_records = {p: digest(p) for p in acc.glob('*seed42*.run.json')}
assert len(existing_records) == 52
reference_path = acc / 'census_kdd_seed42_ctgan_xonly_dnn_synthetic.run.json'
reference = json.loads(reference_path.read_text())
synthetic_path = Path(reference['synthetic_path'])
dev_path = synthetic_path.with_suffix('.dnn.json')
archive_root = ROOT / '.cache/merge_resolution_20261001T222728Z'
archived_dev = archive_root / dev_path.relative_to(ROOT)
conflicted = Path(str(archived_dev) + '.conflicted')
ours = Path(str(archived_dev) + '.ours')
conflict_lines = conflicted.read_text().splitlines()
assert conflict_lines[48] == '<<<<<<< HEAD'
assert digest(dev_path) == digest(ours), 'Resolved report differs from original server report'
dev = json.loads(dev_path.read_text())
assert dev['seed'] == 42 and dev['dataset'] == 'census_kdd'
assert dev['quality_gate_passed'] and dev['quality_gate_enforced'] and dev['converged']
assert dev == reference['dnn_dev_report']

report_path = acc / (name + '.report.txt')
predictions_path = acc / (name + '.predictions.csv')
epochs_path = acc / (name + '.epochs.csv')
weight_path = ROOT / 'output/corrected_v2/census_kdd/weight' / (name + '.weights.pth')
assert weight_path.is_file()
report_text = report_path.read_text()
loss_text, scores_text = report_text.removeprefix('Testing statistic: loss: ').split(', scores: ', 1)
test_loss = float(loss_text)
test_scores = ast.literal_eval(scores_text)
epochs = pd.read_csv(epochs_path)
selected_epoch = None
best_loss = float('inf')
for row in epochs.itertuples():
    if best_loss > row.dev_loss + 1e-5:
        best_loss = row.dev_loss
        selected_epoch = int(row.global_round)
assert selected_epoch is not None
assert len(epochs) - selected_epoch == 31, 'Patience replay does not match early stopping'
predictions = pd.read_csv(predictions_path)
assert predictions.source_id.is_unique
assert predictions.source_id.tolist() == reference['split_manifest']['splits']['test']
y, pred, score = (predictions[col].to_numpy() for col in ('y_true', 'y_pred', 'score'))
recomputed = {
    'accuracy': metrics.accuracy_score(y, pred),
    'balanced_accuracy': metrics.balanced_accuracy_score(y, pred),
    'pr_auc': metrics.average_precision_score(y, score),
    'roc_auc': metrics.roc_auc_score(y, score),
}
for metric, function in [('f1', metrics.f1_score), ('precision', metrics.precision_score), ('recall', metrics.recall_score)]:
    for average in ('binary', 'macro', 'micro', 'weighted'):
        recomputed[metric + '_' + average] = function(y, pred, average=average, zero_division=0)
assert set(recomputed) == set(test_scores)
for metric, value in recomputed.items():
    assert np.isclose(value, test_scores[metric], atol=1e-7, rtol=0), (metric, value, test_scores[metric])
assert np.array_equal(pred, (score > 0.5).astype(float))
synthetic_y = pd.read_csv(synthetic_path, usecols=['income'])['income']
synthetic_counts = {str(label): int(count) for label, count in synthetic_y.value_counts().items()}
assert synthetic_counts == reference['synthetic_label_counts']
artifact_paths = [report_path, predictions_path, epochs_path, weight_path, dev_path, synthetic_path]
artifact_hashes = {str(p): digest(p) for p in artifact_paths}
scratch = ROOT / '.cache/census_record_recovery_20261002'
scratch.mkdir(exist_ok=True)
pending = scratch / (name + '.run.json')
write_run_record(
    pending, dataset='census_kdd', seed=42, train_option='mix', augment_option='dnn',
    synthetic_path=str(synthetic_path), synthetic_label_counts=synthetic_counts,
    split_manifest_path=reference['split_manifest_path'], classifier=reference['classifier'],
    batch_size=reference['batch_size'], learning_rate=reference['learning_rate'],
    epoch_budget=reference['epoch_budget'], selected_epoch=selected_epoch,
    selection_metric='loss', test_loss=test_loss, test_scores=test_scores,
    predictions_path=str(predictions_path), weight_path=str(weight_path),
)
record = json.loads(pending.read_text())
record['record_recovery'] = {
    'recovered_at': datetime.now().astimezone().isoformat(),
    'reason': 'Git conflict markers in DNN dev report at original record-write time',
    'method': 'Original test report and predictions; checkpoint epoch replayed from saved dev-loss history',
    'retrained': False, 'reevaluated_test': False,
    'code_metadata_scope': 'Code and package metadata captured at record recovery, not asserted as original training metadata',
    'original_reference_record': str(reference_path),
    'original_reference_code_version': reference['code_version'],
    'original_reference_source_sha256': reference['source_sha256'],
    'artifact_sha256': artifact_hashes,
    'conflicted_report_sha256': digest(conflicted),
    'selected_dev_loss': best_loss,
}
pending.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
assert not record_path.exists()
os.replace(pending, record_path)
assert all(digest(p) == value for p, value in existing_records.items())
assert all(digest(Path(p)) == value for p, value in artifact_hashes.items())
failure_path = ROOT / 'output/corrected_v2/failures_classifiers_census_kdd_42.json'
if failure_path.exists():
    (scratch / failure_path.name).write_bytes(failure_path.read_bytes())
subprocess.run([sys.executable, 'scripts/run_corrected_matrix.py', '--dataset', 'census_kdd',
                '--seed', '42', '--stage', 'classifiers', '--resume'], cwd=ROOT, check=True)
assert len(list(acc.glob('*seed42*.run.json'))) == 53
result = {'recovered_record': str(record_path), 'completed': 53, 'expected': 53,
          'selected_epoch': selected_epoch, 'test_f1_binary': test_scores['f1_binary'],
          'retrained': False, 'existing_records_and_artifacts_unchanged': True,
          'cause': 'Git conflict markers at line 49 of seed-42 CTGAN X-only DNN dev report'}
(scratch / 'recovery_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
