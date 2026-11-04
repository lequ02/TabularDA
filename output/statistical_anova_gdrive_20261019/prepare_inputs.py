"""Freeze and verify small archived score records; no experiment computation."""
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ARCHIVE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'inputs'
rows = list(csv.DictReader((ARCHIVE / 'comparisons/scores_valbest.csv').open(newline='', encoding='utf-8-sig')))
assert len(rows) == 602
keys = ['dataset', 'seed', 'generator', 'construction', 'labeler', 'mode']
assert len({tuple(r[k] for k in keys) for r in rows}) == 602
evidence, scores = [], []
checked_values = 0
for row in rows:
    result_path = ARCHIVE / row['result_path']
    original_path = ARCHIVE / row['original_record_path']
    result = json.loads(result_path.read_text())
    original = json.loads(original_path.read_text())
    assert result['dataset'] == row['dataset'] and result['seed'] == int(row['seed'])
    assert result['original_checkpoint_tensors_exact'] is True
    assert result['test_ids_and_targets_match'] is True
    for metric, value in result['test_scores'].items():
        cell = row.get('test_' + metric, '')
        if cell:
            assert math.isclose(float(cell), value, rel_tol=1e-12, abs_tol=1e-12)
            checked_values += 1
    if row['dataset'] == 'news':
        for record in [result, original]:
            norm = record['target_normalization']
            assert norm['split'] == 'test' and norm['ddof'] == 0
            assert norm['rows'] > 0 and len(norm['target_table_sha256']) == 64
            assert math.isclose(record['test_scores']['nmae_sigma'],
                                record['test_scores']['mae'] / norm['sigma_y'], rel_tol=1e-12)
    for selection, record, path in [('valbest', result, result_path), ('original', original, original_path)]:
        evidence.append(dict(selection=selection, archive_path=str(path),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        item = {k: row[k] for k in keys}
        item.update(selection=selection, retained_status=row['retained_status'],
                    caveats_json=row['caveats_json'], archive_path=path.relative_to(ARCHIVE).as_posix(),
                    selection_metric=record.get('selection_metric', result['original_selection_metric']))
        for metric in ['f1_macro', 'accuracy', 'f1_binary', 'r2', 'mae', 'nmae_sigma', 'd2_absolute_error']:
            item[metric] = record['test_scores'].get(metric, '')
        scores.append(item)
for name, values in [('scores.csv', scores), ('record_hashes.csv', evidence)]:
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
for name in ['scores_valbest.csv', 'valbest.md', 'valbest_archive.json']:
    shutil.copyfile(ARCHIVE / 'comparisons' / name, OUT / name)
manifest = dict(archive=str(ARCHIVE), checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                valbest_records=602, original_records=602, verified_score_values=checked_values,
                all_original_checkpoint_tensor_checks=True, all_test_ids_targets_match=True,
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
