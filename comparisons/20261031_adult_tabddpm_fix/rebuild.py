"""Rebuild the original-selection comparison reports with the Adult TabDDPM posterior-fix records.

Identical to `scripts/workspace.py reports --selection original`, except that:
- the 28 selected Adult TabDDPM arm records come from `tabddpm_adult_500ep_fix_20261030_v1`
  (archived in `adult/tabddpm_fix_20261030/`, hashes checked against the remote evidence);
- the TabDDPM snapshot used is a copy of `audit/comparison_tabddpm_snapshot.json` with only those
  entries and their four generator provenances replaced (written next to this script).
The held pre-fix records and earlier reports are not modified.
Run from the workspace root: .venv/Scripts/python.exe comparisons/20261031_adult_tabddpm_fix/rebuild.py
"""
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from workspace import artifact_map, read_rows  # noqa: E402

ARCHIVE = ROOT / 'adult/tabddpm_fix_20261030'
REMOTE = '/home/thuy/Research/minh_data_synth/TabularDA/'
SELECTED = ('tabddpm', 'tabddpm_compare_rf', 'tabddpm_compare_xgb', 'tabddpm_compare_dnn',
            'tabddpm_rf', 'tabddpm_xgb', 'tabddpm_dnn')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

evidence = json.loads((ARCHIVE / 'report_evidence_20261031.json').read_text())
old = json.loads((ROOT / 'audit/comparison_tabddpm_snapshot.json').read_text())
assert evidence['source_sha256'] in old['verified_source_hashes']
new = [r for r in evidence['new_records'] if r['method'] in SELECTED]
assert len(new) == 28
for record in new:
    assert sha(ARCHIVE / 'records' / Path(record['path']).name) == record['sha256'], record['path']
for path, generator in evidence['generators'].items():
    assert sha(ARCHIVE / 'generators' / Path(path).name) == generator['sha256'], path

snapshot = copy.deepcopy(old)
held = [r for r in old['selected_records'] if r['dataset'] == 'adult' and r['mode'] != 'original']
assert len(held) == 28 and all(r['path'].startswith('output/tabddpm_500ep_20261010_v1/') for r in held)
held_paths = {r['path'] for r in held}
snapshot['selected_records'] = [r for r in old['selected_records'] if r['path'] not in held_paths] + [
    {k: r[k] for k in ('path', 'sha256', 'dataset', 'seed', 'mode', 'method', 'source_sha256')} for r in new]
snapshot['generators'] = {k: v for k, v in old['generators'].items()
                          if '/tabddpm_500ep_20261010_v1/adult/' not in k}
snapshot['generators'].update(evidence['generators'])
snapshot['adult_recovery'] = dict(
    namespace=evidence['namespace'], checked_at_chicago=evidence['checked_at_chicago'],
    source_sha256=evidence['source_sha256'], verified_completed=evidence['verified_completed'],
    planned_total=evidence['planned_total'], reused_baselines=evidence['reused_baselines'],
    replaced_selected_records=sorted(held_paths),
    handoff='audit/TAB_DDPM_ADULT_RERUN_2026_10_30.md')
snapshot_path = HERE / 'comparison_tabddpm_snapshot_20261031.json'
snapshot_path.write_text(json.dumps(snapshot, indent=1) + '\n', encoding='utf-8')

mapping = artifact_map()
for record in new:
    mapping[record['path']] = ARCHIVE / 'records' / Path(record['path']).name
destination = HERE / 'reports'
if destination.exists():
    raise FileExistsError('Choose a new output directory; existing reports are preserved')
with tempfile.TemporaryDirectory(prefix='adult_fix_report_', dir=ROOT / '.cache') as temporary:
    view = Path(temporary)
    for relative, stored in mapping.items():
        if relative.startswith('output/') and relative.endswith(('.run.json', '.d2.json')):
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(stored, target)
    (view / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/plot_recent_comparison.py', view / 'scripts/plot_recent_comparison.py')
    (view / 'audit').mkdir()
    for name in ['comparison_rerun_verification_latest.json', 'comparison_news_log_fit_table_hashes.json',
                 'comparison_housing_fit_table_hashes.json', 'latest_comparison_snapshot.json',
                 'latest_comparison_mix_snapshot.json']:
        shutil.copy2(ROOT / 'audit' / name, view / 'audit' / name)
    shutil.copy2(snapshot_path, view / 'audit/comparison_tabddpm_snapshot.json')
    env = dict(os.environ, MPLCONFIGDIR=str(view / 'matplotlib'))
    for mode in ['synthetic', 'mix']:
        subprocess.run([sys.executable, str(view / 'scripts/plot_recent_comparison.py'),
                        '--train-option', mode], check=True, env=env)
    destination.mkdir()
    for source in (view / 'output/comparisons').iterdir():
        target = destination / source.name
        if source.suffix == '.md':  # builder links assume output/comparisons/; this folder is one level deeper
            target.write_text(source.read_text(encoding='utf-8').replace('](../../', '](../../../'), encoding='utf-8')
            continue
        if source.suffix != '.csv':
            shutil.copy2(source, target)
            continue
        rows = read_rows(source)
        for row in rows:
            for field, value in row.items():
                if value and (field.endswith('_record') or field.endswith('_d2_metadata')):
                    row[field] = Path(os.path.relpath(mapping[value], destination)).as_posix()
        with target.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
hashes = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
hashes['comparison_tabddpm_snapshot_20261031.json'] = sha(snapshot_path)
(destination / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
print(destination)
