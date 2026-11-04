"""Check small report metadata and plotted scores against completed result records."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
from datetime import datetime
from zoneinfo import ZoneInfo

base = Path('D:/SummerResearch/output/comparison_devf1_20261015_v1')
reports = base / 'reports'
recipe = runpy.run_path('D:/SummerResearch/.cache/comparison_devf1_20261015/plot_recipe.py')
for name, expected in json.loads((reports / 'artifact_hashes.json').read_text()).items():
    assert hashlib.sha256((reports / name).read_bytes()).hexdigest() == expected, name
results = {r['record_relative_path']: r for p in (base / 'jobs').glob('*/result.json')
           for r in [json.loads(p.read_text())]}
assert len(results) == 602
assert all(r['original_checkpoint_tensors_exact'] and r['test_ids_and_targets_match'] for r in results.values())
assert sum(not r['same_checkpoint'] for r in results.values()) == 308
all_sources = set()
checked_cells = 0
report_counts = {}
for mode in ('synthetic', 'mix'):
    filename = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '') + '_val_f1.csv'
    with (reports / filename).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    data = [r for r in rows if not r['configuration'].startswith('Paper ')]
    assert len(data) == 22
    labels = [r['configuration'] for r in data]
    assert 'Real-only (TabDDPM run)' not in labels
    assert labels[0] == 'Original data only'
    assert labels[1] == 'CTGAN, generated target'
    assert labels[8] == 'TVAE, generated target'
    assert labels[15] == 'TabDDPM, generated target'
    sources = set()
    for row in data:
        for panel, info in recipe['PANELS'].items():
            source = (reports / row[panel + '_record']).resolve()
            source.relative_to(base)
            result = json.loads(source.read_text())
            sources.add(result['record_relative_path'])
            pairs = [(info['metric'], panel + '_' + info['metric'] +
                      ('' if info['metric'] in recipe['UNSCALED_METRICS'] else '_percent'))]
            if info['metric'] == 'f1_binary':
                pairs.append(('f1_macro', panel + '_f1_macro_percent'))
            if info['metric'] == 'r2':
                pairs.append(('d2_absolute_error', panel + '_d2_absolute_error'))
            for metric, field in pairs:
                expected = result['test_scores'][metric] * (1 if metric in recipe['UNSCALED_METRICS'] else 100)
                assert abs(float(row[field]) - expected) < 1e-7, (mode, row['configuration'], field)
                checked_cells += 1
    assert len(sources) == 308
    all_sources.update(sources)
    report_counts[mode] = len(sources)
assert all_sources == set(results)
summary = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
           'completed': len(results), 'distinct_source_records_per_report': report_counts,
           'checked_score_cells': checked_cells, 'reported_configurations_per_panel': 22,
           'artifact_hashes_match': True, 'all_report_scores_match_verified_results': True,
           'original_checkpoint_tensors_exact_all_runs': True,
           'original_test_ids_and_targets_match_all_runs': True}
(base / 'local_artifact_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
