"""Inventory retained report checkpoints without evaluating or training models."""
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path('G:/SummerResearch_Retained_20261013_2221')
OUT = Path('D:/SummerResearch/.cache/comparison_devf1_20261015')
PREFIX = '/home/thuy/Research/minh_data_synth/TabularDA/'
files = {}
with (BASE / 'comparisons/files.csv').open(newline='', encoding='utf-8-sig') as stream:
    for row in csv.DictReader(stream):
        for path in {row['source_remote_path'], *json.loads(row['remote_aliases_json'])}:
            if path.startswith(PREFIX):
                files[path[len(PREFIX):]] = row
scores = {}
for score_file in BASE.glob('*/scores.csv'):
    with score_file.open(newline='', encoding='utf-8-sig') as stream:
        scores.update({r['record_relative_path']: r for r in csv.DictReader(stream)})
records = {}
for mode in ('synthetic', 'mix'):
    with (BASE / f'comparisons/{mode}.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['configuration'] == 'Real-only (TabDDPM run)':
                continue
            for column, relative in row.items():
                if not column.endswith('_record') or not relative:
                    continue
                if relative in records:
                    records[relative]['report_cells'].append({'mode': mode, 'configuration': row['configuration'], 'column': column})
                    continue
                record = scores[relative]
                # Adult binary F1 matches the ongoing audit; all multiclass models
                # and the established weighted Census protocol use macro F1.
                metric = ('r2' if record['dataset'] in ('news', 'california_housing')
                          else 'f1_binary' if record['dataset'] == 'adult' else 'f1_macro')
                history_path = relative.removesuffix('.run.json') + '.epochs.csv'
                with (BASE / files[history_path]['storage_path']).open(newline='') as hist:
                    history = list(csv.DictReader(hist))
                best = max(history, key=lambda x: float(x['dev_' + metric]))
                selected_epoch = int(best['global_round'])
                records.setdefault(relative, {
                    'record_relative_path': relative,
                    'retained_record': files[relative]['storage_path'],
                    'retained_history': files[history_path]['storage_path'],
                    'dataset': record['dataset'], 'seed': int(record['seed']),
                    'mode': record['mode'], 'generator': record['generator'],
                    'original_selection_metric': record['selection_metric'],
                    'original_epoch': int(record['selected_dev_epoch']),
                    'selection_metric': metric, 'target_epoch': selected_epoch,
                    'history_epochs': len(history),
                    'replay_epochs': max(selected_epoch, int(record['selected_dev_epoch'])),
                    'same_checkpoint': selected_epoch == int(record['selected_dev_epoch']),
                    'source_sha256': record['source_sha256'], 'report_cells': [],
                })['report_cells'].append({'mode': mode, 'configuration': row['configuration'], 'column': column})
(OUT / 'files_index.json').write_text(json.dumps(files, indent=2))
(OUT / 'inventory.json').write_text(json.dumps(list(records.values()), indent=2))
summary = {}
for dataset in sorted({x['dataset'] for x in records.values()}):
    group = [x for x in records.values() if x['dataset'] == dataset]
    summary[dataset] = {'records': len(group), 'unchanged_checkpoints': sum(x['same_checkpoint'] for x in group),
                        'epochs_if_replay_needed': sum(x['replay_epochs'] for x in group if not x['same_checkpoint']),
                        'original_criteria': dict(Counter(x['original_selection_metric'] for x in group))}
(OUT / 'inventory_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
