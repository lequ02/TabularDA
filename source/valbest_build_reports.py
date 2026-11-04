"""Build separate reports using only successfully re-evaluated selected states."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
OUT = BASE / 'reports'
OUT.mkdir(exist_ok=True)
inventory = json.loads((BASE / 'inventory.json').read_text())
recipe = runpy.run_path(str(BASE / 'plot_recipe.py'))
PANELS = recipe['PANELS']
UNSCALED = recipe['UNSCALED_METRICS']
labels = dict(recipe['ROWS'])
order = ['Original data only', 'CTGAN, generated target']
order += [x for x in labels if x.startswith(('CTGAN full', 'CTGAN X-only'))]
order += ['TVAE, generated target']
order += [x for x in labels if x.startswith(('TVAE full', 'TVAE X-only'))]
order += ['TabDDPM, generated target']
order += [x for x in labels if x.startswith(('TabDDPM full', 'TabDDPM X-only'))]
assert len(order) == 22 and len(set(order)) == 22
results = {}
for path in (BASE / 'jobs').glob('*/result.json'):
    result = json.loads(path.read_text())
    assert result['original_checkpoint_tensors_exact'] and result['test_ids_and_targets_match']
    results[result['record_relative_path']] = (path, result)
complete = len(results) == len(inventory)
status = {'completed': len(results), 'planned': len(inventory), 'complete': complete,
          'updated_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat()}
(OUT / 'status.json').write_text(json.dumps(status, indent=2))
audit = []
for item in inventory:
    result = results.get(item['record_relative_path'])
    audit.append({**{k: v for k, v in item.items() if k != 'report_cells'},
                  'status': 'completed' if result else 'pending',
                  'old_test_selection_metric': result[1]['original_test_scores'][item['selection_metric']] if result else '',
                  'new_test_selection_metric': result[1]['test_scores'][item['selection_metric']] if result else '',
                  'result': result[0].relative_to(BASE).as_posix() if result else ''})
with (OUT / 'checkpoint_selection.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
    writer.writeheader()
    writer.writerows(audit)

for mode in ('synthetic', 'mix'):
    with (BASE / (mode + '.csv')).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        original = {r['configuration']: r for r in reader}
    rows_out = [original[label].copy() for label in order]
    for row in rows_out:
        for panel, info in PANELS.items():
            record_col = panel + '_record'
            if record_col not in row or not row[record_col]:
                continue
            old_relative = row[record_col]
            score_fields = [(info['metric'], panel + '_' + info['metric'] + ('' if info['metric'] in UNSCALED else '_percent'))]
            if info['metric'] == 'f1_binary':
                score_fields.append(('f1_macro', panel + '_f1_macro_percent'))
            if info['metric'] == 'r2':
                score_fields.append(('d2_absolute_error', panel + '_d2_absolute_error'))
            if old_relative in results:
                result_path, result = results[old_relative]
                row[record_col] = '../' + result_path.relative_to(BASE).as_posix()
                for metric, column in score_fields:
                    row[column] = f"{result['test_scores'][metric] * (1 if metric in UNSCALED else 100):.8f}"
                if info['metric'] == 'r2':
                    row[panel + '_d2_metadata'] = row[record_col]
            else:
                row[record_col] = ''
                for _, column in score_fields:
                    row[column] = ''
                if info['metric'] == 'r2':
                    row[panel + '_d2_metadata'] = ''
    basename = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '') + '_val_f1'
    target_csv = OUT / (basename + '.csv')
    with target_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
        for label, row in original.items():
            if label.startswith('Paper '):
                writer.writerow(row)
    table = [f'# {mode.capitalize()} training: development F1 / R² checkpoint selection', '',
             f"Status: {len(results)}/{len(inventory)} distinct source runs re-evaluated. " + ('Complete.' if complete else 'Pending cells are blank; these tables are provisional.'), '',
             'Checkpoint rule: highest real-development binary F1 for Adult; highest macro F1 for Census KDD, Covertype, MNIST12 and MNIST28; highest raw-scale development R² for News and Housing. Ties select the earliest epoch. Selection is restricted to each original completed history; training budgets, splits, thresholds, synthetic tables and losses are preserved.', '',
             'Every displayed test result was evaluated again on the server. MNIST panels continue to display test accuracy; binary and regression panels retain both displayed metrics. News and Housing select the same checkpoints under R² and squared error. Census retains its weighted loss. Mix concatenates all original training rows and 100,000 synthetic rows.', '',
             'Sources: G:/SummerResearch_Retained_20261013_2221. Original server input files are checked against retained records and saved artifact hashes. Historical generated tables are reused; generators and labelers are not refitted.', '',
             'Some older synthetic-table hashes come from the October 13 audit rather than training-time records. Later hashes alone cannot certify historical training-time bytes; exact history and checkpoint reproduction are additionally required for recovered states.', '',
             'Caveats remain: Adult TabDDPM has an unresolved historical generator-source defect; MNIST12 TVAE retains the documented feature-collision caveat. These alternate evaluations do not clear either caveat. Known broken configurations excluded from the retained snapshot remain excluded.', '',
             'Paper references are unchanged and use different splits and classifier averages. References in binary panels apply to binary F1; regression references apply to R². Original comparison files are preserved.', '']
    values, macro, d2 = {}, {}, {}
    for row in rows_out:
        label = row['configuration']
        values[label], macro[label], d2[label] = {}, {}, {}
        for panel, info in PANELS.items():
            metric = info['metric']
            field = panel + '_' + metric + ('' if metric in UNSCALED else '_percent')
            values[label][panel] = float(row[field]) if row.get(field) else None
            macro[label][panel] = float(row[panel + '_f1_macro_percent']) if row.get(panel + '_f1_macro_percent') else None
            d2[label][panel] = float(row[panel + '_d2_absolute_error']) if row.get(panel + '_d2_absolute_error') else None
    for dataset in ('adult', 'covertype', 'mnist12', 'mnist28', 'census_kdd', 'news', 'california_housing'):
        panels = [p for p, info in PANELS.items() if info['dataset'] == dataset and info['metric'] != 'nmae_sigma']
        table += [f'## {dataset}', '', '| Configuration | Seed 42 | Seed 43 |', '|---|---:|---:|']
        for label in order:
            cells = []
            for panel in panels:
                info = PANELS[panel]
                value = values[label][panel]
                if value is None:
                    cells.append('Pending')
                elif info['metric'] == 'r2':
                    cells.append(f'{value:.4f} / {d2[label][panel]:.4f}')
                elif info['metric'] == 'f1_binary':
                    cells.append(f'{value:.2f}% / {macro[label][panel]:.2f}%')
                else:
                    cells.append(f'{value:.2f}%')
            table.append('| ' + ' | '.join([label, *cells]) + ' |')
        for paper in ('CTGAN', 'TVAE', 'Real'):
            cells = []
            for panel in panels:
                info = PANELS[panel]
                ref = info['paper'].get(paper)
                cells.append('Not reported' if ref is None else
                             f'{ref:.3f} / —' if info['metric'] == 'r2' else
                             f'{ref:.1f}%' + (' / —' if info['metric'] == 'f1_binary' else ''))
            table.append('| ' + ' | '.join([f'Paper {paper} (reference)', *cells]) + ' |')
        table.append('')
    (OUT / (basename + '.md')).write_text('\n'.join(table) + '\n', encoding='utf-8')
    if '--tables-only' in sys.argv:
        continue
    if not complete:
        raise RuntimeError('Final figures require every planned test evaluation to complete')
    source = (BASE / 'plot_recipe.py').read_text(encoding='utf-8')
    block = textwrap.dedent(source[source.index('    plt.rcParams.update('):source.index('    print(target_csv)')])
    block = block.replace('for divider in (0.5, 2.5, 5.5, 8.5, 11.5, 14.5, 15.5, 16.5, 19.5):',
                          'for divider in (0.5, 1.5, 4.5, 7.5, 8.5, 11.5, 14.5, 15.5, 18.5):')
    block = block.replace('Each row pairs one dataset: seed 42 left, seed 43 right · empty panels have no completed results',
                          'Alternate checkpoints: highest validation F1 / R² · seed 42 left, seed 43 right')
    block = block.replace("Census: weighted loss + dev macro-F1 selection. News: fresh log-target rerun, no normalization; raw-scale dev MSE selection. Paper CTGAN is labeled 'TGAN'.",
                          'Selection: Adult binary F1; Census/Covertype/MNIST macro F1; News/Housing R². Best epoch within each original completed history.')
    block = block.replace('MNIST: ten-class evaluator repaired. Housing: fresh generators retain coordinates. Seeds may share the same holdout.',
                          'Historical caveats retained: Adult TabDDPM generator source; MNIST12 TVAE feature collisions. Seeds may share the same holdout.')
    rows = [(label, labels[label] if mode == 'synthetic' or labels[label].endswith('real_original')
             else labels[label].removesuffix('_synthetic') + '_mix') for label in order]
    namespace = {**recipe, 'rows': rows, 'values': values, 'macro_values': macro, 'd2_values': d2,
                 'train_option': mode, 'basename': basename, 'DEST': OUT}
    exec(compile(block, str(BASE / 'plot_recipe.py'), 'exec'), namespace)
if complete and '--tables-only' not in sys.argv:
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.csv', '.md', '.png')}
    (OUT / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2))
print(json.dumps(status), flush=True)
