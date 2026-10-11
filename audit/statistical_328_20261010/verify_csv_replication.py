"""Focused scalar/CSV checks; no statistical model fitting or experiments."""
import csv
import hashlib
import itertools
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'output/statistical_analysis_328_20261010'
BUNDLE = REPORT / 'csv_replication'
AUDIT = ROOT / 'audit/statistical_328_20261010'
CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
ARMS = ['synthetic', 'mix']


def read(name):
    with (BUNDLE / name).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), (a, b)


def avg(rows, metric):
    assert rows
    return statistics.mean(float(r[metric]) for r in rows)


tables = json.loads((ROOT / 'tmp/statistical_328_csv_replication/tables.json').read_text(encoding='utf-8'))
numeric_cells = 0
for table in tables:
    with (BUNDLE / table['filename']).open(encoding='utf-8', newline='') as handle:
        matrix = list(csv.reader(handle))
    assert matrix[0] == table['headers']
    assert len(matrix) == len(table['rows']) + 1
    assert all(len(row) == len(table['headers']) for row in matrix)
    for row, expected in zip(matrix[1:], table['rows']):
        for actual, original in zip(row, expected):
            if original is None:
                assert actual == ''
            elif isinstance(original, (int, float)):
                # Exact binary-float equality after parsing, not a tolerance:
                # every exported numeric scalar retains its precision.
                assert float(actual) == original, (table['filename'], actual, original)
                numeric_cells += 1
            else:
                assert actual == original, (table['filename'], actual, original)

files = sorted(BUNDLE.glob('*.csv'))
assert len(files) == 21
manifest = read('manifest.csv')
assert len(manifest) == 20
assert {r['file'] for r in manifest} == {p.name for p in files if p.name != 'manifest.csv'}
for row in manifest:
    target = BUNDLE / row['file']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']
    records = read(row['file'])
    assert len(records) == int(row['rows'])
    assert len(records[0]) == int(row['columns'])

inputs = {arm: read(f'{arm}_input.csv') for arm in ARMS}
for arm, rows in inputs.items():
    assert len(rows) == 210
    assert len({r['record'] for r in rows}) == 210
    factorial = [r for r in rows if r['task'] == 'classification' and r['row_role'] == 'relabeled']
    actual = {(r['dataset'], int(r['seed']), r['generator'], int(r['has_y']), r['labeler']) for r in factorial}
    expected = set(itertools.product(CLASS, [42, 43], ['ctgan', 'tvae'], [0, 1], ['rf', 'xgb', 'dnn']))
    assert actual == expected and len(factorial) == 120
    assert sum(r['task'] == 'classification' and r['row_role'] == 'generated_target_baseline' for r in rows) == 20
    assert sum(r['task'] == 'regression' for r in rows) == 60
    for row in rows:
        if row['row_role'] == 'real_only_reference':
            assert row['generator'] == row['has_y'] == row['labeler'] == ''
        elif row['row_role'] == 'generated_target_baseline':
            assert row['has_y'] == '1' and row['labeler'] == ''
        else:
            assert row['labeler'] in ['rf', 'xgb', 'dnn']
        if row['task'] == 'classification':
            assert all(row[m] == '' for m in ['r2', 'mae', 'nmae_sigma', 'd2_absolute_error', 'sigma_y', 'median_y', 'baseline_mae', 'd2_prediction_mae'])
        else:
            assert row['macro_F1'] == ''
            near(row['nmae_sigma'], float(row['mae']) / float(row['sigma_y']))
            near(row['d2_absolute_error'], 1 - float(row['d2_prediction_mae']) / float(row['baseline_mae']))

sets = [{r['record'] for r in inputs[arm]} for arm in ARMS]
assert len(sets[0] | sets[1]) == 406
assert len(sets[0] & sets[1]) == 14
assert all(r['row_role'] == 'real_only_reference' for r in inputs['synthetic'] if r['record'] in sets[1])
by_record = {r['record']: r for arm in ARMS for r in inputs[arm]}
assert {r['record'] for r in read('provenance.csv')} == set(by_record)

for row in read('matched_relabeling_scores.csv'):
    left = by_record[row['record']]
    right = by_record[row['record_generated']]
    assert left['row_role'] == 'relabeled' and right['row_role'] == 'generated_target_baseline'
    assert all(left[k] == right[k] for k in ['dataset', 'seed', 'generator'])
    near(row['macro_F1'], left['macro_F1'])
    near(row['value_generated'], right['macro_F1'])
    near(row['difference'], float(left['macro_F1']) - float(right['macro_F1']))

dataset_results = {(r['analysis'], r['contrast'], r['dataset']): r for r in read('dataset_effects.csv')}
planned = read('planned_contrasts.csv')
for arm in ARMS:
    for name in ['relabel_minus_generated', 'has_y1_minus_has_y0', 'has_y1_minus_generated', 'has_y0_minus_generated']:
        deltas, lefts, rights = [], [], []
        for dataset in CLASS:
            rows = [r for r in inputs[arm] if r['dataset'] == dataset]
            relabel = [r for r in rows if r['row_role'] == 'relabeled']
            generated = [r for r in rows if r['row_role'] == 'generated_target_baseline']
            means = {h: avg([r for r in relabel if r['has_y'] == str(h)], 'macro_F1') for h in [0, 1]}
            pairs = {'relabel_minus_generated': (avg(relabel, 'macro_F1'), avg(generated, 'macro_F1')),
                     'has_y1_minus_has_y0': (means[1], means[0]),
                     'has_y1_minus_generated': (means[1], avg(generated, 'macro_F1')),
                     'has_y0_minus_generated': (means[0], avg(generated, 'macro_F1'))}
            left, right = pairs[name]
            recorded = dataset_results[(arm, name, dataset)]
            near(left, recorded['left']); near(right, recorded['right']); near(left-right, recorded['difference'])
            lefts.append(left); rights.append(right); deltas.append(left-right)
        recorded = next(r for r in planned if r['analysis'] == arm and r['contrast'] == name)
        mean = statistics.mean(deltas); sd = statistics.stdev(deltas); se = sd/math.sqrt(5)
        near(recorded['left_mean'], statistics.mean(lefts)); near(recorded['right_mean'], statistics.mean(rights))
        near(recorded['score_change'], mean); near(recorded['mean'], mean)
        near(recorded['percentage_points'], 100*mean)
        near(recorded['relative_percent'], 100*mean/statistics.mean(rights))
        near(recorded['t'], mean/se); near(recorded['cohen_dz'], mean/sd)
        # Student t(.975, 4), used in the existing report's unadjusted intervals.
        near(recorded['ci_low'], mean-2.7764451051977987*se)
        near(recorded['ci_high'], mean+2.7764451051977987*se)
        signs = list(itertools.product([-1, 1], repeat=5))
        exact = sum(abs(statistics.mean(a*b for a,b in zip(sign, deltas))) >= abs(mean)-1e-12 for sign in signs)/32
        near(recorded['p_exact'], exact)

for row in read('labeler_effects.csv'):
    deltas = []
    for dataset in CLASS:
        rows = [r for r in inputs[row['analysis']] if r['dataset'] == dataset and r['generator'] == row['generator']]
        left = [r for r in rows if r['row_role'] == 'relabeled' and r['has_y'] == row['has_y'] and r['labeler'] == row['labeler']]
        right = [r for r in rows if r['row_role'] == 'generated_target_baseline']
        deltas.append(avg(left, 'macro_F1') - avg(right, 'macro_F1'))
    near(row['mean'], statistics.mean(deltas))

for row in read('regression_summary.csv'):
    rows = [r for r in inputs[row['analysis']] if r['dataset'] == row['dataset']]
    for column, subset in {
        'original': [r for r in rows if r['row_role'] == 'real_only_reference'],
        'generated': [r for r in rows if r['row_role'] == 'generated_target_baseline'],
        'has_y0': [r for r in rows if r['row_role'] == 'relabeled' and r['has_y'] == '0'],
        'has_y1': [r for r in rows if r['row_role'] == 'relabeled' and r['has_y'] == '1'],
    }.items():
        near(row[column], avg(subset, row['metric']))

for filename, raw_column, adjusted in [('factorial_effects.csv', 'p', 'p_holm'), ('planned_contrasts.csv', 'p_t', 'p_t_holm'),
                                       ('planned_contrasts.csv', 'p_exact', 'p_exact_holm'), ('labeler_effects.csv', 'p_t', 'p_t_holm')]:
    for arm in ARMS:
        rows = [r for r in read(filename) if r['analysis'] == arm]
        sorted_rows = sorted(rows, key=lambda r: float(r[raw_column]))
        previous = 0
        for i, row in enumerate(sorted_rows):
            previous = max(previous, min(1, (len(rows)-i)*float(row[raw_column])))
            near(row[adjusted], previous)

significance_rows = 0
for filename in ['anova_results.csv', 'joint_tests.csv', 'factorial_effects.csv', 'planned_contrasts.csv', 'labeler_effects.csv']:
    for row in read(filename):
        p = float(row['p_for_significance']) if row['p_for_significance'] else 1
        stars = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '.' if p < .10 else ''
        assert row['significance'] == stars
        assert int(row['significant_05']) == int(p < .05)
        significance_rows += 1

summary = dict(csv_files=21, rows_per_input=210, classification_model_rows_per_analysis=120,
               distinct_run_records=406, shared_real_only_references=14,
               regression_sidecars_verified=116, exact_numeric_roundtrip_cells=numeric_cells,
               paired_baseline_rows_checked=240, dataset_differences_checked=40,
               regression_summary_rows_checked=12, significance_rows_checked=significance_rows,
               contrast_arithmetic='passed', holm_families='passed',
               original_statistical_results_preserved=True, manifest_hashes='passed',
               model_refits_or_experiments_performed=False)
(AUDIT / 'csv_replication_validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
archive = REPORT / 'real_data_328_csv_replication.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
    for target in files:
        package.write(target, f'csv_replication/{target.name}')
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    assert len(package.namelist()) == 21
    for target in files:
        assert package.read(f'csv_replication/{target.name}') == target.read_bytes()
print(json.dumps(summary))
print(f'Created {archive}')
