"""Build manuscript tables and vector figures from saved, small artifacts only."""
from pathlib import Path
import csv
import hashlib
import json
import re
from statistics import mean
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modeling.constants import method_parts

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

frozen = ROOT / 'output/statistical_two_contrasts_20261010/frozen_inputs/scores.csv'
meta = json.loads((frozen.parent.parent / 'analysis_metadata.json').read_text())
assert digest(frozen) == meta['source_hashes']['scores.csv']
scores = read_csv(frozen)
assert len({r['record'] for r in scores}) == 406
index = {}
for row in scores:
    key = tuple(row[k] for k in ('dataset', 'seed', 'generator', 'source', 'labeler', 'mode', 'metric'))
    assert key not in index, key
    index[key] = row
d2_path = frozen.parent / 'regression_d2_scores.csv'
assert digest(d2_path) == meta['source_hashes']['regression_d2_scores.csv']
d2_scores = read_csv(d2_path)
assert len(d2_scores) == 116
for row in d2_scores:
    base = tuple(row[k] for k in ('dataset', 'seed', 'generator', 'source', 'labeler', 'mode'))
    original = index[base + ('r2',)]
    assert row['record_sha256'] == original['record_sha256']
    assert row['record'] == original['record']
    index[base + ('d2_absolute_error',)] = row
snapshot = json.loads((HERE / 'tabddpm_snapshot.json').read_text(encoding='utf-8-sig'))
ddpm_d2_evidence = json.loads((HERE / 'tabddpm_d2_snapshot.json').read_text())
assert ddpm_d2_evidence['remote_files_changed'] is False
ddpm_d2 = {row['record']: row for row in ddpm_d2_evidence['records']}
ddpm = {}
for row in snapshot['records']:
    if row['train_option'] == 'original':
        gen, source, label, mode = 'real', 'real', 'original', 'original'
    else:
        gen, source, label = method_parts(row['augment_option'])
        mode = row['train_option']
        gp = row['generator_provenance']
        assert gp['parameters']['epochs'] == 500
        assert gp['sample_size'] == 100000
    assert row['selected_dev_epoch'] is not None
    if row['dataset'] in ('news', 'california_housing'):
        norm = row['target_normalization']
        assert norm['split'] == 'test' and norm['ddof'] == 0
        assert 'nmae_sigma' in row['test_scores']
        derived = ddpm_d2[row['record']]
        assert derived['record_sha256'] == row['record_sha256']
        assert derived['target_table_sha256'] == norm['target_table_sha256']
    key = (row['dataset'], str(row['seed']), source, label, mode)
    assert key not in ddpm
    ddpm[key] = row

tasks = {
    'adult': ('Adult', ('f1_binary', 'f1_macro'), 'Binary F1 / macro F1 (\\%, higher is better)'),
    'census_kdd': ('Census KDD', ('f1_binary', 'f1_macro'), 'Binary F1 / macro F1 (\\%, higher is better)'),
    'covertype': ('Covertype', ('f1_macro',), 'Macro F1 (\\%, higher is better)'),
    'mnist12': ('MNIST12', ('accuracy',), 'Accuracy (\\%, higher is better)'),
    'mnist28': ('MNIST28', ('accuracy',), 'Accuracy (\\%, higher is better)'),
    'news': ('News', ('r2', 'd2_absolute_error'), '$R^2$ / $D^2$ (higher is better)'),
    'california_housing': ('California Housing', ('r2', 'd2_absolute_error'), '$R^2$ / $D^2$ (higher is better)'),
}
arms = [('full', 'generated', 'Generated target'),
        ('full', 'rf', 'Joint + RF'), ('full', 'xgb', 'Joint + XGB'), ('full', 'dnn', 'Joint + DNN'),
        ('xonly', 'rf', 'X-only + RF'), ('xonly', 'xgb', 'X-only + XGB'), ('xonly', 'dnn', 'X-only + DNN')]
comparisons = {mode: {r['configuration']: r for r in read_csv(ROOT / 'output/comparisons' / name)}
               for mode, name in [('synthetic', 'dataset_pipeline_comparison.csv'),
                                  ('mix', 'dataset_pipeline_comparison_mix.csv')]}
cells = {}
def values(dataset, seed, gen, source, label, mode):
    metrics = tasks[dataset][1]
    if gen == 'tabddpm':
        record = ddpm.get((dataset, str(seed), source, label, mode))
        if record is None:
            return None
        data = [float(ddpm_d2[record['record']]['d2_absolute_error'] if m == 'd2_absolute_error'
                      else record['test_scores'][m]) for m in metrics]
        record_path = record['record']
    else:
        selected = [index[(dataset, str(seed), gen, source, label, mode, m)] for m in metrics]
        data = [float(r['value']) for r in selected]
        record_path = selected[0]['record']
        cfg = ('Original data only' if mode == 'original' else
               f'{gen.upper()}, generated target' if label == 'generated' else
               f'{gen.upper()} {"full" if source == "full" else "X-only"} features + {label.upper()}')
        comparison = comparisons['synthetic' if mode == 'original' else mode][cfg]
        for metric, value in zip(metrics, data):
            column = (f'{dataset}_corrected_{"seed43_" if seed == 43 else ""}{metric}' +
                      ('_percent' if metric in ('accuracy', 'f1_binary', 'f1_macro') else ''))
            scale = 100 if metric in ('accuracy', 'f1_binary', 'f1_macro') else 1
            assert abs(float(comparison[column]) - value * scale) < 0.000001, column
    key = (dataset, seed, gen, source, label, mode)
    cells[key] = dict(dataset=dataset, seed=seed, generator=gen, source=source,
                      labeler=label, mode=mode, metrics=metrics, values=data, record=record_path)
    if 'd2_absolute_error' in metrics:
        if gen == 'tabddpm':
            cells[key]['d2_provenance'] = dict(snapshot='tabddpm_d2_snapshot.json',
                                             record_sha256=record['record_sha256'])
        else:
            derived = index[(dataset, str(seed), gen, source, label, mode, 'd2_absolute_error')]
            cells[key]['d2_provenance'] = {k: derived[k] for k in ('sidecar', 'sidecar_sha256', 'record_sha256')}
    return data

GENERATORS = ('ctgan', 'tvae', 'tabddpm')
LABELERS = ('rf', 'xgb', 'dnn')
table_summaries = []

def average_cell(dataset, gen, source, label, mode):
    samples = {seed: values(dataset, seed, gen, source, label, mode) for seed in (42, 43)}
    samples = {seed: data for seed, data in samples.items() if data is not None}
    if not samples:
        return None
    return {'means': [mean(data[j] for data in samples.values()) for j in range(len(tasks[dataset][1]))],
            'seeds': list(samples)}

def numeric_text(value, metric, delta=False):
    percent = metric in ('accuracy', 'f1_binary', 'f1_macro')
    scaled = value * 100 if percent else value
    precision = 2 if percent else 4
    rounded = round(scaled, precision)
    if rounded == 0:
        rounded = 0.0
    return format(rounded, ('+' if delta else '') + f'.{precision}f')

PANELS = [('Classification', tuple(list(tasks)[:5])),
          ('Regression', ('news', 'california_housing'))]

def panel_header(title, datasets, first_column):
    columns = 1 + sum(len(tasks[d][1]) for d in datasets)
    rows = [r'\setlength{\tabcolsep}{' + ('6pt' if len(datasets) == 5 else '16pt') + '}',
            r'\begin{tabular}{@{}l' + 'r' * (columns-1) + r'@{}}',
            r'\toprule', r'\multicolumn{' + str(columns) + r'}{@{}l}{\textbf{' + title + r'}} \\',
            r'\addlinespace']
    groups, rules, labels = [], [], []
    position = 2
    for dataset in datasets:
        count = len(tasks[dataset][1])
        name = 'Housing' if dataset == 'california_housing' else tasks[dataset][0]
        groups.append(r'\multicolumn{' + str(count) + r'}{c}{\textbf{' + name + '}}')
        rules.append(r'\cmidrule(lr){' + f'{position}-{position+count-1}' + '}')
        labels.extend({'f1_binary': 'Binary F1', 'f1_macro': 'Macro F1', 'accuracy': 'Accuracy',
                       'r2': '$R^2$', 'd2_absolute_error': '$D^2$'}[m] for m in tasks[dataset][1])
        position += count
    rows += [r'\textbf{' + first_column + '} & ' + ' & '.join(groups) + r' \\',
             ''.join(rules), ' & ' + ' & '.join(labels) + r' \\', r'\midrule']
    return rows

def top_two(results):
    """Dense ranks at displayed precision, including the real-only row."""
    ranks = {}
    for dataset in tasks:
        complete = [row[dataset] for row in results.values()
                    if row[dataset] is not None and len(row[dataset]['seeds']) == 2]
        ranks[dataset] = [sorted({float(numeric_text(r['means'][j], metric)) for r in complete},
                                reverse=True)[:2] for j, metric in enumerate(tasks[dataset][1])]
    return ranks

def render_metric(result, metric, j, ranks, delta=False):
    if result is None:
        return '---'
    text = numeric_text(result['means'][j], metric, delta)
    if len(result['seeds']) == 2:
        if ranks and float(text) == ranks[0]:
            text = r'\textbf{' + text + '}'
        elif len(ranks) > 1 and float(text) == ranks[1]:
            text = r'\underline{' + text + '}'
    if len(result['seeds']) == 1:
        text += r'$^{\dagger}$'
    return text

def panel_cells(datasets, results, ranks, delta=False):
    return [render_metric(results[d], metric, j, ranks[d][j], delta)
            for d in datasets for j, metric in enumerate(tasks[d][1])]

for mode, regime in [('synthetic', 'Synthetic-only training'), ('mix', 'Real + synthetic training')]:
    methods = [('Real only', 'real', 'real', 'original', 'original')]
    methods += [(gen.upper() if gen != 'tabddpm' else 'TabDDPM', gen, 'full', 'generated', mode)
                for gen in GENERATORS]
    methods += [(f'{gen.upper() if gen != "tabddpm" else "TabDDPM"}-X + {label.upper()}',
                 gen, 'xonly', label, mode) for gen in GENERATORS for label in LABELERS]
    results = {name: {dataset: average_cell(dataset, gen, source, label, training_mode)
                      for dataset in tasks} for name, gen, source, label, training_mode in methods}
    ranks = top_two(results)
    rows = [r'\begin{table*}[!t]', r'\centering']
    for p, (panel, datasets) in enumerate(PANELS):
        rows += panel_header(f'({chr(97+p)}) {panel}' + (r' (\%)' if p == 0 else ''), datasets, 'Method')
        columns = 1 + sum(len(tasks[d][1]) for d in datasets)
        for i, (name, gen, source, label, training_mode) in enumerate(methods):
            if not any(result is not None for result in results[name].values()):
                continue
            if i in (1, 4):
                rows += [r'\midrule', r'\multicolumn{' + str(columns) + r'}{@{}l}{\textbf{' +
                         ('Baselines: generated targets' if i == 1 else 'Our methods: X-only + predicted targets') + r'}} \\']
            elif i in (7, 10):
                rows += [r'\addlinespace']
            rows.append(name + ' & ' + ' & '.join(panel_cells(datasets, results[name], ranks)) + r' \\')
        rows += [r'\bottomrule', r'\end{tabular}']
        if p == 0:
            rows += [r'\par\medskip']
    for name, gen, source, label, training_mode in methods:
        for dataset in tasks:
            table_summaries.append({'table': mode, 'method': name, 'dataset': dataset,
                                    'metrics': tasks[dataset][1], 'result': results[name][dataset]})
    caption = (regime + r'. Means over seeds 42/43; higher is better for every metric. $D^2$ uses absolute error relative to the test-median baseline. G-X fits generator G to features only, then uses the named labeler. Bold: best; underline: second-best distinct displayed mean, including real-only and ties. $\dagger$: seed 42 only, excluded from ranking; ---: no completed entry in the October 11, 2026, 00:25 Chicago TabDDPM snapshot. TabDDPM X-only results are unavailable. MNIST12 TVAE retains the feature-collision caveat.'
               if mode == 'synthetic' else
               r'Real + synthetic training. Metrics, means, notation, and highlighting follow Table~\ref{tab:real-synthetic}. Mixing concatenates all real training rows and 100,000 synthetic rows. Daggers identify seed-42-only results.')
    rows += [r'\caption{' + caption + '}',
             r'\label{tab:real-' + mode + '}', r'\end{table*}']
    (HERE / 'tables' / f'real_{mode}.tex').write_text('\n'.join(rows) + '\n')

# Reuse verified analysis outputs; do not rerun models or change the original report.
analysis = frozen.parent.parent
analysis_hashes = json.loads((analysis / 'deliverable_hashes.json').read_text())
contrast_hashes = {}
for filename in ('classification_contrasts.csv', 'task_contrasts.csv'):
    path = analysis / filename
    assert digest(path) == analysis_hashes[path.relative_to(ROOT).as_posix()]
    contrast_hashes[filename] = digest(path)
classification = read_csv(analysis / 'classification_contrasts.csv')
task_contrasts = [row for row in read_csv(analysis / 'task_contrasts.csv')
                  if row['metric'] in ('f1_macro', 'r2', 'd2_absolute_error')]
assert len(classification) == 4 and len(task_contrasts) == 36
contrast_names = {'relabel_minus_generated': r'Relabeled $-$ generated',
                  'has_y1_minus_has_y0': r'Joint $-$ X-only'}
task_index = {}
for row in task_contrasts:
    dataset, mode, metric, contrast = (row[k] for k in ('dataset', 'training', 'metric', 'contrast'))
    assert contrast in contrast_names
    def score(seed, gen, source, label):
        return float(index[(dataset, str(seed), gen, source, label, mode, metric)]['value'])
    relabeled = [score(seed, gen, source, label) for seed in (42, 43)
                 for gen in ('ctgan', 'tvae') for source in ('full', 'xonly') for label in LABELERS]
    generated = [score(seed, gen, 'full', 'generated') for seed in (42, 43) for gen in ('ctgan', 'tvae')]
    joint = [score(seed, gen, 'full', label) for seed in (42, 43) for gen in ('ctgan', 'tvae') for label in LABELERS]
    xonly = [score(seed, gen, 'xonly', label) for seed in (42, 43) for gen in ('ctgan', 'tvae') for label in LABELERS]
    left, right = (mean(relabeled), mean(generated)) if contrast == 'relabel_minus_generated' else (mean(joint), mean(xonly))
    assert abs(left - float(row['left_mean'])) < 1e-12
    assert abs(right - float(row['right_mean'])) < 1e-12
    assert abs(left - right - float(row['native_difference'])) < 1e-12
    key = (mode, metric, dataset, contrast)
    assert key not in task_index
    task_index[key] = row
for row in classification:
    differences = [float(task_index[(row['training'], 'f1_macro', dataset, row['contrast'])]['native_difference'])
                   for dataset in PANELS[0][1]]
    assert int(row['n_datasets']) == 5 and int(row['df']) == 4
    assert abs(mean(differences) - float(row['mean'])) < 1e-12
    assert sum(value > 0 for value in differences) == int(row['wins'])

rows = [r'\begin{table*}[t]', r'\centering', r'\setlength{\tabcolsep}{8pt}',
        r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
        r'\textbf{Training} & \textbf{Contrast} & \textbf{Gain (pp)} & \textbf{95\% CI} & \textbf{Raw $p$} \\',
        r'\midrule']
for i, row in enumerate(classification):
    if i == 2:
        rows += [r'\midrule']
    training = ('Synthetic-only' if row['training'] == 'synthetic' else 'Mix') if i % 2 == 0 else ''
    gain = numeric_text(float(row['mean']), 'f1_macro', delta=True)
    interval = '$[' + numeric_text(float(row['ci1_low']), 'f1_macro', delta=True) + ', ' + numeric_text(float(row['ci1_high']), 'f1_macro', delta=True) + ']$'
    rows.append(' & '.join([training, contrast_names[row['contrast']], gain, interval,
                            f"{float(row['p_raw']):.4f}"]) + r' \\')
rows += [r'\bottomrule', r'\end{tabular}',
         r'\caption{Classification contrasts across Adult, Census KDD, Covertype, MNIST12, and MNIST28. Gains and pointwise 95\% confidence intervals are in macro-F1 percentage points. Each dataset contributes one contrast, after averaging equally over CTGAN/TVAE, RF/XGB/DNN, and seeds 42/43. Raw $p$ values use exploratory two-sided paired $t$ tests across the five dataset contrasts ($df=4$), with no multiplicity adjustment.}',
         r'\label{tab:classification-contrasts}', r'\end{table*}']
(HERE / 'tables/classification_contrasts.tex').write_text('\n'.join(rows) + '\n')

regression_path = HERE / 'statistics/regression_p_values.json'
regression = json.loads(regression_path.read_text())
assert len(regression['verified_records']) == 112 and len(regression['tests']) == 16
assert regression['resamples'] == 99999 and regression['multiplicity_adjustment'] is None
assert regression['script_sha256'] == digest(HERE / 'statistics/regression_p_values.py')
assert regression['requests_sha256'] == digest(HERE / 'statistics/requests.json')
assert regression['group_contrasts_sha256'] == digest(HERE / 'statistics/group_contrasts.csv')
assert regression['source_hashes']['frozen_inputs/scores.csv'] == digest(frozen)
assert regression['source_hashes']['frozen_inputs/regression_d2_scores.csv'] == digest(d2_path)
regression_index = {}
for test in regression['tests']:
    key = tuple(test[k] for k in ('training', 'metric', 'dataset', 'contrast'))
    assert key not in regression_index
    assert abs(test['native_difference'] - float(task_index[key]['native_difference'])) < 1e-12
    assert abs(test['prediction_difference'] - test['native_difference']) < 1e-6
    assert test['p_raw'] == (1+test['extreme_resamples'])/(1+test['resamples'])
    assert 0 < test['p_raw'] <= 1
    regression_index[key] = test

def p_text(value):
    return r'$<0.0001$' if value < 0.0001 else f'{value:.4f}'

rows = [r'\begin{table*}[t]', r'\centering', r'\setlength{\tabcolsep}{9pt}',
        r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule']
for panel, metric, heading in [('a', 'r2', r'$R^2$'), ('b', 'd2_absolute_error', r'$D^2$ absolute error')]:
    if panel == 'b':
        rows += [r'\midrule']
    rows += [r'\multicolumn{6}{c}{\textbf{(' + panel + ') ' + heading + r'}} \\',
             r'\textbf{Training} & \textbf{Contrast} & \multicolumn{2}{c}{\textbf{News}} & \multicolumn{2}{c}{\textbf{Housing}} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
             r' & & Gain & Raw $p$ & Gain & Raw $p$ \\', r'\midrule']
    for mode in ('synthetic', 'mix'):
        if mode == 'mix':
            rows += [r'\midrule']
        for i, contrast in enumerate(contrast_names):
            training = ('Synthetic-only' if mode == 'synthetic' else 'Mix') if i == 0 else ''
            entries = []
            for dataset in PANELS[1][1]:
                test = regression_index[(mode, metric, dataset, contrast)]
                entries += [numeric_text(test['native_difference'], metric, delta=True), p_text(test['p_raw'])]
            rows.append(' & '.join([training, contrast_names[contrast]] + entries) + r' \\')
rows += [r'\bottomrule', r'\end{tabular}',
         r'\caption{Regression contrasts with unadjusted, two-sided paired permutation $p$ values (99,999 group sign flips). Gains are unscaled; positive values favor the first construction. CTGAN/TVAE, RF/XGB/DNN, and seeds 42/43 receive equal weight. Tests use squared error for $R^2$ and absolute error for $D^2$ on 7,929 News and 4,128 Housing test rows. Inference is exploratory and conditional on the fitted models; it does not measure retraining variability.}',
         r'\label{tab:regression-contrasts}', r'\end{table*}']
(HERE / 'tables/regression_contrasts.tex').write_text('\n'.join(rows) + '\n')
contrast_provenance = {
    'source_hashes': contrast_hashes, 'frozen_scores_sha256': digest(frozen),
    'frozen_d2_scores_sha256': digest(d2_path), 'classification_p_values': 'unadjusted two-sided paired t; df=4',
    'confidence_intervals': 'pointwise 95%; ci1_low/ci1_high', 'scope': 'exploratory',
    'contrasts': list(contrast_names), 'classification_metric': 'f1_macro',
    'classification': [{key: row[key] for key in ('training', 'contrast', 'n_datasets', 'df', 'mean',
                                                'se', 't', 'p_raw', 'ci1_low', 'ci1_high', 'wins', 'losses')}
                       for row in classification],
    'task_contrasts': task_contrasts, 'task_contrasts_checked': len(task_contrasts),
    'regression': regression['tests'], 'regression_evidence_sha256': digest(regression_path),
    'regression_p_values': regression['method'], 'regression_inference_scope': regression['scope'],
    'regression_null_assumption': regression['null_assumption'], 'regression_resamples': regression['resamples']}
(HERE / 'tables/contrast_provenance.json').write_text(json.dumps(contrast_provenance, indent=2) + '\n')
(HERE / 'tables/summary_provenance.json').write_text(json.dumps(table_summaries, indent=2) + '\n')
table_design = {'files': ['real_synthetic.tex', 'real_mix.tex', 'classification_contrasts.tex', 'regression_contrasts.tex'],
                'table_count': 4, 'dataset_columns': list(tasks), 'aggregation': 'mean over seeds 42/43',
                'panels': ['classification', 'regression'],
                'regression_metrics': ['r2', 'd2_absolute_error'],
                'ranking': 'best bold; second-best distinct displayed mean underlined; includes real-only',
                'contrast_source_hashes': contrast_hashes, 'contrasts': list(contrast_names),
                'contrast_p_values': 'raw, unadjusted', 'contrast_intervals': 'pointwise 95%',
                'regression_raw_tests': 16, 'regression_p_values_evidence_sha256': digest(regression_path),
                'partial_results': 'marked with dagger; excluded from ranking',
                'unavailable_rows': 'omitted when every dataset is missing; status retained in captions'}

if '--tables-only' in sys.argv:
    with zipfile.ZipFile(ROOT / 'papers/AuthorKit27.zip') as kit:
        for name in ('aaai2027.sty', 'aaai2027.bst'):
            assert (HERE / name).read_bytes() == kit.read('AuthorKit27/' + name)
    validation = json.loads((HERE / 'validation.json').read_text())
    validation.pop('paper_pdf', None)
    validation['comparison_cells_checked'] = len(cells)
    validation['regression_d2_scores_sha256'] = digest(d2_path)
    validation['tabddpm_d2_snapshot_sha256'] = digest(HERE / 'tabddpm_d2_snapshot.json')
    validation['real_table_design'] = table_design
    (HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    (HERE / 'tables/cell_provenance.json').write_text(json.dumps(list(cells.values()), indent=2) + '\n')
    print(json.dumps({'tables': 4, 'source_cells_checked': len(cells), 'summary_cells': len(table_summaries)}))
    raise SystemExit(0)

# Use BIF schema, rather than inferring ordinal meaning from encoded states.
schema = [r'\begin{table}[t]', r'\centering', r'{\small', r'\begin{tabular}{lrlr}',
          r'\toprule', r'Network & Features & Target & States \\', r'\midrule']
for name, target in [('asia', 'dysp'), ('alarm', 'BP'), ('child', 'Disease'), ('insurance', 'Accident')]:
    bif = (ROOT / 'simulated_eval/oracles' / f'{name}.bif').read_text()
    nodes = re.findall(r'variable\s+(\w+)\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]', bif)
    states = dict(nodes)
    schema.append(f'{name.title()} & {len(nodes)-1} & \\texttt{{{target}}} & {states[target]}' + r' \\')
schema += [r'\bottomrule', r'\end{tabular}', '}',
           r'\caption{Categorical Bayesian-network datasets. Every non-target node is observed. Original conditional probability tables define each joint distribution.}',
           r'\label{tab:simulated-schema}', r'\end{table}']
(HERE / 'tables/simulated_schema.tex').write_text('\n'.join(schema) + '\n')

font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9,
                     'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42,
                     'axes.linewidth': .6, 'savefig.dpi': 300})
illustration_sources = []
colors = ['#2878b8', '#e17832']
fig, axes = plt.subplots(2, 3, figsize=(7, 4.1))
for j, dataset in enumerate(('gaussian', 'grid', 'ring')):
    path = ROOT / 'output/simulated_all_verified' / dataset / 'seed_7/train.csv'
    data = pd.read_csv(path)
    assert len(data) == 2000
    illustration_sources.append({'path': str(path), 'sha256': digest(path), 'rows': len(data)})
    lim = float(np.abs(data[['feature_0', 'feature_1']].to_numpy()).max()) * 1.06
    xline = np.linspace(-lim, lim, 200)
    for c in (0, 1):
        ax = axes[c, j]
        subset = data[data['feature_3'] == c]
        for y in (0, 1):
            group = subset[subset['target'] == y]
            ax.scatter(group['feature_0'], group['feature_1'], s=3.2, marker='o' if y == 0 else '^', color=colors[y], alpha=.60,
                       edgecolors='none', rasterized=False)
        ax.plot(xline, -(1 + .75 * (2*c-1)) * xline, '--', color='#172a42', linewidth=.9)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='$X_0$', ylabel='$X_1$',
               title=f'{dataset.title()}, $C={c}$', aspect='equal')
        ax.grid(True, linewidth=.35, color='#dbe2e9')
        ax.set_axisbelow(True)
        ax.tick_params(length=2, pad=2)
fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=colors[0], markersize=3, label='$Y=0$'),
                    Line2D([], [], marker='^', linestyle='none', color=colors[1], markersize=3, label='$Y=1$'),
                    Line2D([], [], linestyle='--', color='#172a42', label='Bayes boundary')],
           loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.005))
fig.subplots_adjust(left=.07, right=.99, bottom=.10, top=.89, wspace=.37, hspace=.56)
for extension in ('pdf', 'png'):
    fig.savefig(HERE / 'figures' / f'simulated_construction.{extension}')
plt.close(fig)

bif = (ROOT / 'simulated_eval/oracles/asia.bif').read_text()
edges = []
for child, parents in re.findall(r'probability\s*\(\s*(\w+)\s*\|\s*([^)]*)\)', bif):
    edges.extend((parent.strip(), child) for parent in parents.split(','))
positions = {'asia': (.13, .87), 'smoke': (.71, .87), 'tub': (.13, .62),
             'lung': (.48, .62), 'bronc': (.87, .48), 'either': (.31, .35),
             'xray': (.13, .09), 'dysp': (.65, .09)}
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for parent, child in edges:
    ax.annotate('', xy=positions[child], xytext=positions[parent],
                arrowprops=dict(arrowstyle='->', lw=.8, color='#172a42', shrinkA=12, shrinkB=12))
for name, (x, y) in positions.items():
    box = dict(boxstyle='square,pad=.2', fc='white', ec='#172a42' if name == 'dysp' else 'none', lw=.8)
    ax.text(x, y, name, ha='center', va='center', fontsize=9, bbox=box,
            fontweight='bold' if name == 'dysp' else 'normal')
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis('off')
fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
for extension in ('pdf', 'png'):
    fig.savefig(HERE / 'figures' / f'asia_graph.{extension}')
plt.close(fig)

sim_path = ROOT / 'output/simulated_all_20260930/per_run.csv'
assert digest(sim_path) == '0bc6d31a97584b71da43c5c4a14928dd8641d6bd4424f7306e0a73e2ba0c3db2'
sim = read_csv(sim_path)
assert len(sim) == 378
sim_index = {(r['dataset'], r['seed'], r['method']): r for r in sim}
assert len(sim_index) == 378
oracle = [r'\begin{table}[ht]', r'\centering', r'{\small', r'\setlength{\tabcolsep}{2pt}',
          r'\begin{tabular}{lrrrrr}', r'\toprule',
          r'Task & Acc. & $\Delta$Acc. & $\Delta h^*$ & $\Delta L_{\rm syn}$ & $\Delta L_{\rm test}$ \\',
          r'\midrule']
for dataset in ('gaussian', 'grid', 'ring', 'asia', 'alarm', 'child', 'insurance'):
    baseline = [sim_index[(dataset, str(seed), gen)] for gen in ('ctgan', 'tvae') for seed in (42, 43)]
    generated_acc = mean(float(r['accuracy']) for r in baseline) * 100
    deltas = []
    for metric in ('accuracy', 'h_star_agreement', 'l_syn', 'l_test'):
        effects = [float(sim_index[(dataset, str(seed), f'{gen}-joint-{label}')][metric])
                   - float(sim_index[(dataset, str(seed), gen)][metric])
                   for gen in ('ctgan', 'tvae') for seed in (42, 43) for label in ('rf', 'xgb', 'dnn')]
        delta = mean(effects)
        deltas.append(f'{delta * 100:+.2f}' if metric in ('accuracy', 'h_star_agreement') else f'{delta:+.4f}')
    oracle.append(dataset.title() + f' & {generated_acc:.2f} & ' + ' & '.join(deltas) + r' \\')
oracle += [r'\bottomrule', r'\end{tabular}', '}',
           r'\caption{Known-distribution benchmark. Acc. is generated-target accuracy (\%), averaged over two generators and two seeds. Changes compare joint relabeling on the same features with generated targets: RF/XGB/DNN, two generators, two seeds (12 changes per task). Accuracy and $h^*$ agreement changes are percentage points. Density changes use native scores; $L_{\rm test}$ is a refit proxy. BN scores use the full-joint epsilon convention.}',
           r'\label{tab:oracle}', r'\end{table}']
(HERE / 'tables/oracle.tex').write_text('\n'.join(oracle) + '\n')
template_hashes = {}
with zipfile.ZipFile(ROOT / 'papers/AuthorKit27.zip') as kit:
    for name in ('aaai2027.sty', 'aaai2027.bst'):
        assert (HERE / name).read_bytes() == kit.read('AuthorKit27/' + name)
        template_hashes[name] = digest(HERE / name)
validation = {'template_unchanged': template_hashes, 'real_records': 406,
              'real_scores_sha256': digest(frozen), 'comparison_cells_checked': len(cells),
              'regression_d2_scores_sha256': digest(d2_path),
              'tabddpm_d2_snapshot_sha256': digest(HERE / 'tabddpm_d2_snapshot.json'),
              'real_table_design': table_design,
              'tabddpm_snapshot_chicago': snapshot['snapshot_chicago'],
              'tabddpm_records': len(snapshot['records']), 'simulated_records': len(sim),
              'simulated_scores_sha256': digest(sim_path), 'illustration_sources': illustration_sources,
              'illustration_is_original_process': True, 'training_performed': False}
(HERE / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
(HERE / 'tables/cell_provenance.json').write_text(json.dumps(list(cells.values()), indent=2) + '\n')
print(json.dumps(validation, indent=2))
