"""Simulated-data tables in the real-data layout (datasets as columns, methods as rows).

Reads only output/simulated_all_20260930/per_run.csv (378 evaluations, hash-checked).
Training is synthetic-only; every row is evaluated by a fresh downstream DNN.
"""
import csv
import hashlib
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / 'output/simulated_all_20260930/per_run.csv'
assert hashlib.sha256(SIM.read_bytes()).hexdigest() == \
    '0bc6d31a97584b71da43c5c4a14928dd8641d6bd4424f7306e0a73e2ba0c3db2'
with open(SIM, newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 378
index = {(r['dataset'], int(r['seed']), r['method']): r for r in rows}
assert len(index) == 378

DATASETS = ('gaussian', 'grid', 'ring', 'asia', 'alarm', 'child', 'insurance')
SEEDS = (42, 43)
METHODS = [('Real only', 'original')]
for gen in ('ctgan', 'tvae'):
    METHODS.append((f'{gen.upper()}, generated $Y$', gen))
    for source, tag in (('joint', 'joint'), ('features', 'X-only')):
        for labeler in ('rf', 'xgb', 'dnn'):
            METHODS.append((f'{gen.upper()} {tag} + {labeler.upper()}', f'{gen}-{source}-{labeler}'))


def value(dataset, method, metric):
    return mean(float(index[(dataset, seed, method)][metric]) for seed in SEEDS)


def header(title_cells):
    return [r'\textbf{Method} & ' + ' & '.join(r'\textbf{' + d.title() + '}' for d in DATASETS) + r' \\',
            r'\midrule']


def panel(title, metric, scale, digits, highlight, reference=None):
    lines = [r'\multicolumn{8}{@{}l}{\textbf{' + title + r'}} \\', r'\addlinespace[2pt]']
    values = {method: [value(d, method, metric) * scale for d in DATASETS] for _, method in METHODS}
    shown = {m: [round(v, digits) for v in vals] for m, vals in values.items()}
    ranks = [sorted({shown[m][j] for m in shown}, reverse=True)[:2] for j in range(len(DATASETS))]
    if reference:
        lines.append(reference[0] + ' & ' + ' & '.join(f'{v:.{digits}f}' for v in reference[1]) + r' \\')
        lines.append(r'\midrule')
    for i, (name, method) in enumerate(METHODS):
        if i == 1:
            lines.append(r'\midrule')
        elif i == 8:
            lines.append(r'\addlinespace')
        cells = []
        for j, v in enumerate(shown[method]):
            t = f'{v:.{digits}f}'
            if highlight and v == ranks[j][0]:
                t = r'\textbf{' + t + '}'
            elif highlight and v == ranks[j][1]:
                t = r'\underline{' + t + '}'
            cells.append(t)
        lines.append(name + ' & ' + ' & '.join(cells) + r' \\')
    return lines


def table(panels, caption, label):
    lines = [r'\begin{table*}[p]', r'\centering', r'\small', r'\setlength{\tabcolsep}{7pt}',
             r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule']
    lines += header(None)
    for k, p in enumerate(panels):
        if k:
            lines.append(r'\midrule')
        lines += p
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


bayes = [value(d, 'original', 'accuracy_ceiling') * 100 for d in DATASETS]
prediction = table(
    [panel('(a) Test accuracy (\\%)', 'accuracy', 100, 2, True, ('Bayes accuracy', bayes)),
     panel('(b) Agreement with the Bayes rule $h^*$ (\\%)', 'h_star_agreement', 100, 2, True)],
    r'Simulated data, synthetic-only training: prediction. Each training table (10,000 rows) trains a '
    r'fresh downstream DNN that is scored on the same independent 10,000-row original test set; means '
    r'over seeds 42/43. (a) Accuracy against the observed test targets; the Bayes accuracy is 90\% for '
    r'the three mixed datasets and the mean maximum oracle posterior for the Bayesian networks. '
    r'(b) Fraction of test rows on which the DNN predicts the Bayes decision $h^*(x)$. Construction '
    r'names follow Table~\ref{tab:valbest-synthetic}. Bold/underline: best/second-best displayed value, '
    r'including real only.', 'tab:sim-prediction')
fidelity = table(
    [panel(r'(a) $L_{\rm syn}$: mean $\log p$ of the training rows under the true distribution',
           'l_syn', 1, 2, False),
     panel(r'(b) $L_{\rm test}$: mean $\log q$ of real test rows, $q$ refitted to the training table',
           'l_test', 1, 2, False)],
    r'Simulated data: fidelity of each training table to the true joint distribution $p$ of '
    r'$(X, C, Y)$ or of all network nodes. Both are mean log densities, but they answer different '
    r'questions. $L_{\rm syn}$ asks whether the synthetic rows are plausible under $p$ and rewards '
    r'rows concentrated in high-density regions even if modes are missed, so values above real only signal '
    r'concentration rather than better fidelity. $L_{\rm test}$ fits a fixed '
    r'density family to the table and asks whether it explains real test rows, so it penalizes missing '
    r'modes and missing target noise. Real only gives the reference value for each task. Means over '
    r'seeds 42/43; each task has its own log-density scale, so values are not comparable '
    r'across tasks. Bayesian-network scores use $\log(p+10^{-8})$ on the full joint probability.',
    'tab:sim-fidelity')
(HERE / 'tables/sim_prediction.tex').write_text(prediction)
(HERE / 'tables/sim_fidelity.tex').write_text(fidelity)


def effect(metric, source):
    return {d: mean(float(index[(d, s, f'{g}-{source}-{l}')][metric]) - float(index[(d, s, g)][metric])
                    for g in ('ctgan', 'tvae') for s in SEEDS for l in ('rf', 'xgb', 'dnn'))
            for d in DATASETS}


for metric in ('accuracy', 'h_star_agreement', 'l_syn', 'l_test'):
    print(metric, {d: round(v, 4) for d, v in effect(metric, 'joint').items()})
