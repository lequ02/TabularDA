"""Build the October 30, 2026 draft exhibits from validation-best results.

Reads only saved small files: comparisons/scores_valbest.csv (602 evaluations,
completed October 18, 2026) and the October 19 ANOVA results. No fitting.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORES = ROOT / 'comparisons/scores_valbest.csv'
ANOVA = ROOT / 'output/statistical_anova_gdrive_20261019'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


manifest = json.loads((ANOVA / 'inputs/manifest.json').read_text())
assert digest(SCORES) == manifest['files']['scores_valbest.csv']
assert digest(ANOVA / 'inputs/scores_valbest.csv') == digest(SCORES)

rows = read_csv(SCORES)
assert len(rows) == 602
index = {}
for row in rows:
    key = (row['dataset'], int(row['seed']), row['generator'], row['construction'], row['labeler'], row['mode'])
    assert key not in index
    index[key] = row

tasks = {
    'adult': ('Adult', ('f1_binary', 'f1_macro')),
    'census_kdd': ('Census KDD', ('f1_binary', 'f1_macro')),
    'covertype': ('Covertype', ('f1_macro',)),
    'mnist12': ('MNIST12', ('accuracy',)),
    'mnist28': ('MNIST28', ('accuracy',)),
    'news': ('News', ('r2', 'd2_absolute_error')),
    'california_housing': ('Housing', ('r2', 'd2_absolute_error')),
}
PANELS = [('Classification', ('adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28')),
          ('Regression', ('news', 'california_housing'))]
LABELS = {'f1_binary': 'Bin.\\ F1', 'f1_macro': 'Mac.\\ F1', 'accuracy': 'Acc.',
          'r2': '$R^2$', 'd2_absolute_error': '$D^2$'}
GEN_NAMES = {'ctgan': 'CTGAN', 'tvae': 'TVAE', 'tabddpm': 'TabDDPM'}
PERCENT = ('accuracy', 'f1_binary', 'f1_macro')
# Flagged cells: shown with a dagger and excluded from ranking.
FLAGGED = {('adult', 'tabddpm')}


def cell(dataset, gen, construction, labeler, mode):
    """Two-seed means, verifying caveats and that both seeds exist."""
    records = [index[(dataset, seed, gen, construction, labeler, mode)] for seed in (42, 43)]
    for record in records:
        assert record['retained_status'] in ('passes_current_validity_checks',
                                             'usable_with_documented_caveat',
                                             'completed_with_source_bug_caveat')
        if record['retained_status'] == 'completed_with_source_bug_caveat':
            assert (dataset, gen) in FLAGGED
        if dataset == 'news':
            norm = json.loads(record['news_target_normalization_json'])
            assert norm['split'] == 'test' and norm['ddof'] == 0
    return [mean(float(r['test_' + m]) for r in records) for m in tasks[dataset][1]]


def text(value, metric):
    if metric in PERCENT:
        return f'{value * 100:.2f}'
    rounded = round(value, 3)
    return f'{0.0 if rounded == 0 else rounded:.3f}'


def methods(mode):
    out = [('Real only', 'real', 'real', 'observed', 'original')]
    for gen in ('ctgan', 'tvae', 'tabddpm'):
        name = GEN_NAMES[gen]
        out.append((f'{name}, generated $Y$', gen, 'full', 'generated', mode))
        for construction, tag in (('full', 'joint'), ('xonly', 'X-only')):
            for labeler in ('rf', 'xgb', 'dnn'):
                out.append((f'{name} {tag} + {labeler.upper()}', gen, construction, labeler, mode))
    return out


summary = []


def build_table(groups, caption, label, filename, colsep, place='!t', size='small'):
    """groups: list of (mode, datasets); one column per mode/dataset/metric."""
    cols = [(mode, d, j, metric) for mode, datasets in groups for d in datasets
            for j, metric in enumerate(tasks[d][1])]
    results = {(mode, d, i): cell(d, gen, construction, labeler, m)
               for mode, datasets in groups for d in datasets
               for i, (_, gen, construction, labeler, m) in enumerate(methods(mode))}
    ranks = {}
    for mode, d, j, metric in cols:
        shown = {float(text(results[(mode, d, i)][j], metric))
                 for i, (_, gen, *_) in enumerate(methods(mode)) if (d, gen) not in FLAGGED}
        ranks[(mode, d, j)] = sorted(shown, reverse=True)[:2]
    lines = [r'\begin{table*}[' + place + ']', r'\centering', '\\' + size, r'\setlength{\tabcolsep}{' + colsep + '}',
             r'\begin{tabular}{@{}l' + 'r' * len(cols) + r'@{}}', r'\toprule']
    if len(groups) > 1:
        head, rules, pos = [], [], 2
        for mode, datasets in groups:
            n = sum(len(tasks[d][1]) for d in datasets)
            head.append(r'\multicolumn{' + str(n) + r'}{c}{\textbf{' +
                        ('Synthetic only' if mode == 'synthetic' else 'Real + synthetic') + '}}')
            rules.append(r'\cmidrule(lr){' + f'{pos}-{pos + n - 1}' + '}')
            pos += n
        lines += [' & ' + ' & '.join(head) + r' \\', ''.join(rules)]
    head, rules, pos = [], [], 2
    for mode, datasets in groups:
        for d in datasets:
            n = len(tasks[d][1])
            head.append(r'\multicolumn{' + str(n) + r'}{c}{\textbf{' + tasks[d][0] + '}}')
            rules.append(r'\cmidrule(lr){' + f'{pos}-{pos + n - 1}' + '}')
            pos += n
    lines += [r'\textbf{Method} & ' + ' & '.join(head) + r' \\', ''.join(rules),
              ' & ' + ' & '.join(LABELS[metric] for *_, metric in cols) + r' \\', r'\midrule']
    for i, (name, gen, *_) in enumerate(methods('synthetic')):
        if i in (1, 8, 15):
            lines.append(r'\midrule' if i == 1 else r'\addlinespace')
        cells = []
        for mode, d, j, metric in cols:
            value = results[(mode, d, i)][j]
            t = text(value, metric)
            if (d, gen) in FLAGGED:
                t += r'$^{\dagger}$'
            elif float(t) == ranks[(mode, d, j)][0]:
                t = r'\textbf{' + t + '}'
            elif len(ranks[(mode, d, j)]) > 1 and float(t) == ranks[(mode, d, j)][1]:
                t = r'\underline{' + t + '}'
            cells.append(t)
            summary.append(dict(mode=mode, method=name, dataset=d, metric=metric, mean=value))
        lines.append(name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{' + caption + '}', r'\label{' + label + '}', r'\end{table*}']
    (HERE / 'tables' / filename).write_text('\n'.join(lines) + '\n')


CLASSIFICATION = ('adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28')
REGRESSION = ('news', 'california_housing')
CAPTION = (r'Classification, synthetic-only training (validation-best checkpoints; \%). Test means '
           r'over seeds 42/43; higher is better. Joint: features from the full-table generator with '
           r'$Y$ replaced by the named labeler; X-only: a separate features-only generator followed by '
           r'the labeler. Bold/underline: best/second-best distinct displayed value, including real only. '
           r'$\dagger$: Adult TabDDPM has an unresolved historical generator-source caveat and is '
           r'excluded from ranking. MNIST12 TVAE retains a documented exact held-out feature-collision '
           r'caveat; MNIST12 TabDDPM shows a diagnosed generated-input shift that is retained as a result.')
build_table([('synthetic', CLASSIFICATION)], CAPTION, 'tab:valbest-synthetic',
            'valbest_synthetic.tex', '6pt', place='p', size='footnotesize')
build_table([('mix', CLASSIFICATION)],
            r'Classification, real + synthetic training (all real training rows plus 100,000 synthetic '
            r'rows; \%). Checkpoints, notation, highlighting, and caveats follow '
            r'Table~\ref{tab:valbest-synthetic}.', 'tab:valbest-mix', 'valbest_mix.tex', '6pt', place='p', size='footnotesize')
build_table([('synthetic', REGRESSION), ('mix', REGRESSION)],
            r'Regression, both training regimes. Test means over seeds 42/43; higher is better. '
            r'$D^2$ is absolute-error skill relative to the constant test-median predictor; both '
            r'metrics are unscaled. Construction names and highlighting follow '
            r'Table~\ref{tab:valbest-synthetic}.', 'tab:valbest-regression', 'valbest_regression.tex', '6pt', place='p')


def p_text(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


QUESTION = {'hybrid_minus_generated': 'Relabeled $-$ generated',
            'full_minus_xonly': 'Joint $-$ X-only'}
SHORT = {'hybrid_minus_generated': r'relab.\ $-$ gen.', 'full_minus_xonly': 'joint $-$ X-only'}
primary = read_csv(ANOVA / 'results/primary_anova.csv')
generator = read_csv(ANOVA / 'results/generator_contrasts.csv')


def anova_row(label, row):
    return (f"{label} & {float(row['difference']) * 100:+.2f} & "
            f"[{float(row['ci_low']) * 100:+.1f}, {float(row['ci_high']) * 100:+.1f}] & "
            f"{p_text(float(row['p']))} & {p_text(float(row['p_holm']))} \\\\")


lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{2pt}',
         r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
         r'Contrast & $\Delta$ & 95\% CI & $p$ & $p_{\rm Holm}$ \\']
for mode in ('synthetic', 'mix'):
    lines += [r'\midrule', r'\multicolumn{5}{@{}l}{\emph{' +
              ('Synthetic only' if mode == 'synthetic' else 'Real + synthetic') + r'}} \\']
    lines += [anova_row(QUESTION[r['question']], r) for r in primary if r['mode'] == mode]
    lines += [anova_row(r'\quad ' + GEN_NAMES[r['generator']] + ' ' + SHORT[r['question']], r)
              for r in generator if r['mode'] == mode]
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{Repeated-measures ANOVA on five classification tasks, $F(1,4)$, macro F1 in '
          r'percentage points. Seeds, labelers (RF/XGB/DNN), and generators are averaged with equal '
          r'weight before inference; the dataset is the unit. Relabeled averages both feature sources. '
          r'Pooled rows: Holm over the four primary tests; indented generator rows: Holm over '
          r'12 tests. Intervals are unadjusted.}',
          r'\label{tab:anova}', r'\end{table}']
(HERE / 'tables/valbest_anova.tex').write_text('\n'.join(lines) + '\n')

sensitivity = read_csv(ANOVA / 'results/sensitivity_anova.csv')
COHORT = {'all_five': 'All five', 'exclude_adult': 'No Adult', 'exclude_mnist12': 'No MNIST12',
          'exclude_adult_and_mnist12': 'Neither'}
lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
         r'& & \multicolumn{2}{c}{Relabeled $-$ gen.} & \multicolumn{2}{c}{Joint $-$ X-only} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r'Checkpoint & Tasks & Synth. & Mix & Synth. & Mix \\', r'\midrule']
for selection in ('valbest', 'original'):
    for cohort in COHORT:
        values = []
        for question in ('hybrid_minus_generated', 'full_minus_xonly'):
            for mode in ('synthetic', 'mix'):
                [row] = [r for r in sensitivity if r['selection'] == selection and r['cohort'] == cohort
                         and r['question'] == question and r['mode'] == mode]
                values.append(f"{float(row['difference']) * 100:+.2f}")
        rule = 'Dev.\\ metric' if selection == 'valbest' else 'Dev.\\ loss'
        lines.append(f"{rule} & {COHORT[cohort]} & "
                     + ' & '.join(values) + r' \\')
    if selection == 'valbest':
        lines.append(r'\addlinespace')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{Sensitivity of the two primary contrasts (macro F1, percentage points) to the '
          r'checkpoint rule and to excluding caveated tasks. No contrast passes Holm correction in any cohort.}',
          r'\label{tab:sensitivity}', r'\end{table}']
assert all(float(r['p_holm']) > 0.05 for r in sensitivity)
(HERE / 'tables/valbest_sensitivity.tex').write_text('\n'.join(lines) + '\n')

# Per-dataset relabeling effect by generator, both regimes (classification macro F1; regression R^2).
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
order = list(tasks)
for ax, mode in zip(axes, ('synthetic', 'mix')):
    for k, gen in enumerate(('ctgan', 'tvae', 'tabddpm')):
        effects = []
        for d in order:
            metric = 'f1_macro' if d not in ('news', 'california_housing') else 'r2'
            scale = 100
            def score(construction, labeler):
                return mean(float(index[(d, s, gen, construction, labeler, mode)]['test_' + metric])
                            for s in (42, 43))
            relabeled = mean(score(c, l) for c in ('full', 'xonly') for l in ('rf', 'xgb', 'dnn'))
            effects.append((relabeled - score('full', 'generated')) * scale)
        xs = [i + (k - 1) * 0.26 for i in range(len(order))]
        bars = ax.bar(xs, effects, width=0.26, label=GEN_NAMES[gen],
                      color=('#1f3b73', '#5b8bd0', '#b9cbe8')[k], edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([tasks[d][0].replace('Census KDD', 'Census') for d in order], rotation=40,
                       ha='right', fontsize=7)
    ax.set_title('Synthetic only' if mode == 'synthetic' else 'Real + synthetic', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
axes[0].set_ylabel('Relabeled $-$ generated\n(points; $R^2\\times100$)', fontsize=7)
axes[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(HERE / 'figures/valbest_effects.pdf')
fig.savefig(HERE / 'figures/valbest_effects.png', dpi=200)

(HERE / 'tables/valbest_summary.json').write_text(json.dumps(dict(
    scores=str(SCORES.relative_to(ROOT)).replace('\\', '/'), scores_sha256=digest(SCORES),
    anova_manifest=manifest['files'], cells=summary), indent=1) + '\n')
print('ok', len(summary))
