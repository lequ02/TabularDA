"""Downstream score vs generator fit time, per dataset and training regime.

Inputs: fit_times.csv (collect_fit_times.py), comparisons/scores.csv, and the Adult TabDDPM
posterior-fix scores (adult/tabddpm_fix_20261030/scores.csv) replacing the held pre-fix Adult rows.
Relabeled points average RF/XGB/DNN; all points average seeds 42/43.
Credit is excluded: most arms have binary F1 = 0 (all-negative predictions), as in the comparison panels.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRIC = {'adult': ('test_f1_binary', 'binary F1'), 'census_kdd': ('test_f1_binary', 'binary F1'),
          'covertype': ('test_f1_macro', 'macro F1'),
          'mnist12': ('test_accuracy', 'accuracy')}
TITLE = {'adult': 'Adult', 'census_kdd': 'Census KDD', 'credit': 'Credit', 'covertype': 'Covertype',
         'mnist12': 'MNIST12'}
COLOR = {'ctgan': '#2a78d6', 'tvae': '#eb6834', 'tabddpm': '#1baf7a'}
NAME = {'ctgan': 'CTGAN', 'tvae': 'TVAE', 'tabddpm': 'TabDDPM'}
CONS = {'generated': ('o', 'full (X,Y), generated Y'), 'full_relabel': ('s', 'full (X,Y), relabeled Y'),
        'xonly_relabel': ('^', 'X-only, relabeled Y')}

times = pd.read_csv(HERE / 'fit_times.csv')
scores = pd.read_csv(ROOT / 'comparisons/scores.csv')
fix = pd.read_csv(ROOT / 'adult/tabddpm_fix_20261030/scores.csv')
scores = pd.concat([scores[~((scores.dataset == 'adult') & (scores.generator == 'tabddpm'))],
                    fix[fix.generator == 'tabddpm']], ignore_index=True)
scores = scores[scores.dataset.isin(METRIC)]
scores['score'] = [getattr(r, METRIC[r.dataset][0]) for r in scores.itertuples()]
scores['cons'] = None
scores.loc[(scores.construction == 'full') & (scores.labeler == 'generated'), 'cons'] = 'generated'
relabel = scores.labeler.isin(['rf', 'xgb', 'dnn'])
scores.loc[relabel & (scores.construction == 'full'), 'cons'] = 'full_relabel'
scores.loc[relabel & (scores.construction == 'xonly'), 'cons'] = 'xonly_relabel'

rows = []
for (ds, gen, mode, cons), g in scores.dropna(subset=['cons']).groupby(['dataset', 'generator', 'mode', 'cons']):
    t = times[(times.dataset == ds) & (times.generator == gen) &
              (times.construction == ('xonly' if cons == 'xonly_relabel' else 'full'))]
    if t.empty:
        continue
    assert sorted(g.seed.unique()) == [42, 43] and sorted(t.seed) == [42, 43], (ds, gen, mode, cons)
    rows.append(dict(dataset=ds, generator=gen, mode=mode, construction=cons, n_runs=len(g),
                     score=g.score.mean(), fit_hours=t.fit_seconds.mean() / 3600))
points = pd.DataFrame(rows)
points.to_csv(HERE / 'tradeoff_points.csv', index=False)
real = scores[scores.generator == 'real'].groupby('dataset').score.mean()

datasets = list(METRIC)
fig, axes = plt.subplots(2, len(datasets), figsize=(17, 7), constrained_layout=True)
for i, mode in enumerate(['synthetic', 'mix']):
    for j, ds in enumerate(datasets):
        ax = axes[i, j]
        p = points[(points.dataset == ds) & (points['mode'] == mode)]
        for gen, gp in p.groupby('generator'):
            gp = gp.set_index('construction')
            if {'generated', 'full_relabel'} <= set(gp.index):
                ax.plot([gp.fit_hours['generated']] * 2, [gp.score['generated'], gp.score['full_relabel']],
                        color=COLOR[gen], lw=1, alpha=0.6, zorder=1)
            for cons, r in gp.iterrows():
                ax.scatter(r.fit_hours, r.score, marker=CONS[cons][0], s=70, color=COLOR[gen],
                           edgecolor='white', linewidth=1.5, zorder=3)
        ax.axhline(real[ds], color='#8a8984', ls='--', lw=1, zorder=0)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 60:.0f} min' if v < 1 else f'{v:g} h'))
        ax.grid(True, color='#e6e5e0', lw=0.6)
        ax.tick_params(colors='#52514e', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#c3c2b7')
        if i == 0:
            ax.set_title(TITLE[ds], fontsize=11)
        if i == 1:
            ax.set_xlabel('generator fit time (log scale)', fontsize=9, color='#52514e')
        ax.set_ylabel(f'{mode}: test {METRIC[ds][1]}', fontsize=9, color='#52514e')

handles = [plt.Line2D([], [], ls='', marker='o', ms=8, color=c, label=NAME[g]) for g, c in COLOR.items()]
handles += [plt.Line2D([], [], ls='', marker=m, ms=8, color='#52514e', label=l) for m, l in CONS.values()]
handles += [plt.Line2D([], [], ls='--', color='#8a8984', label='real-only baseline')]
fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.07))
fig.suptitle('Downstream score vs generator fit time (500 epochs; seeds 42/43 averaged; relabeled = mean of RF/XGB/DNN)',
             fontsize=12)
fig.savefig(HERE / 'fit_time_tradeoff.png', dpi=150, bbox_inches='tight')
print(points.to_string())
