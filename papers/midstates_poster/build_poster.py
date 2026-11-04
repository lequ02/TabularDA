"""Editable Midstates poster (48 x 36 in, landscape PPTX) with its figures.

Reads only saved small files, after checking their hashes:
- comparisons/20261031_adult_tabddpm_fix/valbest/scores_valbest.csv: 602 validation-best
  evaluations, with the 28 posterior-fix Adult TabDDPM arms (October 31, 2026).
- output/simulated_all_20260930/per_run.csv: 378 simulated-benchmark evaluations.
No fitting. Statistics follow output/statistical_anova_gdrive_20261019/analyze.py. The Ring panel
draws a fresh illustrative sample from simulated_eval/oracle.py (no generator or labeler).
"""
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulated_eval.oracle import Oracle  # noqa: E402

SCORES = ROOT / 'comparisons/20261031_adult_tabddpm_fix/valbest/scores_valbest.csv'
SIM = ROOT / 'output/simulated_all_20260930/per_run.csv'
HASHES = {SCORES: 'eebcc5d1c64bcb18043287451ce15f41f0a983072d1a90510d117f2999208780',
          SIM: '0bc6d31a97584b71da43c5c4a14928dd8641d6bd4424f7306e0a73e2ba0c3db2'}
for path, expected in HASHES.items():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
# `build_poster.py --figures DIR` writes only the figures, to DIR, and leaves the PPTX alone.
FIGURES_ONLY = sys.argv[1:2] == ['--figures']
FIG = HERE / (sys.argv[2] if FIGURES_ONLY else 'figures')
FIG.mkdir(exist_ok=True)
POSTER = HERE / 'ffts_midstates_poster.pptx'

FONT = 'Calibri'
for name in ('calibri.ttf', 'calibrib.ttf', 'calibrii.ttf', 'calibriz.ttf'):
    font_manager.fontManager.addfont(str(Path('C:/Windows/Fonts') / name))
plt.rcParams.update({'font.family': FONT, 'font.size': 26, 'mathtext.fontset': 'custom',
                     'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold',
                     'mathtext.bfit': f'{FONT}:italic:bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.6,
                     'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.major.size': 8,
                     'ytick.major.size': 8, 'axes.titleweight': 'bold', 'axes.titlesize': 28,
                     'savefig.dpi': 200})
NAVY, INK, GREY, RED, PALE = '#12355B', '#1E2A36', '#8A96A3', '#C0392B', '#EEF2F7'
GEN_COLOR = {'ctgan': '#D1495B', 'tvae': '#2E86AB', 'tabddpm': '#E3A21A'}
GEN_NAME = {'ctgan': 'CTGAN', 'tvae': 'TVAE', 'tabddpm': 'TabDDPM'}
CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
REG = ['california_housing', 'news']
NAMES = dict(adult='Adult', census_kdd='Census KDD', covertype='Covertype', mnist12='MNIST12',
             mnist28='MNIST28', news='News', california_housing='Housing')
GEN = ['ctgan', 'tvae', 'tabddpm']
LABEL = ['rf', 'xgb', 'dnn']

# ---------------------------------------------------------------- real-data statistics
scores = pd.read_csv(SCORES)
assert len(scores) == 602
assert set(scores.retained_status) <= {'passes_current_validity_checks', 'usable_with_documented_caveat',
                                       'passes_runner_completion_check_posterior_fix'}
adult_ddpm = scores[(scores.dataset == 'adult') & (scores.generator == 'tabddpm')]
assert len(adult_ddpm) == 28 and adult_ddpm.retained_status.eq('passes_runner_completion_check_posterior_fix').all()
experimental = scores[scores['mode'].isin(['synthetic', 'mix'])].copy()
experimental['treatment'] = np.where(experimental.labeler.eq('generated'), 'generated',
                                     'hybrid_' + experimental.construction)
assert len(experimental) == 588
real = scores[scores.generator.eq('real')].groupby('dataset')[['test_f1_macro', 'test_r2']].mean()


def cells(mode, metric, datasets):
    part = experimental[experimental['mode'].eq(mode) & experimental.dataset.isin(datasets)]
    assert part[metric].notna().all()
    return part.groupby(['dataset', 'generator', 'treatment'])[metric].mean().unstack()


def paired(left, right):
    diff = np.asarray(left) - np.asarray(right)
    half = stats.t.ppf(.975, len(diff) - 1) * stats.sem(diff)
    return dict(diff=diff.mean(), lo=diff.mean() - half, hi=diff.mean() + half,
                p=float(stats.ttest_rel(left, right).pvalue), base=np.mean(right))


# Primary family of analyze.py: relabeled (mean of joint and X-only) minus generated, and joint
# minus X-only, in both regimes, Holm over the four tests. FFTS (X-only minus generated) and
# generator-specific contrasts are descriptive.
results = {}
for mode in ('synthetic', 'mix'):
    table = cells(mode, 'test_f1_macro', CLASS)
    for gen in ['all'] + GEN:
        sub = table if gen == 'all' else table.xs(gen, level='generator', drop_level=False)
        w = sub.groupby('dataset').mean().reindex(CLASS)
        results[(mode, gen, 'relabeled')] = paired((w.hybrid_full + w.hybrid_xonly) / 2, w.generated)
        results[(mode, gen, 'joint_xonly')] = paired(w.hybrid_full, w.hybrid_xonly)
        results[(mode, gen, 'ffts')] = paired(w.hybrid_xonly, w.generated)
primary = [(m, 'all', q) for m in ('synthetic', 'mix') for q in ('relabeled', 'joint_xonly')]
for key, adjusted in zip(primary, multipletests([results[k]['p'] for k in primary], method='holm')[1]):
    results[key]['holm'] = adjusted
r = results

ffts_syn = cells('synthetic', 'test_f1_macro', CLASS)
wins = {mode: int((t.hybrid_xonly > t.generated).sum())
        for mode, t in (('synthetic', ffts_syn), ('mix', cells('mix', 'test_f1_macro', CLASS)))}
ctgan_tvae = experimental[experimental.generator.isin(['ctgan', 'tvae']) & experimental.dataset.isin(CLASS)
                          & experimental['mode'].eq('synthetic')]
ct_gen = ctgan_tvae[ctgan_tvae.treatment.eq('generated')].test_f1_macro.mean()
ct_ffts = ctgan_tvae[ctgan_tvae.treatment.eq('hybrid_xonly')].test_f1_macro.mean()
tvae_ffts = ffts_syn.xs('tvae', level='generator').hybrid_xonly
tvae_gap = (real.test_f1_macro.reindex(CLASS) - tvae_ffts.reindex(CLASS))
generated_short = int((ffts_syn.generated.groupby('dataset').max().reindex(CLASS)
                       < real.test_f1_macro.reindex(CLASS)).sum())
reg_cells = {mode: cells(mode, 'test_r2', REG) for mode in ('synthetic', 'mix')}
housing = {mode: t.xs('california_housing', level='dataset') for mode, t in reg_cells.items()}
news_max = experimental[experimental.dataset.eq('news')].groupby(['generator', 'treatment', 'mode']).test_r2.mean().abs().max()

# ---------------------------------------------------------------- simulated statistics
sim = pd.read_csv(SIM)
assert len(sim) == 378
SIM_SETS = ['gaussian', 'grid', 'ring', 'asia', 'alarm', 'child', 'insurance']
sim_idx = sim.set_index(['dataset', 'seed', 'method'])


def sim_effect(metric):
    return pd.Series({d: np.mean([sim_idx.loc[(d, s, f'{g}-features-{l}'), metric] - sim_idx.loc[(d, s, g), metric]
                                  for g in ('ctgan', 'tvae') for s in (42, 43) for l in LABEL])
                      for d in SIM_SETS})


sim_acc, sim_ltest = sim_effect('accuracy'), sim_effect('l_test')
sim_gap = pd.Series({d: np.mean([sim_idx.loc[(d, s, 'original'), 'accuracy'] -
                                 sim_idx.loc[(d, s, f'{g}-features-{l}'), 'accuracy']
                                 for g in ('ctgan', 'tvae') for s in (42, 43) for l in LABEL])
                     for d in SIM_SETS})


def pts(x, sign=True):
    return (f'{100 * x:+.1f}' if sign else f'{100 * x:.1f}').replace('-', '\u2212')


def num(x, fmt):
    return format(x, fmt).replace('-', '\u2212')


def pval(p):
    return f'{p:.2f}' if p >= 0.1 else f'{p:.3f}'


# ---------------------------------------------------------------- figures
def dumbbell(datasets, metric, scale, xlim, xticks, xlabel, size, path, legend):
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True, gridspec_kw=dict(wspace=0.08))
    positions, labels = [], []
    for i, d in enumerate(datasets):
        for k, gen in enumerate(GEN):
            positions.append(i * 4.0 + k); labels.append(GEN_NAME[gen])
    for ax, mode in zip(axes, ('synthetic', 'mix')):
        table = cells(mode, metric, datasets) * scale
        for i, d in enumerate(datasets):
            y0 = i * 4.0
            ax.axhspan(y0 - 0.6, y0 + 2.6, color='#F3F5F8' if i % 2 == 0 else 'white', zorder=0)
            ax.plot([real.loc[d, metric] * scale] * 2, [y0 - 0.55, y0 + 2.55], color=INK, linewidth=3.2,
                    linestyle=(0, (2.5, 1.5)), zorder=2)
            for k, gen in enumerate(GEN):
                g, f = table.loc[(d, gen), 'generated'], table.loc[(d, gen), 'hybrid_xonly']
                ax.annotate('', xy=(f, y0 + k), xytext=(g, y0 + k), zorder=3,
                            arrowprops=dict(arrowstyle='-|>', color=GEN_COLOR[gen], linewidth=3.4,
                                            mutation_scale=30, shrinkA=9, shrinkB=4))
                ax.scatter([g], [y0 + k], s=260, facecolor='white', edgecolor=GEN_COLOR[gen], linewidth=3.2,
                           zorder=4)
                ax.scatter([f], [y0 + k], s=300, color=GEN_COLOR[gen], zorder=4)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.grid(axis='x', color='#D5DAE0', linewidth=1.2, zorder=0)
        if legend:
            ax.set_title('Synthetic only' if mode == 'synthetic' else 'Original + synthetic', pad=16)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(positions, labels, fontsize=24)
    for i, d in enumerate(datasets):
        axes[0].text(-0.33, i * 4.0 + 1, NAMES[d], transform=axes[0].get_yaxis_transform(), rotation=90,
                     ha='center', va='center', fontsize=26, fontweight='bold', color=NAVY)
    axes[0].set_ylim(len(datasets) * 4.0 - 1.3, -0.8)
    bottom = 0.15 if legend else 0.3
    if legend:
        handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=17, markerfacecolor='white',
                              markeredgecolor=INK, markeredgewidth=3, label='Baseline'),
                   plt.Line2D([], [], marker='o', linestyle='', markersize=17, color=INK,
                              label='LTA (mean of RF/XGB/DNN)'),
                   plt.Line2D([], [], color=INK, linewidth=3.2, linestyle=(0, (2.5, 1.5)), label='Original data only')]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=25,
                   bbox_to_anchor=(0.575, 0.0), handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.93 if legend else 0.97, bottom=bottom)
    fig.savefig(path)
    plt.close(fig)


dumbbell(CLASS, 'test_f1_macro', 100, (18, 100), [20, 40, 60, 80, 100], 'Test macro F1 (%)', (14.6, 11.2),
         FIG / 'dumbbell_classification.png', legend=True)
dumbbell(REG, 'test_r2', 1, (-0.1, 0.9), [0, 0.2, 0.4, 0.6, 0.8], 'Test $R^2$', (14.6, 4.6),
         FIG / 'dumbbell_regression.png', legend=False)

fig, ax = plt.subplots(figsize=(7.4, 9.0))
for mode, marker in (('synthetic', 'o'), ('mix', 's')):
    part = experimental[experimental['mode'].eq(mode) & experimental.dataset.isin(CLASS)
                        & experimental.labeler.isin(LABEL)]
    wide = part.groupby(['dataset', 'generator', 'labeler', 'construction']).test_f1_macro.mean().unstack() * 100
    for gen in GEN:
        g = wide.xs(gen, level='generator')
        ax.scatter(g.xonly, g.full, s=170, marker=marker, facecolor=GEN_COLOR[gen] if mode == 'synthetic' else 'none',
                   edgecolor=GEN_COLOR[gen], linewidth=2.4, alpha=0.9, zorder=3)
ax.plot([0, 100], [0, 100], color=INK, linewidth=2, zorder=2)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel('LTA: $G$ fits $\mathbfit{X}$ only (macro F1 %)')
ax.set_ylabel('$G$ fits ($\mathbfit{X}$, $\mathbfit{y}$), $L$ relabels $\mathbfit{y}$ (%)')
ax.grid(color='#E3E7EB', linewidth=1.1, zorder=0)
ax.set_aspect('equal')
legend = [plt.Line2D([], [], marker='o', linestyle='', markersize=15, color=GEN_COLOR[g], label=GEN_NAME[g])
          for g in GEN]
legend += [plt.Line2D([], [], marker='o', linestyle='', markersize=15, color=GREY, label='Synthetic only'),
           plt.Line2D([], [], marker='s', linestyle='', markersize=15, markerfacecolor='none',
                      markeredgecolor=GREY, markeredgewidth=2.4, label='Original + synthetic')]
fig.legend(handles=legend[:3], loc='lower center', bbox_to_anchor=(0.55, 0.055), ncol=3, frameon=False,
           fontsize=23, handletextpad=0.2, columnspacing=1.0)
fig.legend(handles=legend[3:], loc='lower center', bbox_to_anchor=(0.55, -0.01), ncol=2, frameon=False,
           fontsize=23, handletextpad=0.2, columnspacing=1.0)
fig.subplots_adjust(left=0.2, right=0.97, top=0.98, bottom=0.27)
fig.savefig(FIG / 'joint_vs_xonly.png')
plt.close(fig)

oracle = Oracle(dataset='ring')
sample = oracle.sample(2400, np.random.default_rng(7))
fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6), sharey=True, gridspec_kw=dict(wspace=0.08))
grid = np.linspace(-2.6, 2.6, 2)
for ax, c in zip(axes, (0, 1)):
    s = 2 * c - 1
    part = sample[sample.feature_3.eq(c)]
    slope = -(1 + oracle.beta * s)
    ax.fill_between(grid, slope * grid, 2.6, color='#FCE9E6', zorder=0)
    for y, color in ((0, '#2E86AB'), (1, '#D1495B')):
        rows = part[part.target.eq(y)]
        ax.scatter(rows.feature_0, rows.feature_1, s=22, color=color, alpha=0.75, linewidth=0, zorder=2)
    ax.plot(grid, slope * grid, color=INK, linewidth=3, linestyle=(0, (4, 2)), zorder=3)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal')
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.set_xlabel('$X_0$')
    ax.set_title(f'S = {"+1" if s > 0 else chr(0x2212) + "1"}', pad=10)
axes[0].set_ylabel('$X_1$')
handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=14, color='#D1495B', label='$y$ = 1'),
           plt.Line2D([], [], marker='o', linestyle='', markersize=14, color='#2E86AB', label='$y$ = 0'),
           plt.Line2D([], [], color=INK, linewidth=3, linestyle=(0, (4, 2)), label='Bayes rule h*')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=23, bbox_to_anchor=(0.53, -0.01),
           handletextpad=0.3, columnspacing=1.4)
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.25)
fig.savefig(FIG / 'ring_distribution.png')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0), sharey=True, gridspec_kw=dict(wspace=0.12))
ypos = np.arange(len(SIM_SETS))
axes[0].barh(ypos, sim_acc * 100, height=0.6, color=NAVY)
axes[0].set_xlabel('Accuracy change (points)')
axes[0].set_title('Prediction: better')
axes[1].barh(ypos, sim_ltest, height=0.6, color=np.where(sim_ltest < 0, RED, '#7FA7D1'))
axes[1].axvline(0, color=INK, linewidth=1.8)
axes[1].set_xlabel('$L_{\\mathrm{test}}$ change (nats per row)')
axes[1].set_title('Coverage of original data: mixed')
axes[0].set_yticks(ypos, [d.title() for d in SIM_SETS])
axes[0].invert_yaxis()
for ax in axes:
    ax.grid(axis='x', color='#E3E7EB', linewidth=1.1)
    ax.tick_params(axis='y', length=0)
fig.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.18)
fig.savefig(FIG / 'simulated_effects.png')
plt.close(fig)

if FIGURES_ONLY:
    sys.exit()

# ---------------------------------------------------------------- PPTX helpers
def rgb(hex_color):
    return RGBColor.from_string(hex_color.lstrip('#'))


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(48), Inches(36)
slide = prs.slides.add_slide(prs.slide_layouts[6])
BODY, NOTE, HEAD = 30, 25, 48


def runs_into(paragraph, runs, size, color):
    for item in runs:
        text, style = (item, {}) if isinstance(item, str) else item
        run = paragraph.add_run()
        run.text = text
        font = run.font
        font.name = FONT
        font.size = Pt(style.get('size', size))
        font.bold = style.get('b', False)
        font.italic = style.get('i', False)
        font.color.rgb = rgb(style.get('color', color))
        if style.get('sub'):
            run._r.get_or_add_rPr().set('baseline', '-25000')
        if style.get('sup'):
            run._r.get_or_add_rPr().set('baseline', '30000')


def text(x, y, w, h, paragraphs, size=BODY, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, after=10):
    """paragraphs: list of run lists; a run list starting with '*' is a bullet."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, runs in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        p.line_spacing = 1.05
        bullet = runs and runs[0] == '*'
        if bullet:
            runs = runs[1:]
            ppr = p._p.get_or_add_pPr()
            ppr.set('marL', str(Emu(Inches(0.42))))
            ppr.set('indent', str(-Emu(Inches(0.42))))
            char = OxmlElement('a:buChar')
            char.set('char', '\u2022')
            ppr.append(char)
        runs_into(p, runs, size, color)
    return box


def rect(x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, width=2.5):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(width)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def node(x, y, w, h, paragraphs, fill='#FFFFFF', line=NAVY, size=27, color=INK):
    s = rect(x, y, w, h, fill, line, MSO_SHAPE.ROUNDED_RECTANGLE)
    s.adjustments[0] = 0.12
    frame = s.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.1)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, runs in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        runs_into(p, runs, size, color)
    return s


def arrow(x1, y1, x2, y2, color=NAVY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(4)
    tail = OxmlElement('a:tailEnd')
    tail.set('type', 'triangle'); tail.set('w', 'med'); tail.set('len', 'med')
    c.line._get_or_add_ln().append(tail)


def head(x, y, w, title):
    text(x, y, w, 0.85, [[(title, B)]], size=HEAD, color=NAVY)
    rect(x, y + 0.9, w, 0.07, RED)
    return y + 1.25


def picture(path, x, y, w):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


B = {'b': True}
SUB = {'sub': True}
XS = [('X', {'i': True}), ('syn', SUB)]
YS = [('Y', {'i': True}), ('syn', SUB)]
X, Y = ('X', {'i': True}), ('Y', {'i': True})

# ---------------------------------------------------------------- banner
rect(0, 5.35, 48, 0.25, NAVY)
text(7.0, 0.45, 34.0, 1.6, [[('Synthetic Data Generation via Factorized Feature-Target Synthesis', B)]], size=88,
     color=NAVY, align=PP_ALIGN.CENTER)
text(7.0, 2.25, 34.0, 1.2, [['Generate as many rows as you need, label them with a model trained on real data, '
                             'and the synthetic data can stand in for the real data']],
     size=40, color=RED, align=PP_ALIGN.CENTER)
text(7.0, 3.2, 34.0, 1.65, [['Minh Le, Thuy Do'], ['Luther College']], size=50, color=INK, align=PP_ALIGN.CENTER,
     after=0)
LOGO_W, LOGO_H, LOGO_TOP = 5.62, 4.0, 0.6
slide.shapes.add_picture(str(HERE / 'logos/luther_college.png'), Inches(48 - 1.0 - LOGO_W), Inches(LOGO_TOP),
                         Inches(LOGO_W), Inches(LOGO_H))
left_logo = HERE / 'logos/left.png'
if left_logo.exists():
    slide.shapes.add_picture(str(left_logo), Inches(1.0), Inches(LOGO_TOP), height=Inches(LOGO_H))
else:
    holder = rect(1.0, LOGO_TOP, LOGO_W, LOGO_H, '#FFFFFF', GREY)
    holder.line.dash_style = 4
    runs_into(holder.text_frame.paragraphs[0], ['Logo'], 40, GREY)
    holder.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

TOP = 6.2
LX, LW = 0.7, 13.4
MX, MW = 14.8, 18.4
RX, RW = 33.9, 13.4

# ---------------------------------------------------------------- left column
y = head(LX, TOP, LW, 'Why factorize?')
text(LX, y, LW, 5.6, [
    ['A supervised table has features ', X, ' and a target ', Y,
     '. A model trained on synthetic data can only learn the ', X, '\u2013', Y, ' relation that the table contains.'],
    ['Generators such as ', ('CTGAN', B), ', ', ('TVAE', B), ' and ', ('TabDDPM', B), ' model (', X, ', ', Y,
     ') jointly and treat ', Y, ' as one more column. Nothing ties a generated ', Y, ' to its generated ', X, '.'],
    [('FFTS', B), ' generates the features, then assigns every target with a model fitted to the real data.'],
], after=14)

y = head(LX, 11.0, LW, 'Method')
text(LX, y, LW, 0.6, [[('Standard: joint synthesis', B)]], size=28, color=GREY)
row = y + 0.75
node(LX, row, 3.0, 1.7, [['Real data'], ['(', X, ', ', Y, ')']], fill=PALE)
node(LX + 4.1, row, 3.6, 1.7, [['Generator'], ['fits (', X, ', ', Y, ')']])
node(LX + 8.8, row, 4.6, 1.7, [['Synthetic (', *XS, ', ', *YS, ')'], [*YS, ' is generated']], fill='#F4F4F4', line=GREY)
arrow(LX + 3.0, row + 0.85, LX + 4.1, row + 0.85)
arrow(LX + 7.7, row + 0.85, LX + 8.8, row + 0.85, color=GREY)
line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(LX), Inches(row + 2.25), Inches(LX + LW),
                                  Inches(row + 2.25))
line.line.color.rgb = rgb(GREY); line.line.width = Pt(2); line.line.dash_style = 4
text(LX, row + 2.5, LW, 0.6, [[('FFTS: factorized feature\u2013target synthesis', B)]], size=28, color=RED)
row2 = row + 3.3
node(LX, row2 + 0.6, 3.0, 2.6, [['Real data'], ['(', X, ', ', Y, ')']], fill=PALE)
node(LX + 4.1, row2, 3.6, 1.6, [[('1. ', B), 'Generator'], ['fits ', X, ' only']])
node(LX + 4.1, row2 + 2.2, 3.6, 2.0, [[('2. ', B), 'Labeler ', ('h', {'i': True})], ['fits ', Y, ' | ', X],
                                      [('RF, XGB or DNN', {'size': 23})]])
node(LX + 8.8, row2 + 0.6, 4.6, 2.6, [[('Synthetic', B)], [('(', B), ('X', {'b': True, 'i': True}), ('syn', {'b': True, 'sub': True}),
                                                          (', ', B), ('h', {'b': True, 'i': True}), ('(', B),
                                                          ('X', {'b': True, 'i': True}), ('syn', {'b': True, 'sub': True}),
                                                          ('))', B)],
                                      ['any number of rows']], fill='#FCE9E6', line=RED)
arrow(LX + 3.0, row2 + 1.3, LX + 4.1, row2 + 0.8)
arrow(LX + 3.0, row2 + 2.5, LX + 4.1, row2 + 3.2)
arrow(LX + 7.7, row2 + 0.8, LX + 8.8, row2 + 1.4, color=RED)
arrow(LX + 7.7, row2 + 3.2, LX + 8.8, row2 + 2.4, color=RED)
text(LX + 7.75, row2 + 0.0, 1.0, 0.6, [XS], size=26, color=RED, align=PP_ALIGN.CENTER)
text(LX + 7.75, row2 + 3.4, 1.0, 0.6, [[('h', {'i': True})]], size=26, color=RED, align=PP_ALIGN.CENTER)

y = row2 + 4.6
text(LX, y, LW, 3.0, [
    ['*', 'The generator samples synthetic features ', *XS, ' (100,000 rows here).'],
    ['*', 'The labeler ', ('h', {'i': True}), ' learns ', Y, ' from ', X, ' on the real training rows.'],
    ['*', 'Ablation: keep a joint generator\u2019s ', *XS, ' and replace its ', *YS, ' with ',
     ('h', {'i': True}), '(', *XS, ').'],
], after=8)

y = head(LX, y + 2.4, LW, 'Evaluation')
text(LX, y, LW, 5.6, [
    ['*', ('Classification: ', B), 'Adult, Census KDD, Covertype, MNIST12, MNIST28 (macro F1).'],
    ['*', ('Regression: ', B), 'California Housing, News (', ('R', {'i': True}), ('2', {'sup': True}), ').'],
    ['*', 'A downstream neural network trains on ', ('synthetic only', B), ' or ', ('real + synthetic', B),
     ', and is tested on held-out real rows.'],
    ['*', 'Generators, labelers and preprocessing see training rows only; checkpoints are chosen on a real '
          'development split. 2 seeds; 588 downstream runs.'],
], after=8)

rl = r[('synthetic', 'all', 'relabeled')]
y = head(LX, y + 4.6, LW, 'Takeaways')
text(LX, y, LW, 35.3 - y, [
    ['*', 'Labeling synthetic features with a model trained on real data is a simple add-on to any generator.'],
    ['*', 'Exception: TabDDPM on MNIST. Its digits drift from real ones, so a real-data labeler mislabels them.'],
    ['*', ('Caveats: ', B), 'the wins over real data likely distill a strong labeler into many rows, and the '
          'real-only network is weak on Housing. With 5 tasks, the primary test (',
     f'{pts(rl["diff"])} points, ', ('p', {'i': True}), f' = {pval(rl["p"])}) is not significant after Holm '
     'correction (', ('p', {'i': True}), f' = {pval(rl["holm"])}).'],
], size=27, after=8)

# ---------------------------------------------------------------- middle column
y = head(MX, TOP, MW, 'Result: synthetic data that rivals real data')
callout = rect(MX, y, MW, 3.25, PALE)
text(MX + 0.4, y + 0.2, MW - 0.8, 2.9, [
    [('Synthetic-only training, classification: ', {}), ('FFTS adds ', B),
     (f'{pts(r[("synthetic", "all", "ffts")]["diff"])} macro-F1 points', {'b': True, 'color': RED}),
     (' over generated targets', B)],
    [(f'95% CI {pts(r[("synthetic", "all", "ffts")]["lo"], False)} to {pts(r[("synthetic", "all", "ffts")]["hi"], False)}; '
      f'better in {wins["synthetic"]} of 15 task\u2013generator pairs; CTGAN and TVAE: '
      f'{pts(ct_ffts - ct_gen)} points ({100 * (ct_ffts - ct_gen) / ct_gen:.0f}% relative).', {'size': NOTE})],
    [('With TVAE features, FFTS comes within ', {}),
     (f'{100 * tvae_gap.max():.1f} points', B), (' of real-data training on every classification task, and ', {}),
     ('beats it', B), (f' on Covertype ({pts(-tvae_gap["covertype"])}) and Housing (', {}),
     ('R', {'i': True}), ('2', {'sup': True}),
     (f' {housing["synthetic"].loc["tvae", "hybrid_xonly"]:.2f} vs. {real.loc["california_housing", "test_r2"]:.2f}).', {})],
], size=29, align=PP_ALIGN.CENTER, after=10)
y += 3.55
picture(FIG / 'dumbbell_classification.png', MX, y, MW)
y += MW * 11.2 / 14.6
text(MX, y + 0.05, MW, 0.6, [[('Regression', B), ' (same layout; dashed line = real data only)']], size=NOTE,
     color=NAVY)
picture(FIG / 'dumbbell_regression.png', MX, y + 0.6, MW)
y += 0.6 + MW * 4.6 / 14.6 + 0.15
text(MX, y, MW, 1.0, [[
    'Open circle: the generator\u2019s own targets. Filled circle: FFTS. Averaged over 2 seeds (FFTS also over '
    f'3 labelers). Generated targets fall short of real data on {generated_short} of 5 classification tasks for '
    'every generator; News has no predictable signal for any method, including real data (|',
    ('R', {'i': True}), ('2', {'sup': True}), f'| \u2264 {news_max:.2f}).']], size=NOTE)

# ---------------------------------------------------------------- right column
y = head(RX, TOP, RW, 'Does the generator need Y?')
picture(FIG / 'joint_vs_xonly.png', RX, y, 6.2)
jx = r[('synthetic', 'all', 'joint_xonly')]
text(RX + 6.5, y + 0.1, RW - 6.5, 7.4, [
    ['Each point is one task, generator and labeler. On the diagonal, fitting the generator to (', X, ', ', Y,
     ') instead of ', X, ' alone makes no difference.'],
    ['Joint minus FFTS: ', (f'{pts(jx["diff"])} points', B),
     f' (95% CI {pts(jx["lo"])} to {pts(jx["hi"])}; Holm ', ('p', {'i': True}), f' = {pval(jx["holm"])}).'],
    ['Fitting the generator to ', Y, ' as well adds no detectable benefit.'],
], size=28, after=14)

y = head(RX, y + 6.2 * 9.0 / 7.4 + 0.3, RW, 'Simulated data with known truth')
text(RX, y, RW, 2.3, [
    [('Ring', B), ': 8 Gaussian clusters in (', ('X', {'i': True}), ('0', SUB), ', ', ('X', {'i': True}), ('1', SUB),
     '). A binary feature ', ('S', {'i': True}), ' sets the slope of the Bayes boundary ', ('h', {'i': True}),
     '*; 10% of labels are flipped. Six more datasets: Gaussian, Grid and 4 Bayesian networks.'],
], size=NOTE)
picture(FIG / 'ring_distribution.png', RX, y + 2.0, RW)
y += 2.0 + RW * 6.6 / 13.4
picture(FIG / 'simulated_effects.png', RX, y, RW)
y += RW * 6.0 / 13.4 + 0.1
rect(RX, y, RW, 2.9, PALE)
text(RX + 0.3, y + 0.2, RW - 0.6, 2.6, [
    [('L', {'b': True, 'i': True}), ('test', {'b': True, 'sub': True}), (': fit a simple density model to the synthetic '
     'table, then average its log-density over real test rows. Higher means the synthetic table covers the real data '
     'better.', {})],
    [f'FFTS raises accuracy on all 7 datasets ({pts(sim_acc.min())} to {pts(sim_acc.max())} points, within '
     f'{100 * sim_gap.max():.1f} of real data) but lowers ', ('L', {'i': True}), ('test', SUB),
     f' on {int((sim_ltest < 0).sum())}: hard labels erase the target\u2019s natural randomness.'],
], size=NOTE, after=8)

prs.save(POSTER)

print('ffts', {k: round(v, 4) for k, v in r[('synthetic', 'all', 'ffts')].items()})
print('relabeled syn', {k: round(v, 4) for k, v in r[('synthetic', 'all', 'relabeled')].items()})
print('joint syn', {k: round(v, 4) for k, v in jx.items()})
print('wins', wins, 'ct/tvae', round(100 * (ct_ffts - ct_gen), 2), 'generated_short', generated_short)
print('tvae gap', (100 * tvae_gap).round(2).to_dict())
print('housing', {m: t.round(3).to_dict() for m, t in housing.items()})
print('news max', round(news_max, 4), 'sim acc', (100 * sim_acc).round(2).to_dict())
print('sim L_test', sim_ltest.round(3).to_dict(), 'gap', (100 * sim_gap).round(2).to_dict())
