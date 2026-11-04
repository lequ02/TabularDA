"""Poster v7 from the hand-edited v6 (comments in ffts_midstates_poster_v6.pdf).

"Generated y" becomes "Baseline"; the table gains an average row and a paragraph below it; Evaluation
names the baselines and splits classification/regression; Takeaways becomes Conclusion; the simulated
section gets equations. Figures come from `build_poster.py --figures figures_v5`.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from make_v6 import (GEN_NAMES, GENERATORS, NAVY, PALE, PINK, Builder, M, SUB, SUP, T, V, XS,
                     cell_text, headline_numbers, set_text)

HERE = Path(__file__).resolve().parent
SRC = HERE / 'ffts_midstates_poster_v6.pptx'
DST = HERE / 'ffts_midstates_poster_v7.pptx'
FIG = HERE / 'figures_v5'
F1, R2 = 'test_f1_macro', 'test_r2'


def averages(n):
    for metric in (F1, R2):
        for mode in ('synthetic', 'mix'):
            for key in ('generated', 'hybrid_xonly'):
                n[(metric, mode, 'avg', key)] = sum(n[(metric, mode, g, key)] for g in GENERATORS) / 3
    return n


def headline_table(slide, old):
    n = averages(headline_numbers(regression=('california_housing', 'news')))
    fmt = {F1: lambda v: f'{100 * v:.1f}', R2: lambda v: f'{v:.2f}'}
    cols = [(metric, mode, key) for metric in (F1, R2) for mode in ('synthetic', 'mix')
            for key in ('generated', 'hybrid_xonly')]
    rows = GENERATORS + ['avg']
    shape = slide.shapes.add_table(8, 9, old.left, old.top, old.width, Inches(4.0))
    shape.name = 'Headline table'
    old._element.addnext(shape._element)
    old._element.getparent().remove(old._element)
    tbl = shape.table
    tbl.first_row = tbl.horz_banding = False
    tbl.columns[0].width = Inches(2.8)
    for j in range(1, 9):
        tbl.columns[j].width = Inches(1.95)
    for i in range(8):
        tbl.rows[i].height = Inches(0.5)
        for j in range(9):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            fill = NAVY if i < 3 else PALE if i == 7 else PINK if j and cols[j - 1][2] == 'hybrid_xonly' else 'FFFFFF'
            cell.fill.fore_color.rgb = RGBColor.from_string(fill)
    white = dict(color='FFFFFF')
    tbl.cell(0, 0).merge(tbl.cell(2, 0))
    cell_text(tbl.cell(0, 0), [T('Generator', b=True, **white)], align=PP_ALIGN.LEFT)
    tbl.cell(0, 1).merge(tbl.cell(0, 4))
    cell_text(tbl.cell(0, 1), [T('Classification: macro F1 (%)', b=True, **white)])
    tbl.cell(0, 5).merge(tbl.cell(0, 8))
    cell_text(tbl.cell(0, 5), [T('Regression: ', b=True, **white), M('R', b=True, **white),
                               SUP('2', b=True, **white)])
    for j in (1, 3, 5, 7):
        tbl.cell(1, j).merge(tbl.cell(1, j + 1))
        cell_text(tbl.cell(1, j), [T('Synthetic only' if j in (1, 5) else 'Original + synthetic', **white)], size=24)
        cell_text(tbl.cell(2, j), [T('Baseline', **white)], size=22)
        cell_text(tbl.cell(2, j + 1), [T('LTA', b=True, **white)], size=22)
    for i, gen in enumerate(rows, start=3):
        avg = gen == 'avg'
        cell_text(tbl.cell(i, 0), [T('Average' if avg else GEN_NAMES[gen], b=avg)], size=24, align=PP_ALIGN.LEFT)
        for j, (metric, mode, key) in enumerate(cols, start=1):
            cell_text(tbl.cell(i, j), [T(fmt[metric](n[(metric, mode, gen, key)]), b=key == 'hybrid_xonly')],
                      size=24)
    cell_text(tbl.cell(7, 0), [T('Original data only')], size=24, align=PP_ALIGN.LEFT)
    for j, metric in ((1, F1), (5, R2)):
        tbl.cell(7, j).merge(tbl.cell(7, j + 3))
        cell_text(tbl.cell(7, j), [T(fmt[metric](n[(metric, 'real')]))], size=24)
    print({k: round(v, 4) for k, v in n.items()})
    return n


def gain(n, metric, mode):
    return n[(metric, mode, 'avg', 'hybrid_xonly')] - n[(metric, mode, 'avg', 'generated')]


def pts(x):
    return f'{100 * x:+.1f}'.replace('-', '−')


def sizes(shape, size):
    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            run.font.size = Pt(size)


def amend(slide):
    b = Builder(slide)
    sh = b.by_id
    n = headline_table(slide, sh[7])
    ct = [n[(F1, mode, g, 'hybrid_xonly')] for mode in ('synthetic', 'mix') for g in ('ctgan', 'tvae')]
    real_f1, real_r2 = n[(F1, 'real')], n[(R2, 'real')]
    assert max(ct[:2]) <= real_f1 + 0.0005 and min(ct[2:]) > real_f1  # matches it alone, beats it when added
    reg = [n[(R2, mode, g, 'hybrid_xonly')] for mode in ('synthetic', 'mix') for g in GENERATORS]
    assert min(reg) > real_r2

    # Paragraph between the table and the figure.
    s = b.clone(45, 14.8, 11.55, 18.4, 1.6, 'Table summary')
    set_text(s, [[
        T('On average, LTA beats the baseline by '), T(f'{pts(gain(n, F1, "synthetic"))} macro F1 points', b=True),
        T(' with synthetic data only and '), T(f'{pts(gain(n, F1, "mix"))}', b=True),
        T(' with original + synthetic (regression '), M('R'), SUP('2'),
        T(f': {gain(n, R2, "synthetic"):+.2f}, {gain(n, R2, "mix"):+.2f}). With CTGAN or TVAE, LTA matches '
          f'original data on its own ({100 * ct[0]:.1f}, {100 * ct[1]:.1f} vs. {100 * real_f1:.1f}) and beats '
          f'it when added to it ({100 * ct[2]:.1f}, {100 * ct[3]:.1f}). In regression, every LTA set beats '
          f'original data ('), M('R'), SUP('2'), T(f' ≥ {min(reg):.2f} vs. {real_r2:.2f}).')]])
    sizes(s, 27)
    for i in (42, 43, 44, 45):
        sh[i].top += Inches(1.9)
    set_text(sh[45], [[
        T('Table and figure: averaged over 2 seeds (LTA also over RF, XGB and DNN); table cells also average '
          'over the 5 classification or 2 regression tasks. Open circle: baseline, the generator’s own targets. '
          'Filled circle: LTA.')]])

    # Method diagram: the standard row is the baseline.
    set_text(sh[63], [[T('Baseline: joint synthesis', b=True)]])

    set_text(sh[34], [
        [T('Baselines: ', b=True), T('each generator with its own generated '), V('y'), T('; original data only.')],
        [T('Classification: ', b=True), T('Adult, Census KDD, Covertype, MNIST12, MNIST28 (macro F1).')],
        [T('Regression: ', b=True), T('California Housing, News ('), M('R'), SUP('2'), T(').')],
        [T('LTA: ', b=True), M('G'), T(' = each of the three generators (100,000 rows); '),
         M('L'), T(' = RF, XGB or DNN.')],
        [T('Scoring: ', b=True), T('a fresh neural network per training set, tested on held-out original rows.')],
        [T('Training sets: ', b=True), T('synthetic only, or synthetic added to the original data.')],
        [T('Ablation: ', b=True), M('L'), T(' relabels the '), XS(), T(' of a joint '), M('G'),
         T(' (right). 2 seeds; 588 runs.')],
    ])
    sizes(sh[34], 27)

    set_text(sh[35], [[T('Conclusion', b=True)]])
    joint = [r.text for r in sh[49].text_frame.paragraphs[1].runs]  # '+0.5 points', ' (95% CI ... Holm ', 'p', ...
    ci = joint[2].split(';')[0].lstrip(' (')
    set_text(sh[37], [
        [T('LTA beats the baseline by '), T(f'{pts(gain(n, F1, "synthetic"))} macro F1 points', b=True),
         T(f' on average ({pts(gain(n, F1, "mix"))} added to original data), as an add-on to any generator.')],
        [T('The base generator matters: with CTGAN or TVAE, LTA matches or beats original data; with '
           'TabDDPM it does not.')],
        [T('Fitting '), M('G'), T(' to '), V('y'), T(f' as well does not matter ({joint[1]}, {ci}).')],
        [T('Future work: more regression tasks (News has no predictable target for any method).')],
    ])
    for top, i in ((21.35, 32), (22.25, 33), (22.55, 34), (26.75, 35), (27.65, 36), (28.0, 37)):
        sh[i].top = Inches(top)

    # References: one line each, smaller, at the bottom of the left column.
    refs = [
        'Xu et al. Modeling tabular data using conditional GAN. NeurIPS 2019. (CTGAN, TVAE)',
        'Kotelnikov et al. TabDDPM: Modelling tabular data with diffusion models. ICML 2023.',
        'Breiman. Random forests. Machine Learning, 2001.',
        'Chen and Guestrin. XGBoost: A scalable tree boosting system. KDD 2016.',
    ]
    set_text(sh[83], [[T(f'{i}. {r}')] for i, r in enumerate(refs, start=1)])
    sizes(sh[83], 21)
    for para in sh[83].text_frame.paragraphs:
        para.space_after = Pt(2)
    for top, i, h in ((31.75, 81, None), (32.65, 82, None), (32.95, 83, 1.8)):
        sh[i].top = Inches(top)
        if h:
            sh[i].height = Inches(h)

    # Simulated data: equations for the Ring and for both bar charts.
    sub = lambda t: SUB(t)  # noqa: E731
    set_text(sh[52], [
        [T('Ring', b=True), T(': ('), M('X'), sub('0'), T(', '), M('X'), sub('1'),
         T(') from 8 Gaussian clusters on a circle, a binary feature '), M('S'), T(' ∈ {−1, +1}, and')],
        [M('P'), T('('), M('S'), T(' = +1 | '), M('X'), T(') = σ(2'), M('X'), sub('0'), M('X'), sub('1'),
         T('),     Bayes rule '), M('h'), T('* = 1['), M('X'), sub('1'), T(' + (1 + 0.75'), M('S'), T(')'),
         M('X'), sub('0'), T(' > 0],')],
        [M('y'), T(' = '), M('h'), T('* with 10% of labels flipped (Bayes accuracy 90%). Six more datasets: '
                                   'Gaussian, Grid and 4 Bayesian networks.')],
    ])
    sh[52].text_frame.paragraphs[1].alignment = PP_ALIGN.CENTER
    for i in (53, 54, 55, 56):
        sh[i].top += Inches(0.55)
    sh[55].height = Inches(3.65)
    sh[56].height = Inches(3.45)
    acc = [r.text for r in sh[56].text_frame.paragraphs[1].runs]
    set_text(sh[56], [
        [T('Each bar: LTA minus baseline, averaged over CTGAN and TVAE, RF/XGB/DNN and 2 seeds. '),
         T('Prediction: ', b=True), T('accuracy on original test rows.')],
        [T('Coverage: ', b=True), M('L'), sub('test'), T(' = (1/'), M('n'), T(') Σ'), sub('i'), T(' log '),
         M('q'), T('('), M('z'), sub('i'), T('), where '), M('q'), T(' is a simple density fitted to the synthetic '
                                                               'table and '), M('z'), sub('1'), T(', …, '),
         M('z'), sub('n'), T(' are original test rows. Higher means the synthetic table covers the original '
                             'data better.')],
        [T(acc[0]), M('L'), sub('test'), T(acc[3])],
    ])

    for pic in slide.shapes:
        if pic.shape_type == 13 and (FIG / pic._element.nvPicPr.cNvPr.get('descr')).exists():
            name = pic._element.nvPicPr.cNvPr.get('descr')
            slide.part.related_part(pic._element.blipFill.blip.rEmbed)._blob = (FIG / name).read_bytes()


def main():
    assert not DST.exists(), f'{DST} exists; it may hold manual edits'
    prs = Presentation(SRC)
    assert len(prs.slides) == 1
    amend(prs.slides[0])
    prs.save(DST)
    print(DST)


if __name__ == '__main__':
    main()
