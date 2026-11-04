"""Poster v5 from the hand-edited v3: the v4 changes plus the user's feedback on v4.

Method shows concepts only: two parallel rows (standard vs. LTA) split into explicit stages,
"1. Fit to original data" and "2. Generate", with generator G and labeler L. Implementation details
move to Evaluation. "Real" becomes "original" throughout. A headline table (macro F1, matching the
figure and the pre-registered analysis) replaces the results paragraph above the central figure.
Figures come from `build_poster.py --figures figures_v4`.
"""
import copy
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.shapes.connector import Connector
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
SRC = HERE / 'ffts_midstates_poster_v3.pptx'
DST = HERE / 'ffts_midstates_poster_v5.pptx'
FIG = HERE / 'figures_v4'
SCORES = HERE.parents[1] / 'comparisons/20261031_adult_tabddpm_fix/valbest/scores_valbest.csv'
SCORES_SHA256 = 'eebcc5d1c64bcb18043287451ce15f41f0a983072d1a90510d117f2999208780'  # as in build_poster.py
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
NAVY, GREY, RED, INK, PALE, PINK = '12355B', '8A96A3', 'C0392B', '1E2A36', 'EEF2F7', 'FCE9E6'


# Segment helpers: (text, options). Options: b, i, base ('sub' or 'sup'), sz, color.
def T(t, **o):
    return (t, o)


def V(t, **o):  # bold-italic vector/matrix: X, y
    return (t, {'b': True, 'i': True, **o})


def M(t, **o):  # italic math symbol: G, L, p
    return (t, {'i': True, **o})


def SUB(t='syn', **o):
    return (t, dict(base='sub', **o))


def SUP(t, **o):
    return (t, dict(base='sup', **o))


def XS(**o):
    return [V('X', **o), SUB(**{k: v for k, v in o.items() if k != 'b'})]


def YS(**o):
    return [V('y', **o), SUB(**{k: v for k, v in o.items() if k != 'b'})]


def flat(segs):
    out = []
    for s in segs:
        out.extend(s if isinstance(s, list) else [s])
    return out


def set_para(para, segs, base_r=None):
    """Replace a paragraph's runs, cloning the formatting of its first run (or base_r)."""
    runs = list(para.runs)
    base = copy.deepcopy(base_r if base_r is not None else runs[0]._r)
    for r in runs:
        para._p.remove(r._r)
    end = para._p.find(f'{A}endParaRPr')
    for text, o in flat(segs):
        r = copy.deepcopy(base)
        rpr = r.find(f'{A}rPr')
        rpr.set('b', '1' if o.get('b') else '0')
        rpr.set('i', '1' if o.get('i') else '0')
        rpr.attrib.pop('baseline', None)
        if o.get('base'):
            rpr.set('baseline', '-25000' if o['base'] == 'sub' else '30000')
        if o.get('sz'):
            rpr.set('sz', str(int(o['sz'] * 100)))
        if o.get('color'):
            rpr.find(f'{A}solidFill')[0].set('val', o['color'])
        r.find(f'{A}t').text = text
        (end.addprevious(r) if end is not None else para._p.append(r))


def set_text(shape, paragraphs):
    """Set a shape's paragraphs; reuses existing paragraphs, cloning the last one when more are needed."""
    tf = shape.text_frame
    base_r = copy.deepcopy(tf.paragraphs[0].runs[0]._r)
    while len(tf.paragraphs) < len(paragraphs):
        tf.paragraphs[-1]._p.addnext(copy.deepcopy(tf.paragraphs[-1]._p))
    while len(tf.paragraphs) > len(paragraphs):
        tf._txBody.remove(tf.paragraphs[-1]._p)
    for para, segs in zip(tf.paragraphs, paragraphs):
        set_para(para, segs, base_r)


class Builder:
    def __init__(self, slide):
        self.slide = slide
        self.tree = slide.shapes._spTree
        self.by_id = {sh.shape_id: sh for sh in slide.shapes}
        self.next_id = max(self.by_id) + 1
        self.anchor = self.by_id[31]._element  # new diagram shapes go just before the method bullets

    def clone(self, src_id, x, y, w, h, name):
        el = copy.deepcopy(self.by_id[src_id]._element)
        cnv = el.find('.//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr')
        cnv.set('id', str(self.next_id)); cnv.set('name', name)
        self.next_id += 1
        self.anchor.addprevious(el)
        shape = next(s for s in self.slide.shapes if s._element is el)
        if isinstance(shape, Connector):
            shape.begin_x, shape.begin_y = Inches(x), Inches(y)
            shape.end_x, shape.end_y = Inches(w), Inches(h)
        else:
            shape.left, shape.top, shape.width, shape.height = Inches(x), Inches(y), Inches(w), Inches(h)
        return shape


# Source shapes in v3 used as style templates.
LABEL, LTA_LABEL, BOX_REAL, BOX_MODEL, BOX_STD_OUT, BOX_LTA_OUT = 13, 20, 14, 15, 16, 24
ARROW_NAVY, ARROW_GREY, ARROW_RED, DASH = 17, 18, 27, 19
OLD_DIAGRAM = range(13, 31)

# Columns (inches): input | 1. fit | 2. generate.
CA, WA = 0.70, 2.50
CB, WB = 4.05, 4.90
CC, WC = 9.80, 4.30
DIV1, DIV2 = (CA + WA + CB) / 2, (CB + WB + CC) / 2


def method(b):
    s = b.clone(LABEL, CB, 12.25, WB, 0.6, 'Stage 1 header')
    set_text(s, [[T('1. Fit to original data', b=True, color=NAVY)]])
    s.text_frame.paragraphs[0].alignment = 2
    s = b.clone(LABEL, CC, 12.25, WC, 0.6, 'Stage 2 header')
    set_text(s, [[T('2. Generate', b=True, color=NAVY)]])
    s.text_frame.paragraphs[0].alignment = 2
    for x in (DIV1, DIV2):  # stage dividers span the box rows only, not the row labels
        for top, bottom in ((13.5, 15.2), (16.2, 19.9)):
            b.clone(DASH, x, top, x, bottom, 'Stage divider')

    # Standard: joint synthesis.
    s = b.clone(LABEL, CA, 12.95, 8.0, 0.6, 'Standard label')
    set_text(s, [[T('Standard: joint synthesis', b=True)]])
    y0, h0 = 13.6, 1.5
    s = b.clone(BOX_REAL, CA, y0, WA, h0, 'Standard real')
    set_text(s, [[T('Original data')], [T('('), V('X'), T(', '), V('y'), T(')')]])
    s = b.clone(BOX_MODEL, CB, y0, WB, h0, 'Standard generator')
    set_text(s, [[T('Generator '), M('G'), T(' learns')], [T('the distribution of ('), V('X'), T(', '), V('y'), T(')')]])
    s = b.clone(BOX_STD_OUT, CC, y0, WC, h0, 'Standard output')
    set_text(s, [[M('G'), T(' samples ('), XS(), T(', '), YS(), T(')')],
                 [YS(), T(' is generated')]])
    b.clone(ARROW_NAVY, CA + WA, y0 + h0 / 2, CB, y0 + h0 / 2, 'Arrow')
    b.clone(ARROW_GREY, CB + WB, y0 + h0 / 2, CC, y0 + h0 / 2, 'Arrow')

    b.clone(DASH, CA, 15.45, CC + WC, 15.45, 'Row divider')

    # LTA: learned target assignment.
    s = b.clone(LTA_LABEL, CA, 15.65, 8.0, 0.6, 'LTA label')
    set_text(s, [[T('LTA: learned target assignment', b=True)]])
    yt, yb, h = 16.3, 18.3, 1.5
    s = b.clone(BOX_REAL, CA, yt, WA, yb + h - yt, 'LTA real')
    set_text(s, [[T('Original data')], [T('('), V('X'), T(', '), V('y'), T(')')]])
    s = b.clone(BOX_MODEL, CB, yt, WB, h, 'LTA generator')
    set_text(s, [[T('Generator '), M('G'), T(' learns')], [T('the distribution of '), V('X')]])
    s = b.clone(BOX_MODEL, CB, yb, WB, h, 'LTA labeler')
    set_text(s, [[T('Labeler '), M('L'), T(' learns')], [V('y'), T(' | '), V('X')]])
    s = b.clone(BOX_LTA_OUT, CC, yt, WC, h, 'LTA features')
    set_text(s, [[M('G', b=True), T(' samples ', b=True), XS(b=True)]])
    s = b.clone(BOX_LTA_OUT, CC, yb, WC, h, 'LTA targets')
    set_text(s, [[YS(b=True), T(' = ', b=True), M('L', b=True), T('(', b=True), XS(b=True), T(')', b=True)],
                 [YS(), T(' is assigned')]])
    mid = (yt + yb + h) / 2
    b.clone(ARROW_NAVY, CA + WA, mid - 0.25, CB, yt + h / 2, 'Arrow')
    b.clone(ARROW_NAVY, CA + WA, mid + 0.25, CB, yb + h / 2, 'Arrow')
    b.clone(ARROW_RED, CB + WB, yt + h / 2, CC, yt + h / 2, 'Arrow')
    b.clone(ARROW_RED, CB + WB, yb + h / 2, CC, yb + h / 2, 'Arrow')
    b.clone(ARROW_RED, CC + WC / 2, yt + h, CC + WC / 2, yb, 'Arrow')

    for i in OLD_DIAGRAM:
        b.tree.remove(b.by_id[i]._element)


CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']


def headline_numbers():
    """Means behind the table, computed exactly as build_poster.py builds the figure cells."""
    assert hashlib.sha256(SCORES.read_bytes()).hexdigest() == SCORES_SHA256
    scores = pd.read_csv(SCORES)
    exp = scores[scores['mode'].isin(['synthetic', 'mix'])].copy()
    assert len(exp) == 588 and set(exp.labeler) == {'generated', 'rf', 'xgb', 'dnn'}
    exp['treatment'] = np.where(exp.labeler.eq('generated'), 'generated', 'hybrid_' + exp.construction)
    real = scores[scores.generator.eq('real')].groupby('dataset')[['test_f1_macro', 'test_r2']].mean()
    out = {}
    for mode in ('synthetic', 'mix'):
        for metric, datasets in (('test_f1_macro', CLASS), ('test_r2', ['california_housing'])):
            part = exp[exp['mode'].eq(mode) & exp.dataset.isin(datasets)]
            table = part.groupby(['dataset', 'generator', 'treatment'])[metric].mean().unstack()
            assert len(table) == 3 * len(datasets) and table.notna().all().all()
            for treatment in ('generated', 'hybrid_xonly'):
                out[(metric, mode, treatment)] = table[treatment].mean()
    for metric, datasets in (('test_f1_macro', CLASS), ('test_r2', ['california_housing'])):
        out[(metric, 'real')] = real.loc[datasets, metric].mean()
    return out


def cell_text(cell, segs, size=26, color=INK, align=PP_ALIGN.CENTER):
    para = cell.text_frame.paragraphs[0]
    para.alignment = align
    for text, o in flat(segs):
        run = para.add_run()
        run.text = text
        run.font.size, run.font.name = Pt(size), 'Calibri'
        run.font.bold, run.font.italic = bool(o.get('b')), bool(o.get('i'))
        run.font.color.rgb = RGBColor.from_string(o.get('color', color))
        if o.get('base'):
            run._r.get_or_add_rPr().set('baseline', '-25000' if o['base'] == 'sub' else '30000')
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.12)
    cell.margin_top = cell.margin_bottom = Inches(0.03)


def headline_table(slide, panel):
    n = headline_numbers()
    f1 = lambda v: f'{100 * v:.1f}'
    r2 = lambda v: f'{v:.2f}'
    rows = [
        ([T('Standard: generated '), V('y')], 'generated', 'FFFFFF', False),
        ([T('LTA: '), YS(), T(' = '), M('L'), T('('), XS(), T(')')], 'hybrid_xonly', PINK, True),
    ]
    shape = slide.shapes.add_table(5, 5, panel.left, panel.top, panel.width, Inches(3.3))
    shape.name = 'Headline table'
    tbl = shape.table
    tbl.first_row = tbl.horz_banding = False
    for j, w in enumerate((5.6, 3.2, 3.2, 3.2, 3.2)):
        tbl.columns[j].width = Inches(w)
    for i in range(5):
        tbl.rows[i].height = Inches(0.66)
    for i in range(5):
        for j in range(5):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(NAVY if i < 2 else 'FFFFFF')
    tbl.cell(0, 1).merge(tbl.cell(0, 2))
    tbl.cell(0, 3).merge(tbl.cell(0, 4))
    tbl.cell(0, 0).merge(tbl.cell(1, 0))
    white = dict(color='FFFFFF')
    cell_text(tbl.cell(0, 0), [T('Training data', b=True, **white)], align=PP_ALIGN.LEFT)
    cell_text(tbl.cell(0, 1), [T('Classification: macro F1 (%)', b=True, **white)])
    cell_text(tbl.cell(0, 3), [T('Housing: ', b=True, **white), M('R', b=True, **white),
                               SUP('2', b=True, **white)])
    for j in (1, 3):
        cell_text(tbl.cell(1, j), [T('Synthetic only', **white)], size=24)
        cell_text(tbl.cell(1, j + 1), [T('Original + synthetic', **white)], size=24)
    for i, (label, key, fill, bold) in enumerate(rows, start=2):
        for j in range(5):
            tbl.cell(i, j).fill.fore_color.rgb = RGBColor.from_string(fill)
        cell_text(tbl.cell(i, 0), label, align=PP_ALIGN.LEFT)
        for j, (metric, mode, fmt) in enumerate((('test_f1_macro', 'synthetic', f1), ('test_f1_macro', 'mix', f1),
                                                 ('test_r2', 'synthetic', r2), ('test_r2', 'mix', r2)), start=1):
            cell_text(tbl.cell(i, j), [T(fmt(n[(metric, mode, key)]), b=bold)])
    for j in range(5):
        tbl.cell(4, j).fill.fore_color.rgb = RGBColor.from_string(PALE)
    tbl.cell(4, 1).merge(tbl.cell(4, 2))
    tbl.cell(4, 3).merge(tbl.cell(4, 4))
    cell_text(tbl.cell(4, 0), [T('Original data only')], align=PP_ALIGN.LEFT)
    cell_text(tbl.cell(4, 1), [T(f1(n[('test_f1_macro', 'real')]))])
    cell_text(tbl.cell(4, 3), [T(r2(n[('test_r2', 'real')]))])
    print({k: round(v, 4) for k, v in n.items()})


def amend(slide):
    b = Builder(slide)
    sh = b.by_id
    method(b)

    set_text(sh[10], [
        [T('A supervised ML problem has input features '), V('X'), T(' and a target '), V('y'),
         T('. Synthetic data helps a downstream model when it preserves the dependency of '), V('y'),
         T(' on '), V('X'), T('.')],
        [T('CTGAN', b=True), T(', '), T('TVAE', b=True), T(' and '), T('TabDDPM', b=True), T(' model ('),
         V('X'), T(', '), V('y'), T(') jointly, unaware of the target: '), V('y'),
         T(' is one more feature column, and nothing ties a generated '), V('y'), T(' to its '), V('X'), T('.')],
        [T('LTA', b=True), T(' models this dependency directly: a generator '), M('G'),
         T(' samples the features, and a labeler '), M('L'), T(' fitted to original ('), V('X'), T(', '), V('y'),
         T(') assigns their targets.')],
    ])
    set_text(sh[31], [
        [M('G'), T(' can be any tabular generator and '), M('L'), T(' any supervised model.')],
        [T('Both are fitted to original training rows only.')],
    ])
    sh[31].top = Inches(20.15)
    set_text(sh[34], [
        [T('Data: ', b=True), T('Adult, Census KDD, Covertype, MNIST12, MNIST28 (macro F1); '
                                'California Housing, News ('), M('R'), SUP('2'), T(').')],
        [T('Models: ', b=True), M('G'), T(' = CTGAN, TVAE or TabDDPM (100,000 rows); '),
         M('L'), T(' = RF, XGB or DNN.')],
        [T('Scoring: ', b=True), T('each training set trains a fresh neural network, which is tested on '
                                   'held-out original rows.')],
        [T('Training sets: ', b=True), T('original data only, or synthetic data with generated '), V('y'),
         T(' (standard) or '), YS(), T(' = '), M('L'), T('('), XS(), T(') (LTA), alone or added to the '
                                                              'original data.')],
        [T('Ablation: ', b=True), M('L'), T(' relabels the '), XS(), T(' of a joint '), M('G'),
         T(' (right). 2 seeds; 588 runs.')],
    ])
    for i, top in ((32, 21.6), (33, 22.5), (34, 22.85), (35, 29.6), (36, 30.5), (37, 30.85)):
        sh[i].top = Inches(top)

    # Right column: name the generator G here too.
    set_text(sh[49], [
        [T('Each point is one task, generator and labeler. On the diagonal, fitting '), M('G'), T(' to ('),
         V('X'), T(', '), V('y'), T(') instead of '), V('X'), T(' alone makes no difference.')],
        [T('Joint minus LTA: '), T(sh[49].text_frame.paragraphs[1].runs[1].text, b=True),
         T(sh[49].text_frame.paragraphs[1].runs[2].text), M('p'), T(sh[49].text_frame.paragraphs[1].runs[4].text)],
        [T('Fitting '), M('G'), T(' to '), V('y'), T(' as well adds no detectable benefit.')],
    ])

    # Takeaways: drop the caveats.
    set_text(sh[37], [
        [T('Labeling synthetic features with a model trained on original data is a simple add-on to any '
           'generator.')],
        [T('Exception: TabDDPM on MNIST. Its digits drift from the original ones, so a labeler trained on '
           'original data mislabels them.')],
    ])
    set_text(sh[45], [[
        T('Table and figure: averaged over 2 seeds (LTA also over RF, XGB and DNN); table rows also average '
          'over tasks and generators. Open circle: the generator’s own targets. Filled circle: LTA. '
          'Generated targets fall short of original data on 5 of 5 classification tasks for every '
          'generator; News has no predictable signal for any method, including original data (|'),
        M('R'), SUP('2'), T('| ≤ 0.03).')]])

    headline_table(slide, sh[40])
    for i in (40, 41):  # results paragraph and its panel, replaced by the table
        b.tree.remove(sh[i]._element)

    # "Real" becomes "original" everywhere else.
    for shape in slide.shapes:
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    run.text = re.sub(r'\b(R|r)eal\b', lambda m: 'Original' if m.group(1) == 'R' else 'original',
                                      run.text)

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
