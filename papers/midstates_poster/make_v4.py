"""Poster v4 from the hand-edited v3 and the reviewer comments in ffts_midstates_poster_v3.pdf.

Method now shows concepts only: two parallel rows (standard vs. LTA) split into explicit stages,
"1. Fit to real data" and "2. Generate", with generator G and labeler L. Implementation details
(which generators/labelers, row counts, ablation, baselines) move to Evaluation.
Figures come from `build_poster.py --figures figures_v3`.
"""
import copy
from pathlib import Path

from pptx import Presentation
from pptx.shapes.connector import Connector
from pptx.util import Inches

HERE = Path(__file__).resolve().parent
SRC = HERE / 'ffts_midstates_poster_v3.pptx'
DST = HERE / 'ffts_midstates_poster_v4.pptx'
FIG = HERE / 'figures_v3'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
NAVY, GREY, RED, INK = '12355B', '8A96A3', 'C0392B', '1E2A36'


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
CA, WA = 0.70, 2.70
CB, WB = 4.35, 4.10
CC, WC = 9.40, 4.70
DIV1, DIV2 = (CA + WA + CB) / 2, (CB + WB + CC) / 2


def method(b):
    s = b.clone(LABEL, CB, 12.25, WB, 0.6, 'Stage 1 header')
    set_text(s, [[T('1. Fit to real data', b=True, color=NAVY)]])
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
    set_text(s, [[T('Real data')], [T('('), V('X'), T(', '), V('y'), T(')')]])
    s = b.clone(BOX_MODEL, CB, y0, WB, h0, 'Standard generator')
    set_text(s, [[T('Generator '), M('G')], [T('learns '), M('p'), T('('), V('X'), T(', '), V('y'), T(')')]])
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
    set_text(s, [[T('Real data')], [T('('), V('X'), T(', '), V('y'), T(')')]])
    s = b.clone(BOX_MODEL, CB, yt, WB, h, 'LTA generator')
    set_text(s, [[T('Generator '), M('G')], [T('learns '), M('p'), T('('), V('X'), T(')')]])
    s = b.clone(BOX_MODEL, CB, yb, WB, h, 'LTA labeler')
    set_text(s, [[T('Labeler '), M('L')], [T('learns '), V('y'), T(' from '), V('X')]])
    s = b.clone(BOX_LTA_OUT, CC, yt, WC, h, 'LTA features')
    set_text(s, [[M('G', b=True), T(' samples ', b=True), XS(b=True)], [T('any number of rows')]])
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
         T(' samples the features, and a labeler '), M('L'), T(' fitted to real ('), V('X'), T(', '), V('y'),
         T(') assigns their targets.')],
    ])
    set_text(sh[31], [
        [M('G'), T(' can be any tabular generator and '), M('L'), T(' any supervised model.')],
        [T('Both are fitted to real training rows only.')],
    ])
    sh[31].top = Inches(20.15)
    set_text(sh[34], [
        [T('Data: ', b=True), T('Adult, Census KDD, Covertype, MNIST12, MNIST28 (macro F1); '
                                'California Housing, News ('), M('R'), SUP('2'), T(').')],
        [T('Models: ', b=True), M('G'), T(' = CTGAN, TVAE or TabDDPM (100,000 rows); '),
         M('L'), T(' = RF, XGB or DNN.')],
        [T('Utility test: ', b=True), T('train a fresh neural network on each training set (synthetic only, '
                                        'or real + synthetic) and score it on held-out real rows.')],
        [T('Baselines: ', b=True), T('real data only; standard synthesis (generated '), V('y'), T('). '),
         T('Ablation: ', b=True), M('L'), T(' relabels the '), XS(), T(' of a joint '), M('G'),
         T('. 2 seeds; 588 runs.')],
    ])
    for i, top in ((32, 22.75), (33, 23.65), (34, 24.0)):
        sh[i].top = Inches(top)

    # Right column: name the generator G here too.
    set_text(sh[49], [
        [T('Each point is one task, generator and labeler. On the diagonal, fitting '), M('G'), T(' to ('),
         V('X'), T(', '), V('y'), T(') instead of '), V('X'), T(' alone makes no difference.')],
        [T('Joint minus LTA: '), T(sh[49].text_frame.paragraphs[1].runs[1].text, b=True),
         T(sh[49].text_frame.paragraphs[1].runs[2].text), M('p'), T(sh[49].text_frame.paragraphs[1].runs[4].text)],
        [T('Fitting '), M('G'), T(' to '), V('y'), T(' as well adds no detectable benefit.')],
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
