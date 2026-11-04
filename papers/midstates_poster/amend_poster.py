"""Apply the Nov 2 reviewer comments (ffts_midstates_poster.pdf) to a new copy of the poster.

Renames FFTS to Learned Target Assignment (LTA), switches to bold X / bold lowercase y notation, and
swaps in figures_v2/ (from `build_poster.py --figures figures_v2`). Edits text runs in place on every
slide; the original .pptx and figures/ are untouched.
"""
import copy
from pathlib import Path

from pptx import Presentation

HERE = Path(__file__).resolve().parent
SRC = HERE / 'ffts_midstates_poster.pptx'
DST = HERE / 'ffts_midstates_poster_v3.pptx'
FIG = HERE / 'figures_v2'
TITLE = 'Learned Target Assignment for Synthetic Tabular Data'

# Segments: plain str, or (text, 'v') for a bold-italic variable, or (text, 'b') for bold.
WHY_HEADING = ['Why generate ', ('X', 'v'), ' and ', ('y', 'v'), ' separately?']
WHY_PARAS = [
    ['A supervised ML problem has input features ', ('X', 'v'), ' and a target ', ('y', 'v'),
     '. Synthetic data is only useful if it keeps the relationship between ', ('X', 'v'), ' and ',
     ('y', 'v'), '.'],
    [('CTGAN', 'b'), ', ', ('TVAE', 'b'), ' and ', ('TabDDPM', 'b'), ' model (', ('X', 'v'), ', ', ('y', 'v'),
     ') jointly, unaware of the target: ', ('y', 'v'), ' is one more feature column, and nothing ties a generated ',
     ('y', 'v'), ' to its ', ('X', 'v'), '.'],
    [('LTA', 'b'), ' is target-aware: it generates ', ('X', 'v'),
     ', then labels each row with a model fitted to the real data.'],
]


def walk(shapes):
    for sh in shapes:
        if sh.shape_type == 6:
            yield from walk(sh.shapes)
        else:
            yield sh


def set_runs(para, segments, keep_bold=False):
    """Rebuild a paragraph's runs from segments, cloning the first plain run's formatting."""
    runs = list(para.runs)
    base = next(r for r in runs if keep_bold or not r.font.bold and not r.font.italic)._r
    for r in runs:
        para._p.remove(r._r)
    end = para._p.find('{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr')
    for seg in segments:
        text, kind = (seg, None) if isinstance(seg, str) else seg
        r = copy.deepcopy(base)
        if end is not None:
            end.addprevious(r)
        else:
            para._p.append(r)
        run = next(x for x in para.runs if x._r is r)
        run.text = text
        run.font.bold = keep_bold or kind in ('v', 'b')
        run.font.italic = kind == 'v'


def amend(slide):
    for sh in walk(slide.shapes):
        if not sh.has_text_frame:
            continue
        paras = sh.text_frame.paragraphs
        full = sh.text_frame.text
        if full == 'Synthetic Data Generation via Factorized Feature-Target Synthesis':
            paras[0].runs[0].text = TITLE
            continue
        if full == 'FFTS: factorized feature–target synthesis':
            paras[0].runs[0].text = 'LTA: learned target assignment'
            continue
        if full == 'Why factorize?':
            set_runs(paras[0], WHY_HEADING, keep_bold=True)
            continue
        if full.startswith('A supervised table has features'):
            for para, segs in zip(paras, WHY_PARAS):
                set_runs(para, segs)
            continue
        if full == 'Does the generator need Y?':
            r = paras[0].runs[0]
            r.text = 'Does the generator need '
            y = copy.deepcopy(r._r)
            r._r.addnext(y)
            q = copy.deepcopy(r._r)
            y.addnext(q)
            runs = paras[0].runs
            runs[1].text, runs[1].font.italic = 'y', True
            runs[2].text = '?'
            continue
        is_ring = full.startswith('Ring')
        for para in paras:
            runs = para.runs
            for i, run in enumerate(runs):
                run.text = run.text.replace('FFTS', 'LTA')
                # Notation: bold X for the feature matrix, bold lowercase y for the target.
                if run.font.italic and run.text in ('X', 'Y'):
                    if is_ring:  # X0, X1 are scalar coordinates in the Ring description
                        continue
                    run.text = run.text.replace('Y', 'y')
                    run.font.bold = True


def replace_figures(slide):
    for sh in slide.shapes:
        if sh.shape_type != 13:  # picture
            continue
        name = sh._element.nvPicPr.cNvPr.get('descr')
        if (FIG / name).exists():
            slide.part.related_part(sh._element.blipFill.blip.rEmbed)._blob = (FIG / name).read_bytes()


def main():
    # v3 has since been hand-edited; never overwrite it.
    assert not DST.exists(), f'{DST} exists; it may hold manual edits'
    prs = Presentation(SRC)
    for slide in prs.slides:
        amend(slide)
        replace_figures(slide)
    prs.save(DST)
    print(DST)


if __name__ == '__main__':
    main()
