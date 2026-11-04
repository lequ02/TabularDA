"""Format existing archived scores as seed-averaged tables; no model fitting."""
import csv
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
source = ROOT / 'inputs/scores_valbest.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['files']['scores_valbest.csv']
archive_source = Path(__file__).resolve().parents[2] / 'comparisons/scores_valbest.csv'
assert hashlib.sha256(archive_source.read_bytes()).hexdigest() == manifest['files']['scores_valbest.csv']
DATASETS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
NAMES = ['Adult', 'Census KDD', 'Covertype', 'MNIST12', 'MNIST28']
GEN = ['ctgan', 'tvae', 'tabddpm']
LABEL = ['rf', 'xgb', 'dnn']
methods = [(g, 'full', 'generated') for g in GEN]
methods += [(g, c, l) for g in GEN for c in ['full', 'xonly'] for l in LABEL]
cells = defaultdict(list)
with source.open(newline='', encoding='utf-8-sig') as stream:
    for r in csv.DictReader(stream):
        if r['dataset'] in DATASETS and r['mode'] in ['synthetic', 'mix']:
            cells[(r['mode'], r['dataset'], r['generator'], r['construction'], r['labeler'])].append((int(r['seed']), float(r['test_f1_macro'])))

intro = 'Macro F1 (0–1), arithmetic mean of seeds 42 and 43, using the same validation-best checkpoints as the ANOVA. Hybrids use RF, XGB or DNN to predict targets. Full: generator fitted on (X,Y); X-only: generator fitted on X. Bold indicates the highest score and underline the second-highest distinct score per dataset and training mode, across all displayed synthetic methods. Ranking uses unrounded values; all exact ties share the same rank. Real-only scores are excluded from ranking. Emphasis indicates numerical rank, not statistical significance. News and California Housing are regression tasks, so F1 does not apply. Credit has no validation-best evaluation in this selection. Adult TabDDPM and MNIST12 TVAE retain the documented source/collision caveats.'
md = ['# F1 averaged over seeds 42 and 43', '', intro]
web = ['<!doctype html><meta charset="utf-8"><title>F1 by method and dataset</title><style>body{font:15px/1.5 system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 24px;color:#233747}table{border-collapse:collapse;width:100%;margin-bottom:36px}th,td{padding:7px 14px;text-align:right;border-bottom:1px solid #e0e5e9;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#eef3f6}.hybrid-start td{border-top:3px solid #344e60}.baseline td:first-child{font-weight:600}h2{margin-top:32px}</style>', '<h1>F1 averaged over seeds 42 and 43</h1><p>'+html.escape(intro)+'</p>']
for mode in ['synthetic', 'mix']:
    top = {d: sorted({mean(v for _, v in cells[(mode, d, g, c, l)]) for g, c, l in methods}, reverse=True)[:2] for d in DATASETS}
    title = 'Synthetic-only training' if mode == 'synthetic' else 'Mixed training'
    md += ['', '## '+title, '', '| Method | '+' | '.join(NAMES)+' |', '|---|'+'---:|'*5]
    web += ['<h2>'+title+'</h2><table><thead><tr><th>Method</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr></thead><tbody>']
    for i, (g, c, l) in enumerate(methods):
        if i == 3:
            md.append('| **Hybrids ↓** | **────** | **────** | **────** | **────** | **────** |')
        label = {'ctgan':'CTGAN', 'tvae':'TVAE', 'tabddpm':'TabDDPM'}[g]
        if l != 'generated':
            label += ' + '+l.upper()+' ('+('full' if c == 'full' else 'X-only')+')'
        values = []
        for dataset in DATASETS:
            data = cells[(mode, dataset, g, c, l)]
            assert len(data) == 2 and {s for s, _ in data} == {42, 43}
            score = mean(v for _, v in data)
            value = f'{score:.4f}'
            values.append('**'+value+'**' if score == top[dataset][0] else '<u>'+value+'</u>' if score == top[dataset][1] else value)
        md.append('| '+label+' | '+' | '.join(values)+' |')
        css = 'hybrid-start' if i == 3 else ('baseline' if i < 3 else '')
        web.append('<tr class="'+css+'"><td>'+html.escape(label)+'</td>'+''.join('<td>'+v.replace('**', '<strong>', 1).replace('**', '</strong>', 1)+'</td>' for v in values)+'</tr>')
    web.append('</tbody></table>')
(OUT / 'f1_seed_means.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
(OUT / 'f1_seed_means.html').write_text('\n'.join(web), encoding='utf-8')
# Preserve the compact labeler-average view as an explicitly descriptive summary.
compact_methods = [(g, 'full', 'generated') for g in GEN]
compact_methods += [(g, c, 'average') for c in ['full', 'xonly'] for g in GEN]
averaged = {(m, d, g, c): mean(v for l in LABEL for _, v in cells[(m, d, g, c, l)])
            for m in ['synthetic', 'mix'] for d in DATASETS for g in GEN for c in ['full', 'xonly']}
with source.open(newline='', encoding='utf-8-sig') as stream:
    references = list(csv.DictReader(stream))
def compact_score(mode, dataset, g, c, l):
    return averaged[(mode, dataset, g, c)] if l == 'average' else mean(v for _, v in cells[(mode, dataset, g, c, l)])
compact_note = 'Descriptive summary only: hybrid cells average the scores of RF, XGB and DNN as well as seeds 42/43. This is not a selected labeler or an ensemble. '+intro
compact_md = ['# Compact F1 summary', '', compact_note]
compact_html = [web[0], '<h1>Compact F1 summary</h1><p>'+html.escape(compact_note)+'</p>']
for mode in ['synthetic', 'mix']:
    top = {d: sorted({compact_score(mode, d, *method) for method in compact_methods}, reverse=True)[:2] for d in DATASETS}
    title = 'Synthetic-only training' if mode == 'synthetic' else 'Mixed training'
    compact_md += ['', '## '+title, '', '| Method | '+' | '.join(NAMES)+' |', '|---|'+'---:|'*5]
    compact_html.append('<h2>'+title+'</h2><table><thead><tr><th>Method</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr></thead><tbody>')
    real = [f'{mean(float(r["test_f1_macro"]) for r in references if r["dataset"] == d and r["mode"] == "original"):.4f}' for d in DATASETS]
    compact_md.append('| Real only | '+' | '.join(real)+' |')
    compact_html.append('<tr><td>Real only</td>'+''.join('<td>'+v+'</td>' for v in real)+'</tr>')
    for i, (g, c, l) in enumerate(compact_methods):
        if i == 3:
            compact_md.append('| **Hybrids** | **────** | **────** | **────** | **────** | **────** |')
        name = {'ctgan':'CTGAN', 'tvae':'TVAE', 'tabddpm':'TabDDPM'}[g]
        if l == 'average':
            name += ': '+('full' if c == 'full' else 'X-only')+' (labeler mean)'
        values = []
        for d in DATASETS:
            score = compact_score(mode, d, g, c, l)
            value = f'{score:.4f}'
            values.append('**'+value+'**' if score == top[d][0] else '<u>'+value+'</u>' if score == top[d][1] else value)
        compact_md.append('| '+name+' | '+' | '.join(values)+' |')
        css = 'hybrid-start' if i == 3 else ''
        compact_html.append('<tr class="'+css+'"><td>'+html.escape(name)+'</td>'+''.join('<td>'+v.replace('**', '<strong>', 1).replace('**', '</strong>', 1)+'</td>' for v in values)+'</tr>')
    compact_html.append('</tbody></table>')
(OUT / 'f1_compact_ranked.md').write_text('\n'.join(compact_md)+'\n', encoding='utf-8')
(OUT / 'f1_compact_ranked.html').write_text('\n'.join(compact_html), encoding='utf-8')
for markup in [web, compact_html]:
    result = '\n'.join(markup)
    assert result.count('<strong>') == result.count('</strong>') == 10
    assert result.count('<u>') == result.count('</u>') == 10
    assert not any('<strong>' in row or '<u>' in row for row in markup if '<td>Real only</td>' in row)
sys.stdout.reconfigure(encoding='utf-8')
print('Verified: top two distinct unrounded scores per dataset/mode; real-only excluded. Updated full and compact HTML/Markdown tables.')
