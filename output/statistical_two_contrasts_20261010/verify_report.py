"""Focused artifact/provenance validation; remote statistics are not rerun locally."""
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import pymupdf

ROOT = Path('D:/SummerResearch')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/statistical_two_contrasts_20261010'
PDF = ROOT / 'output/pdf/statistical_two_contrasts_20261010/real_data_two_contrasts.pdf'

def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def equal(first, second):
    return math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-12)

meta = json.loads((OUT / 'analysis_metadata.json').read_text())
assert meta['contrasts'] == ['relabel_minus_generated', 'has_y1_minus_has_y0']
assert meta['selected_records'] == 406 and meta['d2_sidecars'] == 116
for name, sha in meta['source_hashes'].items():
    assert digest(OUT / name if name == 'analyze.py' else OUT / 'frozen_inputs' / name) == sha
raw = rows(OUT / 'frozen_inputs/scores.csv')
lookup = {(r['record'], r['metric']): r for r in raw}
d2 = rows(OUT / 'frozen_inputs/regression_d2_scores.csv')
d2lookup = {r['record']: r for r in d2}
sidecars = json.loads((OUT / 'frozen_inputs/d2_verification.json').read_text())['sidecars']
for record in sidecars:
    row = d2lookup[record['record']]
    assert equal(row['value'], record['value']) and row['sidecar_sha256'] == record['sidecar_sha256']
matched_values = 0
for path in OUT.glob('*_matched_scores.csv'):
    arm = path.name.split('_', 1)[0]
    metric = path.name[len(arm)+1:-len('_matched_scores.csv')]
    for row in rows(path):
        source = d2lookup[row['record']] if metric == 'd2_absolute_error' else lookup[row['record'], metric]
        baseline = d2lookup[row['record_generated']] if metric == 'd2_absolute_error' else lookup[row['record_generated'], metric]
        assert equal(row['value'], source['value']) and equal(row['value_generated'], baseline['value'])
        assert source['mode'] == baseline['mode'] == arm
        assert source['labeler'] in ['rf', 'xgb', 'dnn'] and baseline['labeler'] == 'generated'
        assert baseline['source'] == 'full'
        assert all(source[k] == baseline[k] for k in ['dataset', 'seed', 'generator'])
        assert int(row['has_y']) == {'full': 1, 'xonly': 0}[source['source']]
        matched_values += 2
assert matched_values == 1056
tests = rows(OUT / 'classification_contrasts.csv')
effects = rows(OUT / 'task_contrasts.csv')
seed_effects = rows(OUT / 'seed_contrasts.csv')
assert len(tests) == 4 and len(effects) == 44 and len(seed_effects) == 88
for row in effects:
    assert equal(row['native_difference'], float(row['left_mean']) - float(row['right_mean']))
    assert int(row['orientation']) == (-1 if row['metric'] == 'nmae_sigma' else 1)
    assert equal(row['benefit'], float(row['orientation']) * float(row['native_difference']))
    seeds = [r for r in seed_effects if all(r[k] == row[k] for k in ['training', 'metric', 'dataset', 'contrast'])]
    assert len(seeds) == 2 and equal(row['benefit'], sum(float(r['benefit']) for r in seeds) / 2)
for row in tests:
    task = [r for r in effects if r['metric'] == 'f1_macro' and all(r[k] == row[k] for k in ['training', 'contrast'])]
    assert len(task) == 5 and equal(row['mean'], sum(float(r['benefit']) for r in task) / 5)
    prior = ROOT / 'audit/statistical_328_20261010/explicit_results' / row['training'] / 'planned_contrasts.csv'
    original = next(r for r in rows(prior) if r['contrast'] == row['contrast'])
    assert equal(row['mean'], original['mean']) and equal(row['p_raw'], original['p_t'])
    assert float(row['p_raw']) <= float(row['p_raw_holm2']) <= float(row['p_raw_holm4']) <= 1
    if row['contrast'] == 'relabel_minus_generated':
        assert equal(row['p_signflip_exact'], .0625) and equal(row['p_wilcoxon_exact'], .0625)
    assert all(float(row[f'ci{family}_low']) <= float(row['mean']) <= float(row[f'ci{family}_high']) for family in [1, 2, 4])
checks = rows(OUT / 'independent_F_checks.csv')
assert len(checks) == 202
for row in checks:
    assert math.isclose(float(row['anova_F']), float(row['restricted_OLS_F']), rel_tol=2e-7, abs_tol=1e-8)
models = rows(OUT / 'model_summary.csv')
assert len(models) == 18
for row in models:
    assert int(row['residual_df']) == int(row['n']) - int(row['rank'])

manifest = json.loads((HERE / 'report_manifest.json').read_text())
class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None
    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.table = []
        elif tag == 'tr': self.row = []
        elif tag in ['td', 'th']: self.cell = ''
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in ['td', 'th']: self.row.append(self.cell); self.cell = None
        elif tag == 'tr': self.table.append(self.row); self.row = None
        elif tag == 'table': self.tables.append(self.table); self.table = None
parser = Tables()
parser.feed((OUT / 'analysis.html').read_text(encoding='utf-8'))
assert len(parser.tables) == len(manifest['tables'])
for actual, expected in zip(parser.tables, manifest['tables']):
    assert actual == [expected['headers']] + expected['rows']
with pymupdf.open(PDF) as pdf:
    assert len(pdf) == manifest['pages'] == 6
    for page, section in zip(pdf, manifest['sections']):
        text = page.get_text()
        assert section in text
        assert all(label in text for label in ['Research question:', 'Method:', 'Finding:'])
        for word in page.get_text('words'):
            x0, y0, x1, y1 = word[:4]
            assert x0 >= 47 and x1 <= 565 and y0 >= 30 and y1 <= 772, (page.number, word)
old_manifest = json.loads((ROOT / 'audit/statistical_328_20261010/deliverable_manifest.json').read_text())
for name, sha in old_manifest['files'].items():
    assert digest(ROOT / name) == sha, 'Prior deliverable changed: ' + name
validation = dict(source_hashes=522, source_records=406, source_d2_sidecars=116,
                  source_metric_values=3770, matched_exported_values=matched_values,
                  task_contrasts=44, seed_contrasts=88, headline_tests=4, hypotheses_per_regime=2,
                  independently_checked_F_statistics=202, models=18, report_pages=6,
                  html_tables='matched report manifest', pdf_layout_bounds='passed',
                  prior_deliverables='all manifest hashes unchanged',
                  visual_review='passed' if '--reviewed' in sys.argv else 'pending')
(HERE / 'validation.json').write_text(json.dumps(validation, indent=2))
if '--reviewed' in sys.argv:
    for name in ['verify_report.py', 'validation.json', 'report_manifest.json']:
        shutil.copy2(HERE / name, OUT / name)
    files = [path for path in OUT.rglob('*') if path.is_file() and path.suffix != '.zip'] + [PDF]
    hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in files}
    (OUT / 'deliverable_hashes.json').write_text(json.dumps(hashes, indent=2))
    package = OUT / 'two_contrast_analysis_package.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(OUT)) if path.is_relative_to(OUT) else path.name)
        archive.write(OUT / 'deliverable_hashes.json', 'deliverable_hashes.json')
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist()))
    for name, sha in hashes.items():
        assert digest(ROOT / name) == sha
print(json.dumps(validation, indent=2))
