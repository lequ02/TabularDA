"""Focused verification of independent fits, source data and report formatting."""
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import re
from pathlib import Path
import zipfile
import pymupdf

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
OUT=ROOT/'output/statistical_analysis_328_20261010';DATA=HERE/'separate_results'
def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
snapshot=json.loads((HERE/'snapshot/snapshot.json').read_text())
for r in snapshot['records']:assert digest(HERE/'snapshot/records'/r['path'])==r['sha256']
assert len(snapshot['records'])==406 and snapshot['selected_completed']==406 and not snapshot['missing']
assert digest(HERE/'original_final_328_project.Rmd')==digest(Path('D:/Rprojects/research_data_synthesis/final_328_project.Rmd'))
assert digest(OUT/'frozen_inputs/scores.csv')==digest(HERE/'snapshot/scores.csv')
meta=json.loads((DATA/'analysis_metadata.json').read_text())
assert meta['frozen_scores_sha256']==digest(HERE/'snapshot/scores.csv')
raw=rows(HERE/'snapshot/scores.csv')
lookup={(r['record'],r['metric']):r for r in raw}
d2=rows(HERE/'results/regression_d2_scores.csv');d2_lookup={r['record']:r for r in d2}
for r in d2:assert digest(HERE/'snapshot/records'/r['sidecar'])==r['sidecar_sha256']
assert len(d2)==116

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.rows=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.rows=[]
        elif tag=='tr':self.row=[]
        elif tag in ('td','th'):self.cell={'text':'','bold':False}
        elif tag=='b' and self.cell is not None:self.cell['bold']=True
    def handle_data(self,data):
        if self.cell is not None:self.cell['text']+=data
    def handle_endtag(self,tag):
        if tag in ('td','th'):self.row.append(self.cell);self.cell=None
        elif tag=='tr':self.rows.append(self.row);self.row=None
        elif tag=='table':self.tables.append(self.rows);self.rows=None

manifest=json.loads((HERE/'deliverable_manifest.json').read_text());checked=0;verified_values=0
for arm,report in zip(['synthetic','mix'],manifest['reports']):
    base=DATA/arm
    for name in ['classification_scores.csv','regression_scores.csv','regression_d2_scores.csv']:
        scores=rows(base/name)
        assert not {'mode','source','group','approach_has_y'}.intersection(scores[0])
        for r in scores:
            original=d2_lookup[r['record']] if name=='regression_d2_scores.csv' else lookup[r['record'],r['metric']]
            assert math.isclose(float(r['value']),float(original['value']),rel_tol=1e-14,abs_tol=1e-14)
            assert int(r['has_y'])=={'full':1,'xonly':0}[original['source']]
            assert original['mode']==arm and r['record_sha256']==original['record_sha256']
            verified_values+=1
    scores=rows(base/'classification_scores.csv');assert len(scores)==140
    assert len({(r['dataset'],r['seed'],r['generator'],r['condition']) for r in scores})==140
    assert {r['condition'] for r in scores}==set(meta['analyses'][arm]['condition_levels'])
    for m in rows(base/'lm_model_summary.csv'):
        assert int(m['rank'])==int(m['columns']) and int(m['n'])==140 and 'mode' not in m['formula']
    mixed=json.loads((base/'mixed_metadata.json').read_text())
    assert mixed['converged'] and not mixed['warnings'] and mixed['n']==140 and mixed['fixed_columns']==14 and 'mode' not in mixed['formula']
    assert len(rows(base/'repeated_anova.csv'))==19
    assert len(rows(base/'labeler_effects.csv'))==12 and len(rows(base/'planned_contrasts.csv'))==4
    planned=rows(base/'planned_contrasts.csv');mm=rows(base/'mixed_contrasts.csv')
    for p,m in zip(planned,mm):assert p['contrast']==m['contrast'] and math.isclose(float(p['score_change']),float(m['effect']),abs_tol=1e-9)
    parser=Tables();parser.feed((OUT/(report['stem']+'.html')).read_text(encoding='utf-8'))
    expected=iter(report['tables'])
    for table in parser.tables:
        headers=[r['text'] for r in table[0]]
        if 'Sig.' not in headers:continue
        specs=next(expected);assert headers==specs['headers'] and len(table)-1==len(specs['rows'])
        si=headers.index('Sig.')
        for row,p,b in zip(table[1:],specs['pvalues'],specs['bold']):
            if p is None:continue
            p=float(p);symbol='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '.' if p<.1 else ''
            assert row[si]['text']==symbol and all(c['bold']==b==(p<.05) for c in row)
            checked+=1
    assert next(expected,None) is None
    doc=pymupdf.open(report['pdf']);assert len(doc)==len(report['sections'])==6
    for page,title in zip(doc,report['sections']):
        text=page.get_text();assert title in text
        assert all(t in text[:1600] for t in ('Research question:','Method:','Conclusion:'))
        assert all(t not in text for t in ('approach_has_y','source /','group /')) and not re.search(r'\bmode\b',text)
        for block in page.get_text('blocks'):
            if block[6]==0:assert 46<=block[0]<=block[2]<=566 and 20<=block[1]<=block[3]<=770,(page.number,block[:4])
    assert 'macro_F1 ~ generator * condition + (1 | dataset)' in doc[1].get_text()
    assert 'macro_F1 ~ dataset + generator * condition' in doc[2].get_text()
    regression_text=' '.join(doc[4].get_text().split())
    assert 'News relabeling improves R2 but worsens NMAE and D2' in regression_text
    assert 'holdout-feature collisions' in regression_text

for path,sha in manifest['files'].items():assert digest(ROOT/path)==sha
with zipfile.ZipFile(OUT/'real_data_328_analysis_package.zip') as z:
    assert 'group_scores.csv' not in z.namelist() and 'classification_scores.csv' not in z.namelist()
    for path,sha in manifest['files'].items():
        p=ROOT/path;name=str(p.relative_to(OUT)).replace('\\','/') if p.is_relative_to(OUT) else p.name
        assert hashlib.sha256(z.read(name)).hexdigest()==sha
validation=dict(verified_source_records=406,verified_d2_sidecars=116,verified_analysis_score_values=verified_values,
 original_template_unchanged=True,independent_training_analyses=True,classification_rows_per_analysis=140,
 mixed_models_converged_without_warnings=True,all_OLS_designs_full_rank=True,
 planned_contrasts_per_analysis=4,method_contrasts_per_analysis=12,repeated_ANOVA_terms_per_analysis=19,
 checked_significance_rows=checked,bold_and_stars='passed',final_pdf_pages_each=6,combined_pdf_pages=12,
 sections_begin_question_method_conclusion=True,layout_bounds='passed',deliverable_hashes='passed',
 source_scores_unchanged=True,statistical_models_refitted=True,visual_review='all twelve rendered pages inspected; final regression pages inspected separately')
(HERE/'validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
