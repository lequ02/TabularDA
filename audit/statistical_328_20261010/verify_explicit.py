"""Focused score, design, ANOVA, presentation and package verification."""
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
import itertools
from pathlib import Path
import re
import sys
import zipfile
import pymupdf

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent;DATA=HERE/'explicit_results'
FINALOUT=ROOT/'output/statistical_analysis_328_20261010';FINALPDF=ROOT/'output/pdf/statistical_analysis_328_20261010'
STAGE=ROOT/'tmp/pdfs/statistical_328_explicit/deliverables'
staged='--stage' in sys.argv;OUT=STAGE/'report' if staged else FINALOUT;PDF=STAGE/'pdf' if staged else FINALPDF
def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def equal(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def actual(path):
    p=ROOT/path
    if staged:return OUT/p.relative_to(FINALOUT) if p.is_relative_to(FINALOUT) else PDF/p.name
    return p
snapshot=json.loads((HERE/'snapshot/snapshot.json').read_text())
for r in snapshot['records']:assert digest(HERE/'snapshot/records'/r['path'])==r['sha256']
assert len(snapshot['records'])==406 and not snapshot['missing']
assert digest(HERE/'original_final_328_project.Rmd')==digest(Path('D:/Rprojects/research_data_synthesis/final_328_project.Rmd'))
assert digest(OUT/'frozen_inputs/scores.csv')==digest(HERE/'snapshot/scores.csv')
raw=rows(HERE/'snapshot/scores.csv');lookup={(r['record'],r['metric']):r for r in raw}
d2=rows(HERE/'results/regression_d2_scores.csv');d2lookup={r['record']:r for r in d2}
for r in d2:assert digest(HERE/'snapshot/records'/r['sidecar'])==r['sidecar_sha256']
assert len(d2)==116
latest=json.loads((OUT/'latest_source_verification.json').read_text());assert latest['checked_files']==522 and not latest['mismatches']
meta=json.loads((OUT/'analysis_metadata.json').read_text())
assert meta['frozen_scores_sha256']==digest(HERE/'snapshot/scores.csv')

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

manifest=json.loads((HERE/'deliverable_manifest.json').read_text());checked=0;values=0;fchecks=0;pdf_texts=[]
for arm,report in zip(['synthetic','mix'],manifest['reports']):
    base=OUT/arm
    for name in ['classification_scores.csv','generated_baselines.csv','regression_scores.csv','regression_d2_scores.csv']:
        for r in rows(base/name):
            assert not {'condition','mode','source','group','approach_has_y'}.intersection(r)
            original=d2lookup[r['record']] if name=='regression_d2_scores.csv' else lookup[r['record'],r['metric']]
            assert equal(r['value'],original['value']) and r['record_sha256']==original['record_sha256']
            assert original['mode']==arm and int(r['has_y'])=={'full':1,'xonly':0}[original['source']]
            values+=1
    primary=rows(base/'classification_scores.csv');baselines=rows(base/'generated_baselines.csv')
    assert len(primary)==120 and len(baselines)==20
    assert {r['labeler'] for r in primary}=={'rf','xgb','dnn'} and 'labeler' not in baselines[0]
    assert len({(r['dataset'],r['seed'],r['generator'],r['has_y'],r['labeler']) for r in primary})==120
    models=rows(base/'model_summary.csv')
    assert [int(r['rank']) for r in models]==[28,17,37]
    for r in models:assert int(r['rank'])==int(r['columns']) and int(r['n'])==120
    assert equal(models[0]['df_resid'],92) and all(x in models[0]['formula'] for x in ['dataset','generator','has_y','labeler','seed'])
    assert len(rows(base/'factorial_anova.csv'))==17
    for r in rows(base/'independent_F_checks.csv'):
        assert math.isclose(float(r['anova_F']),float(r['restricted_OLS_F']),rel_tol=2e-7,abs_tol=1e-8);fchecks+=1
    paired=rows(base/'matched_relabeling_scores.csv');assert len(paired)==120
    baseline_by_record={r['record']:r for r in baselines}
    for r in paired:
        b=baseline_by_record[r['record_generated']]
        assert all(r[k]==b[k] for k in ['dataset','seed','generator'])
        assert equal(r['value_generated'],b['value']) and equal(r['difference'],float(r['value'])-float(b['value']))
    assert len(rows(base/'labeler_effects.csv'))==12 and len(rows(base/'planned_contrasts.csv'))==4
    previous={r['contrast']:r for r in rows(HERE/'separate_results'/arm/'planned_contrasts.csv')}
    for r in rows(base/'planned_contrasts.csv'):
        assert all(equal(r[k],previous[r['contrast']][k]) for k in ['score_change','relative_percent','ci_low','ci_high','p_t_holm'])
    for r in rows(base/'estimated_means.csv'):
        matches=[float(s['value']) for s in primary if all(s[k]==r[k] for k in ['generator','has_y','labeler'])]
        assert len(matches)==10 and equal(r['mean'],sum(matches)/10)
    for r in rows(base/'factorial_effects.csv'):
        assert equal(r['effect'],float(r['first_mean'])-float(r['second_mean'])) and float(r['ci_low'])<=float(r['effect'])<=float(r['ci_high'])
    regmodels=rows(base/'regression_model_summary.csv');assert len(regmodels)==6
    for r in regmodels:
        pooled=r['scope']=='pooled'
        assert int(r['n'])==(48 if pooled else 24)
        assert int(r['rank'])==int(r['columns'])==(25 if pooled else 13)
        assert equal(r['df_resid'],23 if pooled else 11)
        assert '*C(seed, Sum)' in r['formula'] if pooled else '+ C(generator, Sum)*C(has_y, Sum)*C(labeler, Sum)' in r['formula']
        assert len(rows(base/(r['outcome']+'_'+r['scope']+'_anova.csv')))==(17 if pooled else 9)
    regscores={}
    for outcome in ['R2','D2']:
        data=rows(base/(outcome+'_scores.csv'));regscores[outcome]=data
        assert len(data)==48
        expected=set(itertools.product(['news','california_housing'],['42','43'],['ctgan','tvae'],['0','1'],['rf','xgb','dnn']))
        assert {(r['dataset'],r['seed'],r['generator'],r['has_y'],r['labeler']) for r in data}==expected
        for r in data:
            saved=lookup[r['record'],'r2'] if outcome=='R2' else d2lookup[r['record']]
            assert equal(r['value'],saved['value']) and equal(r[outcome],saved['value'])
            assert r['record_sha256']==saved['record_sha256'];values+=1
        baselinefile='regression_scores.csv' if outcome=='R2' else 'regression_d2_scores.csv'
        baseline_by_record={r['record']:r for r in rows(base/baselinefile) if r['labeler']=='generated' and (outcome=='D2' or r['metric']=='r2')}
        matched=rows(base/(outcome+'_matched_baselines.csv'));assert len(matched)==48
        for r in matched:
            b=baseline_by_record[r['record_generated']]
            assert all(r[k]==b[k] for k in ['dataset','seed','generator'])
            assert equal(r['value_generated'],b['value']) and equal(r['difference'],float(r['value'])-float(b['value']))
    factor={'tvae_minus_ctgan':('generator','tvae','ctgan'),'has_y1_minus_has_y0':('has_y','1','0'),
      'seed43_minus_seed42':('seed','43','42'),'rf_minus_dnn':('labeler','rf','dnn'),'xgb_minus_dnn':('labeler','xgb','dnn')}
    effects=rows(base/'regression_factor_effects.csv');assert len(effects)==30
    for r in effects:
        data=[s for s in regscores[r['outcome']] if r['scope']=='pooled' or s['dataset']==r['scope']]
        v,a,b=factor[r['contrast']]
        left=[float(s['value']) for s in data if s[v]==a];right=[float(s['value']) for s in data if s[v]==b]
        assert equal(r['effect'],sum(left)/len(left)-sum(right)/len(right))
        assert float(r['ci_low'])<float(r['effect'])<float(r['ci_high'])
    for r in rows(base/'regression_independent_F_checks.csv'):
        assert math.isclose(float(r['anova_F']),float(r['restricted_OLS_F']),rel_tol=2e-7,abs_tol=1e-8);fchecks+=1
    planned={r['contrast']:r for r in rows(base/'planned_contrasts.csv')}
    for r in rows(base/'contrast_uncertainty.csv'):
        p=planned[r['contrast']];assert equal(r['raw_p'],p['p_t']) and equal(r['holm_p'],p['p_t_holm'])
        assert int(r['n_datasets'])==5 and int(r['family_size'])==4 and equal(r['first_holm_cutoff'],.0125)
        assert equal(float(p['score_change'])/float(p['t']),r['se'])
    for r in rows(base/'regression_baseline_effects.csv'):
        assert equal(r['difference'],float(r['relabeled_mean'])-float(r['generated_mean'])) and r['inference']=='descriptive'
    parser=Tables();parser.feed((OUT/(report['stem']+'.html')).read_text(encoding='utf-8'));expected=iter(report['tables'])
    for tab in parser.tables:
        headers=[r['text'] for r in tab[0]]
        if 'Sig.' not in headers:continue
        specs=next(expected);assert headers==specs['headers'] and len(tab)-1==len(specs['rows'])
        for row,p,b in zip(tab[1:],specs['pvalues'],specs['bold']):
            if p is None:continue
            p=float(p);symbol='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '.' if p<.1 else ''
            assert row[headers.index('Sig.')]['text']==symbol and all(c['bold']==b==(p<.05) for c in row);checked+=1
    assert next(expected,None) is None
    tasktable=next(t for t in parser.tables if t[0][0]['text']=='Dataset / outcome')
    for row,(outcome,dataset) in zip(tasktable[1:],itertools.product(['R2','D2'],['news','california_housing'])):
        a={r['term']:r for r in rows(base/(outcome+'_'+dataset+'_anova.csv'))}
        for cell,v in zip(row[1:],['generator','has_y','labeler','seed']):
            p=float(a['C('+v+', Sum)']['PR(>F)'])
            symbol='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '.' if p<.1 else ''
            assert cell['bold']==(p<.05) and cell['text'].strip().endswith(symbol);checked+=1
    doc=pymupdf.open(PDF/Path(report['pdf']).name);assert len(doc)==len(report['sections'])==8
    pdf_texts.extend(page.get_text() for page in doc)
    for page,title in zip(doc,report['sections']):
        text=page.get_text();flat=' '.join(text.split());assert title in flat
        assert all(t in text[:1900] for t in ('Research question:','Method:','Conclusion:'))
        assert not re.search(r'\b(condition|mode|group|approach_has_y)\b',text,re.I)
        assert 'source / has_y' not in text
        for block in page.get_text('blocks'):
            if block[6]==0:assert 46<=block[0]<=block[2]<=566 and 20<=block[1]<=block[3]<=770,(page.number,block[:4])
    assert 'macro_F1 ~ dataset + generator * has_y * labeler * seed' in ' '.join(doc[1].get_text().split())
    assert 'News relabeling improves R2 but worsens NMAE and D2' in ' '.join(doc[4].get_text().split())
    assert 'holdout-feature collisions' in ' '.join(doc[4].get_text().split())
    assert 'Holm cutoff .05/4 = .0125' in ' '.join(doc[2].get_text().split())
    assert 'zero residual df' in ' '.join(doc[5].get_text().split())
    for page,outcome in [(6,'R2'),(7,'D2')]:
        flat=' '.join(doc[page].get_text().split())
        assert outcome+' ~ dataset + generator * has_y * labeler * seed' in flat
        assert '48 observed scores; 25 identifiable coefficients; 23 residual df' in flat

combined=pymupdf.open(PDF/'real_data_328_analysis.pdf')
assert len(combined)==16 and [p.get_text() for p in combined]==pdf_texts

for path,sha in manifest['files'].items():assert digest(actual(path))==sha
with zipfile.ZipFile(OUT/'real_data_328_analysis_package.zip') as z:
    assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
    assert all(not x.endswith(('mixed_model.txt','repeated_anova.csv','group_scores.csv')) for x in z.namelist())
    for path,sha in manifest['files'].items():
        p=ROOT/path;name=str(p.relative_to(FINALOUT)).replace('\\','/') if p.is_relative_to(FINALOUT) else p.name
        assert hashlib.sha256(z.read(name)).hexdigest()==sha
validation=dict(verified_source_records=406,verified_d2_sidecars=116,latest_remote_verification=latest,
 verified_analysis_score_values=values,original_template_unchanged=True,independent_training_analyses=True,
 factorial_rows_per_analysis=120,generated_baselines_per_analysis=20,primary_rank=28,primary_residual_df=92,
 all_designs_full_rank=True,independently_reproduced_F_statistics=fchecks,matched_baseline_checks='passed',
 unchanged_planned_effects_and_regression_inputs=True,checked_significance_rows=checked,bold_and_stars='passed',
 regression_outcomes=['R2','D2'],regression_rows_per_outcome_per_analysis=48,
 regression_pooled_rank=25,regression_pooled_residual_df=23,regression_within_task_rank=13,
 regression_within_task_residual_df=11,regression_factor_effects_checked=60,
 final_pdf_pages_each=8,combined_pdf_pages=16,sections_begin_question_method_conclusion=True,
 layout_bounds='passed',deliverable_hashes='passed',no_composite_factor_or_seed_numeric_covariate=True,
 source_scores_unchanged=True,statistical_models_refitted=True,visual_review='pending')
(HERE/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))
