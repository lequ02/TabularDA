"""Stage reports of explicit factorial fits, with no composite treatment factor."""
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import zipfile
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from pypdf import PdfReader,PdfWriter

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent;DATA=HERE/'explicit_results';REG=HERE/'regression_anova_results'
STAGE=ROOT/'tmp/pdfs/statistical_328_explicit/deliverables';OUT=STAGE/'report';PDFOUT=STAGE/'pdf'
FINALOUT=ROOT/'output/statistical_analysis_328_20261010';FINALPDF=ROOT/'output/pdf/statistical_analysis_328_20261010'
OUT.mkdir(parents=True,exist_ok=True);PDFOUT.mkdir(parents=True,exist_ok=True)
DS={'adult':'Adult','census_kdd':'Census KDD','covertype':'Covertype','mnist12':'MNIST12','mnist28':'MNIST28','news':'News','california_housing':'Housing'}
CN={'relabel_minus_generated':'Relabeled - generated','has_y1_minus_has_y0':'has_y=1 - has_y=0','has_y1_minus_generated':'has_y=1 - generated','has_y0_minus_generated':'has_y=0 - generated'}
FN={'tvae_minus_ctgan':'TVAE - CTGAN','has_y1_minus_has_y0':'has_y=1 - has_y=0','seed43_minus_seed42':'Seed 43 - seed 42'}
FN.update({'rf_minus_dnn':'RF - DNN','xgb_minus_dnn':'XGB - DNN'})
def read(name):
    with (DATA/arm/name).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def readreg(name):
    with (REG/arm/name).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
verified=json.loads((HERE/'latest_source_verification.json').read_text())['checked_at_chicago']
from datetime import datetime
verified_label=datetime.fromisoformat(verified).strftime('%b %d, %Y at %I:%M %p Chicago')
def f(x,n=4):
    value=float(x)
    return f'{value:.2e}' if n>0 and 0<abs(value)<.5*10**(-n) else f'{value:.{n}f}'
def sign(x):
    value=float(x)
    return f'{value:+.2e}' if 0<abs(value)<.00005 else f'{value:+.4f}'
def pv(x):return f'{float(x):.2e}' if float(x)<.0001 else f'{float(x):.4f}'
def stars(x):return '***' if float(x)<.001 else '**' if float(x)<.01 else '*' if float(x)<.05 else '.' if float(x)<.1 else ''
def ci(r):return '['+sign(r['ci_low'])+', '+sign(r['ci_high'])+']'
def term(t):return re.sub(r'C\(([^,)]+), Sum\)',r'\1',t).replace(':',' x ')
def markup(t):return re.sub(r'(?<!\*)\*\*(?!\*)(.+?)(?<!\*)\*\*(?!\*)',r'<b>\1</b>',html.escape(str(t)))
INK=colors.HexColor('#193247');TEAL=colors.HexColor('#237b8a');PALE=colors.HexColor('#edf4f6')
styles={
 'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=INK,spaceAfter=11),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.3,leading=13,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.2,leading=11.5,textColor=INK,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=10.5,textColor=INK),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8,leading=10.5,textColor=INK),
 'equation':ParagraphStyle('equation',fontName='Courier',fontSize=8.5,leading=12,textColor=INK,spaceAfter=8)}
def para(t,style='body'):
    story.append(Paragraph(markup(t),styles[style]));md.append(t+'\n');web.append('<p class="'+style+'">'+markup(t)+'</p>')
def eq(t):
    story.append(Paragraph(html.escape(t),styles['equation']));md.append('```text\n'+t+'\n```\n');web.append('<pre>'+html.escape(t)+'</pre>')
def section(title,q,method,conclusion):
    if sections:story.append(PageBreak());web.append('<hr>')
    sections.append(title);story.append(Paragraph(markup(title),styles['h1']))
    md.append('# '+title+'\n');web.append('<h1>'+markup(title)+'</h1>')
    para('**Research question:** '+q);para('**Method:** '+method);para('**Conclusion:** '+conclusion)
def table(headers,rows,widths,pvals=None,pad=3,cellp=None):
    assert sum(widths)==516
    bold=[p is not None and float(p)<.05 for p in pvals] if pvals else [False]*len(rows)
    cells=[[Paragraph(html.escape(str(x)),styles['head']) for x in headers]]
    cellbold=[[b or (cellp is not None and cellp[i][j] is not None and float(cellp[i][j])<.05) for j in range(len(row))] for i,(row,b) in enumerate(zip(rows,bold))]
    cells += [[Paragraph('<b>'+html.escape(str(x))+'</b>' if b else html.escape(str(x)),styles['cell']) for x,b in zip(row,bs)] for row,bs in zip(rows,cellbold)]
    tab=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,TEAL),
      ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d7e3e9')),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),
      ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    story.extend([tab,Spacer(1,7)]);md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    for row,bs in zip(rows,cellbold):
        escaped=[str(x).replace('*','\\*') for x in row]
        md.append('| '+' | '.join('**'+x+'**' if b else x for x,b in zip(escaped,bs))+' |')
    md.append('');web.append('<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+('<b>'+html.escape(str(x))+'</b>' if b else html.escape(str(x)))+'</td>' for x,b in zip(row,bs))+'</tr>' for row,bs in zip(rows,cellbold))+'</tbody></table>')
    if pvals:tables.append(dict(headers=headers,rows=rows,pvalues=pvals,bold=bold))
def legend(basis):
    t='Symbols: *** p < .001; ** p < .01; * p < .05; . p < .10. Bold: p < .05. Uses '+basis+'.'
    para(t,'small');md[-1]=t.replace('*','\\*')+'\n'
def figure(name,caption,width=500):
    path=DATA/arm/name
    with PILImage.open(path) as im:w,h=im.size
    story.append(Image(str(path),width=width,height=width*h/w));para(caption,'small')
    md.append('!['+caption+']('+arm+'/'+name+')\n');web.append('<img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
def write_atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(payload);tmp.replace(path)
def copy_exact(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    payload=src.read_bytes()
    if not dst.exists() or dst.read_bytes()!=payload:write_atomic(dst,payload)
    assert dst.read_bytes()==payload
HTML_HEAD='''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:15px/1.5 system-ui;color:#193247;max-width:1000px;margin:40px auto;padding:0 24px}h1{font-size:25px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 20px}th{background:#edf4f6;text-align:left}
td,th{padding:7px;border-bottom:1px solid #d7e3e9;vertical-align:top}img{max-width:100%}pre{white-space:pre-wrap;background:#edf4f6;padding:10px;font-size:13px}.small{font-size:13px}hr{margin:40px 0}</style></head><body>'''
reports=[]
for arm in ['synthetic','mix']:
    title='Synthetic-only' if arm=='synthetic' else 'Mix: real + synthetic'
    story=[];md=[];web=[];sections=[];tables=[]
    meta=json.loads((DATA/arm/'analysis_metadata.json').read_text());tests=read('planned_contrasts.csv');labels=read('labeler_effects.csv')
    main=tests[0];hy=tests[1];anova=read('factorial_anova.csv');marginal=read('factorial_effects.csv');joint=read('joint_tests.csv')
    section('1. '+title+': design and factors','What choices does the experiment vary?',
      'Analyze this training dataset independently. The factorial grid is 5 datasets x 2 generators x 2 has_y values x 3 labelers x 2 seeds = 120 macro-F1 scores.',
      'All relabeled combinations are observed. The 20 generated-target runs and 10 real-only classification runs remain reference comparisons, outside this factorial ANOVA.')
    table(['Factor','Meaning','Levels'],[
      ['dataset','Real-data prediction task.','Adult; Census KDD; Covertype; MNIST12; MNIST28.'],
      ['generator','Model fitted on real training data to generate synthetic rows.','CTGAN; TVAE.'],
      ['has_y','Whether the generator is fitted with target Y.','0: fit X only. 1: fit (X,Y).'],
      ['labeler','Predictor fitted on real training data to replace synthetic targets.','RF: random forest. XGB: XGBoost. DNN: neural network.'],
      ['seed','Categorical experiment random-number setting.','42; 43. This is not a continuous covariate.']],[84,234,198],pad=4)
    table(['Reference','Construction','Role'],[
      ['Generated targets','Fit (X,Y), retain generated Y. No labeler.','Two generators x two seeds x five tasks = 20 scores.'],
      ['Real only','Train downstream model on original real training rows.','Two seeds x five tasks = 10 scores.']],[100,210,206])
    para('**Training rows:** '+meta['training_rows']+'. Mix concatenates all rows; it is not a fixed 50/50 mixture. Both reports use the same experimental scope, with separate statistical fits.')
    para('**X** = features; **Y** = target. has_y describes generator fitting, not whether the final training table contains Y. Every relabeled table contains X and predicted Y. Generated targets have no has_y=0 case and no labeler; that nonexistent combination is never estimated.','small')
    para('**Macro F1** is the equal-weight average of F1 across target classes, on a 0-1 scale. The downstream evaluator is a dataset-specific neural model. RF/XGB/DNN are labelers. News and Housing regression scores are analyzed separately on page 5.','small')

    label_p=next(x['PR(>F)'] for x in anova if x['term']=='C(labeler, Sum)')
    section('2. ANOVA: which factors affect macro F1?',
      'Do generator, has_y, labeler, seed or their interactions change relabeled-data utility?',
      'Type II ANOVA: macro_F1 ~ dataset + generator * has_y * labeler * seed. All five factors are categorical; sum-to-zero coding is used.',
      '**Labeler differs** (nominal p = '+pv(label_p)+'). Generator and has_y have small average effects; seed and treatment interactions are nonsignificant in this model.')
    eq('macro_F1 ~ dataset + generator * has_y * labeler * seed')
    table(['Term','SS','df','F','Nominal p','Sig.'],[[term(x['term']),f(x['sum_sq']),f(x['df'],0),f(x['F'],2) if x['F'] else '',pv(x['PR(>F)']) if x['PR(>F)'] else '',stars(x['PR(>F)']) if x['PR(>F)'] else ''] for x in anova],[213,65,33,64,97,44],pvals=[x['PR(>F)'] or None for x in anova],pad=2)
    legend('nominal Type II ANOVA p')
    para('* includes all main effects, two-way, three-way and four-way interactions among generator, has_y, labeler and seed. For example, has_y x labeler asks whether including Y helps RF, XGB and DNN differently. Dataset is an additive block in this equation.','small')
    table(['Marginal comparison','Mean F1 change','95% model CI'],[[FN[x['contrast']],sign(x['effect']),ci(x)] for x in marginal],[210,120,186])
    para('SS = sum of squares; df = numerator degrees of freedom; F = term mean square / residual mean square. Residual df = 92. Intervals use the ordinary model assumptions and are unadjusted. Shared test partitions make nominal ANOVA inference exploratory.','small')

    section('3. Does replacing generated targets improve utility?',
      'Do RF/XGB/DNN targets outperform the generator\'s own targets, and does has_y matter?',
      'Match within dataset, seed and generator. delta[d] = mean(s,g,labeler) F1_relabel[d] - mean(s,g) F1_generated[d]. Test five dataset differences with t = mean(delta)/(SD(delta)/sqrt(5)), df=4.',
      'Relabeling gains **'+sign(main['score_change'])+' macro F1 ('+f(main['relative_percent'],2)+'% relative)**. has_y=1 minus 0 is **'+sign(hy['score_change'])+'**. Neither main comparison passes Holm-adjusted p < .05 across tasks.')
    table(['Comparison','First mean','Second mean','F1 change','Relative'],[[CN[x['contrast']],f(x['left_mean']),f(x['right_mean']),sign(x['score_change']),f(x['relative_percent'],2)+'%'] for x in tests],[180,82,82,90,82])
    table(['Comparison','95% paired-t CI','Holm p','Sig.'],[[CN[x['contrast']],ci(x),pv(x['p_t_holm']),stars(x['p_t_holm'])] for x in tests],[220,160,92,44],pvals=[x['p_t_holm'] for x in tests])
    legend('Holm-adjusted dataset-level p over four planned contrasts')
    uncertainty=readreg('contrast_uncertainty.csv')[0]
    para('**Why the gain misses Holm significance:** the five task gains range from '+f(100*float(uncertainty['minimum']),2)+' to '+f(100*float(uncertainty['maximum']),2)+' percentage points; SE = '+f(100*float(uncertainty['se']),2)+' points. Raw p = '+pv(uncertainty['raw_p'])+'; Holm p = '+pv(uncertainty['holm_p'])+'. The smallest of four raw p-values is '+pv(uncertainty['smallest_family_p'])+', above the first Holm cutoff .05/4 = .0125. None passes. Large gains do not imply precise estimates.','small')
    figure('dataset_effects.png','Five dataset means show the main contrasts. Positive values favor relabeling or has_y=1.')
    para('All datasets, seeds and generators receive equal weight; relabeling also averages the three labelers. has_y compares only relabeled runs. The generated baseline is matched once per dataset/seed/generator and is not treated as six independent copies.','small')
    para('Relative gain = F1 change / second mean; .01 F1 is one percentage point. Confidence intervals are unadjusted. Exact sign-flip results are supplied; with five tasks the minimum two-sided p is .0625. This is not evidence that every future task improves.','small')

    best=max(labels,key=lambda x:float(x['mean']));passing=sum(float(x['p_t_holm'])<.05 for x in labels)
    section('4. How large is each labeler\'s improvement?',
      'For each generator and has_y, how much do RF, XGB and DNN improve on generated targets?',
      'delta[d] = mean(s) F1_relabel[d,g,has_y,labeler] - mean(s) F1_generated[d,g]. Paired t-test across five datasets, df=4; Holm over twelve comparisons.',
      'Largest descriptive gain: '+best['generator'].upper()+' / '+best['labeler'].upper()+' / has_y='+best['has_y']+' (**'+sign(best['mean'])+'**). '+str(passing)+' of twelve baseline comparisons pass Holm-adjusted p < .05.')
    table(['Generator','Labeler','has_y','F1 mean','F1 gain','Relative','Holm p','Sig.'],[[x['generator'].upper(),x['labeler'].upper(),x['has_y'],f(x['relabeled']),sign(x['mean']),f(x['relative_percent'],2)+'%',pv(x['p_t_holm']),stars(x['p_t_holm'])] for x in labels],[74,55,48,68,78,77,72,44],pvals=[x['p_t_holm'] for x in labels])
    legend('Holm-adjusted dataset-level p across twelve baseline comparisons')
    baselines=read('generated_baselines.csv')
    baseline_means={g:sum(float(x['value']) for x in baselines if x['generator']==g)/10 for g in ['ctgan','tvae']}
    table(['Matched generated-target comparator','Mean macro F1'],[[g.upper(),f(v)] for g,v in baseline_means.items()],[310,206])
    para('These are comparisons against each generator\'s own generated targets, not pairwise tests between labelers. The labeler ANOVA on page 2 tests differences among RF/XGB/DNN. An observed largest gain does not establish a universally best labeler.','small')
    para('Means average seeds within tasks and then give each task equal weight. Per-task differences, confidence intervals, exact tests and fitted marginal means are in the supplied tables. Generated baselines remain outside the factorial fit.','small')

    reg=read('regression_summary.csv')
    section('5. Regression: what changes on News and Housing?',
      'Does relabeling improve regression utility under the same constructions?',
      'Compare saved scores with generated targets. Means average seeds, generators and, for relabeled runs, RF/XGB/DNN. Pages 7-8 separately test factor effects within relabeled runs.',
      'Housing relabeling improves all three average metrics. News relabeling improves R2 but worsens NMAE and D2. Baseline gains below are descriptive; ANOVA tests are exploratory.')
    eq('R2 = 1 - sum((Y - prediction)^2) / sum((Y - mean(Y))^2)')
    eq('NMAE = mean(abs(Y - prediction)) / SD_test(Y), ddof = 0')
    eq('D2 = 1 - MAE_prediction / MAE_test_median_baseline')
    table(['Dataset / metric','Real only','Generated Y','Relabeled has_y=0','Relabeled has_y=1'],[[DS[x['dataset']]+' / '+{'r2':'R2','nmae_sigma':'NMAE','d2_absolute_error':'D2'}[x['metric']],f(x['original']),f(x['generated']),f(x['has_y0']),f(x['has_y1'])] for x in reg],[132,76,88,110,110])
    gains={ (r['dataset'],r['outcome']):r for r in readreg('regression_baseline_effects.csv') if r['comparison']=='relabeled'}
    table(['Relabeled - generated','R2 change','D2 change'],[[DS[d],sign(gains[d,'R2']['difference']),sign(gains[d,'D2']['difference'])] for d in ['news','california_housing']],[220,148,148])
    para('Higher R2/D2 is better; lower NMAE is better. R2 measures squared error against the test-mean baseline; D2 measures absolute error against the test-median baseline. Negative R2/D2 means worse than that baseline.','small')
    para('News uses log-target reruns scored in original units; Housing uses no-Faker reruns. NMAE and D2 read saved test normalization and verified sidecars; denominators are not recomputed for this report. Real-only runs are references.','small')
    para('**Scope:** 406 verified selected records across both training analyses: 290 classification and 116 regression, plus 116 verified D2 sidecars. The 14 real-only runs are shared references. Credit, Intrusion, NB/PCA-GMM and simulated experiments are excluded.','small')
    para('**MNIST caveat:** output heads were repaired; retained MNIST12 TVAE synthetic tables still have documented holdout-feature collisions. Seeds share prepared test partitions. Neither seed supplies an independent new test sample.','small')

    seed_check=joint[0];dataset_check=joint[1];summary=meta['models'][0]
    section('6. Model checks: do common effects describe each task?',
      'Do classification effects vary by task, and which regression factors matter within News or Housing?',
      'Use nested F tests for the classification equations below. For each regression dataset, fit R2 or D2 ~ seed + generator * has_y * labeler.',
      'Classification dataset interactions are strong (p = '+pv(dataset_check['p'])+'). Regression effects also depend on the task; use the within-task p-values below and signed changes on pages 7-8.')
    eq('Seed block: macro_F1 ~ dataset + seed + generator * has_y * labeler')
    eq('Dataset check: macro_F1 ~ dataset * (has_y * labeler)')
    eq('                         + generator * has_y * labeler + seed')
    table(['Added terms','df num / den','F','Nominal p','Sig.'],[['Seed interactions',f(seed_check['df_num'],0)+' / '+f(seed_check['df_den'],0),f(seed_check['F'],3),pv(seed_check['p']),stars(seed_check['p'])],
      ['Dataset interactions',f(dataset_check['df_num'],0)+' / '+f(dataset_check['df_den'],0),f(dataset_check['F'],3),pv(dataset_check['p']),stars(dataset_check['p'])]],[200,90,80,102,44],pvals=[seed_check['p'],dataset_check['p']])
    legend('nominal nested-model p; these are supporting checks')
    eq('Within each regression dataset: R2 or D2 ~ seed + generator * has_y * labeler')
    task_p=[];task_cellp=[]
    for outcome in ['R2','D2']:
        for dataset in ['news','california_housing']:
            a={r['term']:r for r in readreg(outcome+'_'+dataset+'_anova.csv')}
            ps=[a['C('+v+', Sum)']['PR(>F)'] for v in ['generator','has_y','labeler','seed']]
            task_p.append([DS[dataset]+' / '+outcome]+[pv(p)+' '+stars(p) for p in ps])
            task_cellp.append([None]+ps)
    table(['Dataset / outcome','Generator p','has_y p','Labeler p','Seed p'],task_p,[140,94,94,94,94],cellp=task_cellp)
    legend('nominal within-task ANOVA p; bold cells mark p < .05')
    para('Each regression task has 24 relabeled scores, 13 coefficients and 11 residual df. A full generator * has_y * labeler * seed fit within one task would be saturated: 24 coefficients, zero residual df. The stated within-task equation keeps seed additive; its omitted seed interactions form the residual. Full tables are supplied.','small')
    para('Primary model: 120 scores, 28 identifiable coefficients, 92 residual df. Seed-block rank = 17; dataset-check rank = 37. All ANOVA F values were independently reproduced. Primary residual Shapiro p = '+pv(summary['shapiro_p'])+'; adjusted R-squared = '+f(summary['adjusted_r_squared'])+'.','small')
    para('The ordinary ANOVA assumes independent errors with common variance. Shared test sets, nonnormal residuals and dataset-specific treatment effects limit its p-values; page 3 therefore reports paired evidence across five dataset means. Seed is fixed at 42/43, not a sampled population of seeds.','small')
    para('**Provenance:** original score snapshot Oct 10, 2026, 1:08:42 a.m. Chicago; all 522 source/sidecar hashes rechecked '+verified_label+'. Generator settings remain 500 epochs and 100,000 rows. Census selects checkpoints by weighted macro-F1; others use development loss. Experiments were not rerun.','small')

    for page,outcome in [(7,'R2'),(8,'D2')]:
        ra=readreg(outcome+'_pooled_anova.csv')
        byterm={r['term']:r for r in ra}
        significant=[v for v in ['generator','has_y','labeler','seed'] if float(byterm['C('+v+', Sum)']['PR(>F)'])<.05]
        if significant:
            finding='Nominal main effects: '+', '.join(v+' (p = '+pv(byterm['C('+v+', Sum)']['PR(>F)'])+')' for v in significant)+'.'
        else:finding='No treatment main effect or interaction passes nominal p < .05 in this common-effects model.'
        finding+=' has_y has no detected common effect; a significant term means choices differ, not that every choice improves utility.'
        section(str(page)+'. Regression ANOVA: which factors affect '+outcome+'?',
          'Among relabeled runs, do generator, has_y, labeler, seed or their interactions change '+outcome+'?',
          'Type II ANOVA: '+outcome+' ~ dataset + generator * has_y * labeler * seed. All factors are categorical with sum-to-zero coding; synthetic and mix are fitted independently.',finding)
        eq(outcome+' ~ dataset + generator * has_y * labeler * seed')
        table(['Term','SS','df','F','Nominal p','Partial eta2','Sig.'],[[term(r['term']),f(r['sum_sq']),f(r['df'],0),f(r['F'],2) if r['F'] else '',pv(r['PR(>F)']) if r['PR(>F)'] else '',f(r['partial_eta_sq'],3) if r['partial_eta_sq'] else '',stars(r['PR(>F)']) if r['PR(>F)'] else ''] for r in ra],[195,59,28,58,84,48,44],pvals=[r['PR(>F)'] or None for r in ra],pad=2)
        legend('nominal exploratory ANOVA p; no multiplicity adjustment')
        effects={(r['scope'],r['contrast']):r for r in readreg('regression_factor_effects.csv') if r['outcome']==outcome}
        table(['Marginal comparison','Pooled change','News change','Housing change'],[[FN[name]]+[sign(effects[scope,name]['effect']) for scope in ['pooled','news','california_housing']] for name in FN],[195,107,107,107])
        para('Positive changes favor the first choice. Pooled changes give the two tasks equal weight. Within-task model p-values are on page 6; unadjusted 95% model intervals and contrast p-values are supplied in regression_factor_effects.csv.','small')
        para('48 observed scores; 25 identifiable coefficients; 23 residual df. Partial eta2 = SS_term / (SS_term + SS_residual). These exploratory tests use two fixed datasets and shared test partitions. They do not test relabeled versus generated targets; those baseline changes are on page 5. Dataset SS reflects different task score levels.','small')
    stem='synthetic_only_328_analysis' if arm=='synthetic' else 'mix_328_analysis';pdf=PDFOUT/(stem+'.pdf');temporary=pdf.with_suffix('.tmp.pdf')
    def footer(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(INK)
        c.drawString(48,26,title+' | Explicit factors | Sources verified Oct 10, 2026');c.drawRightString(564,26,str(doc.page));c.restoreState()
    SimpleDocTemplate(str(temporary),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=42,title=title+' - 328 explicit-factor analysis',author='Minh Le').build(story,onFirstPage=footer,onLaterPages=footer)
    temporary.replace(pdf)
    text='\n'.join(md);header='---\ntitle: "'+title+' - 328 explicit-factor analysis"\noutput:\n  html_document:\n    toc: true\n  pdf_document: default\n---\n\n'
    rcode='''\n# Optional equivalent R refit

**Research question:** How can the explicit-factor fits be reproduced?

**Method:** Use the equations below on the relabeled score table.

**Conclusion:** Numerical fits ran remotely in Python. This optional R code has not been executed.

```{r optional-refit, eval=FALSE, echo=TRUE}
library(car)
d <- read.csv("ARM/classification_scores.csv")
for (v in c("dataset", "generator", "has_y", "labeler", "seed")) {
 d[[v]] <- factor(d[[v]])
 contrasts(d[[v]]) <- contr.sum(nlevels(d[[v]]))
}
fit <- lm(macro_F1 ~ dataset + generator * has_y * labeler * seed, data=d)
fit_seed_block <- lm(macro_F1 ~ dataset + seed + generator * has_y * labeler, data=d)
fit_dataset <- lm(macro_F1 ~ dataset * (has_y * labeler) +
 generator * has_y * labeler + seed, data=d)
car::Anova(fit, type=2)
anova(fit_seed_block, fit)
anova(fit_seed_block, fit_dataset)
e <- read.csv("ARM/dataset_effects.csv")
p <- sapply(unique(e$contrast), function(name)
 t.test(e$difference[e$contrast == name], mu=0)$p.value)
p.adjust(p, method="holm")
for (response in c("R2", "D2")) {
 r <- read.csv(paste0("ARM/", response, "_scores.csv"))
 for (v in c("dataset", "generator", "has_y", "labeler", "seed")) {
  r[[v]] <- factor(r[[v]])
  contrasts(r[[v]]) <- contr.sum(nlevels(r[[v]]))
 }
 pooled <- lm(as.formula(paste(response,
  "~ dataset + generator * has_y * labeler * seed")), data=r)
 car::Anova(pooled, type=2)
 for (task in levels(r$dataset)) {
  within_task <- lm(as.formula(paste(response,
   "~ seed + generator * has_y * labeler")), data=r[r$dataset == task, ])
  car::Anova(within_task, type=2)
 }
}
```
'''.replace('ARM',arm)
    for ext,content in [('.md',text),('.Rmd',header+text+rcode),('.html',HTML_HEAD+'\n'.join(web)+'</body></html>')]:write_atomic(OUT/(stem+ext),content.encode('utf-8'))
    reports.append(dict(stem=stem,pdf=str(FINALPDF/pdf.name),pages=len(PdfReader(pdf).pages),sections=sections,tables=tables,html='\n'.join(web),md=text,rmd=header+text+rcode))

for arm in ['synthetic','mix']:
    for path in (DATA/arm).rglob('*'):
        if path.is_file():copy_exact(path,OUT/arm/path.relative_to(DATA/arm))
    for path in (REG/arm).rglob('*'):
        if path.is_file():copy_exact(path,OUT/arm/path.relative_to(REG/arm))
copy_exact(REG/'regression_analysis_metadata.json',OUT/'regression_analysis_metadata.json')
for name in ['analysis_metadata.json','original_references.csv']:copy_exact(DATA/name,OUT/name)
for name in ['analyze_explicit.py','analyze_regression.py','remote_regression_io.py','build_explicit_report.py','verify_explicit.py','EXPLICIT_ANALYSIS_HANDOFF.md','REGRESSION_ANALYSIS_HANDOFF.md']:copy_exact(HERE/name,OUT/name)
copy_exact(ROOT/'audit/statistical_review_20261006/analyze_results.py',OUT/'analyze_results.py')
for src,dst in [(HERE/'snapshot/snapshot.json',OUT/'source_snapshot.json'),(HERE/'snapshot/scores.csv',OUT/'frozen_inputs/scores.csv'),
 (HERE/'results/regression_d2_scores.csv',OUT/'frozen_inputs/regression_d2_scores.csv'),(HERE/'results/d2_verification.json',OUT/'d2_verification.json'),
 (HERE/'original_final_328_project.Rmd',OUT/'original_final_328_project.Rmd'),(HERE/'latest_source_verification.json',OUT/'latest_source_verification.json')]:copy_exact(src,dst)
combined=PdfWriter()
for r in reports:combined.append(PDFOUT/Path(r['pdf']).name)
combined_pdf=PDFOUT/'real_data_328_analysis.pdf';buffer=io.BytesIO();combined.write(buffer);write_atomic(combined_pdf,buffer.getvalue())
for ext,content in [('.html',HTML_HEAD+reports[0]['html']+'<hr>'+reports[1]['html']+'</body></html>'),('.md','\n\n'.join(r['md'] for r in reports)),
 ('.Rmd',reports[0]['rmd']+'\n\n'+reports[1]['rmd'].split('---',2)[2])]:write_atomic(OUT/('real_data_328_analysis'+ext),content.encode('utf-8'))
readme='''There are TWO analyses: synthetic-only and mix, each with its own factorial fit.
The combined real_data_328_analysis PDF/HTML contains both; it is not a third analysis.
Primary equation, with categorical seed:
macro_F1 ~ dataset + generator * has_y * labeler * seed
This complete grid contains RF/XGB/DNN relabeled runs only: 120 scores per analysis.
Twenty generated-target scores are explicit matched baseline comparisons. Real-only
scores are references. No nonexistent generated-target/has_y=0 cell is estimated.
No composite treatment factor is used. Current tables are only in synthetic/ and mix/.
The former mixed model and repeated-measures appendix are removed from these reports.
Original 328 template structure: ANOVA, model checks, estimated means and effect contrasts.
Numerical fits ran remotely in Python; optional R code has not been executed.
Bold and stars use the p-value stated for each table. Confidence intervals are unadjusted.
Frozen input files preserve the historical schema verbatim as evidence, not model factors.
R2 and D2 have separate exploratory ANOVAs on 48 relabeled scores per training analysis:
outcome ~ dataset + generator * has_y * labeler * seed (rank 25, residual df 23).
Within each regression task: outcome ~ seed + generator * has_y * labeler
(rank 13, residual df 11). A within-task full four-factor fit would have zero residual df.
Regression ANOVA and model contrast p-values are nominal, not Holm-adjusted.
Generated-target improvements are descriptive comparisons outside those ANOVAs.
Only News and Housing are available; these tests do not establish effects across new tasks.
All 406 records and 116 D2 sidecars remain unchanged; see latest_source_verification.json.
Remote cache: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010
Regression cache: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_regression_20261010
Never run experiments locally. Prior deliverables are preserved in the audit backup.
'''
write_atomic(OUT/'README.txt',readme.encode('utf-8'))
files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='real_data_328_analysis_package.zip']+list(PDFOUT.glob('*.pdf'))
manifest=dict(reports=[{k:v for k,v in r.items() if k not in ['html','md','rmd']} for r in reports],combined_pdf=str(FINALPDF/combined_pdf.name),
 combined_pages=len(PdfReader(combined_pdf).pages),files={str((FINALOUT/p.relative_to(OUT) if p.is_relative_to(OUT) else FINALPDF/p.name).relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(HERE/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
package=OUT/'real_data_328_analysis_package.zip';tmp=package.with_suffix('.tmp.zip')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in files:archive.write(p,str(p.relative_to(OUT)) if p.is_relative_to(OUT) else p.name)
    archive.write(HERE/'deliverable_manifest.json','deliverable_manifest.json')
tmp.replace(package)
print(json.dumps(dict(staging=str(STAGE),reports=[dict(stem=r['stem'],pages=r['pages']) for r in reports],combined_pages=manifest['combined_pages']),indent=2))
