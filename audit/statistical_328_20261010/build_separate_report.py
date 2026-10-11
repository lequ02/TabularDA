"""Publish two independent real-data analyses from remotely fitted models."""
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader, PdfWriter

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent;DATA=HERE/'separate_results'
OUT=ROOT/'output/statistical_analysis_328_20261010';PDFOUT=ROOT/'output/pdf/statistical_analysis_328_20261010'
DS={'adult':'Adult','census_kdd':'Census KDD','covertype':'Covertype','mnist12':'MNIST12','mnist28':'MNIST28','news':'News','california_housing':'Housing'}
CN={'relabel_minus_generated':'Relabeled - generated','has_y1_minus_has_y0':'has_y=1 - has_y=0','has_y1_minus_generated':'has_y=1 - generated','has_y0_minus_generated':'has_y=0 - generated'}
def read(name):
    with (DATA/arm/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def f(x,n=4):return f'{float(x):.{n}f}'
def sign(x):return f'{float(x):+.4f}'
def pv(x):return f'{float(x):.2e}' if float(x)<.0001 else f'{float(x):.4f}'
def stars(x):return '***' if float(x)<.001 else '**' if float(x)<.01 else '*' if float(x)<.05 else '.' if float(x)<.1 else ''
def ci(r):return '['+sign(r['ci_low'])+', '+sign(r['ci_high'])+']'
def markup(t):return re.sub(r'(?<!\*)\*\*(?!\*)(.+?)(?<!\*)\*\*(?!\*)',r'<b>\1</b>',html.escape(str(t)))
def term(t):return re.sub(r'C\(([^)]+)\)',r'\1',t).replace(':',' x ')
INK=colors.HexColor('#193247');TEAL=colors.HexColor('#237b8a');PALE=colors.HexColor('#edf4f6')
styles={
 'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=INK,spaceAfter=11),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.3,leading=13,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8,leading=11,textColor=INK,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=10.5,textColor=INK),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8,leading=10.5,textColor=INK),
 'equation':ParagraphStyle('equation',fontName='Courier',fontSize=8.2,leading=11.5,textColor=INK,spaceAfter=8)}
def para(t,style='body'):
    story.append(Paragraph(markup(t),styles[style]));md.append(t+'\n');web.append('<p class="'+style+'">'+markup(t)+'</p>')
def eq(t):
    story.append(Paragraph(html.escape(t),styles['equation']));md.append('```text\n'+t+'\n```\n');web.append('<pre>'+html.escape(t)+'</pre>')
def section(title,q,method,conclusion):
    if sections:story.append(PageBreak());web.append('<hr>')
    sections.append(title);story.append(Paragraph(markup(title),styles['h1']))
    md.append('# '+title+'\n');web.append('<h1>'+markup(title)+'</h1>')
    para('**Research question:** '+q);para('**Method:** '+method);para('**Conclusion:** '+conclusion)
def table(headers,rows,widths,pvals=None,pad=3):
    assert sum(widths)==516
    bold=[p is not None and float(p)<.05 for p in pvals] if pvals else [False]*len(rows)
    cells=[[Paragraph(html.escape(str(x)),styles['head']) for x in headers]]
    cells += [[Paragraph('<b>'+html.escape(str(x))+'</b>' if b else html.escape(str(x)),styles['cell']) for x in row] for row,b in zip(rows,bold)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d7e3e9')),
      ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,7)]);md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    for row,b in zip(rows,bold):
        escaped=[str(x).replace('*','\\*') for x in row]
        md.append('| '+' | '.join('**'+x+'**' if b else x for x in escaped)+' |')
    md.append('');web.append('<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+('<b>'+html.escape(str(x))+'</b>' if b else html.escape(str(x)))+'</td>' for x in row)+'</tr>' for row,b in zip(rows,bold))+'</tbody></table>')
    if pvals:tables.append(dict(headers=headers,rows=rows,pvalues=pvals,bold=bold))
def legend(basis):
    t='Symbols: *** p < .001; ** p < .01; * p < .05; . p < .10. Bold: p < .05. Uses '+basis+'.'
    para(t,'small');md[-1]=t.replace('*','\\*')+'\n'
def figure(name,caption,width=500):
    path=DATA/arm/name
    with PILImage.open(path) as im:w,h=im.size
    story.append(Image(str(path),width=width,height=width*h/w));para(caption,'small')
    md.append('!['+caption+']('+arm+'/'+name+')\n');web.append('<img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
def anova(name):
    data=read(name)
    table(['Term','SS','df','F','Nominal p','Sig.'],[[term(r['term']),f(r['sum_sq']),f(r['df'],0),f(r['F'],2) if r['F'] else '',pv(r['PR(>F)']) if r['PR(>F)'] else '',stars(r['PR(>F)']) if r['PR(>F)'] else ''] for r in data],[213,72,36,60,91,44],pvals=[r['PR(>F)'] or None for r in data])
HTML_HEAD='''<!doctype html><html><head><meta charset="utf-8"><style>
body{font:15px/1.5 system-ui;color:#193247;max-width:1000px;margin:40px auto;padding:0 24px}
h1{font-size:25px}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 20px}th{background:#edf4f6;text-align:left}td,th{padding:7px;border-bottom:1px solid #d7e3e9;vertical-align:top}img{max-width:100%}pre{white-space:pre-wrap;background:#edf4f6;padding:10px;font-size:13px}.small{font-size:13px}hr{margin:40px 0}</style></head><body>'''
reports=[]
for arm in ['synthetic','mix']:
    title='Synthetic-only' if arm=='synthetic' else 'Mix: real + synthetic'
    story=[];md=[];web=[];sections=[];tables=[]
    tests=read('planned_contrasts.csv');mixed=read('mixed_contrasts.csv');labels=read('labeler_effects.csv')
    main=tests[0];hy=tests[1];meta=json.loads((DATA/arm/'analysis_metadata.json').read_text())
    section('1. '+title+': design and terms','Which experimental choices are being compared?',
      'One analysis of this training dataset: 5 datasets x 2 seeds x 2 generators x 7 conditions = 140 classification scores.',
      'Each generator has one generated-target condition and six relabeled conditions. Real-only training is a separate reference.')
    table(['Term','Definition','Levels'],[
      ['dataset','Real-data prediction task.','Classification: Adult, Census KDD, Covertype, MNIST12, MNIST28. Regression: News, Housing.'],
      ['generator','Model fitted to real training data to generate synthetic rows.','CTGAN; TVAE.'],
      ['has_y','Whether Y is included when fitting the generator.','0: fit X only. 1: fit (X,Y). This does not describe whether the final table has a target.'],
      ['labeler','Model trained on real training data to predict replacement targets.','RF: random forest. XGB: XGBoost. DNN: neural network.'],
      ['condition','Complete target construction: generated Y, or labeler plus has_y.','Seven levels listed below.'],
      ['seed','Experiment random-number setting.','42; 43. Both reuse the prepared heldout split.'],
      ['original','Real-only downstream training reference.','Outside the seven-condition ANOVA.']],[84,216,216],pad=3)
    table(['condition','has_y','Target Y used in synthetic rows'],[['generated','1','Retain generator output.']]+[[label+'_has_y'+str(h),str(h),'Replace with '+label.upper()+' predictions.'] for label in ['rf','xgb','dnn'] for h in [0,1]],[180,60,276],pad=2)
    para('**Training rows:** '+meta['training_rows']+'. Mix concatenates all rows; it is not a fixed 50/50 mixture. Synthetic-only and mix have separate fits throughout.','small')
    para('**X** = features; **Y** = target. Both has_y constructions ultimately contain X and Y. Generated Y has no has_y=0 cell. condition represents the seven observed constructions without inventing that cell. RF/XGB/DNN are labelers; the downstream evaluator is a neural model.','small')

    section('2. Does relabeling or has_y improve utility?',
      'Does predicted Y beat generated Y, and does including Y during generator fitting help?',
      'macro_F1 ~ generator * condition + (1 | dataset). Contrast seven condition means with equal generator weights; check each contrast with a paired t-test on five dataset means.',
      'Relabeling: **'+sign(main['score_change'])+' macro F1 ('+f(main['relative_percent'],2)+'% relative)**. has_y=1 minus 0: **'+sign(hy['score_change'])+'**. Relabeling is significant under the mixed model; neither contrast is significant across datasets after Holm adjustment.')
    eq('macro_F1 ~ generator * condition + (1 | dataset)')
    eq('F1[d,s,g,c] = mu + G[g] + C[c] + GC[g,c] + u[d] + e[d,s,g,c]')
    para('u[d] ~ Normal(0, tau^2); e ~ Normal(0, sigma^2), conditionally independent. d = dataset, s = seed, g = generator, c = condition. REML fit; normal-Wald contrasts. This assumes common treatment effects across datasets.','small')
    table(['Comparison','First mean','Second mean','F1 change','Relative'],[[CN[x['contrast']],f(x['left_mean']),f(x['right_mean']),sign(x['score_change']),f(x['relative_percent'],2)+'%'] for x in tests],[180,82,82,90,82])
    para('**Mixed-model contrasts** (95% normal-Wald intervals)')
    table(['Comparison','95% CI','Holm p','Sig.'],[[CN[x['contrast']],ci(x),pv(x['p_wald_holm']),stars(x['p_wald_holm'])] for x in mixed],[220,160,92,44],pvals=[x['p_wald_holm'] for x in mixed])
    para('**Dataset-level check:** delta[d] = sum(c) w[c] x mean(s,g) F1[d,s,g,c]; t = mean(delta) / (SD(delta) / sqrt(5)), df = 4. This asks whether the gain is consistent across tasks.','small')
    table(['Comparison','95% paired-t CI','Holm p','Sig.'],[[CN[x['contrast']],ci(x),pv(x['p_t_holm']),stars(x['p_t_holm'])] for x in tests],[220,160,92,44],pvals=[x['p_t_holm'] for x in tests])
    legend('Holm-adjusted p over four contrasts within this analysis')
    para('Contrast order: generated, RF0, RF1, XGB0, XGB1, DNN0, DNN1. Relabeling weights: (-1, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6). has_y weights: (0, -1/3, 1/3, -1/3, 1/3, -1/3, 1/3). The other two compare the three has_y=1 or three has_y=0 conditions with generated Y.','small')
    para('CI limits are unadjusted. Macro F1 averages target-class F1 scores equally (0-1). Relative = change / second mean. A change of .01 is one percentage point.','small')

    interaction=next(x for x in read('lm_dataset_interaction_anova.csv') if x['term']=='C(dataset):C(condition)')
    section('3. ANOVA: do effects depend on dataset?',
      'Do CTGAN and TVAE differ, do conditions differ, and do condition effects change across datasets?',
      'Type II ANOVA: macro_F1 ~ dataset + generator * condition; macro_F1 ~ dataset * condition + generator. These follow the original-project fixed-effect models.',
      'The dataset x condition term is significant (nominal p = **'+pv(interaction['PR(>F)'])+'**): construction effects differ across tasks.')
    eq('Blocked: macro_F1 ~ dataset + generator * condition')
    eq('F1[d,s,g,c] = mu + D[d] + G[g] + C[c] + GC[g,c] + e[d,s,g,c]')
    anova('lm_blocked_anova.csv')
    eq('Interaction: macro_F1 ~ dataset * condition + generator')
    eq('F1[d,s,g,c] = mu + D[d] + C[c] + DC[d,c] + G[g] + e[d,s,g,c]')
    anova('lm_dataset_interaction_anova.csv');legend('nominal ANOVA p')
    para('generator: CTGAN-TVAE difference. condition: any difference among seven constructions. generator x condition: construction effects depend on generator. dataset x condition: construction effects depend on task. * includes main effects and their interaction.','small')
    para('SS = sum of squares; df = degrees of freedom; F = term mean square / residual mean square. Scores within a dataset share heldout rows, so nominal p-values are exploratory. The dataset-level check on page 2 governs claims across tasks.','small')
    model=meta['ols_models'][2]
    para('**Diagnostics:** interaction-model residual Shapiro p = '+pv(model['shapiro_p'])+'; adjusted R-squared = '+f(model['adjusted_r_squared'])+'. Residual plots and the full ANOVA macro_F1 ~ dataset * generator * condition are supplied. They check model assumptions; they do not add a research question.','small')

    best=max(labels,key=lambda x:float(x['mean']));count=sum(float(x['p_t_holm'])<.05 for x in labels)
    section('4. Which labeler produces the largest gains?',
      'How do RF, XGB and DNN compare with their own generator baseline?',
      'For each generator and has_y: delta[d] = mean(s) F1[d,s,labeler] - mean(s) F1[d,s,generated]. Test mean(delta)=0 using t = mean(delta)/(SD(delta)/sqrt(5)), df=4.',
      'Largest descriptive mean gain: '+best['generator'].upper()+' / '+best['labeler'].upper()+' / has_y='+best['has_y']+' (**'+sign(best['mean'])+'**). '+str(count)+' of 12 comparisons pass Holm-adjusted p < .05.')
    table(['Generator','Labeler','has_y','F1 change','Relative gain','Holm p','Sig.'],[[x['generator'].upper(),x['labeler'].upper(),x['has_y'],sign(x['mean']),f(x['relative_percent'],2)+'%',pv(x['p_t_holm']),stars(x['p_t_holm'])] for x in labels],[88,65,53,95,90,81,44],pvals=[x['p_t_holm'] for x in labels])
    legend('Holm-adjusted p across twelve comparisons in this analysis')
    figure('dataset_effects.png','Five dataset means show where the main effects occur. Positive values favor relabeling or has_y=1.')
    para('Seeds are averaged within each task; all five tasks have equal weight. The largest observed gain does not establish a universally best labeler. Exact sign-flip tests and confidence intervals are supplied.','small')

    reg=read('regression_summary.csv')
    section('5. Regression: do constructions improve prediction?',
      'What happens on News and Housing when generated targets are replaced?',
      'Report means across two seeds and two generators; relabeled means also average RF/XGB/DNN. Keep the three metric equations below separate.',
      'Housing relabeling improves all three average metrics. News relabeling improves R2 but worsens NMAE and D2. The News conclusion depends on the loss function. Two tasks support descriptive comparisons only.')
    eq('R2 = 1 - sum((Y - prediction)^2) / sum((Y - mean(Y))^2)')
    eq('NMAE = mean(abs(Y - prediction)) / SD_test(Y), ddof = 0')
    eq('D2 = 1 - MAE_prediction / MAE_test_median_baseline')
    table(['Dataset / metric','Real only','Generated Y','Relabeled has_y=0','Relabeled has_y=1'],[[DS[x['dataset']]+' / '+{'r2':'R2','nmae_sigma':'NMAE','d2_absolute_error':'D2'}[x['metric']],f(x['original']),f(x['generated']),f(x['has_y0']),f(x['has_y1'])] for x in reg],[132,76,88,110,110])
    para('**Direction:** higher R2 and D2 are better; lower NMAE is better. R2 uses the test-mean baseline. D2 uses a constant test-median baseline and absolute error. Zero matches the baseline; negative is worse.','small')
    para('News uses its log-target rerun, scored in original units; Housing uses the no-Faker rerun. Read saved test normalization and verified D2 sidecars. Denominators are not recomputed. Real-only values are reference runs, outside the generator conditions.','small')
    para('**Scope and provenance:** 406 verified selected runs across both analyses: 290 classification and 116 regression; 116 verified D2 sidecars. The shared 14 real-only runs appear in both reports. Frozen source: Oct 10, 2026, 1:08:42 a.m. Chicago. Credit, Intrusion, NB/PCA-GMM labelers and simulated experiments are outside this analysis.','small')
    para('Production generators use 500 epochs and 100,000 synthetic rows. Census uses weighted macro-F1 development selection; other datasets use development loss. The package supplies source hashes and model metadata. Completion of the broader matrix is not claimed.','small')
    para('**MNIST caveat:** output heads were repaired, but retained MNIST12 TVAE synthetic tables still have documented holdout-feature collisions. Interpret their utility with that limitation. Seeds share test partitions; they are not independent test samples.','small')

    repeated=read('repeated_anova.csv');count=sum(float(x['p_GG_holm'])<.05 for x in repeated)
    section('Appendix. What does the repeated-measures check do?',
      'Are ANOVA patterns visible when seed runs are treated as repeated units?',
      'Use the contrast-projection ANOVA below, with dataset/seed units and a separate residual stratum for each treatment contrast. Apply Greenhouse-Geisser (GG) correction, then Holm across 19 terms.',
      str(count)+' of 19 terms pass GG plus Holm adjustment. This checks variation between seed runs on the same tasks. Shared test splits limit independence; it does not replace the five-dataset contrasts.')
    eq('z[d,s,k] = 100 * sum(g,c) q[k,g,c] * F1[d,s,g,c]')
    eq('z[d,s,k] = beta[d,k] + e[d,s,k]; test H0: C * beta = 0')
    para('q is an orthonormal generator x condition contrast. Residuals are dataset:seed x that contrast (5 residual seed df per coordinate). C averages datasets for main effects or compares them for dataset interactions. Factor equation: F1 ~ dataset * generator * condition with Error(dataset:seed / (generator * condition)).','small')
    table(['Term','df num','df den','F','GG + Holm p','Sig.'],[[term(x['term']),f(x['df_num'],0),f(x['df_den'],0),f(x['F'],2),pv(x['p_GG_holm']),stars(x['p_GG_holm'])] for x in repeated],[210,44,44,70,104,44],pvals=[x['p_GG_holm'] for x in repeated],pad=2)
    legend('GG-corrected, Holm-adjusted p; displayed df are before GG')
    para('The six condition df split into: approach (relabeling versus generated Y, 1 df); labeler (RF/XGB/DNN within relabeling, 2 df); has_y (1 df); labeler x has_y (2 df). approach names the page 2 contrast; it is not another fitted factor. Weights, error strata and independent F checks are supplied.','small')

    stem='synthetic_only_328_analysis' if arm=='synthetic' else 'mix_328_analysis';pdf=PDFOUT/(stem+'.pdf')
    def footer(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(INK)
        c.drawString(48,26,title+' | Frozen: Oct 10, 2026, 1:08 a.m. Chicago');c.drawRightString(564,26,str(doc.page));c.restoreState()
    SimpleDocTemplate(str(pdf),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=40,bottomMargin=42,title=title+' - 328 real-data analysis',author='Minh Le').build(story,onFirstPage=footer,onLaterPages=footer)
    text='\n'.join(md);header='---\ntitle: "'+title+' - 328 real-data analysis"\noutput:\n  html_document:\n    toc: true\n  pdf_document: default\n---\n\n'
    rcode='''\n# Optional equivalent R refit

**Research question:** How can the supplied fits be reproduced?

**Method:** Fit the same equations below using this independent score table.

**Conclusion:** Reported fits ran remotely in Python. This optional R refit has not run; use asymptotic Wald inference for comparison.

```{r optional-refit, eval=FALSE, echo=TRUE}
library(car)
library(lme4)
library(emmeans)
d <- read.csv("ARM/classification_scores.csv")
d$dataset <- factor(d$dataset)
d$generator <- factor(d$generator)
d$condition <- factor(d$condition, levels=c("generated", "rf_has_y0", "rf_has_y1",
 "xgb_has_y0", "xgb_has_y1", "dnn_has_y0", "dnn_has_y1"))
fit_full <- lm(macro_F1 ~ dataset * generator * condition, data=d)
fit_block <- lm(macro_F1 ~ dataset + generator * condition, data=d)
fit_interaction <- lm(macro_F1 ~ dataset * condition + generator, data=d)
car::Anova(fit_block, type=2)
car::Anova(fit_interaction, type=2)
fit_mixed <- lmer(macro_F1 ~ generator * condition + (1 | dataset), data=d, REML=TRUE)
emm <- emmeans(fit_mixed, ~ condition, weights="equal", lmer.df="asymptotic")
contrast(emm, method=list(
 relabel_minus_generated=c(-1,rep(1/6,6)),
 has_y1_minus_has_y0=c(0,-1/3,1/3,-1/3,1/3,-1/3,1/3),
 has_y1_minus_generated=c(-1,0,1/3,0,1/3,0,1/3),
 has_y0_minus_generated=c(-1,1/3,0,1/3,0,1/3,0)), adjust="holm")
e <- read.csv("ARM/dataset_effects.csv")
p <- sapply(unique(e$contrast), function(name)
 t.test(e$difference[e$contrast == name], mu=0)$p.value)
p.adjust(p, method="holm")
```
'''.replace('ARM',arm)
    for ext,content in [('.md',text),('.Rmd',header+text+rcode),('.html',HTML_HEAD+'\n'.join(web)+'</body></html>')]:
        (OUT/(stem+ext)).write_text(content,encoding='utf-8')
    reports.append(dict(stem=stem,pdf=str(pdf),pages=len(PdfReader(pdf).pages),sections=sections,tables=tables,html='\n'.join(web),md=text,rmd=header+text+rcode))

def copy_verified(src,dst):
    payload=src.read_bytes()
    if not dst.exists() or dst.read_bytes()!=payload:
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(payload)
    assert dst.read_bytes()==payload
for arm in ['synthetic','mix']:
    for path in (DATA/arm).rglob('*'):
        if path.is_file():copy_verified(path,OUT/arm/path.relative_to(DATA/arm))
for name in ['analysis_metadata.json','contrast_weights.csv','original_references.csv']:copy_verified(DATA/name,OUT/name)
(OUT/'frozen_inputs').mkdir(exist_ok=True)
copy_verified(HERE/'snapshot/scores.csv',OUT/'frozen_inputs/scores.csv')
copy_verified(HERE/'results/regression_d2_scores.csv',OUT/'frozen_inputs/regression_d2_scores.csv')
for name in ['analyze_separate.py','build_separate_report.py']:copy_verified(HERE/name,OUT/name)
for name in ['rebuild_rf_xgb_dnn.py','analyze_results.py']:copy_verified(ROOT/'audit/statistical_review_20261006'/name,OUT/name)
copy_verified(HERE/'snapshot/snapshot.json',OUT/'source_snapshot.json')
copy_verified(HERE/'original_final_328_project.Rmd',OUT/'original_final_328_project.Rmd')
copy_verified(HERE/'results/d2_verification.json',OUT/'d2_verification.json')
combined=PdfWriter()
for r in reports:combined.append(r['pdf'])
combined_pdf=PDFOUT/'real_data_328_analysis.pdf'
stream=io.BytesIO();combined.write(stream)
combined_temp=combined_pdf.with_suffix('.tmp.pdf')
combined_temp.write_bytes(stream.getvalue())
combined_temp.replace(combined_pdf)
(OUT/'real_data_328_analysis.html').write_text(HTML_HEAD+reports[0]['html']+'<hr>'+reports[1]['html']+'</body></html>',encoding='utf-8')
(OUT/'real_data_328_analysis.md').write_text('\n\n'.join(r['md'] for r in reports),encoding='utf-8')
(OUT/'real_data_328_analysis.Rmd').write_text(reports[0]['rmd']+'\n\n'+reports[1]['rmd'].split('---',2)[2],encoding='utf-8')
(OUT/'README.txt').write_text('''Use synthetic_only_328_analysis or mix_328_analysis; the combined report contains both.
Each has six sections beginning with a question, actual method, and conclusion.
Synthetic-only and mix are fitted independently; no pooled effect or training factor.
has_y is the sole name for target inclusion during generator fitting. condition has
seven observed constructions. There is no generated-target has_y=0 condition.
Bold ANOVA rows: p < .05; *** < .001; ** < .01; * < .05; . < .10.
Tables label nominal/GG/Holm p-values; confidence intervals are unadjusted.
Original 328 template: blocked/interacting ANOVA, mixed-model means, planned effects.
Numerical fits ran remotely in Python on frozen verified scores. Optional R code
has not been executed. Never run experiments locally.
Remote cache: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_separate_20261010
Source snapshot: Oct 10, 2026, 1:08:42 a.m. Chicago; 406 runs and 116 D2 sidecars.
Current statistical tables are in synthetic/ and mix/. Earlier pooled CSVs beside
this report are historical; they are excluded from this package.
frozen_inputs/ preserves the original input schema verbatim as evidence. Its legacy
column names are not additional factors in either current analysis.
''',encoding='utf-8')
selected=[p for arm in ['synthetic','mix'] for p in (OUT/arm).rglob('*') if p.is_file()]
selected += list((OUT/'frozen_inputs').glob('*.csv'))
selected += [OUT/name for name in ['analysis_metadata.json','contrast_weights.csv','original_references.csv','analyze_separate.py','build_separate_report.py','rebuild_rf_xgb_dnn.py','analyze_results.py','source_snapshot.json','original_final_328_project.Rmd','d2_verification.json','README.txt','real_data_328_analysis.html','real_data_328_analysis.md','real_data_328_analysis.Rmd']]
selected += [OUT/(r['stem']+ext) for r in reports for ext in ['.html','.md','.Rmd']]
selected += [Path(r['pdf']) for r in reports]+[combined_pdf]
manifest=dict(reports=[{k:v for k,v in r.items() if k not in ['html','md','rmd']} for r in reports],combined_pdf=str(combined_pdf),combined_pages=len(PdfReader(combined_pdf).pages),files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected})
(HERE/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
package=OUT/'real_data_328_analysis_package.zip'
package_temp=package.with_suffix('.tmp.zip')
with zipfile.ZipFile(package_temp,'w',zipfile.ZIP_DEFLATED) as z:
    for p in selected:z.write(p,str(p.relative_to(OUT)) if p.is_relative_to(OUT) else p.name)
    z.write(HERE/'deliverable_manifest.json','deliverable_manifest.json')
package_temp.replace(package)
print(json.dumps(dict(reports=[dict(stem=r['stem'],pages=r['pages']) for r in reports],combined_pages=manifest['combined_pages']),indent=2))
