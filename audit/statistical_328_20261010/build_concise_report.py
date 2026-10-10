"""Revise presentation of the frozen 328 analysis; do not recompute statistics."""
import base64
import csv
import hashlib
import html
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
from pypdf import PdfReader

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
DATA=HERE/'results';OUT=ROOT/'output/statistical_analysis_328_20261010'
PDFOUT=ROOT/'output/pdf/statistical_analysis_328_20261010'
def read(name):
    with (DATA/name).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
tests=read('planned_contrasts.csv');effects=read('dataset_effects.csv');mixed=read('mixed_contrasts.csv')
labels=read('labeler_effects.csv');anova=read('repeated_anova.csv');models=read('lm_model_summary.csv')
groups=read('group_scores.csv');refs=read('original_references.csv');reg=read('regression_scores.csv')
d2=read('regression_d2_summary.csv');snapshot=json.loads((OUT/'source_snapshot.json').read_text())
CLASS=['adult','census_kdd','covertype','mnist12','mnist28']
DS={'adult':'Adult','census_kdd':'Census KDD','covertype':'Covertype','mnist12':'MNIST12','mnist28':'MNIST28',
    'news':'News','california_housing':'Housing'}
NAMES={'new_vs_old':'Relabeled - generated','has_y_vs_no_y':'Full - X-only',
       'full_relabel_vs_generated':'Full relabeled - generated','xonly_relabel_vs_generated':'X-only relabeled - generated'}
def f(value,n=4):return f'{float(value):.{n}f}'
def signed(value,n=4):return f'{float(value):+.{n}f}'
def pv(value):return f'{float(value):.2e}' if float(value)<.0001 else f'{float(value):.4f}'
def stars(value):
    value=float(value)
    return '***' if value<.001 else '**' if value<.01 else '*' if value<.05 else '.' if value<.1 else ''
def interval(row):return f"[{signed(row['ci_low'])}, {signed(row['ci_high'])}]"
def bold(value):return '**'+str(value)+'**'
def markup(text):return re.sub(r'(?<!\*)\*\*(?!\*)(.+?)(?<!\*)\*\*(?!\*)',r'<b>\1</b>',html.escape(str(text)))
def term_name(term):
    term=re.sub(r'C\(([^)]+)\)',r'\1',term)
    return term.replace('training_mode','mode').replace('x_inclusion','has_y').replace(':',' x ')
INK=colors.HexColor('#193247');TEAL=colors.HexColor('#237b8a');PALE=colors.HexColor('#edf4f6')
styles={
 'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=INK,spaceAfter=12),
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.5,leading=14,textColor=INK,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.2,leading=11.5,textColor=INK,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8,leading=10.7,textColor=INK),
 'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=8,leading=10.7,textColor=INK)}
story=[];md=[];web=[];rmd=[];sections=[]
def para(text,style='body'):
    story.append(Paragraph(markup(text),styles[style]));md.append(text+'\n');rmd.append(text+'\n')
    web.append(f'<p class="{style}">{markup(text)}</p>')
def section(title,question,method,conclusion):
    if sections:story.append(PageBreak());web.append('<hr>')
    sections.append(title)
    story.append(Paragraph(markup(title),styles['h1']))
    md.append('# '+title+'\n');rmd.append('# '+title+'\n');web.append('<h1>'+markup(title)+'</h1>')
    para('**Research question:** '+question)
    para('**Method:** '+method)
    para('**Finding:** '+conclusion)
def table(headers,rows,widths=None,padding=4,significant=None):
    widths=widths or [516/len(headers)]*len(headers)
    assert abs(sum(widths)-516)<1e-8
    marked=[]
    for i,row in enumerate(rows):
        escaped=[str(v).replace('*','\\*') if str(v) in ('*','**','***') else str(v) for v in row]
        marked.append([bold(v) for v in escaped] if significant and significant[i] else escaped)
    cells=[[Paragraph(html.escape(str(v)),styles['head']) for v in headers]]
    for i,row in enumerate(rows):
        cells.append([Paragraph('<b>'+html.escape(str(v))+'</b>' if significant and significant[i]
                      else html.escape(str(v)),styles['cell']) for v in row])
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d7e3e9')),
      ('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),
      ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,8)])
    text=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in marked]+['']
    md.extend(text);rmd.extend(text)
    web.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+('<b>'+html.escape(str(v))+'</b>' if significant and significant[i] else html.escape(str(v)))+'</td>' for v in row)+'</tr>' for i,row in enumerate(rows))+'</tbody></table></div>')
def legend(column):
    text='Significance: p < 0.001 = ***; p < 0.01 = **; p < 0.05 = *; p < 0.10 = .; blank otherwise. Bold rows have p < 0.05. Symbols use '+column+'.'
    para(text,'small')
    md[-1]=text.replace('*','\\*')+'\n';rmd[-1]=md[-1]
def figure(name,caption,width=516):
    path=OUT/name
    with PILImage.open(path) as im:w,h=im.size
    story.append(Image(str(path),width=width,height=width*h/w));para(caption,'small')
    md.append(f'![{caption}]({name})\n');rmd.append(f'![{caption}]({name})\n')
    web.append('<img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
def ordinary_anova(name):
    data=read(name)
    table(['Term','SS','df','F','Nominal p','Sig.'],[[term_name(r['term']),f(r['sum_sq']),f(r['df'],0),
        f(r['F'],2) if r['F'] else '',pv(r['PR(>F)']) if r['PR(>F)'] else '',stars(r['PR(>F)']) if r['PR(>F)'] else ''] for r in data],
        [213,72,36,60,91,44],significant=[bool(r['PR(>F)']) and float(r['PR(>F)'])<.05 for r in data])

section('1. Terms and experimental design',
 'What are the factors, and which comparisons do they define?',
 'Define each factor and list its levels before interpreting the models.',
 'Each generator has seven conditions, evaluated with two training modes and two seeds. Real-only runs are separate reference points.')
table(['Term','Meaning','Levels used'],[
 ['dataset','Task being evaluated.','Classification: Adult, Census KDD, Covertype, MNIST12, MNIST28. Regression: News and Housing.'],
 ['generator','Model producing synthetic rows.','CTGAN; TVAE.'],
 ['source / has_y','Columns used to fit the generator.','full / has_y=1: (X,Y). xonly / has_y=0: X. Both generate features X.'],
 ['labeler','Predictor trained on real training data to assign synthetic Y.','RF (random forest); XGB (XGBoost); DNN (neural network).'],
 ['condition','One complete choice of feature source and target construction.','generated; rf_full; rf_xonly; xgb_full; xgb_xonly; dnn_full; dnn_xonly.'],
 ['approach','How the synthetic target is obtained.','Generated Y; relabeled Y (the six RF/XGB/DNN conditions).'],
 ['group / approach_has_y','Three groups used in the mixed model.','generated; hybrid_xonly; hybrid_full. Each hybrid group averages RF/XGB/DNN equally.'],
 ['mode','Rows used for downstream training.','synthetic: 100,000 synthetic rows only. mix: all real training rows plus 100,000 synthetic rows.'],
 ['seed','Experiment random-number setting.','42; 43. The prepared split is the same for both.'],
 ['original','Real-only reference model.','Real training rows only; outside the seven-condition ANOVA.']],[105,187,224],padding=3)
para('**X** = predictors; **Y** = target. Generated Y exists only for a full-table generator; there is no generated-target X-only condition. The downstream evaluator is a neural model in every arm. RF/XGB are labelers, not downstream evaluators.','small')
para('These are **factor levels**, not target class labels. The selected snapshot has 406 runs: 290 classification and 116 regression. Classification models use macro F1; regression metrics are kept separate.','small')
para('**Macro F1** averages the F1 scores of all target classes equally, on a 0-1 scale.','small')

section('2. Does relabeling improve downstream utility?',
 'Do targets predicted from real training data outperform the generator\'s own targets?',
 'Compare the mean of all six relabeled conditions with generated Y. Use a dataset-intercept mixed model, then check the contrast using five dataset means.',
 'Mean macro F1 rises from **0.7090 to 0.8010**: **+0.0920 points (+12.98% relative)**. The model finds a clear effect; evidence across datasets is weaker after adjustment.')
para('**Average effect by training mode**')
new=[r for r in tests if r['contrast']=='new_vs_old']
table(['Training mode','Generated','Relabeled','F1 change','Relative gain'],[[r['mode'],f(r['right_mean']),f(r['left_mean']),signed(r['score_change']),f(r['relative_percent'],2)+'%'] for r in new],[105,85,85,105,136])
para('**Same pooled contrast, two scopes of inference**')
pooled=next(r for r in new if r['mode']=='pooled');mm=next(r for r in mixed if r['contrast']=='new_vs_old')
table(['Test and question answered','95% CI for F1 change','Adjusted p','Sig.'],[
 ['Mixed model: average effect under its dataset-intercept assumptions.',interval(mm),pv(mm['p_wald_holm']),stars(mm['p_wald_holm'])],
 ['Dataset paired t-test: how variable is the effect across five tasks?',interval(pooled),pv(pooled['p_t_holm']),stars(pooled['p_t_holm'])]],
 [240,142,90,44],significant=[float(mm['p_wald_holm'])<.05,float(pooled['p_t_holm'])<.05])
legend('Holm-adjusted p over four planned contrasts')
figure('construction_boxplots.png','Boxes show variation in macro F1 across tasks, seeds and generators. Black diamonds are group means.',470)
para('The dataset-level raw t-test p is 0.0255; exact sign-flip p is 0.0625. The confidence intervals are unadjusted; Holm p accounts for testing four contrasts. Five positive task effects do not establish improvement on every future dataset. Relative gain = score change / generated mean; +0.0920 F1 equals +9.20 percentage points.','small')

section('3. Does including Y in generator fitting help?',
 'Among relabeled constructions, is fitting on (X,Y) better than fitting on X only?',
 'Match full and X-only results within dataset, seed, generator, labeler and mode; report their mean difference and dataset-level confidence interval.',
 'The pooled change is **+0.00248 macro-F1 points (+0.31% relative)**. The confidence interval includes zero; there is no clear full-table advantage.')
source=[r for r in tests if r['contrast']=='has_y_vs_no_y']
table(['Training','X-only','Full','F1 change','95% paired-t CI','Adjusted p','Sig.'],[[r['mode'],f(r['right_mean']),f(r['left_mean']),signed(r['score_change']),interval(r),pv(r['p_t_holm']),stars(r['p_t_holm'])] for r in source],[66,63,63,78,126,76,44],
 significant=[float(r['p_t_holm'])<.05 for r in source])
legend('Holm-adjusted dataset-level p over four contrasts')
table(['Dataset','Relabeled - generated','Full - X-only'],[[DS[d],signed(next(r['difference'] for r in effects if r['dataset']==d and r['mode']=='pooled' and r['contrast']=='new_vs_old')),
 signed(next(r['difference'] for r in effects if r['dataset']==d and r['mode']=='pooled' and r['contrast']=='has_y_vs_no_y'))] for d in CLASS],[156,180,180])
figure('dataset_effects.png','Left: relabeling gains. Right: full-table versus X-only gains. Positive values favor the first construction named.',480)
para('Full and X-only both generate X and replace Y with the same type of labeler. This question concerns including Y during generator fitting. A nonsignificant difference does not prove the two constructions equivalent.','small')

section('4. Do method effects differ across datasets?',
 'Can one common method effect describe all five classification tasks?',
 'Fit the original-project blocked model, then add dataset x condition. These ordinary linear models are exploratory because scores within a dataset are correlated.',
 'The dataset x condition interaction is strong in the exploratory model (p = **3.76e-12**). Method benefits vary by task, so one pooled effect is incomplete.')
para('**Blocked model:** macro F1 ~ dataset + generator * condition * mode.')
ordinary_anova('lm_blocked_anova.csv')
para('**Dataset-interaction model:** macro F1 ~ dataset * condition + generator + mode.')
ordinary_anova('lm_dataset_labeler_interaction_anova.csv')
legend('nominal p; these exploratory tables are not adjusted')
para('**How to read the terms:** generator asks whether CTGAN and TVAE differ on average. condition asks whether the seven constructions differ. mode asks whether synthetic-only and mix differ. dataset x condition asks whether the construction effect changes with the dataset. Other x terms ask the same dependence question for their named factors.','small')
para('In the formulas, **C(term)** means categorical. **a * b** includes a, b and their interaction; **a:b** is the interaction alone. **SS** is variation assigned to a term; **df** counts its independent comparisons; **F** compares term variation with residual variation. **Residual** is the variation the model leaves unexplained.','small')

section('5. Which generator and labeler combinations benefit?',
 'Which relabeled construction has the largest observed improvement over its own generator baseline?',
 'Compare each generator/labeler/source with its matched generated-Y baseline, averaging seeds and giving each of the five tasks equal weight.',
 'CTGAN has larger observed gains than TVAE; DNN gives the largest observed gain for both generators. Individual combinations remain exploratory: none passes Holm adjustment across 24 tests.')
for mode,label in [('synthetic','Synthetic-only'),('mix','Real + synthetic')]:
    para('**'+label+'**')
    data=[r for r in labels if r['mode']==mode]
    table(['Generator','Labeler','Source','F1 change','Relative gain','Adjusted p','Sig.'],[[r['generator'].upper(),r['labeler'].upper(),r['source'],signed(r['mean']),f(r['relative_percent'],1)+'%',pv(r['p_t_holm']),stars(r['p_t_holm'])] for r in data],
      [76,60,62,94,96,84,44],padding=2.5,significant=[float(r['p_t_holm'])<.05 for r in data])
legend('Holm-adjusted dataset-level p over 24 combinations')
para('These are gains over generated Y, not a direct significance test of DNN versus RF/XGB. No best seed or test epoch is selected. Per-combination means, confidence intervals and raw p-values remain in labeler_effects.csv.','small')

section('6. Do the conclusions extend to regression?',
 'How do relabeled tables compare with generated-target tables for News and Housing?',
 'Report separate means for R2, D2 absolute error and saved NMAE. Average seeds and generators equally, and average RF/XGB/DNN within each hybrid group.',
 'Housing relabeling improves all three metrics over generated Y. News improves R2 slightly, while its absolute-error metrics worsen slightly. The two regression tasks are descriptive comparisons.')
table(['Metric','Meaning','Better direction'],[
 ['R2','Squared-error reduction against a constant test-mean baseline.','Higher'],
 ['D2 absolute error','1 - MAE / MAE of the constant test-median baseline.','Higher'],
 ['NMAE','MAE / standard deviation of real test targets (ddof=0).','Lower']],[105,310,101])
regrows=[]
for dataset in ['news','california_housing']:
    for mode in ['synthetic','mix']:
        drow=next(r for r in d2 if r['dataset']==dataset and r['mode']==mode)
        for kind,label in [('original','Real only'),('generated','Generated Y'),('full','Full hybrid'),('xonly','X-only hybrid')]:
            row=[DS[dataset]+' / '+mode,label]
            for metric in ['r2']:
                values=[float(r['value']) for r in reg if r['dataset']==dataset and r['metric']==metric and
                    (r['labeler']=='original' if kind=='original' else r['mode']==mode and
                    (r['labeler']=='generated' if kind=='generated' else r['labeler'] in ('rf','xgb','dnn') and r['source']==kind))]
                row.append(f(sum(values)/len(values)))
            row.append(f(drow[kind]))
            values=[float(r['value']) for r in reg if r['dataset']==dataset and r['metric']=='nmae_sigma' and
                (r['labeler']=='original' if kind=='original' else r['mode']==mode and
                (r['labeler']=='generated' if kind=='generated' else r['labeler'] in ('rf','xgb','dnn') and r['source']==kind))]
            row.append(f(sum(values)/len(values)));regrows.append(row)
table(['Dataset / training','Construction','R2','D2','NMAE'],regrows,[157,107,84,84,84],padding=3)
para('R2 and D2 can be negative, so relative percentage changes in those scores are omitted. Saved test normalization and D2 sidecars are used directly. There are no regression ANOVA significance stars: two tasks do not support the same across-dataset inference as the five-task classification analysis.','small')

section('Appendix A. Why include a repeated-measures ANOVA?',
 'Are factor effects detectable relative to seed-to-seed variation on these fixed datasets?',
 'Treat dataset as fixed and dataset/seed as the repeat unit. Compare all generator/condition/mode scores within each unit; partition condition into approach, labeler and has_y.',
 'Relabeling is significant within this fixed-task model (adjusted p = **0.0043**); full versus X-only is not. This is a supporting check, not another test of generalization to new tasks.')
lookup={r['term']:r for r in anova}
selected=['generator','approach','labeler','x_inclusion','training_mode','dataset:approach','dataset:labeler',
          'dataset:x_inclusion','approach:training_mode','generator:approach']
rows=[];flags=[]
for name in selected:
    r=lookup[name]
    rows.append([term_name(name),f(r['F'],2),f(r['df_num_GG'],2)+', '+f(r['df_den_GG'],2),pv(r['p_GG']),pv(r['p_GG_holm']),stars(r['p_GG_holm'])])
    flags.append(float(r['p_GG_holm'])<.05)
table(['Effect','F','Adjusted df','GG p','Holm p','Sig.'],rows,[197,53,81,70,71,44],significant=flags)
legend('Holm p over all 39 repeated-measures terms')
para('**GG** (Greenhouse-Geisser) adjusts for unequal covariance among repeated comparisons. Adjusted df lists numerator and denominator degrees of freedom. **approach** compares generated Y with the six hybrids; **labeler** and **has_y** apply within hybrids only.','small')
table(['Analysis','What it contributes','Location'],[
 ['Blocked / interaction models','Show which patterns vary with factors and datasets. Their ordinary p-values are exploratory.','Section 4'],
 ['Mixed model','Estimates average construction effects with a dataset random intercept; its covariance assumptions matter.','Section 2'],
 ['Dataset-level contrasts','Use five task means to assess how effects vary across tasks.','Sections 2-3'],
 ['Repeated-measures ANOVA','Tests fixed-task effects against seed variation. It does not create more independent datasets.','This appendix']],[143,294,79])
para('There are ten dataset/seed units and five residual seed degrees of freedom. Both seeds share the holdout, so these tests do not measure test-sample uncertainty. The full 39-term table and partial eta-squared values are retained in repeated_anova.csv.','small')

section('Appendix B. Why keep the model diagnostics?',
 'Do the original-project models adequately describe the score patterns?',
 'Compare full-interaction, blocked and dataset-interaction fits; inspect residual-versus-fitted and normal Q-Q plots.',
 'The full model fits closely, but simpler-model residuals are structured and nonnormal. Their nominal ANOVA p-values are insufficient on their own.')
table(['Model','What it allows','R2','Adjusted R2','Residual df'],[
 ['Full interaction','All dataset/generator/condition/mode interactions.',f(models[0]['r_squared']),f(models[0]['adjusted_r_squared']),f(models[0]['df_resid'],0)],
 ['Blocked','Dataset starting scores; generator/condition/mode interactions.',f(models[1]['r_squared']),f(models[1]['adjusted_r_squared']),f(models[1]['df_resid'],0)],
 ['Dataset interaction','Dataset-specific condition effects; generator and mode main effects.',f(models[2]['r_squared']),f(models[2]['adjusted_r_squared']),f(models[2]['df_resid'],0)]],[105,201,62,76,72])
figure('model_diagnostics.png','Dataset-interaction model residuals: pattern in the left plot and departure from normality in the right plot.',490)
para('Model 3 Shapiro p = 3.50e-21. Macro F1 stays on its original 0-1 scale; no transformation was chosen to improve significance. Model summaries and the full-interaction ANOVA remain in the package.','small')
para('**Mixed-model specification:** macro F1 ~ group * generator * mode + (1 | dataset), fitted to 120 group means. REML estimates the variance components; Wald intervals use a normal approximation. The fit converged. A dataset random intercept allows different starting scores, but no dataset-specific treatment slopes. The dataset-level contrast checks that limitation.','small')

section('Appendix C. Sources, exclusions and reproducibility',
 'Are the report values traceable to completed runs with the intended settings?',
 'Use the verified frozen records, completed-record aggregate and source-linked D2 sidecars. Preserve all statistical values while revising the report presentation.',
 'All 406 selected runs and 116 D2 sidecars verified. This revision changes explanations, ordering and table styling; it does not rerun experiments or statistical models.')
table(['Dataset','Source namespace','Runs'],[[DS[d],scope,'58 / 58'] for d,scope in snapshot['scopes'].items()],[100,337,79])
para('**Source snapshot:** October 10, 2026, 1:08:42 a.m. Chicago. D2 was added from the verified 1:10 a.m. derived-metric update. These are the same frozen results used in the earlier report.','small')
para('**Scope:** CTGAN/TVAE and RF/XGB/DNN; seeds 42/43; synthetic-only/mix. Credit, Intrusion, NB/PCA-GMM, exploratory dropout trials and simulated data are excluded. Completion here does not establish completion of the broader matrix.','small')
para('**Checks and caveats:** models use saved development-selected checkpoints, not test-epoch maxima. Generator budgets remain 500 epochs / batch 500 / 100,000 rows. Splits are shared across seeds. MNIST uses the repaired output heads; its retained MNIST12 TVAE tables still carry the documented holdout-feature collision caveat.','small')
table(['File','Purpose'],[
 ['real_data_328_analysis.Rmd','Editable report and optional equivalent R model specification.'],
 ['planned_contrasts.csv / mixed_contrasts.csv','Dataset-level and mixed-model effects, confidence intervals and p-values.'],
 ['labeler_effects.csv / repeated_anova.csv','All method contrasts and repeated-measures terms.'],
 ['scores.csv / source_snapshot.json','Saved outcomes and source-record hashes.'],
 ['regression_* / d2_snapshot.zip','Regression scores/effects and verified D2 sidecars.']],[233,283])
para('Numerical fits ran remotely in Python. The optional R refit code has not been executed; lmerTest degrees of freedom can differ from the reported normal-Wald intervals.','small')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d5e1e7'));canvas.line(48,37,564,37)
    canvas.setFont('Helvetica',7.3);canvas.setFillColor(INK)
    canvas.drawString(48,25,'328 project | Real data | Frozen results: Oct 10, 2026, Chicago')
    canvas.drawRightString(564,25,str(doc.page))
pdf=PDFOUT/'real_data_328_analysis.pdf'
SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=48,
 title='Real-data 328 statistical analysis',author='Minh Le').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'real_data_328_analysis.md').write_text('\n'.join(md),encoding='utf-8')
css='body{max-width:1050px;margin:40px auto;padding:0 25px;font:16px/1.6 system-ui;color:#193247}h1{color:#237b8a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #d7e3e9}th{background:#edf4f6}.table{overflow-x:auto;margin:18px 0}img{width:100%;height:auto}hr{margin:48px 0;border:0;border-top:1px solid #d7e3e9}.small{font-size:14px}'
(OUT/'real_data_328_analysis.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Real-data 328 analysis</title><style>'+css+'</style></head><body>'+''.join(web)+'</body></html>',encoding='utf-8')
r_header='''---
title: "Statistical Analysis for Research on Synthesis Methods for Structured Data"
author: "Minh Le"
date: "2026-10-10"
output:
  html_document:
    toc: true
    toc_depth: 2
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo=FALSE)
```

'''
r_append='''
## Optional equivalent R refit specification

**Research question:** Can the fitted models be reproduced in R?

**Method:** Use the code below on the supplied score tables. It is disabled during
rendering of this frozen report.

**Finding:** The delivered fits were executed remotely in Python; this optional
R refit has not been run. `lmerTest` degrees of freedom may differ from the reported
normal-Wald intervals.

```{r optional-refit-in-R, eval=FALSE, echo=TRUE}
library(lme4)
library(lmerTest)
library(emmeans)
library(car)
d <- read.csv("classification_scores.csv")
for (v in c("dataset", "generator", "condition", "mode")) d[[v]] <- factor(d[[v]])
model_full_interactions <- lm(score ~ dataset * generator * condition * mode, data=d)
model_block <- lm(score ~ dataset + generator * condition * mode, data=d)
model_dataset_interaction <- lm(score ~ dataset * condition + generator + mode, data=d)
car::Anova(model_block, type=2)
car::Anova(model_dataset_interaction, type=2)
g <- read.csv("group_scores.csv")
g$group <- factor(g$group, levels=c("generated", "hybrid_xonly", "hybrid_full"))
fit <- lmer(score ~ group * generator * mode + (1 | dataset), data=g, REML=TRUE)
emm <- emmeans(fit, ~ group, weights="equal")
contrast(emm, method=list(
 newApproach_vs_oldApproach=c(-1,0.5,0.5), hasY_vs_noY=c(0,-1,1),
 full_vs_generated=c(-1,0,1), xonly_vs_generated=c(-1,1,0)), adjust="holm")
e <- read.csv("dataset_effects.csv")
for (name in unique(e$contrast)) {
 print(t.test(subset(e, mode == "pooled" & contrast == name)$difference, mu=0))
}
```
'''
(OUT/'real_data_328_analysis.Rmd').write_text(r_header+'\n'.join(rmd)+r_append,encoding='utf-8')
readme='''The report uses the original 328 project's research questions and frozen real-data
results. Each section begins with a question, method and finding. Appendix A explains
the supporting repeated-measures check; Appendix B holds model diagnostics.

Bold ANOVA rows use p < .05. *** p < .001; ** p < .01; * p < .05; . p < .10.
Each table states whether its symbols use nominal or adjusted p-values.

Open the HTML or PDF; the Rmd is editable and contains an optional equivalent R refit
specification. The numerical analysis ran remotely in Python; optional R code was not
executed. This revision changes presentation only and preserves all statistical CSVs.

Remote analysis: /home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_20261010
The exact analysis source and dependencies are included. Inspect ongoing jobs before
any remote rerun. Use /home/thuy/miniconda3/envs/env; never run experiments locally.

406 selected configurations and 116 D2 sidecars verified; broader matrix completion
is not claimed. Original source snapshot: October 10, 2026, 1:08:42 a.m. Chicago.
'''
(OUT/'README.txt').write_text(readme)
manifest=dict(source_snapshot_sha256=hashlib.sha256((OUT/'source_snapshot.json').read_bytes()).hexdigest(),
 template_path='D:/Rprojects/research_data_synthesis/final_328_project.Rmd',
 template_sha256=hashlib.sha256((HERE/'original_final_328_project.Rmd').read_bytes()).hexdigest(),
 pdf_pages=len(PdfReader(pdf).pages),sections=sections,
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='real_data_328_analysis_package.zip'},
 pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
(HERE/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT/'real_data_328_analysis_package.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():
        if path.is_file() and path.name!='real_data_328_analysis_package.zip':archive.write(path,path.name)
    archive.write(pdf,pdf.name)
print(json.dumps(dict(pdf=str(pdf),pages=manifest['pdf_pages'],sections=len(sections),words=len(' '.join(md).split())),indent=2))
