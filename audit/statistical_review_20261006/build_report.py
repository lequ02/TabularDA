"""Create a readable PDF and Markdown report from verified statistical tables."""
import csv
import hashlib
import html
import json
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT=Path(__file__).resolve().parents[2]
TABLES=Path(__file__).parent/'results'
OUT=ROOT/'output/statistical_review_20261006'
PDFOUT=ROOT/'output/pdf/statistical_review_20261006'
PDFOUT.mkdir(parents=True,exist_ok=True)
FONT=Path('D:/python/Lib/site-packages/matplotlib/mpl-data/fonts/ttf')
pdfmetrics.registerFont(TTFont('Review',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('ReviewBold',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('Review',normal='Review',bold='ReviewBold',italic='Review',boldItalic='ReviewBold')
INK=colors.HexColor('#172f40');TEAL=colors.HexColor('#176d81');PALE=colors.HexColor('#edf4f6')
styles={
 'title':ParagraphStyle('title',fontName='ReviewBold',fontSize=23,leading=28,textColor=INK,spaceAfter=15),
 'h1':ParagraphStyle('h1',fontName='ReviewBold',fontSize=17,leading=21,textColor=INK,spaceAfter=13),
 'h2':ParagraphStyle('h2',fontName='ReviewBold',fontSize=11.5,leading=16,textColor=TEAL,spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Review',fontSize=9.4,leading=14,textColor=INK,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Review',fontSize=8.0,leading=11.5,textColor=colors.HexColor('#52616b'),spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Review',fontSize=8.0,leading=10.8,textColor=INK),
 'head':ParagraphStyle('head',fontName='ReviewBold',fontSize=8.0,leading=11,textColor=INK),
}
story=[];markdown=[]

def read(name):return list(csv.DictReader((TABLES/name).open()))
tests=read('contrast_tests.csv');effects=read('dataset_effects.csv');anova=read('anova.csv');ga=read('labeler_group_anova.csv')

def result(domain,group,contrast,metric=None):
 return next(r for r in tests if r['analysis']==domain and r['labeler_group']==group and r['contrast']==contrast and (metric is None or r['metric']==metric))
def signed(value,digits=2):return f'{float(value):+.{digits}f}'
def number(value,digits=3):return f'{float(value):.{digits}f}'
def ci(r):return f"[{signed(r['ci_low'])}, {signed(r['ci_high'])}]"
def pvalue(value):
 p=float(value);return '<0.001' if p<.001 else f'{p:.4f}'
def markup(text):
 text=html.escape(text)
 text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
 return text
def para(text,style='body'):
 story.append(Paragraph(markup(text),styles[style]));markdown.append(text+'\n')
def heading(text,level=1):
 story.append(Paragraph(markup(text),styles['h1' if level==1 else 'h2']));markdown.append('#'*level+' '+text+'\n')
def table(headers,rows,widths):
 items=[[Paragraph(markup(str(x)),styles['head' if i==0 else 'cell']) for x in row] for i,row in enumerate([headers]+rows)]
 t=Table(items,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
                       ('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce5e9')),
                       ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                       ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
 story.append(t);story.append(Spacer(1,10))
 markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+[''])
def figure(name,caption,width=516):
 path=OUT/name
 with PILImage.open(path) as im:w,h=im.size
 story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8));para(caption,'small')
 markdown.append(f'![{caption}]({path.as_posix()})\n')
def newpage():story.append(PageBreak());markdown.append('\n---\n')

story.append(Paragraph('Statistical review of hybrid synthetic data',styles['title']))
markdown.append('# Statistical review of hybrid synthetic data\n')
para('Completed real-data and known-distribution results | October 6, 2026','small')
para('**Including Y in generator fitting has no demonstrated utility advantage. Hybrid target replacement helps with RF, XGB and DNN, but does not consistently help across all six labelers.** The simulated benchmark provides stronger evidence of the conditional benefit than the small real-data benchmark. Neither study establishes that the hybrid improves on real-only training.')
para('This report answers two separate questions: full-table (X,Y) versus X-only generator fitting, with the same target labeler; and hybrid targets versus the original CTGAN/TVAE generated targets. "Original approach" in the primary comparisons means the generated-target generator baseline. Real-only training is examined separately.')
para('**Table 1. All six labelers and the RF/XGB/DNN subgroup.** Mean effects are percentage-point differences, not absolute scores or relative percentages. The all-six average includes Gaussian NB, Categorical NB, PCA/GMM, RF, XGB and DNN.','small')
rows=[]
for domain,label in [('real_core','Real macro F1'),('simulated','Simulated accuracy')]:
 for contrast,label2 in [('full_minus_xonly','Full minus X-only'),('full_hybrid_minus_generated','Full hybrid minus baseline'),('xonly_hybrid_minus_generated','X-only hybrid minus baseline')]:
  allsix=result(domain,'all_six',contrast);focused=result(domain,'rf_xgb_dnn',contrast)
  rows.append([label,label2,signed(allsix['mean']),pvalue(allsix['p_exact']),signed(focused['mean']),pvalue(focused['p_exact'])])
table(['Study','Comparison','All six: effect pp','All six: exact p','RF/XGB/DNN: effect pp','RF/XGB/DNN: exact p'],rows,[80,153,65,65,80,73])
para('Exact p-values above are unadjusted. Neither all-six hybrid comparison survives Holm correction. RF/XGB/DNN simulated hybrid comparisons have Holm p = 0.0469 across their three focused comparisons, but p = 0.1875 under the broader 12-comparison correction. Both groups have near-zero full-minus-X-only effects: the two hybrids perform similarly on average, even when their gains against generated targets differ greatly.','small')
para('**Practical interpretation.** The evidence favors accurate target labelers, rather than one generator fitting construction. Preserve full and X-only as distinct methods, but there is no statistical reason to make both equally prominent in a larger experiment matrix. The generic claim that any hybrid is superior is not supported.')
para('All statistical computations used the existing remote environment and a frozen score snapshot. No generators, labelers or downstream research models were trained. Existing experiment results and settings were preserved.','small')

newpage();heading('Design and evidence used')
para('The snapshot was frozen at **12:14 p.m. Chicago on October 6**: 632 real-data records, including 106 earlier unweighted Census records used only for sensitivity analysis, and 378 simulated rows. Real records had their referenced predictions and checkpoint present, a development-selected epoch and finite scores. Saved synthetic budgets matched 500 epochs, batch size 500, CUDA and 100,000 rows. News normalization passed saved-field checks. Scores were not independently recomputed from every prediction file.')
table(['Scope','Data and treatment of incomplete observations'],[
 ['Primary real data','Adult, weighted Census KDD, Covertype and MNIST28. Both synthetic-only and mixed training. Macro F1 throughout.'],
 ['Real matched coverage','132 complete method triplets for all six labelers; 66 for RF/XGB/DNN. Adult and Covertype: both generators, seeds 42/43. MNIST28: both generators, seed 42. Census: CTGAN, seed 42.'],
 ['Balanced real ANOVA','Adult, Covertype and MNIST28 only. Five complete dataset-seed factorial blocks, averaged to three dataset subjects. Census is excluded because its weighted TVAE block is incomplete.'],
 ['Simulated benchmark','Gaussian, Grid, Ring, Asia, Alarm, Child and Insurance; seeds 42/43, 300 generator epochs, 10,000 training/test/synthetic rows. All 378 planned rows; 168 complete hybrid triplets across six labelers.'],
 ['Additional analyses','Credit, MNIST12, earlier unweighted Census, seed-42-only results, News regression and real-only training references. No completed California Housing or Tab-DDPM results enter this snapshot.'],
],[124,392])
para('**Exclusions follow design limitations rather than score direction.** Credit is a sensitivity case because its holdout has ten positives. Intrusion has no completed generator comparisons. MNIST12 replaces MNIST28 in a sensitivity analysis rather than counting the same source images as two independent datasets. News uses separate regression metrics. Zero scores, negative R2 and weak-labeler results were retained.')
para('Primary comparisons require the generated-target baseline, full-table hybrid and X-only hybrid in the same dataset, seed, generator, mode and labeler cell. All three effects therefore use identical matched support. Unmatched Census TVAE X-only results enter supplementary available-pair tables only.')
para('Shared test sets make method scores correlated. Seed 42/43 repeats share the real holdout and are averaged within dataset. Labelers, seeds, modes, generators and datasets receive equal weight in turn. A baseline is reused for contrasts, never counted as six independent baseline replications. Mixed training means all real training rows plus 100,000 synthetic rows, not a fixed 50/50 ratio.','small')
para('Exact two-sided p-values enumerate all 2^n sign reversals of dataset-mean contrasts, assuming independent dataset units and symmetric zero-centered contrasts under the null. Paired-t intervals assume approximately normal contrasts; approximate Wilcoxon results are supplementary. Generalization beyond this convenience benchmark remains limited.','small')

newpage();heading('Full table versus X only')
para('A positive effect favors including Y during generator fitting. The comparison holds generator, seed, downstream mode and labeler fixed. Because full and X-only generators are separately fitted, this estimates the effect of the complete construction, including any induced change in the generated feature distribution.')
figure('full_vs_xonly.png','Average differences are close to zero. The real-data intervals are much wider than the simulated intervals.')
rows=[]
for domain,label in [('real_core','Real macro F1'),('simulated','Simulated accuracy')]:
 for group,name in [('all_six','All six'),('rf_xgb_dnn','RF/XGB/DNN'),('dnn','DNN')]:
  r=result(domain,group,'full_minus_xonly');rows.append([label,name,signed(r['mean']),ci(r),pvalue(r['p_exact'])])
table(['Study','Labelers','Effect pp','95% t interval','Exact p'],rows,[115,96,65,152,88])
para('**There is no supported winner.** For RF/XGB/DNN, the mean full-minus-X-only effect is +0.08 macro-F1 points on real data and +0.12 accuracy points in simulation. Direction differs by dataset. This does not imply identical feature quality, runtime, minority coverage or per-dataset performance.')
para('An exploratory equivalence sensitivity used margins of 0.5, 1 and 2 points. In simulation, the RF/XGB/DNN 90% interval is [-0.159, +0.407] points, inside +/-0.5 (raw TOST p = 0.0209). Real data cannot establish equivalence within +/-1 point (raw p = 0.0982), although the +/-2-point sensitivity passes. These margins were not established by a prospective study design; equivalence refers only to the average benchmark effect.','small')

newpage();heading('Hybrid improvement depends on the labeler')
figure('labeler_effects.png','Each cell is a dataset mean for full-table relabeling versus its generated-target baseline. Positive values favor the hybrid; they are not independent method replications.')
rows=[]
for group,name in [('all_six','All six labelers'),('rf_xgb_dnn','RF/XGB/DNN'),('dnn','DNN'),('gaussian','Gaussian NB'),('categorical','Categorical NB'),('pca_gmm','PCA/GMM')]:
 real=result('real_core',group,'full_hybrid_minus_generated');simr=result('simulated',group,'full_hybrid_minus_generated')
 rows.append([name,signed(real['mean']),real['wins']+'/4',signed(simr['mean']),simr['wins']+'/7'])
table(['Labelers','Real gain pp','Real wins','Simulated gain pp','Simulated wins'],rows,[132,94,73,125,92])
para('**All six labelers do not yield an overall superiority result.** Full-table hybrid means are +1.70 macro-F1 points on real data (exact p = 0.625) and +1.55 accuracy points in simulation (p = 0.2656). RF/XGB/DNN gains are larger: +9.61 and +5.37 points, positive on all four real datasets and all seven simulated tasks. X-only gains are +9.53 and +5.24 points for this group.')
para('The six-labeler average is a summary of a method family, not the performance of an ensemble. It answers whether improvement is generic across the labelers studied. RF/XGB/DNN is a motivated subgroup already emphasized in the earlier comparison plots; it is not selected by taking the best test score for each dataset.')
para('The pooled DNN effect is +12.85 macro-F1 points on real data and +5.31 accuracy points in simulation. Individual RF, XGB and DNN simulated tests each have raw exact p = 0.0156, but p = 0.2813 after correcting the 18 individual-labeler contrasts. The group-level evidence does not prove that one labeler is uniquely best.','small')

newpage();heading('Factorial ANOVA and inference')
para('Repeated-measures ANOVA treats the dataset as the subject. The hybrid-only model has generator, feature source, labeler and real-data training mode factors and all interactions. A separate construction model compares generated targets, full-table hybrid and X-only hybrid after averaging the six labelers. This prevents artificially replicating generated-target baselines. Seeds are averaged, rather than used as extra dataset subjects.')
para('Effects with more than one numerator degree of freedom receive a Greenhouse-Geisser correction for sphericity. Holm corrections then apply within each ANOVA model. These small benchmark samples limit the stability of F tests and partial eta squared; ANOVA is an exploratory factor analysis, not an independent confirmation of the paired findings.')
rows=[]
for model,terms in [('real_hybrid_factorial',['generator','source','labeler','mode']),('simulated_hybrid_factorial',['generator','source','labeler','generator:labeler']),('simulated_construction_factorial',['construction','generator:construction'])]:
 for term in terms:
  r=next(r for r in anova if r['analysis']==model and r['term']==term)
  rows.append(['Real' if model.startswith('real') else 'Simulation',term.replace(':',' x '),number(r['F'],2),pvalue(r['p_uncorrected']),pvalue(r['p_GG']),pvalue(r['p_GG_holm'])])
table(['Study','Factor','F','Raw p','GG p','GG + Holm'],rows,[70,160,51,77,77,81])
para('The source main effect is not significant in either study: F(1,2) = 0.172, p = 0.719 on real data; F(1,6) = 0.573, p = 0.478 in simulation. The six-level labeler effect is suggestive before correction, but does not survive GG plus model-wide Holm correction.')
r=next(r for r in ga if r['analysis']=='simulated_labeler_group_factorial' and r['term']=='generator:labeler_group')
para(f'An additional two-group model contrasts RF/XGB/DNN with NB/PCA-GMM. Its simulated generator-by-labeler-group interaction survives model-wide correction: F(1,6) = {number(r["F"],2)}, raw p = {pvalue(r["p_uncorrected"])}, Holm p = {pvalue(r["p_GG_holm"])}. Generator choice therefore interacts with the type of labeler. This is a subgroup-model finding, not a universal generator ranking.')
para('For the focused RF/XGB/DNN contrasts, simulated hybrid gains are larger against CTGAN (+8.80 points for full-table relabeling) than TVAE (+1.94). After relabeling, their mean utility difference is much smaller. On the balanced real hybrid subset, mixed training increases mean macro F1 by 4.95 points across all labelers, but its dataset-level exact p = 0.25. Seed effects show no reliable direction and have very few real repeat units.','small')

newpage();heading('Simulated utility and distribution fidelity')
para('The oracle-defined simulation permits comparison with the Bayes decision rule, beyond accuracy against observed targets. RF/XGB/DNN full-table relabeling increases mean macro F1 by 6.83 points and Bayes agreement by 7.33 points; both improve on all seven task means. These are secondary endpoints and their raw exact p = 0.0156 values are not new independent confirmations of accuracy.')
figure('utility_and_density.png','All fourteen DNN dataset-generator mean accuracy changes are positive. Eight of the fourteen L_test changes are negative.')
para('For RF/XGB/DNN averaged across generators, full-table relabeling increases L_syn by about 0.323 nats per row but changes L_test by -0.132. The latter has a 95% t interval [-0.405, +0.141] and exact p = 0.3125; there is no evidence of improved density-refit fidelity. X-only L_test changes by -0.288, with exact p = 0.2031. These pooled summaries are descriptive differences across heterogeneous oracle tasks.')
para('**Hard relabeling changes the target distribution.** It can recover an accurate boundary while removing genuine target noise. It cannot restore feature modes absent from the generator. Insurance CTGAN full+DNN retains approximately 48.8% oracle-impossible rows even though downstream accuracy is approximately 94.8%. Predictive utility therefore cannot stand in for faithful synthetic data.')
para('L_test fits a density estimator to the synthetic table and scores independent original test data; it is not the generator likelihood. Mixed datasets use exact densities. Bayesian networks use the historical log(p + 1e-8) convention, once on the full joint probability. The latter is not a normalized-density score for a direct KL identity. Support violations are interpreted on BN tasks separately; mixed-task structural zeros would dilute that summary.','small')

newpage();heading('Sensitivity analyses and real data references')
rows=[]
for scenario,name in [('real_core','Primary matched core'),('real_without_census','Exclude Census'),('real_with_credit','Include Credit'),('real_mnist12_instead','Use MNIST12 instead'),('real_unweighted_census','Use earlier unweighted Census'),('real_seed42_only','Seed 42 only')]:
 full=result(scenario,'rf_xgb_dnn','full_hybrid_minus_generated');source=result(scenario,'rf_xgb_dnn','full_minus_xonly')
 rows.append([name,signed(full['mean']),pvalue(full['p_exact']),signed(source['mean'])])
table(['Real sensitivity','Hybrid gain pp','Exact p','Full minus X-only pp'],rows,[224,93,78,121])
para('Credit changes the apparent source advantage markedly: the RF/XGB/DNN full-minus-X-only effect grows from +0.08 to +3.44 points. This is driven by its favorable full-table CTGAN relabeling arms and severe X-only failures, measured on ten positive test cases. Its inclusion does not establish significance. Excluding it is a measurement-quality choice, not removal of an unfavorable hybrid result.')
para('Replacing weighted Census with the earlier unweighted protocol reduces the RF/XGB/DNN hybrid gain from +9.61 to +4.92 points. Both loss and checkpoint selection changed; the improvement cannot be attributed to weighting alone. Available-pair analysis also changes averages when incomplete TVAE cells enter. The primary triplet restriction avoids that compositional effect.')
para('Collapsing Gaussian, Grid and Ring into one mechanism family reduces the simulated inferential sample from seven tasks to five families. The RF/XGB/DNN hybrid effect remains positive (+5.78 points for full-table relabeling), but exact p rises to 0.0625. Together with the broader multiplicity sensitivity, this makes the simulated significance evidence conditional rather than conclusive.')
heading('The hybrid does not clearly beat real-only training',2)
para('Relative to the real-only student, RF/XGB/DNN full-table hybrids average +1.11 macro-F1 points on real data, with a wide interval [-8.44, +10.66] and exact p = 0.875. In simulation they are essentially level with the original-data student: -0.048 accuracy points, interval [-0.198, +0.103], exact p = 0.4688. The all-six-labeler simulated hybrid average is below real-only training on every task. These references average synthetic-only and mixed real-data arms, so they do not establish a standalone augmentation benefit.')
heading('News remains a separate regression limitation',2)
para('Across both generators and training modes, the four regression labelers increase R2 by 0.0235 with full-table relabeling and 0.0122 with X-only generation. However, NMAE_sigma worsens by 0.0201 and 0.0258 respectively. DNN worsens NMAE_sigma by 0.0364 and 0.0486. Only one completed dataset-seed block exists, so no benchmark-level ANOVA or general superiority p-value is reported. Normalization comes from saved held-out test fields; it was not recomputed for plotting.','small')

newpage();heading('What the evidence supports for the paper')
para('**Supported:** RF/XGB/DNN target replacement produces substantial, consistent improvements over generated-target CTGAN/TVAE baselines on the matched real-data tasks and the known-distribution benchmark. Gains are usually larger against CTGAN. Including Y in generator fitting has no reliable average predictive advantage. Utility improvement does not imply density-fidelity improvement.')
para('**Not established:** superiority for every labeler, superiority to real-only training, a universal preference for full-table or X-only generators, regression superiority, generalization to arbitrary downstream models, or a new generic principle of teacher-labeling synthetic features. All conclusions concern the specified neural downstream evaluations and observed tasks.')
para('The exact paired tests are intentionally conservative about the independent unit. With four real dataset units, the smallest possible two-sided dataset sign-flip p-value is 0.125, even if every dataset improves. Additional seeds can stabilize estimates, but cannot create additional dataset units or new holdouts. Simulated seven-task inference is stronger, but the focused significance is weakened by task-family and multiplicity sensitivity.')
para('A suitable current statement is: **Target replacement with flexible discriminative labelers improves the utility of generated tables across the evaluated tasks, while full-table and X-only constructions have similar average utility; the benefit depends on the labeler and need not improve joint-distribution fidelity.** This is an exploratory benchmark conclusion. Use effect sizes and dataset-specific outcomes as prominently as p-values.')
para('The next scientific priority should be attribution, not expansion of the factorial matrix: evaluate the existing teacher directly and compare with teacher-labeled real-feature resampling before committing to additional generator fits. The statistical results do not justify removing negative observations, selecting a winning labeler independently on each test set, or treating nonsignificance as universal equivalence. No such new experiments were performed for this report.')
heading('Reproducibility and validation',2)
para('The editable report, three figures and their SVG versions accompany the PDF. The audit folder contains the frozen snapshot, source hashes, analysis script, complete ANOVA tables, matched cell differences, dataset effects, exact tests, confidence intervals, equivalence grid, exclusions and sensitivities. Statistics ran remotely with NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3 and statsmodels 0.15.0. Local work prepared the report and figures only.','small')
para('Focused checks verified the snapshot transfer hash, all twelve statistical-table hashes, complete-triplet contrast identities, absence of duplicate cells, recorded budgets and News normalization. The source-effect ANOVA F values independently equal the squared paired-t statistics on their balanced dataset subjects. No uncertainty intervals based on repeated resampling of held-out prediction rows were computed; the reported intervals quantify variability across benchmark dataset means.','small')
heading('Statistical references',2)
para('Demšar, Statistical Comparisons of Classifiers over Multiple Data Sets: https://www.jmlr.org/papers/v7/demsar06a.html\nSciPy paired permutation testing: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html\nstatsmodels balanced repeated-measures ANOVA: https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html','small')

def footer(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dce5e9'));canvas.line(48,39,564,39)
 canvas.setFont('Review',7.2);canvas.setFillColor(colors.HexColor('#52616b'))
 canvas.drawString(48,26,'Frozen results: October 6, 2026, 12:14 p.m. Chicago')
 canvas.drawRightString(564,26,f'{doc.page}');canvas.restoreState()

pdfpath=PDFOUT/'statistical_review.pdf'
doc=SimpleDocTemplate(str(pdfpath),pagesize=letter,leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=53,
                      title='Statistical review of hybrid synthetic data',author='Research analysis')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'statistical_review.md').write_text('\n'.join(markdown),encoding='utf-8')
from pypdf import PdfReader
reader=PdfReader(str(pdfpath));print('PDF',pdfpath,'PAGES',len(reader.pages))
assert len(reader.pages)==8, 'Inspect page overflow before delivery'
for i,page in enumerate(reader.pages,1):
 text=page.extract_text();assert text and len(text)>300,(i,len(text))
 print('PAGE',i,'TEXT_CHARACTERS',len(text))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdfpath,OUT/'statistical_review.md',Path(__file__),Path(__file__).parent/'analyze_results.py']}
(OUT/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
