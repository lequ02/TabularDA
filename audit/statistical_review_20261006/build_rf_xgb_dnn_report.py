"""Prepare the corrected report from remote statistical outputs only."""
import base64
import csv
import hashlib
import html
import json
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(__file__).parent
DATA=AUDIT/'results_rf_xgb_dnn'
OLD=AUDIT/'results'
OUT=ROOT/'output/statistical_review_20261006'
PDFOUT=ROOT/'output/pdf/statistical_review_20261006'
FONT=Path('D:/python/Lib/site-packages/matplotlib/mpl-data/fonts/ttf')
pdfmetrics.registerFont(TTFont('Review',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('ReviewBold',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('Review',normal='Review',bold='ReviewBold',italic='Review',boldItalic='ReviewBold')
INK=colors.HexColor('#172f40');TEAL=colors.HexColor('#176d81');PALE=colors.HexColor('#edf4f6')
styles={
 'title':ParagraphStyle('title',fontName='ReviewBold',fontSize=22,leading=27,textColor=INK,spaceAfter=13),
 'h1':ParagraphStyle('h1',fontName='ReviewBold',fontSize=17,leading=21,textColor=INK,spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='ReviewBold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=8,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Review',fontSize=9.1,leading=13.4,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Review',fontSize=7.9,leading=11.1,textColor=colors.HexColor('#52616b'),spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Review',fontSize=7.6,leading=10.3,textColor=INK),
 'head':ParagraphStyle('head',fontName='ReviewBold',fontSize=7.6,leading=10.3,textColor=INK),
}
story=[];markdown=[];html_body=[]

def read(path):return list(csv.DictReader(path.open()))
methods=read(DATA/'generator_labeler_comparisons.csv')
groups=read(DATA/'matched_group_contrasts.csv')
anovas={domain:read(DATA/f'anova_{domain}.csv') for domain in ['real','simulated']}
meta=json.loads((DATA/'factorial_metadata.json').read_text())

def markup(text):
 text=html.escape(text).replace('\\*','STAR_LITERAL_TOKEN')
 return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text).replace('STAR_LITERAL_TOKEN','*')
def para(text,style='body'):
 story.append(Paragraph(markup(text),styles[style]));markdown.append(text+'\n');html_body.append(f'<p class="{style}">{markup(text)}</p>')
def heading(text,level=1):
 story.append(Paragraph(markup(text),styles['h1' if level==1 else 'h2']));markdown.append('#'*level+' '+text+'\n');html_body.append(f'<h{level}>{markup(text)}</h{level}>')
def table(headers,rows,widths,compact=False):
 items=[[Paragraph(markup(str(x)),styles['head' if i==0 else 'cell']) for x in row] for i,row in enumerate([headers]+rows)]
 t=Table(items,colWidths=widths,repeatRows=1,hAlign='LEFT')
 padding=4 if compact else 6
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
  ('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce5e9')),
  ('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),
  ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
 story.append(t);story.append(Spacer(1,9))
 markdown.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+[''])
 html_body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+markup(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+markup(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def newpage():story.append(PageBreak());markdown.append('\n---\n');html_body.append('<hr>')
def figure(name,caption,width=516):
 path=OUT/name
 with PILImage.open(path) as im:w,h=im.size
 story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,6));para(caption,'small')
 markdown.append(f'![{caption}]({path.as_posix()})\n')
 html_body.append('<img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">')
def f(value,digits=2):return f'{float(value):.{digits}f}'
def signed(value,digits=2):return f'{float(value):+.{digits}f}'
def pv(value):
 value=float(value)
 return '<0.0001' if value<.0001 else f'{value:.4f}'
def significance(value):
 value=float(value)
 return '***' if value<.001 else '**' if value<.01 else '*' if value<.05 else '.' if value<.1 else ''
def effect_name(term):
 labels={'dataset':'Dataset','generator':'Generator','approach':'Approach','labeler':'Labeler',
         'x_inclusion':'Input inclusion','training_mode':'Training mode'}
 return ' x '.join(labels[t] for t in term.split(':'))
def group(domain,name,contrast,generator='pooled'):
 return next(r for r in groups if r['domain']==domain and r['labeler_group']==name and r['contrast']==contrast and r['generator']==generator)
def row(domain,term):return next(r for r in anovas[domain] if r['term']==term)
names={'rf':'RF','xgb':'XGBoost','dnn':'DNN'}

story.append(Paragraph('CTGAN/TVAE versus CTGAN/TVAE + labeler',styles['title']))
markdown.append('# CTGAN/TVAE versus CTGAN/TVAE + labeler\n');html_body.append('<h1>CTGAN/TVAE versus CTGAN/TVAE + labeler</h1>')
para('Corrected statistical report | October 6, 2026 | existing results only','small')
para('**This report compares CTGAN and TVAE with their own RF, XGBoost and DNN relabeling arms.** The original generator keeps generated targets; a hybrid replaces them with targets predicted by a labeler trained on real training data. CTGAN and TVAE remain separate in the method tables.')
para('There are two separate factorial ANOVA tables: **real data** and **simulated data**. Each includes **dataset, generator, labeler and generator-input inclusion**. A separate approach contrast tests generated targets versus hybrid targets. Real-data ANOVA also includes synthetic-only versus mixed training.')
para('**Approach** asks whether replacing generator targets helps: original CTGAN/TVAE versus the average RF/XGB/DNN hybrid (1 df). **Labeler** asks whether RF, XGBoost and DNN differ within the hybrid family (2 df). Labeler excludes the original baseline because that baseline has no relabeler. The two factors answer different questions.','small')
para('**Generator-input inclusion means full (X,Y) versus X-only fitting.** X is included in both; the variable being included or excluded is Y, as confirmed by the user. The ANOVA column is named x_inclusion to match the requested factor list; its contrast is full minus X-only.')
heading('Table 1. Original generators versus hybrids',2)
rows=[]
for domain,label in [('real','Real macro F1'),('simulated','Simulated accuracy')]:
 for contrast,text in [('full_hybrid_minus_generated','Full-table hybrid'),('xonly_hybrid_minus_generated','X-only hybrid'),('hybrid_both_sources_minus_generated','Both sources averaged')]:
  r=group(domain,'rf_xgb_dnn',contrast)
  rows.append([label,text,signed(r['mean']),r['wins']+'/'+r['n_datasets'],pv(r['p_t']),pv(r['p_exact'])])
table(['Study','RF/XGB/DNN construction','Hybrid gain pp','Dataset wins','Paired-t p','Exact p'],rows,[91,167,73,64,61,60])
para('Each gain compares against the matched original CTGAN/TVAE baseline. Dataset means receive equal weight; seeds, generators and modes are averaged within dataset. The three-labeler mean summarizes a method family, not an ensemble. P-values above are unadjusted; Holm adjusts the full-hybrid, X-only-hybrid and source comparisons separately within each study. The both-source row is descriptive and is not another independent primary test.','small')
para('**Scope:** RF/XGB/DNN are the user-requested study family. This restriction follows inspection of earlier results, so the analysis is exploratory rather than prospectively preregistered. Conclusions concern these three labelers. The earlier broader analysis is retained in the audit archive.')
para('The frozen snapshot contains **632 real-data records and 378 simulated rows** at 12:14 p.m. Chicago on October 6. No research models were trained. The analysis follows the factor exploration in D:/Rprojects/research_data_synthesis while using only the corrected production snapshot.','small')

for domain,title,metric in [('real','Real data: generator versus generator + labeler','Macro F1 multiplied by 100'),('simulated','Simulation: generator versus generator + labeler','Accuracy in percent')]:
 newpage();heading(title)
 para(metric+'. Full hybrid uses features from full (X,Y) generator fitting; X-only hybrid uses a separately fitted features-only generator. Both replace targets using the same named labeler. Gains are percentage points versus the original generator, not relative percentages.')
 rows=[]
 for generator in ['ctgan','tvae']:
  for labeler in names:
   full=next(r for r in methods if r['domain']==domain and r['generator']==generator and r['labeler']==labeler and r['source']=='full')
   xonly=next(r for r in methods if r['domain']==domain and r['generator']==generator and r['labeler']==labeler and r['source']=='xonly')
   rows.append([generator.upper(),names[labeler],f(full['baseline_mean']),f(full['hybrid_mean']),signed(full['mean']),f(xonly['hybrid_mean']),signed(xonly['mean']),full['n_datasets']])
 table(['Generator','Labeler','Original score','Full hybrid','Full gain','X-only hybrid','X-only gain','N tasks'],rows,[61,98,63,69,55,73,58,39])
 para('The baseline is repeated in this display for readability; it appears only once per generator/seed/mode in the ANOVA. Every hybrid is matched to the same generator, dataset, seed and training mode. Zero and negative results are retained.','small')
 if domain=='real':
  para('CTGAN coverage: Adult, weighted Census KDD, Covertype and MNIST28. TVAE coverage: Adult, Covertype and MNIST28; weighted Census lacks complete full/X-only triplets. Adult and Covertype have seeds 42/43, MNIST28 seed 42, matched Census CTGAN seed 42. Both training modes are averaged. Mixed training adds 100,000 synthetic rows to all real training rows, rather than a fixed 50/50 mix. Task coverage differs, so these CTGAN/TVAE marginal scores are not a direct generator ranking.','small')
 else:
  para('Both generators cover all seven simulated datasets, seeds 42/43: Gaussian, Grid, Ring, Asia, Alarm, Child and Insurance. All methods use synthetic-only downstream training. Production settings were 300 generator epochs and 10,000 training, test and synthetic rows each.','small')
 for generator in ['ctgan','tvae']:
  strong=group(domain,'rf_xgb_dnn','hybrid_both_sources_minus_generated',generator)
  para(f"**{generator.upper()} + RF/XGB/DNN:** the both-source average gain is {signed(strong['mean'])} points. Gains are measured relative to {generator.upper()} generated targets on matched tasks.")
 para('These summaries do not select a winning labeler on each test set. Generator-specific, labeler-specific and dataset-specific scores and matched tests are available in the companion CSV tables.','small')

for domain,title in [('real','Table 2. Real-data factorial ANOVA'),('simulated','Table 3. Simulated-data factorial ANOVA')]:
 newpage();heading(title)
 if domain=='real':
  para('Balanced repeated coverage is Adult seeds 42/43, Covertype seeds 42/43 and MNIST28 seed 42: **3 datasets, 5 dataset/seed units, 140 scores**. Every unit has CTGAN/TVAE, one original plus six RF/XGB/DNN hybrid constructions per generator, and both training modes. Weighted Census remains in matched method comparisons; its incomplete TVAE block prevents entry into this balanced ANOVA.')
  selected=['dataset','generator','approach','labeler','x_inclusion','training_mode','dataset:generator','dataset:approach','generator:approach','dataset:generator:approach','dataset:labeler','generator:labeler','dataset:x_inclusion','generator:x_inclusion','labeler:x_inclusion','approach:training_mode']
 else:
  para('All seven datasets have seeds 42/43: **7 datasets, 14 dataset/seed units, 196 scores**. Each unit contains CTGAN/TVAE and one original plus six RF/XGB/DNN hybrid constructions per generator. Fourteen real-only reference rows are outside the ANOVA, yielding 210 selected simulated results in total. The complete 378-row production snapshot is preserved.')
  selected=['dataset','generator','approach','labeler','x_inclusion','dataset:generator','dataset:approach','generator:approach','dataset:generator:approach','dataset:labeler','generator:labeler','dataset:x_inclusion','generator:x_inclusion','labeler:x_inclusion','dataset:labeler:x_inclusion','dataset:generator:labeler','dataset:generator:x_inclusion','generator:labeler:x_inclusion','dataset:generator:labeler:x_inclusion']
 rows=[]
 formatted=[]
 for term in selected:
  r=row(domain,term)
  mean_sq=float(r['sum_sq'])/float(r['df_num'])
  raw_stars=significance(r['p_raw'])
  cells=[effect_name(term),f(r['df_num'],0)+' / '+f(r['df_den'],0),f(r['sum_sq']),f(mean_sq),f(r['F']),pv(r['p_raw'])+(' '+raw_stars.replace('*',r'\*') if raw_stars else ''),pv(r['p_GG_holm'])]
  rows.append(['**'+cell+'**' for cell in cells] if float(r['p_GG_holm'])<.05 else cells)
  formatted.append(dict(effect=effect_name(term),df_num=r['df_num'],df_den=r['df_den'],sum_sq=r['sum_sq'],mean_sq=mean_sq,
                        F=r['F'],p_raw=r['p_raw'],raw_significance=raw_stars,p_GG=r['p_GG'],p_adjusted=r['p_GG_holm'],
                        significant_adjusted=float(r['p_GG_holm'])<.05))
 table(['Source','Df (num / den)','Sum Sq','Mean Sq','F value','Pr(>F)','Adj. p'],rows,[160,45,65,65,50,67,64],compact=True)
 with (OUT/f'anova_{domain}_formatted.csv').open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(formatted[0]));writer.writeheader();writer.writerows(formatted)
 para(r'Significance codes beside Pr(>F): \*\*\* p < 0.001; \*\* p < 0.01; \* p < 0.05; . p < 0.10. Stars use raw p. **Bold rows** have adjusted p < 0.05. Adj. p applies Greenhouse-Geisser (GG), then Holm across all '+('39' if domain=='real' else '19')+' model effects. Df is numerator / denominator; Sum Sq and Mean Sq are uncorrected. Each repeated-measures effect has its own error stratum, so there is no single pooled residual row.','small')
 para('Dataset is a fixed between-unit factor; seed is nested within dataset. Generator and constructions repeat within each dataset/seed unit. **Approach = original versus hybrid**; **Labeler = RF versus XGBoost versus DNN within hybrids**. Input inclusion compares full (X,Y) with X-only fitting. The complete CSV contains every interaction.','small')
 a=row(domain,'approach');s=row(domain,'x_inclusion');l=row(domain,'labeler');g=row(domain,'generator:approach')
 para(f"**Interpretation.** The RF/XGB/DNN approach main effect has raw p = {pv(a['p_raw'])} and model-wide Holm p = {pv(a['p_GG_holm'])}; it {'survives' if float(a['p_GG_holm'])<.05 else 'does not survive'} correction. Inclusion has p = {pv(s['p_raw'])}. Labeler Holm p = {pv(l['p_GG_holm'])}; generator-by-approach Holm p = {pv(g['p_GG_holm'])}. These tests are conditional on the fixed benchmark datasets.",'small')
 if domain=='real':
  para('Only **two residual seed degrees of freedom** estimate repeat variability. The covariance estimate and corrected tests are fragile, especially for the singleton MNIST28 group. These are conditional tests on fixed datasets and holdouts, not estimates of test-row uncertainty or evidence from five independent datasets.','small')

newpage();heading('Target inclusion and approach are different questions')
para('The source comparison holds generator and labeler fixed and subtracts the X-only hybrid from the full-table hybrid. It does not subtract an original generator. Both hybrids can gain substantially over the original while differing little from each other.')
rows=[]
for labeler in names:
 real=group('real',labeler,'full_minus_xonly');sim=group('simulated',labeler,'full_minus_xonly')
 rows.append([names[labeler],signed(real['mean']),pv(real['p_exact']),signed(sim['mean']),pv(sim['p_exact'])])
table(['Labeler','Real full - X pp','Real exact p','Sim full - X pp','Sim exact p'],rows,[142,101,86,101,86])
for domain,name in [('real','Real data'),('simulated','Simulation')]:
 r=group(domain,'rf_xgb_dnn','full_minus_xonly')
 para(f"**{name}, RF/XGB/DNN:** mean full-minus-X-only = {signed(r['mean'])} points; 95% paired-t interval [{signed(r['ci_low'])}, {signed(r['ci_high'])}]; exact p = {pv(r['p_exact'])}.")
para('For RF/XGB/DNN, real full-hybrid gain is +9.61 points and X-only gain is +9.53, so their difference is +0.08. Simulation gives +5.37 minus +5.24 = +0.12. The small source difference is not an arithmetic error and does not mean the hybrid gain itself is 0.08 points.')
heading('Which statistical question does each test answer?',2)
para('**Factorial ANOVA:** does a factor or interaction change mean scores relative to experiment-seed repeat variability on these fixed datasets? Dataset is an explicit factor and dataset interactions are estimated. Normality, common covariance across dataset groups and independent RNG repeats are assumptions; real repeats share one held-out split.')
para('**Dataset-level paired comparisons:** is the average improvement consistent across the observed tasks? Seeds are averaged within dataset, so the primary real comparison has four task units and simulation seven. Exact sign-flip p-values assume independent task units and symmetric zero-centered differences under the null. Paired-t p-values assume approximately normal task differences.')
para('The tests have different denominators and different scopes. Small conditional ANOVA p-values do not make new datasets or establish general superiority. Likewise, a nonsignificant inclusion effect is not proof of equivalence for every dataset or labeler.')

newpage();heading('Simulation: utility versus distribution fidelity')
para('The simulated benchmark supplies known joint distributions and Bayes decisions. It therefore tests more than downstream accuracy. RF/XGB/DNN full-table relabeling increases average macro F1 by 6.83 points and Bayes agreement by 7.33 points; these are correlated secondary outcomes, not additional independent confirmations.')
figure('utility_and_density.png','DNN full-table relabeling: each point averages seeds 42/43 for one dataset and generator. All fourteen accuracy gains are positive; eight L_test changes are negative.')
para('For RF/XGB/DNN, average L_syn improves by about 0.323 nats per row while L_test changes by -0.132, with 95% interval [-0.405, +0.141] and exact p = 0.3125. There is no demonstrated improvement in density-refit fidelity. X-only L_test changes by -0.288, p = 0.2031.')
para('Hard relabeling can recover a useful decision boundary while removing real target noise. It cannot recover feature modes absent from the generated features. Insurance CTGAN full+DNN retains approximately 48.8% oracle-impossible rows despite approximately 94.8% downstream accuracy.')
para('L_test scores a density refitted to each training table; it is not generator likelihood. Mixed-data densities are exact. BN scores use historical log(p + 1e-8) once on joint probability, so they do not satisfy the normalized-density KL identity. Oracle probabilities and Bayes decisions remain exact.','small')

newpage();heading('Exclusions, sensitivities and paper claims')
credit=read(DATA/'anova_real_credit_sensitivity.csv')
ca=next(r for r in credit if r['term']=='approach');cs=next(r for r in credit if r['term']=='x_inclusion')
para(f"**Credit:** held out of the primary classification analysis because the real holdout contains only ten positives. Including it raises the RF/XGB/DNN full-minus-X-only effect from +0.08 to +3.44 points. In the separate Credit ANOVA, approach raw p = {pv(ca['p_raw'])}, Holm p = {pv(ca['p_GG_holm'])}; inclusion raw p = {pv(cs['p_raw'])}, Holm p = {pv(cs['p_GG_holm'])}. Both survive correction in that sensitivity. The changed conclusion is driven by a rare-class holdout and severe X-only failures; it does not establish a robust source preference.")
para('**Intrusion:** no completed generator-versus-hybrid comparisons enter the frozen snapshot. **MNIST12:** replaces MNIST28 in sensitivity analysis, rather than counting related source images as independent tasks. **Census:** old unweighted records are separate; weighted CTGAN comparisons remain, while incomplete TVAE cells are excluded from complete-triplet inference. All RF/XGB/DNN outcomes, including negative ones, remain in scope.')
para('**News regression:** only one dataset/seed block exists, so News is separate from classification and has no benchmark-level ANOVA. All three labelers worsen normalized MAE versus generated targets. For DNN, full-table R2 improves by 0.0485 while normalized MAE worsens by 0.0364; X-only R2 improves by 0.0445 while normalized MAE worsens by 0.0486. Saved test-set normalization was used.')
heading('What the current evidence supports',2)
para('RF/XGB/DNN produce consistent utility gains against their original generators on the observed classification tasks, with generally larger gains against CTGAN. This supports a claim about flexible discriminative relabeling with these three methods. Including Y in generator fitting has no clear average utility advantage. Utility gains need not improve distribution fidelity.')
para('Real-only superiority is not established. RF/XGB/DNN full-table hybrids average +1.11 real macro-F1 points relative to the real-only student (exact p = 0.875) and -0.048 simulated accuracy points (p = 0.4688). The real reference averages synthetic-only and mixed arms; it does not alone establish an augmentation benefit.')
para('Across four real task units, the smallest exact two-sided sign-flip p is 0.125. Simulated full and X-only hybrid gains have raw p = 0.0156 and three-comparison Holm p = 0.0469. The corresponding Holm-adjusted paired-t p is 0.0699, showing sensitivity to the test. Grouping Gaussian/Grid/Ring into one family raises exact p to 0.0625. The requested restriction was made after seeing earlier results; this is exploratory evidence within RF/XGB/DNN.')
heading('A simple next step',2)
para('Present both original-generator baselines and all three selected labelers in the paper. Choose a primary labeler using training/development evidence. Before expanding to Tab-DDPM, establish what generated features add beyond direct teacher evaluation and teacher-labeled resampling of real training features. Additional independent datasets address generalization; extra seeds address repeat variability. No new experiments were run for this report.')

newpage();heading('References, provenance and reproducibility')
para('The requested reference project was read at D:/Rprojects/research_data_synthesis. Its final_328_project.Rmd explores dataset, generator (tvae), labeler (y_synth) and target inclusion (has_y), including dataset-by-labeler interactions. Its later deduplicated_analysis_2026_09_18/README.md, analyse.R and legacy_artifacts.R flag baseline/source aliasing, correlated methods and sensitivity to labeler dependence.')
para('This report follows that factor structure while addressing the absent baseline X-only cells through nested construction contrasts. It does not import historical test maxima, the duplicated Adult/Census routes, hypothetical paper-score substitutions or leaked records. Real production results use development-selected checkpoints. Historical files are references, not additional independent observations.')
heading('Exact statistical design',2)
para('Each generator has seven observed constructions: one original generated-target baseline; three full-table relabeling arms; three X-only relabeling arms. An orthonormal construction basis partitions six contrast df into approach (1), hybrid labeler (2), target inclusion within hybrids (1), and labeler-by-inclusion (2). Generator and real training-mode contrasts are crossed with this basis. Labelers exist within the hybrid family; there is one baseline per generator/seed/mode.')
para('Type III sums of squares test equal-dataset marginal effects in the between/within model. Every term uses its own seed-within-dataset error projection. Multi-df within contrasts receive GG corrections; Holm covers the complete 39-term real and 19-term simulated tables. Interactions, error strata and coverage are saved in results_rf_xgb_dnn. These conditional tests retain RF/XGB/DNN and original baselines without treating a shared baseline as three observations.')
heading('Files and validation',2)
para('The report is available as PDF, editable Markdown and standalone HTML with embedded figures. The audit folder contains the frozen snapshot, rebuild_rf_xgb_dnn.py, eleven restricted-family CSVs, both complete ANOVAs, method scores, task-level gains, Credit sensitivity, and independent checks. Earlier broader outputs remain archived. Statistics ran remotely: NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3 and statsmodels 0.15.0. Local work prepared report artifacts only.')
para('Validation checked all source/table hashes, construction-basis orthogonality, exact score reconstruction, every ANOVA sum of squares against an independently restricted least-squares model, every one-dimensional F against statsmodels OLS, no duplicate cells, and no fabricated baseline arms. Referenced predictions/checkpoints and production budgets were checked; scores were not independently recomputed from every prediction file. No test-row bootstrap was performed.')
heading('Method documentation',2)
para('The afex reference documents Type III between/within ANOVA and GG correction. Demsar describes comparisons across datasets. The paired dataset tests and conditional ANOVAs answer different inferential questions.','small')
para('https://search.r-project.org/CRAN/refmans/afex/html/aov_car.html','small')
para('https://search.r-project.org/CRAN/refmans/afex/html/afex_aov-methods.html','small')
para('https://www.jmlr.org/papers/v7/demsar06a.html','small')
para('Frozen snapshot SHA-256: '+meta['snapshot_sha256'],'small')

def footer(canvas,doc):
 canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dce5e9'));canvas.line(48,39,564,39)
 canvas.setFont('Review',7);canvas.setFillColor(colors.HexColor('#52616b'))
 canvas.drawString(48,26,'Corrected factorial report | Frozen results: Oct 6, 2026, 12:14 p.m. Chicago')
 canvas.drawRightString(564,26,str(doc.page));canvas.restoreState()

PDFOUT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
pdfpath=PDFOUT/'statistical_review.pdf'
SimpleDocTemplate(str(pdfpath),pagesize=letter,leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=55,
 title='CTGAN/TVAE versus CTGAN/TVAE + labeler: factorial statistical analysis',author='Research analysis').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'statistical_review.md').write_text('\n'.join(markdown),encoding='utf-8')
css='body{font:16px/1.5 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#172f40}h1{font-size:28px}h2{font-size:22px;color:#176d81}.small{font-size:14px;color:#52616b}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dce5e9}th{background:#edf4f6;font-size:14px}td{font-size:14px}img{max-width:100%}hr{margin:45px 0;border:0;border-top:1px solid #dce5e9}a{color:#176d81}'
(OUT/'statistical_review.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>CTGAN/TVAE versus CTGAN/TVAE + labeler</title><style>'+css+'</style></head><body>'+''.join(html_body)+'</body></html>',encoding='utf-8')
reader=PdfReader(str(pdfpath));print('PDF pages',len(reader.pages))
assert len(reader.pages)==9,'Inspect report pagination before delivery'
for n,p in enumerate(reader.pages,1):
 text=p.extract_text();assert text and len(text)>500;print('Page',n,'characters',len(text))
refs=[Path('D:/Rprojects/research_data_synthesis/final_328_project.Rmd'),Path('D:/Rprojects/research_data_synthesis/deduplicated_analysis_2026_09_18/README.md'),Path('D:/Rprojects/research_data_synthesis/deduplicated_analysis_2026_09_18/analyse.R'),Path('D:/Rprojects/research_data_synthesis/deduplicated_analysis_2026_09_18/legacy_artifacts.R')]
manifest=dict(outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdfpath,OUT/'statistical_review.md',OUT/'statistical_review.html',OUT/'anova_real_formatted.csv',OUT/'anova_simulated_formatted.csv',Path(__file__),AUDIT/'rebuild_rf_xgb_dnn.py']},
              references={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs},snapshot_sha256=meta['snapshot_sha256'],
              statistical_table_sha256=meta['table_sha256'])
(OUT/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved PDF, HTML and Markdown report')
