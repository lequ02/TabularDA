"""Build a focused report from remotely computed tables; no local model fits."""
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
import re
import shutil
import zipfile

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

ROOT = Path('D:/SummerResearch')
HERE = Path(__file__).resolve().parent
DATA = HERE / 'results'
OUT = ROOT / 'output/statistical_two_contrasts_20261010'
PDFOUT = ROOT / 'output/pdf/statistical_two_contrasts_20261010'
OUT.mkdir(parents=True, exist_ok=True)
PDFOUT.mkdir(parents=True, exist_ok=True)
PDF = PDFOUT / 'real_data_two_contrasts.pdf'
DS = {'adult': 'Adult', 'census_kdd': 'Census KDD', 'covertype': 'Covertype',
      'mnist12': 'MNIST12', 'mnist28': 'MNIST28', 'news': 'News', 'california_housing': 'Housing'}
ARM = {'synthetic': 'Synthetic-only', 'mix': 'Mix'}
CN = {'relabel_minus_generated': 'Relabeled - generated', 'has_y1_minus_has_y0': 'has_y=1 - has_y=0'}

def rows(name):
    with (DATA / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

tests = rows('classification_contrasts.csv')
effects = rows('task_contrasts.csv')
anova = rows('exploratory_anova.csv')
joint = rows('classification_joint_checks.csv')
meta = json.loads((DATA / 'analysis_metadata.json').read_text())
checked = datetime.fromisoformat(meta['checked_at_chicago']).strftime('%B %d, %Y, %I:%M %p Chicago')
INK = colors.HexColor('#193247')
PALE = colors.HexColor('#edf4f6')
TEAL = colors.HexColor('#237b8a')
style = {
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=INK, spaceAfter=12),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=14, textColor=INK, spaceAfter=9),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8.8, leading=12, textColor=INK, spaceAfter=8),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.4, leading=11.5, textColor=INK),
    'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=8.4, leading=11.5, textColor=INK),
    'eq': ParagraphStyle('eq', fontName='Courier', fontSize=8.6, leading=12, textColor=INK, spaceAfter=9)}
story, md, web, sections, table_manifest = [], [], [], [], []

def markup(value):
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html.escape(str(value)))

def para(value, kind='body'):
    story.append(Paragraph(markup(value), style[kind]))
    md.append(value + '\n')
    web.append('<p class="' + kind + '">' + markup(value) + '</p>')

def section(title, question, method, conclusion):
    if sections:
        story.append(PageBreak())
        web.append('<hr>')
    sections.append(title)
    story.append(Paragraph(html.escape(title), style['h1']))
    md.append('# ' + title + '\n')
    web.append('<h1>' + html.escape(title) + '</h1>')
    para('**Research question:** ' + question)
    para('**Method:** ' + method)
    para('**Finding:** ' + conclusion)

def table(headers, values, widths):
    assert sum(widths) == 516
    cells = [[Paragraph(html.escape(str(value)), style['head']) for value in headers]]
    cells += [[Paragraph(html.escape(str(value)), style['cell']) for value in row] for row in values]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), PALE), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
                           ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d7e3e9')),
                           ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                           ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    story.extend([t, Spacer(1, 10)])
    md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in values)
    md.append('')
    web.append('<table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) +
               '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>'
               for v in row) + '</tr>' for row in values) + '</tbody></table>')
    table_manifest.append(dict(headers=headers, rows=values))

def eq(value):
    story.append(Paragraph(html.escape(value), style['eq']))
    md.append('```text\n' + value + '\n```\n')
    web.append('<pre>' + html.escape(value) + '</pre>')

def number(value):
    return f'{float(value):.4f}'

def signed(value, scale=1, decimals=4):
    return f'{scale*float(value):+.{decimals}f}'

def p(value):
    return f'{float(value):.2e}' if float(value) < .0001 else f'{float(value):.4f}'

def ci(row, family):
    return '[' + signed(row[f'ci{family}_low'], 100, 2) + ', ' + signed(row[f'ci{family}_high'], 100, 2) + ']'

section('1. Two questions, seven tasks', 'Does relabeling improve utility, and does fitting the generator with Y help?',
        'Use verified completed CTGAN/TVAE runs, seeds 42/43 and RF/XGB/DNN. Keep synthetic-only and mix separate.',
        'Classification gains favor relabeling but do not pass Holm across the two questions. Average has_y effects are small. Regression depends on the task and metric.')
table(['Design choice', 'Definition'], [
    ['Classification tasks', 'Adult, Census KDD, Covertype, MNIST12, MNIST28. Outcome: macro F1 for all five.'],
    ['Regression tasks', 'News and California Housing. Outcomes: saved R2, NMAE and absolute-error D2, separately.'],
    ['Relabeled', 'Average both has_y settings, both generators and RF/XGB/DNN; keep generated X and predict Y.'],
    ['Generated-target comparator', 'Fit each generator to (X,Y), retaining its generated Y. Match dataset, seed and generator.'],
    ['has_y=1 versus has_y=0', 'Compare generator fitted to (X,Y) versus fitted to X only. Both arms receive predicted targets.'],
    ['Training regimes', 'Synthetic-only: 100,000 synthetic rows. Mix: all real training rows plus 100,000 synthetic rows.'],
    ['Coverage', '406 selected completed records: 290 classification and 116 regression, including 14 shared real-only references.']
], [138, 378])
eq('C_relabel[d] = mean(relabel[d]) - mean(generated[d])')
eq('C_has_y[d] = mean(relabel[d, has_y=1]) - mean(relabel[d, has_y=0])')
para('Means give equal weight to seeds, generators and labelers. Classification inference then gives each of the five datasets equal weight. Each generated baseline is a shared comparator, not six independent observations.', 'small')
para('The two questions were selected after earlier analyses of these results. This redo is exploratory; it is not a newly preregistered confirmatory analysis. It excludes Credit, Intrusion, NB/PCA-GMM, simulated benchmarks and active TabDDPM runs.', 'small')

section('2. Classification: gains and evidence', 'Are either of the two average macro-F1 contrasts supported across tasks?',
        'Two-sided paired t-tests on five dataset differences, df=4. Holm2 covers both questions within a regime; Holm4 covers all four tests across both regimes.',
        'Relabeling gains 11.63 points for synthetic-only and 6.77 for mix; all five task gains are positive. Neither relabeling nor has_y passes the Holm2 threshold of .05.')
table(['Training', 'Contrast', 'Gain (points)', 'Raw p', 'Holm2 p', 'Holm4 p'], [
    [ARM[r['training']], CN[r['contrast']], signed(r['mean'], 100, 2), p(r['p_raw']),
     p(r['p_raw_holm2']), p(r['p_raw_holm4'])] for r in tests
], [78, 150, 78, 70, 70, 70])
task_index = {(r['training'], r['dataset'], r['contrast']): r for r in effects if r['metric'] == 'f1_macro'}
table(['Task', 'Synthetic relabel', 'Mix relabel', 'Synthetic has_y', 'Mix has_y'], [
    [DS[d]] + [signed(task_index[a, d, c]['benefit'], 100, 2)
                for c, a in [('relabel_minus_generated', 'synthetic'), ('relabel_minus_generated', 'mix'),
                             ('has_y1_minus_has_y0', 'synthetic'), ('has_y1_minus_has_y0', 'mix')]]
    for d in ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
], [108, 102, 102, 102, 102])
para('Every change in the second table is in macro-F1 percentage points; positive favors relabeling or has_y=1. The has_y contrast is positive on three of five synthetic-only tasks and two of five mix tasks.', 'small')
para('Synthetic-only mean F1: relabeled .7855 versus generated .6692. Mix: .8166 versus .7488. The corresponding relative improvements are 17.38% and 9.04%; relative percentages and percentage points are different quantities.', 'small')
para('Raw synthetic relabel p=.0350 is below .05; its Holm2 p=.0700 is above .05. Holm changes the inferential threshold, not the gain. No detected has_y effect does not establish equivalence.', 'small')

section('3. Regression: News and Housing', 'Do the same constructions improve regression utility?',
        'Compare native-unit means across seeds, generators and RF/XGB/DNN separately for each task. These baseline contrasts are descriptive; no seven-task pooled score is computed.',
        'Housing relabeling improves R2, NMAE and D2. News relabeling improves R2 but worsens NMAE and D2. has_y effects are smaller and have mixed directions.')
reg_index = {(r['training'], r['dataset'], r['metric'], r['contrast']): r for r in effects if r['metric'] != 'f1_macro'}
for arm in ['synthetic', 'mix']:
    para('**' + ARM[arm] + '**')
    values = []
    for dataset in ['news', 'california_housing']:
        for metric, label in [('r2', 'R2'), ('nmae_sigma', 'NMAE'), ('d2_absolute_error', 'D2')]:
            rel = reg_index[arm, dataset, metric, 'relabel_minus_generated']
            has = reg_index[arm, dataset, metric, 'has_y1_minus_has_y0']
            values.append([DS[dataset] + ' / ' + label, number(rel['right_mean']), number(rel['left_mean']),
                           number(has['right_mean']), number(has['left_mean']),
                           signed(rel['native_difference']), signed(has['native_difference'])])
    table(['Task / metric', 'Generated', 'Relabeled', 'has_y=0', 'has_y=1', 'Relabel change', 'has_y change'],
          values, [94, 68, 68, 68, 68, 75, 75])
para('Higher R2 and D2 is better; lower NMAE is better. The change columns retain native signs: negative NMAE change is improvement. R2 uses squared error against a test-mean baseline; D2 uses absolute error against a test-median baseline.', 'small')
para('NMAE = saved MAE / saved SD_test(Y), with population SD (ddof=0). Saved record normalization and verified D2 sidecars are read directly; denominators are not recomputed. News log-target runs are scored in original units; Housing uses no-Faker runs.', 'small')
para('Only two fixed regression tasks and two seeds are available. Descriptive improvement on Housing does not establish a population effect on new regression tasks. Testing multiple regression outcomes would also require an explicit family if used for joint significance claims.', 'small')

section('4. Exploratory factor checks', 'What do factorial models add beyond the two research contrasts?',
        'Refit categorical Type II ANOVAs using sum-to-zero coding. Generated targets stay outside the relabeled factorial grid. All p-values on this page are nominal and exploratory.',
        'Labeler choices differ in the classification model and treatment effects vary strongly by task. These tests do not establish improvement over generated targets.')
eq('macro_F1 ~ dataset + generator * has_y * labeler * seed')
para('Classification: 120 relabeled scores per regime, model rank 28 and residual df 92. The ANOVA assumes independent, homoskedastic residuals; shared heldout rows and task-specific effects limit its interpretation.', 'small')
def anova_p(arm, outcome, scope, factor):
    selected = [r for r in anova if r['training'] == arm and r['outcome'] == outcome
                and r['scope'] == scope and r['term'] == f'C({factor}, Sum)']
    assert len(selected) == 1
    return p(selected[0]['PR(>F)'])
table(['Classification term', 'Synthetic nominal p', 'Mix nominal p'], [
    [factor, anova_p('synthetic', 'macro_F1', 'factorial', factor),
     anova_p('mix', 'macro_F1', 'factorial', factor)] for factor in ['dataset', 'generator', 'has_y', 'labeler', 'seed']
], [216, 150, 150])
table(['Supporting nested test', 'Synthetic nominal p', 'Mix nominal p'], [
    [name.replace('_', ' ')] + [p(next(r['p_nominal'] for r in joint if r['training'] == arm and r['test'] == name))
                               for arm in ['synthetic', 'mix']]
    for name in ['seed_interactions', 'dataset_interactions']
], [216, 150, 150])
eq('R2 or D2 ~ dataset + generator * has_y * labeler * seed')
para('Regression pooled checks: 48 relabeled scores per outcome/regime, rank 25, residual df 23. The two tasks remain fixed study tasks; pooled coefficients are not evidence about a population of regression tasks.', 'small')
table(['Training / outcome', 'Generator p', 'has_y p', 'Labeler p', 'Seed p'], [
    [ARM[arm] + ' / ' + outcome] + [anova_p(arm, outcome, 'pooled', factor)
                                   for factor in ['generator', 'has_y', 'labeler', 'seed']]
    for arm in ['synthetic', 'mix'] for outcome in ['R2', 'D2']
], [160, 89, 89, 89, 89])
para('Complete ANOVA tables and within-task regression models are supplied as CSVs. Within each regression task: outcome ~ seed + generator * has_y * labeler, rank 13 and residual df 11. A full four-factor within-task model is saturated and has no residual degrees of freedom.', 'small')
para('Nominal ANOVA p-values support exploration, not additional adjusted headline claims. A significant labeler term means differences among RF/XGB/DNN, not superiority to generated targets.', 'small')

section('5. Uncertainty and exact sensitivity checks', 'How precise are the gains, and do small-sample checks agree?',
        'Show pointwise 95% paired-t intervals and Bonferroni simultaneous intervals for both questions within each regime. Enumerate all 32 sign patterns; compute exact Wilcoxon signed-rank p-values.',
        'Simultaneous intervals for all four effects contain zero. Exact two-sided relabeling p=.0625 in each regime, before multiplicity adjustment.')
table(['Training / contrast', 'Pointwise 95% CI', 'Simultaneous 95% CI (2)'], [
    [ARM[r['training']] + ' / ' + CN[r['contrast']], ci(r, 1), ci(r, 2)] for r in tests
], [240, 138, 138])
para('Intervals are in macro-F1 percentage points. Simultaneous intervals use t_(1-.05/(2*2),4); they are Bonferroni intervals, not an inversion of Holm. The package also includes simultaneous intervals for all four tests.', 'small')
table(['Training / contrast', 'Exact sign-flip p', 'Exact Wilcoxon p', 'Sign-flip Holm2'], [
    [ARM[r['training']] + ' / ' + CN[r['contrast']], p(r['p_signflip_exact']),
     p(r['p_wilcoxon_exact']), p(r['p_signflip_exact_holm2'])] for r in tests
], [240, 92, 92, 92])
para('With five nonzero task effects, the minimum attainable two-sided exact p is 2/32=.0625. The prior normal-approximation Wilcoxon value for the all-positive relabeling effects was .0431; the exact value is .0625. Neither exact check rejects at .05.', 'small')
para('The paired-t test assumes independent task-level differences with a suitable normal model; exact sign-flip and signed-rank checks assume independent differences symmetric about zero under the null. The task methods were not randomly assigned, so this is a symmetry-based sensitivity analysis, not a design-based randomization test.', 'small')
para('MNIST12 and MNIST28 share a task family and prepared split, and MNIST12 TVAE tables retain documented holdout-feature collisions. Both facts weaken broad task-generalization claims. Additional experiment seeds do not supply additional independent datasets.', 'small')
para('Two observed seeds share heldout partitions; the reported intervals do not quantify uncertainty from resampling new test rows. The results show observed gains with limited evidence about generalization, not evidence that the two constructions are equivalent.', 'small')

section('6. Is Holm required for contrasts?', 'Does using a contrast automatically require Holm correction?',
        'Distinguish the individual estimand from the family of claims. Use the two-question family specified for this redo; also report the four-test family when drawing conclusions across both training regimes.',
        'No. A contrast is a way of defining a comparison; multiplicity comes from making several inferential claims. Holm is a defensible choice for the two related questions, not the only valid procedure.')
table(['Situation', 'Implication'], [
    ['One prespecified primary hypothesis', 'Ordinary p-value and interval can be used for that single claim; no additional contrast adjustment is needed.'],
    ['Several related claims', 'Define the family before interpreting significance. Holm controls the chance of at least one false rejection within that family, under valid marginal tests.'],
    ['Prespecified or orthogonal contrasts', 'Planning improves focus but does not automatically give familywise error control. Correlated contrasts do not invalidate Holm.'],
    ['Your two questions in one regime', 'Holm2. The smaller p must first be <=.025. Synthetic relabel raw p=.0350 misses this cutoff; mix raw p=.0566 also misses it.'],
    ['Claims across both regimes', 'Holm4 covers the four reported classification tests. Separate model fits and shared data do not by themselves justify separate testing families.'],
    ['Descriptive or exploratory results', 'Unadjusted p-values may be shown if labeled clearly; they should not be presented as familywise-controlled discoveries.'],
    ['Current results already examined', 'Restricting to two questions now is transparent and useful, but cannot retroactively create prespecification. Retain two-sided tests in this redo.']
], [152, 364])
para('The previous report tested four construction contrasts per regime. This report tests only relabeled versus generated and has_y=1 versus 0, as requested. It does not add separate full-relabel-versus-generated or X-only-relabel-versus-generated significance claims.', 'small')
para('Holm2 and Holm4 are family-specific results, not interchangeable thresholds to choose after viewing p-values. If the scientific claim concerns either question in either regime, use Holm4. A joint conjunctive claim requiring all component tests to succeed is a different hypothesis and would need its own stated decision rule.', 'small')
para('**Provenance and checks:** ' + checked + '. All 522 record/sidecar hashes matched; 3,770 frozen metric values matched source records. Eighteen models were refitted remotely with one numerical thread; all 202 ANOVA F statistics were independently reproduced by restricted least squares. No experiment, source record, split or research setting was changed.', 'small')
para('References: R stats p.adjust documentation (stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html); NIST, Bonferroni method (itl.nist.gov/div898/handbook/prc/section4/prc473.htm); SciPy, wilcoxon documentation (docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).', 'small')

def footer(canvas, doc):
    canvas.setFillColor(INK)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(48, 25, 'Real-data results | two research contrasts | ' + checked)
    canvas.drawRightString(564, 25, str(doc.page))

doc = SimpleDocTemplate(str(PDF), pagesize=letter, leftMargin=48, rightMargin=48,
                        topMargin=42, bottomMargin=44)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
markdown = '\n'.join(md)
(OUT / 'analysis.md').write_text(markdown, encoding='utf-8')
html_head = '<!doctype html><html><head><meta charset="utf-8"><title>Two research contrasts</title><style>body{font:15px/1.5 system-ui;color:#193247;max-width:1050px;margin:32px auto;padding:0 24px}h1{font-size:25px}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 20px}th{background:#edf4f6;text-align:left}td,th{padding:7px;border-bottom:1px solid #d7e3e9;vertical-align:top}.small{font-size:13px}pre{white-space:pre-wrap;background:#edf4f6;padding:10px}hr{margin:38px 0}</style></head><body>'
(OUT / 'analysis.html').write_text(html_head + '\n'.join(web) + '</body></html>', encoding='utf-8')
for path in DATA.glob('*'):
    if path.is_file():
        shutil.copy2(path, OUT / path.name)
for name in ['analyze.py', 'remote_io.py', 'build_report.py', 'deployment.json']:
    shutil.copy2(HERE / name, OUT / name)
prior = ROOT / 'audit/statistical_328_20261010'
inputs = {'scores.csv': prior / 'snapshot/scores.csv', 'source_snapshot.json': prior / 'snapshot/snapshot.json',
          'regression_d2_scores.csv': prior / 'results/regression_d2_scores.csv',
          'd2_verification.json': prior / 'results/d2_verification.json'}
frozen = OUT / 'frozen_inputs'
frozen.mkdir(exist_ok=True)
for name, path in inputs.items():
    shutil.copy2(path, frozen / name)
    assert hashlib.sha256((frozen/name).read_bytes()).hexdigest() == meta['source_hashes'][name]
manifest = dict(pdf=str(PDF), pages=len(PdfReader(PDF).pages), sections=sections, tables=table_manifest,
                prior_reports_preserved=True, experiments_changed=False)
(HERE / 'report_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(pdf=str(PDF), pages=manifest['pages'], report=str(OUT)), indent=2))
