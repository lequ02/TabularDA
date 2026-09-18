import json
from pathlib import Path

out = Path(__file__).resolve().parent
runs = json.loads((out / 'reconstructed.json').read_text())['runs']
selected = {}
for r in runs:
    if r['train'] == 'mix' or r['version'] == 'old':
        continue
    if r['dataset'] == 'news' and '/new/' not in r['member']:
        continue
    key = (r['dataset'], r['method'])
    assert key not in selected, key
    selected[key] = r

def val(ds, method, metric, percent=True):
    v = selected.get((ds, method), {}).get('dev_loss_selected', {}).get(metric)
    return '—' if v is None else (f'{v * 100:.2f}' if percent else f'{v:.4f}')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(row) + ' |' for row in rows])

paper = 'https://arxiv.org/pdf/1907.00503'
parts = ['# Paper, replicated CTGAN, and our methods',
'''**Conclusion:** our original-data classifiers perform well on Adult, Credit and MNIST. Several synthetic methods beat our CTGAN runs, but none beats the original-data classifier on these four datasets by macro-F1. Credit CTGAN needs repair before drawing a general superiority claim.

The screenshot is **Table 5: real-data classifier baselines**, not CTGAN synthetic-data performance. Table 6 provides the synthetic benchmark; its proposed-method row is printed “TGAN(1)” in this arXiv version. Those benchmark scores average classifiers; our logs use one DNN. Cross-paper numbers are context, not a controlled replication comparison. [Source: paper, §5.2 and Tables 5–6](https://arxiv.org/pdf/1907.00503).

All local scores below select the first epoch at minimum logged dev loss. Synthetic-only and original-only runs are kept separate. New MNIST runs are used. These are observed scores, **not leakage-corrected reruns**.''',
'## 1. Screenshot baseline versus our non-augmented data',
'Accuracy (%); screenshot uses its MLP row, the closest model family—not an identical architecture.',
table(['Dataset', 'Paper real-data MLP', 'Our original data', 'Difference (points)'], [
    [ds, f'{p:.2f}', val(ds.lower(), 'original', 'test_accuracy'), f'{float(val(ds.lower(), "original", "test_accuracy")) - p:+.2f}']
    for ds,p in [('Adult',85.06),('Credit',99.92),('MNIST12',94.40),('MNIST28',97.28)]
]),
'## 2. Actual paper synthetic benchmark versus local runs',
'F1 and accuracy are percentages; R² is unscaled. Each row uses the same named metric, but paper/local protocols differ.',
table(['Dataset / metric', 'Paper CTGAN¹', 'Our CTGAN', 'Our original', 'Categorical', 'Gaussian', 'PCA-GMM'], [
    [label, paper_score] + [val(ds,m,metric,pct) for m in ['ctgan','original','categorical','gaussian','pca_gmm']]
    for ds,label,metric,paper_score,pct in [
      ('adult','Adult / binary F1','test_f1_binary','60.10',True),
      ('credit','Credit / binary F1','test_f1_binary','67.20',True),
      ('mnist12','MNIST12 / accuracy','test_accuracy','39.40',True),
      ('mnist28','MNIST28 / accuracy','test_accuracy','37.10',True),
      ('news','News / R²','test_r2','−0.4300',False),
    ]
]),
'¹ Table 6 row printed “TGAN(1)”; distinct from the real-data MLP values in the screenshot. Adult PCA-GMM binary F1 was not logged. No matching new News PCA-GMM run is included.',
'## 3. Local comparison across all available methods',
'**Macro-F1 (%)**, consistently applied within this table. RF and XGBoost denote synthetic-data labeling methods; TVAE is another generator. Missing entries are unavailable matching runs, not zeros.',
table(['Dataset','Original','CTGAN','Categorical','Gaussian','PCA-GMM','RF','XGBoost','TVAE'],[
    [label]+[val(ds,m,'test_f1_macro') for m in ['original','ctgan','categorical','gaussian','pca_gmm','rf','xgb','tvae']]
    for ds,label in [('adult','Adult'),('census','Census*'),('credit','Credit'),('covertype','Covertype new'),('mnist12','MNIST12 new'),('mnist28','MNIST28 new')]
]),
'''## Interpretation and limits

- **MNIST:** categorical and PCA-GMM beat our CTGAN; RF/XGBoost do better still, and TVAE is strongest among available synthetic runs. Original data remains best. Our CTGAN exceeds the paper's aggregate MNIST accuracy numerically; the screenshot's much higher numbers were real-data baselines.
- **Credit:** original-data fraud F1 is 78.79%; Gaussian reaches 27.59%, categorical 0%, and CTGAN 0.59%. High accuracy conceals poor fraud detection. CTGAN's synthetic labels are 32.99% fraud versus 0.17% in the test set.
- **Adult:** original data beats CTGAN; CTGAN beats the three core alternatives on macro-F1.
- **News:** original R² 0.0134 and CTGAN 0.0126 are both weak. Categorical, RF (−0.0753), XGBoost (−0.0375), and Gaussian do not improve on original. The screenshot's real-data MLP R² is 0.1492.
- **Coverage:** *local Census duplicates Adult and is not the paper's distinct Census benchmark.* No matched new Covertype CTGAN/original logs or Intrusion logs were available in the reconstructed download. These cannot support a complete eight-dataset replication claim.
- **Credibility:** mixed-run workbook averages are excluded. Known overlap and preprocessing issues remain; no confidence intervals or significance claims are justified from these single runs. Earlier fixed-model overlap-removal bounds do not establish clean retraining performance.

Local source: individual CSV logs from `G:/summer_research/download2`, indexed in `local_log_manifest.json` and reconstructed in `reconstructed.json` alongside this report. See [the detailed leakage review](RESULTS_REVIEW.md) for provenance and overlap bounds. Originals were not modified.''']
datasets = [
    ('adult','Adult','test_f1_binary',(.669,.601,.626)),
    ('census','Census (Adult alias)*','test_f1_binary',(.669,.601,.626)),
    ('census_kdd','Census-KDD','test_f1_binary',(.494,.391,.377)),
    ('credit','Credit','test_f1_binary',(.720,.672,.098)),
    ('covertype','Covertype','test_f1_macro',(.652,.324,.433)),
    ('intrusion','Intrusion','test_f1_macro',(.862,.528,.511)),
    ('mnist12','MNIST12','test_accuracy',(.886,.394,.793)),
    ('mnist28','MNIST28','test_accuracy',(.916,.371,.794)),
    ('news','News','test_r2',(.14,-.43,-.20)),
]
metrics = {'test_f1_binary':'Binary F1','test_f1_macro':'Macro-F1','test_accuracy':'Accuracy','test_r2':'R²'}
baseline_rows = []
method_rows = []
for ds,label,metric,published in datasets:
    baseline_rows.append([label, metrics[metric]] + [f'{v:.3f}' for v in published] + [val(ds,m,metric,False) for m in ['original','ctgan','tvae']])
    method_rows.append([label] + [val(ds,m,metric,False) for m in ['categorical','gaussian','pca_gmm','rf','xgb']])
parts = [
    '# All-dataset comparison: CTGAN, TVAE, original data and our methods',
    '**Dataset mapping:** paper `adult` → local `adult` and its duplicate `census`; paper `census` → local `census_kdd`. Adult and local Census are separate runs on identical data, not independent benchmarks. All eight paper datasets are represented below; the local alias adds a ninth row.',
    '## Published and replicated baselines',
    '**Higher is better.** F1/accuracy use the 0–1 scale; R² is unscaled and can be negative. Paper Identity means training on real data. Local Original means our non-augmented real-data run. “—” means a matching evaluable run/metric is unavailable, not zero.',
    table(['Dataset','Metric','Paper Identity','Paper CTGAN','Paper TVAE','Our Original','Our CTGAN','Our TVAE'],baseline_rows),
    '*The paper Adult reference is repeated for the local Census alias, solely to show the correct mapping. It is not the paper Census-KDD result. Paper CTGAN values come from the row labeled `TGAN(1)` in the supplied screenshot.',
    '## Our other synthetic-only methods',
    'Each row uses the same metric as the baseline table above. RF/XGBoost are labeling methods, not the downstream evaluation classifier.',
    table(['Dataset','Categorical','Gaussian','PCA-GMM','RF','XGBoost'],method_rows),
    '## Conclusions',
    '- **MNIST12/28:** our TVAE accuracies are **92.25% / 93.14%**, versus CTGAN **53.35% / 42.97%**. TVAE beats all available alternative synthetic methods; original data remains best at **95.92% / 97.83%**. Categorical and PCA-GMM beat CTGAN, but do not beat TVAE.',
    '- **Covertype:** PCA-GMM macro-F1 **0.4634** exceeds our TVAE **0.2758**. RF and XGBoost score **0.2040 / 0.4287**. A matching new CTGAN or Original run is missing, so their local ranking is unresolved.',
    '- **Adult / local Census:** CTGAN beats categorical and Gaussian on binary F1; Original performs better. PCA-GMM binary F1 is missing, but its logged macro-F1 trails CTGAN on both. No local TVAE evaluation is available.',
    '- **Credit:** Gaussian beats our CTGAN on fraud F1 (**0.2759 vs 0.0059**), but Original is much stronger (**0.7879**); categorical detects no fraud. CTGAN synthetic labels contain 32.99% fraud versus 0.17% in the real test set. No local TVAE result is available.',
    '- **Census-KDD / Intrusion:** raw data is present, but matching evaluation logs are missing. Published CTGAN/TVAE scores are shown; local winners cannot be established.',
    '- **News:** Original (**0.0134**) and CTGAN (**0.0126**) have weak R²; available alternative methods are worse. No matching local TVAE or new PCA-GMM result is available.',
    '## Scope and credibility',
    'Local runs use the first epoch at minimum recorded dev loss, with new MNIST/Covertype folders and the new News folder. Mixed real-plus-synthetic runs and invalid workbook averages are excluded. Paper scores average downstream classifiers; ours use one DNN, with differing splits/settings. Cross-paper differences are descriptive, not proof of replication or superiority. These are logged results, **not leakage-corrected reruns**, and no repeated-seed uncertainty is available.',
    'The download archive inventory and workspace result-file search found no Census-KDD/Intrusion evaluation CSVs. Dataset and generator files alone cannot supply performance scores. Adult/Census PCA-GMM logs omit binary F1. A headerless, truncated extra MNIST12 RF CSV was not treated as an additional valid run.',
    'Sources: user-supplied Table 6 screenshot; [CTGAN paper, §5.2 and Tables 4–6](https://arxiv.org/pdf/1907.00503); individual logs in `G:/summer_research/download2`, indexed in `local_log_manifest.json` and `reconstructed.json` alongside this report. [Detailed leakage review](RESULTS_REVIEW.md). Original files were unchanged.'
]
(out / 'PAPER_COMPARISON.md').write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
print('Updated PAPER_COMPARISON.md: 9 dataset rows, published and local CTGAN/TVAE, plus all available alternative methods.')
