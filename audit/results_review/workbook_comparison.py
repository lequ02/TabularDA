import json
from pathlib import Path

out = Path(__file__).resolve().parent
sheet = json.loads((out / 'workbook.json').read_text())[0]
headers = sheet['values'][0]
records = {}
dataset = None
for row in sheet['values'][1:]:
    dataset = row[0] or dataset
    records[(dataset, row[2])] = dict(zip(headers, row))

datasets = [
    ('adult','Adult','F1',(.669,.601,.626)),
    ('census','Census (Adult alias)*','F1',(.669,.601,.626)),
    ('census_kdd','Census-KDD','F1',(.494,.391,.377)),
    ('credit','Credit','F1',(.720,.672,.098)),
    ('covertype','Covertype','Macro-F1',(.652,.324,.433)),
    ('intrusion','Intrusion','Macro-F1',(.862,.528,.511)),
    ('mnist12','MNIST12','Accuracy',(.886,.394,.793)),
    ('mnist28','MNIST28','Accuracy',(.916,.371,.794)),
    ('news','News','R²',(.14,-.43,-.20)),
]
metric_keys = {'F1':'test_f1_binary', 'Macro-F1':'test_f1_macro', 'Accuracy':'test_accuracy', 'R²':'test_r2'}
methods = [('original','Our Original'),('ctgan','Our CTGAN'),('categorical','Categorical'),('gaussian','Gaussian'),('pca_gmm','PCA-GMM'),('tvae','Our TVAE')]

def value(ds, method, metric, suffix='end'):
    v = records.get((ds,method),{}).get(metric+'_'+suffix)
    return '—' if v is None else f'{v:.4f}'

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows])

rows = []
for idx,label in enumerate(['Paper Identity','Paper CTGAN','Paper TVAE']):
    rows.append([label]+[f'{d[3][idx]:.3f}' for d in datasets])
for method,label in methods:
    rows.append([label]+[value(ds,method,metric_keys[metric]) for ds,_,metric,_ in datasets])

parts = [
    '# Synthetic-data comparison — workbook-reported results',
    '**Local source:** `D:/SummerResearch/final_results.xlsx`, sheet `final_results!A1:O29`. All “Our” and alternative-method scores below come exclusively from this workbook. The requested `D:/SummerResearch/final/_results.xlsx` does not exist; this is the previously confirmed file.',
    '**Selection:** use the workbook’s `_end` columns consistently. The separate `_max` table below is included for reference; maxima may come from different epochs and are not validation-selected scores. Values are copied as reported, with no leakage adjustment or substitution from individual experiment logs.',
    '## Paper-compatible synthetic table',
    'Columns use the paper’s metrics: binary F1 for Adult/Census/Census-KDD/Credit; macro-F1 for Covertype/Intrusion; accuracy for MNIST; R² for News. Higher is better. F1/accuracy use a 0–1 scale.',
    table(['Method']+[f'{label} ({metric})' for _,label,metric,_ in datasets],rows),
    '*Local Census is an Adult alias, so the paper Adult reference is repeated for that column. The paper’s `census` corresponds to `census_kdd`, shown separately. Paper CTGAN is the supplied screenshot’s `TGAN(1)` row; Identity is its real-data reference.',
    '**Missing means missing:** “—” denotes an absent workbook dataset, method, or metric. The workbook has no Census-KDD, Intrusion, News, TVAE, RF, or XGBoost rows; no MNIST PCA-GMM rows; and some binary-F1 cells are empty, including Adult/Census Original and Census CTGAN. No scores from the downloaded logs were used to fill these cells.',
    '## Consistent local comparison: macro-F1 at end',
    'This supplementary table makes every available original-data row comparable with the workbook’s synthetic rows. Do not compare its binary-dataset macro-F1 directly with the paper’s binary F1.',
    table(['Dataset']+[label for _,label in methods],[[label]+[value(ds,m,'test_f1_macro') for m,_ in methods] for ds,label,_,_ in datasets]),
    '## What the workbook reports',
    '- **Adult and local Census:** CTGAN has the highest synthetic macro-F1; Original is higher still.',
    '- **Covertype:** CTGAN macro-F1 is **0.7227**, above Original **0.6871** and all other listed synthetic methods.',
    '- **Credit:** Gaussian has the highest synthetic fraud F1 (**0.5781**) and macro-F1 (**0.7887**); Original remains higher (**0.8571 / 0.9284**).',
    '- **MNIST12/28:** categorical accuracy (**0.8490 / 0.8193**) exceeds our CTGAN (**0.7370 / 0.5205**) and the published TVAE averages (**0.793 / 0.794**) numerically. Original remains higher (**0.9579 / 0.9776**). This does not establish a controlled win against TVAE: no local TVAE result is in the workbook.',
    '**Interpretation:** these are workbook-reported rankings, not validated synthetic-only performance claims. The audit established that some workbook rows average mixed real-plus-synthetic and synthetic-only runs. Paper scores also average classifiers under a different protocol. Neither issue is corrected by copying the workbook.',
    '## Appendix: workbook maximum scores, paper-compatible metrics',
    table(['Method']+[label for _,label,_,_ in datasets],[[label]+[value(ds,m,metric_keys[metric],'max') for ds,_,metric,_ in datasets] for m,label in methods]),
    'Paper source: user-provided Table 6 image and [CTGAN paper](https://arxiv.org/pdf/1907.00503). [Detailed audit](RESULTS_REVIEW.md). The workbook was read without modification.'
]
(out/'PAPER_COMPARISON.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
assert len(records)==28
assert records[('mnist12','categorical')]['test_accuracy_end']==0.849
assert records[('covertype','ctgan')]['test_f1_macro_end']==0.722667564
print('Updated report using 28 workbook records only; all 9 dataset/alias columns retained.')
