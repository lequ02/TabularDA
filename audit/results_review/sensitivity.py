import csv,json,math,zipfile
from pathlib import Path
import pandas as pd
OUT=Path(__file__).resolve().parent
runs=json.loads((OUT/'reconstructed.json').read_text())['runs']
overlaps=json.loads((OUT/'local_overlap.json').read_text())
comparison=[]
for ds in ['adult','census','credit','mnist12','mnist28']:
    version='new' if ds.startswith('mnist') else 'unspecified'
    group={r['method']:r for r in runs if r['dataset']==ds and r['version']==version and r['train']=='synthetic'}
    if 'ctgan' not in group:continue
    ct=group['ctgan']['dev_loss_selected']
    overlap=next(o for o in overlaps if o['dataset']==ds and o['version']=='current')
    p=overlap['feature_overlap']/overlap['test_rows']
    for method in ['categorical','gaussian','pca_gmm','rf','xgb','tvae']:
        if method not in group:continue
        r=group[method];selected=r['dev_loss_selected'];gap=selected['test_accuracy']-ct['test_accuracy']
        item={'dataset':ds,'method':method,'version':version,'epoch':selected['epoch'],'ctgan_epoch':ct['epoch'],
          'macro_f1':selected['test_f1_macro'],'ctgan_macro_f1':ct['test_f1_macro'],
          'accuracy':selected['test_accuracy'],'ctgan_accuracy':ct['test_accuracy'],
          'duplicates':overlap['feature_overlap'],'test_rows':overlap['test_rows'],
          'accuracy_gap_after_removal_lower_bound':max(-1,(gap-p)/(1-p)),
          'accuracy_gap_after_removal_upper_bound':min(1,(gap+p)/(1-p)),
          'source':r['member'],'ctgan_source':group['ctgan']['member']}
        comparison.append(item);print(json.dumps(item))

def binary_bounds(r):
    s=r['dev_loss_selected'];n=10000;positive=17;negative=n-positive
    errors=round((1-s['test_accuracy'])*n);f=s['test_f1_binary']
    tp_float=f*errors/(2*(1-f));tp=round(tp_float)
    if abs(tp-tp_float)>1e-5:raise ValueError('Cannot recover integer confusion counts')
    fn=positive-tp;fp=errors-fn;tn=negative-fp
    if min(tp,fn,fp,tn)<0:raise ValueError('Inconsistent class counts')
    def score(tp,fn,fp,tn):
        pos=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        neg=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
        return (pos+neg)/2
    if abs(score(tp,fn,fp,tn)-s['test_f1_macro'])>1e-8:raise ValueError('Macro F1 mismatch')
    values=[]
    for remove_tp in range(2):
        for remove_fp in range(50):
            q=(tp-remove_tp,fn-(1-remove_tp),fp-remove_fp,tn-(49-remove_fp))
            if min(q)>=0:values.append(score(*q))
    return {'method':r['method'],'epoch':s['epoch'],'confusion':{'TP':tp,'FN':fn,'FP':fp,'TN':tn},'macro_f1_after_removal_min':min(values),'macro_f1_after_removal_max':max(values)}
bounds=[binary_bounds(r) for r in runs if r['dataset']=='credit' and r['train']=='synthetic']
print('CREDIT_BOUNDS',json.dumps(bounds))

manifest={e['member']:e for e in json.loads((OUT/'archive_manifest.json').read_text())}
label_checks=[]
for method,suffix in [('ctgan',''),('categorical','_categorical'),('gaussian','_gaussian')]:
    member=f'data/data/credit/onehot_credit_sdv{suffix}_100k.csv'
    e=manifest[member]
    with zipfile.ZipFile(e['archive']) as z,z.open(member) as f:
        df=pd.read_csv(f,usecols=['Class'])
    item={'method':method,'counts':df['Class'].value_counts(dropna=False).to_dict(),'rows':len(df)}
    label_checks.append(item);print('CREDIT_SYNTHETIC_LABELS',json.dumps(item))
(OUT/'sensitivity.json').write_text(json.dumps({'comparison':comparison,'credit_bounds':bounds,'credit_synthetic_labels':label_checks},indent=2))
