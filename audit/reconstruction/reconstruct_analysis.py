import csv, hashlib, json, re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from pypdf import PdfReader

ROOT=Path('D:/SummerResearch')
SRC=Path('D:/Rprojects/research_data_synthesis')
OUT=ROOT/'audit/reconstruction'
inventory=[]
for p in sorted(SRC.iterdir()):
    if p.is_file():
        inventory.append(dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if p.suffix.lower()=='.pdf':
            try:
                pdf=PdfReader(p)
                (OUT/(p.stem+'_pdf.txt')).write_text('\n\n'.join(f'PAGE {i+1}\n'+(pg.extract_text() or '') for i,pg in enumerate(pdf.pages)),encoding='utf-8')
            except Exception as e: print('PDF_ERROR',p.name,str(e))
(OUT/'analysis_inventory.json').write_text(json.dumps(inventory,indent=2))

longs={}
for fname in ['Mar23.csv','April02.csv','April29.csv']:
    with (SRC/fname).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
    records=[]
    for r in rows[4:]:
        if not r:continue
        for j,s in enumerate(r[1:],1):
            if s.strip():
                try: val=float(s)
                except ValueError:continue
                records.append(dict(source=fname,source_row=rows.index(r)+1,source_col=j+1,method=r[0],metric=rows[0][j],train=rows[1][j],dataset=rows[2][j],value=val))
    d=pd.DataFrame(records);longs[fname]=d
    d.to_csv(OUT/(Path(fname).stem+'_tidy.csv'),index=False)
    print('PIVOT',fname,'records',len(d),'datasets',sorted(d.dataset.unique()),'methods',len(d.method.unique()))

data=pd.read_csv(SRC/'April29_macro_max_mfa.csv')
nog=pd.read_csv(SRC/'April29_macro_max_mfa_no_gauss.csv')
print('FACTORS',len(data),len(nog),nog.groupby('dataset').size().to_dict(),Counter(nog.method))
wide=longs['April29.csv']
check=[]
for r in data.to_dict('records'):
    matches=wide[(wide.method==r['method'])&(wide.dataset==r['dataset'])&(wide.metric=='test_f1_macro_max')]
    check.append(len(matches)==1 and matches.iloc[0]['train']=='synthetic' and abs(matches.iloc[0].value-r['macro_max'])<1e-12)
print('DERIVATION',sum(check),'/',len(check),'exact synthetic macro-max matches')
assert all(check)
assert len(nog)==len(data[~data.method.str.contains('gauss')])

outcomes={}
for metric in ['test_f1_macro_max','test_f1_macro_end']:
    d=wide[(wide.train=='synthetic')&(wide.metric==metric)].copy()
    d['approach']=np.where(d.method.isin(['ctgan','tvae']),'old','new')
    for exclusion in [True,False]:
        x=d[~d.method.str.contains('gauss')] if exclusion else d
        key=metric+('_no_gauss' if exclusion else '_all')
        means=x.groupby('approach').value.mean().to_dict()
        diff=means['new']-means['old']
        per=x.groupby(['dataset','approach']).value.mean().unstack()
        per['diff']=per['new']-per['old']
        outcomes[key]=dict(means=means,diff=diff,relative=diff/means['old'],by_dataset=per.to_dict('index'))
        print('SUMMARY',key,json.dumps(outcomes[key]))
        per.to_csv(OUT/(key+'_dataset_means.csv'))

print('LATEST_TABLE')
tab=wide[(wide.metric=='test_f1_macro_max')&(wide.train.isin(['original','synthetic']))].pivot(index='method',columns='dataset',values='value')
print(tab.round(5).to_string())
tab.to_csv(OUT/'April29_macro_max_table.csv')
print('BASELINE',wide[(wide.method.isin(['ctgan','tvae','none']))&(wide.metric=='test_f1_macro_max')][['dataset','train','method','value']].to_string(index=False))

# Adult/Census/KDD test-size signatures implied by six-decimal+ accuracy values.
acc=wide[(wide.metric=='test_f1_micro_end')&(wide.train=='synthetic')]
for ds in acc.dataset.unique():
    vals=acc[acc.dataset==ds].value.to_numpy()
    print('DENOM',ds,{n:int(np.sum(np.abs(vals*n-np.round(vals*n))<1e-5)) for n in [9526,9551,10000,14000,116203]})

for name in ['github_bn','github_phuong']:
    c=json.loads((OUT/(name+'_commits.json')).read_text(encoding='utf-8-sig'))
    print('BRANCH',name)
    print('\n'.join(x['sha'][:12]+' '+x['commit']['committer']['date']+' '+x['commit']['message'].splitlines()[0] for x in c[:25]))
    t=json.loads((OUT/(name+'_tree.json')).read_text(encoding='utf-8-sig'))
    paths=[x['path'] for x in t['tree']]
    print('TREE',t['sha'],len(paths),'logs',sum(p.endswith('.acc.csv') for p in paths))
    hits=[p for p in paths if any(k in p.lower() for k in ['april','result','post_process','constants.py','data_loader.py','create_train','classification_train','agents.md']) and not any(k in p for k in ['__pycache__','.png'])]
    print('\n'.join(hits[:75]))
(OUT/'analysis_numbers.json').write_text(json.dumps(outcomes,indent=2))
