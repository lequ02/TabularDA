import ast, csv, hashlib, json, re
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd
OUT=Path(__file__).resolve().parent
trace=(OUT/'aggregation_run_trace.txt').read_text(encoding='utf-8')
pairs=re.findall(r'^Processing (.*?)\.\.\.\nExtracted info - Dataset: (.*?), Train: (.*?), Augment: (.*?)\n',trace,re.M)
counts=Counter((ds,train,method) for _,ds,train,method in pairs)
print('TRACE_ROWS',len(pairs),'UNIQUE_GROUPS',len(counts),'DUPLICATE_GROUPS',[(k,v) for k,v in counts.items() if v>1])
print('COVERTYPE',*[p for p in pairs if p[1]=='Covertype'],sep='\n')
rows=[dict(file=p,dataset=d,train=t,method=m) for p,d,t,m in pairs]
(OUT/'traced_run_paths.json').write_text(json.dumps(rows,indent=2))
wide=pd.read_csv(OUT/'April29_tidy.csv')
print('TEST_SIZE_BY_METHOD')
for r in wide[(wide.dataset=='Covertype')&(wide.metric=='test_f1_micro_end')].to_dict('records'):
    v=r['value'];print(r['train'],r['method'],v,{n:round(abs(v*n-round(v*n)),6) for n in [10000,116203]})
print('GENERATIONS')
for f in ['Mar23','April02']:
    old=pd.read_csv(OUT/(f+'_tidy.csv'))
    m=old.merge(wide,on=['dataset','method','train','metric'],suffixes=('_old','_new'))
    same=(m.value_old-m.value_new).abs()<1e-9
    print(f,'toApril29',len(m),'same',int(same.sum()),'changed',int((~same).sum()))
    print(m[~same].groupby(['dataset','train']).size().to_dict())

d=pd.read_csv('D:/Rprojects/research_data_synthesis/April29_macro_max_mfa.csv')
print('BASELINE_AND_METHOD_MEANS',d.groupby('method').macro_max.mean().round(6).to_dict())
print('GROUPED_SENSITIVITY')
for nog in [True,False]:
    x=d[~d.method.str.contains('gauss')].copy() if nog else d.copy()
    x['dataset_unit']=x.dataset.replace({'Census':'Adult','Census_Kdd':'Adult'})
    unit=x.groupby(['dataset_unit','approach']).macro_max.mean().unstack()
    unit['delta']=unit['new']-unit['old']
    print('no_gauss',nog,unit.to_dict('index'),'difference',float(unit.delta.mean()),'relative',float(unit.delta.mean()/unit.old.mean()))

if (OUT/'github_evidence_manifest.json').exists():
    manifest=json.loads((OUT/'github_evidence_manifest.json').read_text())
    report_results=[]
    for e in manifest:
        if 'local' not in e or not e['path'].endswith('.report.txt'):continue
        text=Path(e['local']).read_text(encoding='utf-8-sig')
        match=re.search(r'\{.*\}',text)
        if not match:continue
        metrics=ast.literal_eval(match[0])
        name=e['path'].rsplit('/',1)[-1]
        info=re.match(r'DNN_(.+?)_train_(.+?)_augment_(.+?)_mix_ratio',name)
        if info:ds,tr,me=info.groups()
        else:
            info=re.match(r'DNN_(.+?)_train_(.+?)_test_',name)
            if not info:continue
            ds,tr=info.groups();me='none'
        for metric,v in metrics.items():
            key='test_'+metric+'_end'
            m=wide[(wide.dataset==ds)&(wide.train==tr)&(wide.method==me)&(wide.metric==key)]
            if len(m)==1:
                diff=float(v)-float(m.iloc[0].value)
                report_results.append(dict(path=e['path'],blob=e['sha'],dataset=ds,train=tr,method=me,metric=key,report_value=v,aggregate_value=float(m.iloc[0].value),difference=diff))
    check=pd.DataFrame(report_results)
    check.to_csv(OUT/'report_matches.csv',index=False)
    print('REPORT_MATCH',len(check),'within1e-8',int((check.difference.abs()<1e-8).sum()),'maxdiff',check.difference.abs().max())
    print('REPORT_MATCH_BY_DATASET',check.assign(matches=check.difference.abs()<1e-8).groupby('dataset').matches.agg(['count','sum']).to_dict('index'))
    print('POINTERS')
    for e in manifest:
        if e['path'].startswith('data/') and 'local' in e:
            ptr=Path(e['local']).read_text();o=re.search(r'oid sha256:(\w+)',ptr)
            local=Path('D:/SummerResearch')/e['path']
            if o and local.exists():
                h=hashlib.file_digest(local.open('rb'),'sha256').hexdigest()
                print(e['path'],o[1],h,h==o[1])

