import ast,csv,hashlib,json,re,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent
D=OUT/'drive'
cloud_manifest=json.loads((D/'log_manifest.json').read_text())
cloud_hashes={hashlib.sha256(Path(e['local']).read_bytes()).hexdigest():e for e in cloud_manifest}
manifest=[]
for e in json.loads((OUT/'local_log_manifest.json').read_text()):
    cloud=cloud_hashes.get(e['sha256'],{})
    manifest.append({**e,'name':Path(e['member']).name,'id':cloud.get('id','local:'+e['sha256'][:16]),'parent':cloud.get('parent','')})
old={'1I87Rpgt_ro5rLajwXH0oS8O3BD2bteZW','1GnYTdAjuhlW_2e9s6JyeM9q0XUHhu2nV','15eDjTcRVcsAgAMeQUjoDt7Q1OkGI1W1Y'}
new={'1m2ajH-MfeiJr4omWZzhYWcrWG8jnfO7m','1S4KDDgbN5QZbnCtNiZJR3y0Jp7RHIRWm','18L3VF2Hfab3b5Xgdd87qEaAnp0X1fhzG'}
records=[]
for e in manifest:
    name=e['name']; ds=re.search(r'DNN_(.*?)_(?:mix_)?train_',name)[1].lower()
    train=re.search(r'_train_(original|synthetic|mix)',name)[1]
    method='original' if train=='original' else re.search(r'_augment_(.*?)(?:_mix_ratio|_test_)',name)[1]
    with Path(e['local']).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows:continue
    metrics=[k for k in rows[0] if k and (k.startswith('test_') or k.startswith('dev_'))]
    nums=[{k:float(row[k]) for k in metrics if row[k] not in (None,'')} for row in rows]
    # Some logs omit test columns from the header but append a literal metrics
    # dict and duplicate scalar values. Recover only the explicit dictionary.
    recovered=0
    lines=Path(e['local']).read_text(encoding='utf-8-sig').splitlines()[1:]
    for i,line in enumerate(lines):
        literal=re.search(r'\{[^}]+\}',line)
        if literal:
            values=ast.literal_eval(literal[0])
            nums[i].update({'test_'+k:float(v) for k,v in values.items()})
            recovered+=1
    metrics=sorted(set().union(*(r.keys() for r in nums)))
    record={**e,'dataset':ds,'train':train,'method':method,'version':'old' if e['parent'] in old else 'new' if e['parent'] in new else 'unspecified','epochs':len(rows),
       'end':nums[-1], 'max':{k:max(r[k] for r in nums if k in r) for k in metrics},'recovered_test_dict_rows':recovered}
    for selection in ['dev_loss','dev_f1_macro','dev_f1_micro']:
        if selection in nums[0]:
            idx=(min if selection=='dev_loss' else max)(range(len(nums)),key=lambda i:nums[i][selection])
            record[selection+'_selected']={'epoch':idx+1,**nums[idx]}
    records.append(record)
with (D/'final_results.csv').open(newline='') as f: final=list(csv.DictReader(f))
workbook=json.loads((OUT/'workbook.json').read_text())[0]['values']
workbook_match=0
for i,r in enumerate(final):
    for j,h in enumerate(workbook[0][3:],3):
        val=workbook[i+1][j]
        if val is not None and r[h] and abs(val-float(r[h]))<=5.1e-10:workbook_match+=1
print('WORKBOOK_CLOUD_MATCH',workbook_match,'numeric cells')
reconciled=[]
for f in final:
    candidates=[r for r in records if r['dataset']==f['Dataset'] and r['method']==f['Augment_Type'] and r['version']!='new']
    if not candidates:continue
    errors=[];matched=[]
    for key,value in f.items():
        if key in ('Dataset','Train_Augment_Option','Augment_Type') or not value:continue
        metric,agg=key.rsplit('_',1)
        vals=[r[agg][metric] for r in candidates if metric in r[agg]]
        if vals:
            calc=statistics.mean(vals)
            if abs(calc-float(value))<1e-10:matched.append(key)
            else:errors.append({'key':key,'workbook':float(value),'reconstructed':calc})
    item={'dataset':f['Dataset'],'method':f['Augment_Type'],'runs':[{k:r[k] for k in ['id','train','method','version']} for r in candidates],'matched_metrics':matched,'mismatches':errors}
    reconciled.append(item)
    print('RECONCILE',json.dumps(item))
print('RUNS')
for r in records:
    if r['train']!='synthetic':continue
    print(json.dumps({k:r[k] for k in ['id','dataset','method','version','epochs']}|{'macro_end':r['end'].get('test_f1_macro'),'accuracy_end':r['end'].get('test_accuracy'),'dev_loss_epoch':r.get('dev_loss_selected',{}).get('epoch'),'macro_at_min_dev_loss':r.get('dev_loss_selected',{}).get('test_f1_macro'),'accuracy_at_min_dev_loss':r.get('dev_loss_selected',{}).get('test_accuracy')}))
(OUT/'reconstructed.json').write_text(json.dumps({'runs':records,'reconciliation':reconciled},indent=2))
