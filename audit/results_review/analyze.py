import csv,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sheet=json.loads((OUT/'workbook.json').read_text())[0]
headers=sheet['values'][0]
records=[]
dataset=None
for i,row in enumerate(sheet['values'][1:],2):
    dataset=row[0] or dataset
    records.append({'row':i,'dataset':dataset,'method':row[2],**dict(zip(headers[3:],row[3:]))})
logs=[]
for p in (ROOT/'src').rglob('*.acc.csv'):
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        rows=list(reader)
    if not rows:continue
    stats={}
    for col in reader.fieldnames:
        try:values=[float(r[col]) for r in rows]
        except (ValueError,TypeError):continue
        stats[col]={'max':max(values),'end':values[-1],'min':min(values),'max_epoch':values.index(max(values))+1}
    logs.append({'file':str(p.relative_to(ROOT)),'rows':len(rows),'stats':stats})
    if len(reader.fieldnames)==5 and rows[0].get(None) and len(rows[0][None])==2 and ('mnist' in str(p)):
        # Explicit alternate interpretation of malformed classification CSVs:
        # 5-column regression header, but 7-field classification data rows.
        with p.open(encoding='utf-8-sig',newline='') as f: raw=list(csv.reader(f))[1:]
        repaired={}
        for j,col in enumerate(['global_round','train_loss','train_acc','train_f1','test_loss','test_acc','test_f1']):
            values=[float(row[j]) for row in raw]
            repaired[col]={'max':max(values),'end':values[-1],'min':min(values),'max_epoch':values.index(max(values))+1}
        logs.append({'file':str(p.relative_to(ROOT))+' [inferred 7-column classification header]','rows':len(rows),'stats':repaired})
matches=[]
for rec in records:
    candidates=[]
    for log in logs:
        if '/'+rec['dataset']+'/' not in log['file'].replace('\\','/'):continue
        match=[]
        for col in headers[3:]:
            v=rec[col]
            if not isinstance(v,(int,float)):continue
            base,agg=col.rsplit('_',1)
            options=[(base,1)]
            if base=='test_accuracy':options += [('test_acc',100),('test_acc',1)]
            if base=='test_f1_macro':options += [('test_f1',1)]
            for logcol,scale in options:
                if logcol in log['stats'] and abs(log['stats'][logcol][agg]/scale-v)<6e-10:
                    match.append(col);break
        if match:candidates.append({'file':log['file'],'matches':match,'count':len(match)})
    candidates.sort(key=lambda c:-c['count'])
    matches.append({'dataset':rec['dataset'],'method':rec['method'],'row':rec['row'],'candidates':candidates[:3]})
    print(json.dumps(matches[-1]))
comparisons=[]
for ds in dict.fromkeys(r['dataset'] for r in records):
    group=[r for r in records if r['dataset']==ds]
    ct=next(r for r in group if r['method']=='ctgan')
    for r in group:
        if r['method'] in ('ctgan','original'):continue
        comparisons.append({'dataset':ds,'method':r['method'],'row':r['row'],'ctgan_row':ct['row'],
          'macro_end':r['test_f1_macro_end'],'ctgan_macro_end':ct['test_f1_macro_end'],
          'macro_end_delta':r['test_f1_macro_end']-ct['test_f1_macro_end'],
          'accuracy_end_delta':r['test_accuracy_end']-ct['test_accuracy_end'],
          'macro_max_delta':r['test_f1_macro_max']-ct['test_f1_macro_max']})
print('COMPARISONS',json.dumps(comparisons))
(OUT/'analysis.json').write_text(json.dumps({'records':records,'matches':matches,'comparisons':comparisons,'logs':logs},indent=2))
print('FORMULAS',sum(bool(f) for row in sheet['formulas'] for f in row))
for r in records:
    f=r['test_f1_binary_end']; a=r['test_accuracy_end']; macro=r['test_f1_macro_end']
    if f is None or f>=1:continue
    tp=f*(1-a)/(2*(1-f)); tn=a-tp
    implied_f0=2*tn/(2*tn+1-a)
    print('BINARY_COHERENCE',json.dumps({'row':r['row'],'dataset':r['dataset'],'method':r['method'],'macro_reported':macro,'macro_implied_by_accuracy_and_binary_f1':(implied_f0+f)/2,'difference':macro-(implied_f0+f)/2}))
