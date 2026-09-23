import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def hashes(path):
    full=[]; features=[]; labels=[]
    for chunk in pd.read_csv(path,chunksize=5000):
        chunk=chunk.loc[:,~chunk.columns.str.startswith('Unnamed:')]
        cols=sorted(c for c in chunk.columns if c!='label')
        a=chunk[cols+['label']].to_numpy(dtype=np.float64)
        # Byte hashing after column ordering and numeric normalization; no rounding.
        a[a==0]=0
        full.extend(hashlib.sha256(row.tobytes()).digest() for row in a)
        features.extend(hashlib.sha256(row[:-1].tobytes()).digest() for row in a)
        labels.extend(a[:,-1].astype(int).tolist())
    return {'full':full,'features':features,'labels':labels}

results=[]
for ds in ['mnist12','mnist28']:
    data={}
    for version in ['current','old_split']:
        base=ROOT/'data'/ds/('' if version=='current' else version)
        for part,name in [('train',f'{ds}_train.csv'),('test',f'{ds}_test.csv'),('ctgan',f'onehot_{ds}_sdv_100k.csv'),('categorical',f'onehot_{ds}_sdv_categorical_100k.csv')]:
            data[(version,part)]=hashes(base/name)
            print(json.dumps({'loaded':str(base/name),'rows':len(data[(version,part)]['full'])}),flush=True)
    for version in ['current','old_split']:
        test=data[(version,'test')]
        for trainversion in ['current','old_split']:
            for part in ['train','ctgan','categorical']:
                train=data[(trainversion,part)]
                matching=set(train['features'])
                mask=[h in matching for h in test['features']]
                label_full=set(train['full'])
                item={'dataset':ds,'test_version':version,'test_rows':len(mask),'compared_version':trainversion,'compared_part':part,
                      'feature_matches':sum(mask),'full_row_matches':sum(h in label_full for h in test['full']),
                      'feature_matches_by_label':{str(c):sum(m for m,y in zip(mask,test['labels']) if y==c) for c in sorted(set(test['labels']))},
                      'test_class_counts':{str(c):test['labels'].count(c) for c in sorted(set(test['labels']))}}
                results.append(item);print(json.dumps(item),flush=True)
    for part in ['train','test','ctgan','categorical']:
        print(json.dumps({'dataset':ds,'part':part,'current_equals_old_normalized':data[('current',part)]['full']==data[('old_split',part)]['full']}),flush=True)
(OUT/'versions.json').write_text(json.dumps(results,indent=2))
