"""Stream only selected CSV members; never extract or execute archive content."""
import hashlib,json,zipfile
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parent
manifest={e['member']:e for e in json.loads((OUT/'archive_manifest.json').read_text())}
cached={}
def read(member):
    e=manifest[member]; full=[]; feat=[]; labels=[]; logical=hashlib.sha256()
    with zipfile.ZipFile(e['archive']) as z, z.open(member) as f:
        header=next(f).rstrip(b'\r\n'); logical.update(header+b'\n')
        for raw in f:
            row=raw.rstrip(b'\r\n'); logical.update(row+b'\n')
            x,y=row.rsplit(b',',1)
            full.append(hashlib.sha256(row).digest()); feat.append(hashlib.sha256(x).digest()); labels.append(y.decode())
    return {'header':header,'full':full,'features':feat,'labels':labels,'logical_sha256':logical.hexdigest()}
results=[]
for ds in ['adult','census','covertype','credit','mnist12','mnist28']:
    for version in (['current','old_split'] if ds.startswith('mnist') else ['current']):
        prefix=f'data/data/{ds}/'+('old_split/' if version=='old_split' else '')
        for part in ['train','test']:
            member=prefix+f'{ds}_{part}.csv';cached[(ds,version,part)]=read(member)
        train=cached[(ds,version,'train')];test=cached[(ds,version,'test')]
        if train['header']!=test['header']:raise ValueError('Column order mismatch: '+prefix)
        s=set(train['features']);fs=set(train['full']);mask=[x in s for x in test['features']]
        item={'dataset':ds,'version':version,'train_rows':len(train['full']),'test_rows':len(test['full']),
          'feature_overlap':sum(mask),'full_overlap':sum(x in fs for x in test['full']),
          'test_counts':dict(Counter(test['labels'])),
          'overlap_counts':dict(Counter(y for y,m in zip(test['labels'],mask) if m)),
          'train_sha256_normalized_newlines':train['logical_sha256'],'test_sha256_normalized_newlines':test['logical_sha256'],
          'source_prefix':prefix}
        results.append(item);print(json.dumps(item),flush=True)
for ds in ['mnist12','mnist28']:
    for tv,rv in [('current','old_split'),('old_split','current')]:
        test=cached[(ds,tv,'test')];train=cached[(ds,rv,'train')]
        s=set(train['features'])
        print(json.dumps({'cross_version':ds,'test':tv,'train':rv,'feature_overlap':sum(h in s for h in test['features']),'test_rows':len(test['full'])}),flush=True)
print('ADULT_CENSUS_EQUAL',cached[('adult','current','train')]['full']==cached[('census','current','train')]['full'],cached[('adult','current','test')]['full']==cached[('census','current','test')]['full'])
(OUT/'local_overlap.json').write_text(json.dumps(results,indent=2))
