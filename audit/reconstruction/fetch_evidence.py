import base64, concurrent.futures, hashlib, json, re, subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
tree=json.loads((OUT/'github_phuong_tree.json').read_text(encoding='utf-8-sig'))['tree']
trace=(OUT/'aggregation_run_trace.txt').read_text(encoding='utf-8')
names={p.rsplit('\\',1)[-1].removesuffix('.acc.csv')+'.report.txt' for p in re.findall(r'^Processing (.*?)\.\.\.$',trace,re.M)}
targets=[e for e in tree if e['type']=='blob' and e['path'].startswith('src/modeling_thuy/output/') and e['path'].rsplit('/',1)[-1] in names and e.get('size',0)<5000]
targets += [e for e in tree if e['type']=='blob' and e['path'].startswith('data/') and e['path'].endswith(('_train.csv','_test.csv')) and e.get('size',0)<500]
dest=OUT/'github_blobs';dest.mkdir(exist_ok=True)
def get(e):
    p=dest/e['sha']
    if not p.exists():
        r=subprocess.run(['D:/GitHub CLI/gh.exe','api','repos/thuydt02/DA/git/blobs/'+e['sha']],capture_output=True,text=True,encoding='utf-8')
        if r.returncode:return {**e,'error':r.stderr[:250]}
        b=base64.b64decode(json.loads(r.stdout)['content'])
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==e['sha']
        p.write_bytes(b)
    return {**e,'local':str(p)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: result=list(pool.map(get,targets))
(OUT/'github_evidence_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Evidence files',len(result),'errors',sum('error' in e for e in result),'matched trace basenames',len(names))
