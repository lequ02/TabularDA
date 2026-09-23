"""Read public, user-supplied Drive folder listings and selected files."""
import concurrent.futures,json,re,sys,urllib.request
from pathlib import Path
OUT=Path(__file__).resolve().parent/'drive'
OUT.mkdir(exist_ok=True)

def listing(fid):
    url='https://drive.google.com/drive/folders/'+fid
    with urllib.request.urlopen(url,timeout=40) as r: html=r.read(8000000).decode('utf-8')
    # Keep only the parsed listing. Drive HTML embeds application credentials.
    m=re.search(r"_DRIVE_ivd'\]\s*=\s*'(.*?)';",html,re.S)
    if not m:raise ValueError('No public file listing: '+fid)
    s=re.sub(r'\\x([0-9a-fA-F]{2})',lambda m:chr(int(m[1],16)),m[1])
    data=json.loads(s.replace("\\'","'"))
    entries=[{'id':a[0],'name':a[2],'type':a[3],'parent':fid} for a in (data[0] or [])]
    (OUT/(fid+'.json')).write_text(json.dumps(entries,indent=2),encoding='utf-8')
    return entries

if __name__=='__main__':
    if sys.argv[1]=='--logs':
        entries={}
        for path in OUT.glob('*.json'):
            data=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data,list):continue
            for e in data:
                if e.get('name','').endswith('.acc.csv'):entries[e['id']]=e
        logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
        def fetch(e):
            p=logdir/(e['id']+'.csv')
            if not p.exists():
                with urllib.request.urlopen('https://drive.google.com/uc?export=download&id='+e['id'],timeout=60) as r:
                    b=r.read(3000000)
                    if 'text/html' in r.headers.get('Content-Type',''):raise ValueError('HTML for '+e['id'])
                p.write_bytes(b)
            return {**e,'local':str(p),'bytes':p.stat().st_size}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(fetch,entries.values()))
        (OUT/'log_manifest.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'logs_downloaded':len(results),'bytes':sum(e['bytes'] for e in results)}))
        sys.exit(0)
    if sys.argv[1]=='--download':
        for arg in sys.argv[2:]:
            fid,name=arg.split(':',1)
            if Path(name).name!=name:raise ValueError('basename required')
            url='https://drive.google.com/uc?export=download&id='+fid
            with urllib.request.urlopen(url,timeout=60) as r:
                data=r.read(25000000)
                content_type=r.headers.get('Content-Type')
            if 'text/html' in (content_type or ''):raise ValueError('HTML instead of file: '+fid)
            (OUT/name).write_bytes(data)
            print(json.dumps({'download':name,'id':fid,'bytes':len(data),'content_type':content_type}),flush=True)
        sys.exit(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for fid,result in zip(sys.argv[1:],pool.map(listing,sys.argv[1:])):
            print(json.dumps({'folder':fid,'entries':result}),flush=True)
