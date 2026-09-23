import csv,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path('G:/summer_research/download2')
OUT=Path(__file__).resolve().parent
manifest=[]
for archive in sorted(ROOT.rglob('*.zip')):
    try:
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist()
            print(json.dumps({'archive':str(archive),'members':len(infos),'uncompressed_bytes':sum(i.file_size for i in infos)}))
            for info in infos:
                if not info.is_dir():manifest.append({'archive':str(archive),'member':info.filename,'size':info.file_size,'crc':info.CRC})
    except Exception as exc:
        print(json.dumps({'archive':str(archive),'error':str(exc)}))
(OUT/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
for e in manifest:
    if e['member'].endswith(('_train.csv','_test.csv','final_results.csv')):
        print(json.dumps(e))
logs=OUT/'local_logs';logs.mkdir(exist_ok=True)
log_manifest=[]
for e in manifest:
    if not e['member'].endswith('.acc.csv'):continue
    key=hashlib.sha256((e['archive']+'!'+e['member']).encode()).hexdigest()[:20]
    with zipfile.ZipFile(e['archive']) as z: data=z.read(e['member'])
    (logs/(key+'.csv')).write_bytes(data)
    log_manifest.append({**e,'local':str(logs/(key+'.csv')),'sha256':hashlib.sha256(data).hexdigest()})
(OUT/'local_log_manifest.json').write_text(json.dumps(log_manifest,indent=2))
print('LOCAL_LOGS',len(log_manifest))
