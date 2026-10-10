"""Send isolated analysis work through the authorized dedicated SSH identity."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path('D:/SummerResearch')
HERE=Path(__file__).parent
SSH=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
     '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
if sys.argv[1]=='collect':
    source=(HERE/'collect_remote.py').read_text()
    result=subprocess.run(SSH+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/thuy/miniconda3/envs/env/bin/python -'],input=source,text=True,capture_output=True)
    if result.stderr:print(result.stderr)
    result.check_returncode()
    archive=HERE/'remote_snapshot.zip'
    archive.write_bytes(base64.b64decode(result.stdout.strip(),validate=True))
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target=HERE/'snapshot'/name
            assert target.resolve().is_relative_to((HERE/'snapshot').resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    snapshot=json.loads((HERE/'snapshot/snapshot.json').read_text())
    for r in snapshot['records']:
        assert hashlib.sha256((HERE/'snapshot/records'/r['path']).read_bytes()).hexdigest()==r['sha256']
    print(json.dumps({k:snapshot[k] for k in ('checked_at_chicago','selected_completed','selected_expected','missing','shared_seed_test_partitions')},indent=2))
else:
    sources={p.name:p.read_text() for p in [HERE/'analyze.py',ROOT/'audit/statistical_review_20261006/rebuild_rf_xgb_dnn.py',ROOT/'audit/statistical_review_20261006/analyze_results.py']}
    source="import json,io,zipfile,base64,runpy\nfrom pathlib import Path\np=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_20261010')\n"
    source+='sources='+repr(sources)+'\nfor name,text in sources.items():\n (p/name).write_text(text)\n'
    source+="import sys\nsys.path.insert(0,str(p))\nrunpy.run_path(str(p/'analyze.py'),run_name='__main__')\nb=io.BytesIO()\nwith zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:\n for f in (p/'results').glob('*'):\n  z.write(f,f.name)\nprint('RESULT_ARCHIVE:'+base64.b64encode(b.getvalue()).decode())\n"
    result=subprocess.run(SSH+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/thuy/miniconda3/envs/env/bin/python -'],input=source,text=True,capture_output=True)
    if result.returncode:print(result.stderr)
    result.check_returncode()
    log,payload=result.stdout.split('RESULT_ARCHIVE:')
    (HERE/'remote_analysis.log').write_text(log+'\n'+result.stderr)
    print(log);print(result.stderr)
    archive=HERE/'remote_results.zip';archive.write_bytes(base64.b64decode(payload.strip(),validate=True))
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target=HERE/'results'/name
            assert target.resolve().is_relative_to((HERE/'results').resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
