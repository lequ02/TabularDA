"""Deploy only this analysis into a new remote cache; retrieve its results."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
SSH=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
 '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
files=[HERE/'analyze_separate.py',ROOT/'audit/statistical_review_20261006/rebuild_rf_xgb_dnn.py',
 ROOT/'audit/statistical_review_20261006/analyze_results.py',HERE/'results/regression_d2_scores.csv']
sources={p.name:p.read_text() for p in files}
source="import io,zipfile,base64,runpy,sys,hashlib\nfrom pathlib import Path\np=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_separate_20261010')\np.mkdir(exist_ok=True)\n"
source+='sources='+repr(sources)+'\n'
source+="for name,text in sources.items():\n target=p/name\n if target.exists():\n  assert target.read_text()==text, 'Existing remote analysis differs: '+name\n else:\n  target.write_text(text)\n assert target.read_text()==text\n"
source+="sys.path.insert(0,str(p))\nrunpy.run_path(str(p/'analyze_separate.py'),run_name='__main__')\nb=io.BytesIO()\nwith zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:\n for f in (p/'results').rglob('*'):\n  if f.is_file():z.write(f,str(f.relative_to(p/'results')))\nprint('RESULT_ARCHIVE:'+base64.b64encode(b.getvalue()).decode())\n"
result=subprocess.run(SSH+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg /home/thuy/miniconda3/envs/env/bin/python -'],input=source,text=True,capture_output=True)
if result.stderr:print(result.stderr)
result.check_returncode()
log,payload=result.stdout.split('RESULT_ARCHIVE:')
(HERE/'remote_separate_analysis.log').write_text(log+'\n'+result.stderr)
archive=HERE/'remote_separate_results.zip';archive.write_bytes(base64.b64decode(payload.strip(),validate=True))
dest=HERE/'separate_results';dest.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  target=dest/name
  assert target.resolve().is_relative_to(dest.resolve())
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
print(log)
print(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),results=str(dest)),indent=2))
