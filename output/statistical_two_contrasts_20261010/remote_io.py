"""Transfer isolated analysis inputs and retrieve results; no experiment edits."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path('D:/SummerResearch')
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / 'audit/statistical_328_20261010'
REMOTE = '/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_two_contrasts_20261010'
inputs = {'analyze.py': HERE / 'analyze.py', 'scores.csv': PRIOR / 'snapshot/scores.csv',
          'source_snapshot.json': PRIOR / 'snapshot/snapshot.json',
          'regression_d2_scores.csv': PRIOR / 'results/regression_d2_scores.csv',
          'd2_verification.json': PRIOR / 'results/d2_verification.json'}
payload = io.BytesIO()
with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, path in inputs.items():
        archive.writestr(name, path.read_bytes())
digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
source = "import base64,hashlib,io,json,runpy,zipfile\nfrom pathlib import Path\n"
source += 'dest=Path(' + repr(REMOTE) + ')\ndest.mkdir(exist_ok=True)\n'
source += 'digests=' + repr(digests) + '\n'
source += 'payload=base64.b64decode(' + repr(base64.b64encode(payload.getvalue()).decode()) + ')\n'
source += "with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n for name in archive.namelist():\n  target=dest/name\n  data=archive.read(name)\n  assert hashlib.sha256(data).hexdigest()==digests[name]\n  if target.exists():\n   assert target.read_bytes()==data, 'Remote file differs: '+name\n  else:\n   target.write_bytes(data)\n  assert hashlib.sha256(target.read_bytes()).hexdigest()==digests[name]\n"
source += "runpy.run_path(str(dest/'analyze.py'),run_name='__main__')\nresult=io.BytesIO()\nwith zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as archive:\n for path in (dest/'results').rglob('*'):\n  if path.is_file(): archive.write(path,str(path.relative_to(dest/'results')))\nprint('RESULT_ARCHIVE:'+base64.b64encode(result.getvalue()).decode())\n"
ssh = ['ssh', '-i', str(ROOT / '.cache/remote_intrusion/server_key'), '-o', 'IdentitiesOnly=yes',
       '-o', 'UserKnownHostsFile=' + str(ROOT / 'audit/ssh_known_hosts'), '-o', 'BatchMode=yes',
       'thuy@10.24.10.133']
command = 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'
response = subprocess.run(ssh + [command], input=source, text=True, capture_output=True)
(HERE / 'remote_analysis.log').write_text(response.stdout.split('RESULT_ARCHIVE:')[0] + response.stderr)
if response.stderr:
    print(response.stderr)
response.check_returncode()
log, encoded = response.stdout.split('RESULT_ARCHIVE:')
result_archive = HERE / 'remote_results.zip'
result_archive.write_bytes(base64.b64decode(encoded.strip(), validate=True))
dest = HERE / 'results'
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(result_archive) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        target = dest / name
        assert target.resolve().is_relative_to(dest.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
(HERE / 'deployment.json').write_text(json.dumps(dict(remote_namespace=REMOTE, input_sha256=digests,
    result_archive_sha256=hashlib.sha256(result_archive.read_bytes()).hexdigest(),
    command=command, log=str(HERE/'remote_analysis.log'), experiment_changes=False), indent=2))
print(log)
