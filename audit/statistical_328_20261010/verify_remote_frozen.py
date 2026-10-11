"""Confirm selected records still match the frozen input; read-only remote check."""
import json
from pathlib import Path
import subprocess
ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
snapshot=json.loads((HERE/'snapshot/snapshot.json').read_text())
d2=json.loads((HERE/'results/d2_verification.json').read_text())
expected={r['path']:r['sha256'] for r in snapshot['records']}
expected.update({r['sidecar']:r['sidecar_sha256'] for r in d2['sidecars']})
source="import hashlib,json\nfrom pathlib import Path\nfrom datetime import datetime\nfrom zoneinfo import ZoneInfo\nroot=Path('/home/thuy/Research/minh_data_synth/TabularDA')\nexpected="+repr(expected)+"\nmismatches=[name for name,sha in expected.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha]\nprint(json.dumps(dict(checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),checked_files=len(expected),mismatches=mismatches)))\nassert not mismatches, mismatches\n"
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes','-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
r=subprocess.run(ssh+['/home/thuy/miniconda3/envs/env/bin/python -'],input=source,text=True,capture_output=True)
if r.stderr:print(r.stderr)
r.check_returncode()
result=json.loads(r.stdout);(HERE/'latest_source_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
