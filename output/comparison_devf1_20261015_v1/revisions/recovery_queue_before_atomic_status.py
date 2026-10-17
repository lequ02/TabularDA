"""Persistent sequential recovery queue; surface failures and preserve completed jobs."""
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
inventory = json.loads((BASE / 'inventory.json').read_text())
order = json.loads((BASE / ('pilot_order.json' if '--pilot' in sys.argv else 'queue_order.json')).read_text())
python = '/home/thuy/miniconda3/envs/env/bin/python'
def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()
for index in order:
    item = inventory[index]
    job_id = item['record_relative_path'].replace('/', '__').removesuffix('.run.json')
    folder = BASE / 'jobs' / job_id
    result = folder / 'result.json'
    if result.exists():
        completed = json.loads(result.read_text())
        if completed['record_relative_path'] != item['record_relative_path']:
            raise ValueError('Completed job provenance differs from queue item')
        continue
    if folder.exists():
        raise RuntimeError(f'Incomplete job requires diagnosis before resume: {folder}')
    (BASE / 'logs').mkdir(exist_ok=True)
    log = BASE / 'logs' / (job_id + '.log')
    command = ['nice', '-n', '19', 'taskset', '-c', '0', python, '-u', str(BASE / 'recover_one.py'), str(index)]
    print(f'{now()} START {index} {item["dataset"]} seed={item["seed"]} {item["mode"]} target_epoch={item["target_epoch"]} log={log}', flush=True)
    (BASE / 'active.json').write_text(json.dumps({'index': index, 'item': item, 'log': str(log), 'command': command, 'started_at_chicago': now()}, indent=2))
    with log.open('w') as stream:
        completed = subprocess.run(command, cwd=BASE, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        failure = {'index': index, 'record': item['record_relative_path'], 'returncode': completed.returncode,
                   'log': str(log), 'failed_at_chicago': now()}
        (BASE / 'failure.json').write_text(json.dumps(failure, indent=2))
        raise RuntimeError(f'Recovery failed: {failure}')
    print(f'{now()} DONE {index} {job_id}', flush=True)
    done = [p for p in (BASE / 'jobs').glob('*/result.json')]
    (BASE / 'progress.json').write_text(json.dumps({'completed': len(done), 'planned': len(inventory), 'updated_at_chicago': now()}, indent=2))
    subprocess.run([python, str(BASE / 'build_reports.py'), '--tables-only'], check=True)
if '--pilot' not in sys.argv:
    subprocess.run([python, str(BASE / 'build_reports.py')], check=True)
    bundle = BASE / 'completed_reports.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (BASE / 'reports').iterdir():
            archive.write(path, path.relative_to(BASE))
        for path in (BASE / 'jobs').glob('*/result.json'):
            archive.write(path, path.relative_to(BASE))
        for name in ('inventory_summary.json', 'inventory.json', 'launch.json',
                     'recover_one.py', 'recovery_queue.py', 'build_reports.py', 'launch.sh'):
            archive.write(BASE / name, name)
    (BASE / 'delivery.json').write_text(json.dumps({'bundle': bundle.name, 'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}, indent=2))
    (BASE / 'completed.json').write_text(json.dumps({'completed_at_chicago': now(), 'jobs': len(inventory)}, indent=2))
print(f'{now()} QUEUE COMPLETE', flush=True)
