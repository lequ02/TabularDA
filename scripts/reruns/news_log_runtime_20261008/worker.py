import json, os, subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
config = json.loads(Path('/home/thuy/Research/minh_data_synth/TabularDA/output/news_log_v1/launch.json').read_text())
status = Path(config['status_path'])
def stamp(): return datetime.now(ZoneInfo('America/Chicago')).isoformat()
state = {'state':'running','started_at_chicago':stamp(),'supervisor_pid':os.getpid()}
status.write_text(json.dumps(state,indent=2)+chr(10))
os.environ.update(config['environment'])
with Path(config['log_path']).open('a') as log:
    log.write('News log-target matrix started '+stamp()+chr(10)); log.flush()
    result = subprocess.run(config['command'],cwd=config['working_directory'],stdout=log,stderr=subprocess.STDOUT)
state.update(state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_at_chicago=stamp())
status.write_text(json.dumps(state,indent=2)+chr(10))
raise SystemExit(result.returncode)
