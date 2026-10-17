import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

base = Path(__file__).resolve().parent
status = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat()}
for name in ('progress', 'active', 'failure', 'completed', 'delivery'):
    path = base / (name + '.json')
    status[name] = json.loads(path.read_text()) if path.exists() else None
print(json.dumps(status))
