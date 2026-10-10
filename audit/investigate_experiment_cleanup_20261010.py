"""Read-only cleanup dependency/hash audit. Execute through SSH on the research host.

Reads metadata, JSON evidence and source text; never loads models or changes experiments.
The only output is a JSON investigation on stdout, with progress on stderr.
"""
import hashlib
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
AREAS = ('data', 'sdv trained model', 'output', '.cache', 'audit', 'scripts', 'src', 'simulated_eval')
PATH_PREFIXES = tuple(area + '/' for area in AREAS)
CURRENT = {'corrected_v2', 'census_kdd_weighted_macro_f1_20261005',
           'mnist_head_fixed_20261009', 'housing_no_faker_20261009', 'news_log_v1'}
inventory = {}
links = []
for area in AREAS:
    for base, dirs, filenames in os.walk(ROOT / area, followlinks=False):
        for name in list(dirs):
            path = Path(base) / name
            if path.is_symlink():
                links.append({'path': path.relative_to(ROOT).as_posix(), 'target': str(path.resolve())})
                dirs.remove(name)
            elif name in {'.git', 'node_modules', 'env', 'venv'}:
                dirs.remove(name)
        for name in filenames:
            if name in {'server_key', 'ssh_known_hosts'} or name.endswith(('.pem', '.key')):
                continue
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            stat = path.stat()
            inventory[rel] = {'bytes': stat.st_size, 'allocated_bytes': stat.st_blocks * 512,
                              'inode': stat.st_ino, 'device': stat.st_dev,
                              'mtime_ns': stat.st_mtime_ns, 'resolved': str(path.resolve())}

by_resolved = defaultdict(list)
for rel, row in inventory.items():
    by_resolved[row['resolved']].append(rel)
refs = defaultdict(set)
directory_refs = defaultdict(set)
missing_refs = defaultdict(set)
records = []
json_errors = []
namespace_mentions = defaultdict(list)
pilot_names = sorted({rel.split('/')[1] for rel in inventory
                      if rel.startswith(('data/', 'output/', 'sdv trained model/'))
                      and ('pilot' in rel.split('/')[1] or 'smoke' in rel.split('/')[1])})


def normalize(value):
    if not isinstance(value, str) or '\n' in value or len(value) > 1000:
        return None
    value = value.replace('\\', '/')
    if value.startswith(str(ROOT) + '/'):
        return Path(value)
    if value.startswith('D:/SummerResearch/'):
        return ROOT / value[len('D:/SummerResearch/'):]
    if value.startswith(PATH_PREFIXES):
        return ROOT / value
    return None


def add_ref(value, source):
    path = normalize(value)
    if path is None:
        return
    resolved = str(path.resolve())
    if resolved in by_resolved:
        refs[resolved].add(source)
    elif path.is_dir():
        directory_refs[str(path)].add(source)
    else:
        missing_refs[str(path)].add(source)


def walk_json(obj, source):
    if isinstance(obj, dict):
        for key, value in obj.items():
            add_ref(key, source)
            walk_json(value, source)
    elif isinstance(obj, list):
        for value in obj:
            walk_json(value, source)
    elif isinstance(obj, str):
        add_ref(obj, source)


json_paths = [rel for rel in inventory if rel.endswith('.json')]
for index, rel in enumerate(json_paths):
    path = ROOT / rel
    try:
        with path.open(encoding='utf-8-sig') as stream:
            saved = json.load(stream)
    except json.JSONDecodeError as error:
        json_errors.append({'path': rel, 'error': str(error)})
        continue
    kind = ('run' if rel.endswith('.run.json') else
            'provenance' if rel.endswith('.provenance.json') else
            'sidecar' if rel.endswith(('.quality.json', '.d2.json', '.dnn.json')) else 'evidence')
    walk_json(saved, kind + ':' + rel)
    if kind == 'run':
        direct = {key: value for key, value in saved.items()
                  if key.endswith('_path') and isinstance(value, str)}
        protocol = saved.get('evaluation_protocol', {})
        origin = protocol.get('source_record')
        records.append({'path': rel, 'dataset': saved.get('dataset'), 'seed': saved.get('seed'),
                        'selected_dev_epoch': saved.get('selected_dev_epoch'),
                        'direct_paths': direct, 'import_source_record': origin,
                        'epoch_budget': saved.get('epoch_budget'),
                        'generator_parameters': saved.get('generator_provenance', {}).get('parameters')
                        if saved.get('generator_provenance') else None})
    if index % 150 == 0:
        print(f'Inspected {index + 1}/{len(json_paths)} JSON files', file=sys.stderr, flush=True)

# Source/document matches identify dynamic loaders and historical evidence beyond saved paths.
# Store compact lines only; content from credentials and arbitrary log files is never read.
source_extensions = {'.py', '.md', '.R', '.Rmd', '.tex', '.sh'}
for rel in inventory:
    path = ROOT / rel
    if path.suffix not in source_extensions or inventory[rel]['bytes'] > 2_000_000:
        continue
    text = path.read_text(encoding='utf-8-sig')
    for lineno, line in enumerate(text.splitlines(), 1):
        for name in pilot_names:
            if name in line:
                namespace_mentions[name].append({'source': rel, 'line': lineno, 'text': line[:260]})
        # Explicit absolute or root-relative file paths written in code/document strings.
        for match in re.finditer(r'[\"\x27`]([^\"\x27`\n]+)[\"\x27`]', line):
            add_ref(match.group(1), 'source:' + rel + ':' + str(lineno))

for rel, meta in inventory.items():
    if not rel.endswith('.csv') or not rel.startswith(('output/', 'audit/', '.cache/')) or meta['bytes'] > 4_000_000:
        continue
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream):
            for cell in row:
                add_ref(cell, 'table:' + rel)

def consumers(rel):
    return sorted(refs[inventory[rel]['resolved']])


def canonical_record(source):
    return source.startswith('run:output/') or source.startswith('run:.cache/')


cross_refs = []
for row in records:
    source_parts = row['path'].split('/')
    source_ns = source_parts[1] if source_parts[0] == 'output' else source_parts[1]
    for field, value in row['direct_paths'].items():
        path = normalize(value)
        if path is None:
            continue
        original = path.relative_to(ROOT).as_posix()
        target_parts = original.split('/')
        target_ns = target_parts[1] if len(target_parts) > 1 else None
        if target_ns != source_ns:
            cross_refs.append({'record': row['path'], 'field': field, 'target': original,
                               'resolved_target': str(path.resolve())})
    if row['import_source_record']:
        cross_refs.append({'record': row['path'], 'field': 'evaluation_protocol.source_record',
                           'target': row['import_source_record']})

# Hash candidate duplicates, never deserialize checkpoints or refit models.
# Restrict the hash pass to identical-size groups containing a pilot/smoke/cache asset.
asset_extensions = ('.pkl', '.pt', '.pth', '.csv', '.zip', '.7z')
size_groups = defaultdict(list)
for rel, row in inventory.items():
    if rel.endswith(asset_extensions) and row['bytes'] > 0:
        size_groups[row['bytes']].append(rel)
hash_groups = []
hash_bytes = 0
for size, group in sorted(size_groups.items(), reverse=True):
    if len(group) < 2 or not any(('pilot' in p or 'smoke' in p or p.startswith('.cache/')) for p in group):
        continue
    digests = defaultdict(list)
    digest_by_inode = {}
    for rel in group:
        meta = inventory[rel]
        inode_key = (meta['device'], meta['inode'])
        if inode_key not in digest_by_inode:
            digest = hashlib.sha256()
            with (ROOT / rel).open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            digest_by_inode[inode_key] = digest.hexdigest()
            hash_bytes += size
        digests[digest_by_inode[inode_key]].append(rel)
        after = (ROOT / rel).stat()
        if (after.st_size, after.st_mtime_ns) != (meta['bytes'], meta['mtime_ns']):
            raise RuntimeError('Candidate changed during inspection: ' + rel)
    for digest, members in digests.items():
        if len(members) > 1:
            hash_groups.append({'bytes_per_file': size, 'sha256': digest,
                                'members': [{'path': p, 'consumers': consumers(p),
                                             'inode': inventory[p]['inode'],
                                             'allocated_bytes': inventory[p]['allocated_bytes']} for p in members]})
    print(f'Hashed candidate group: {len(group)} files of {size} bytes', file=sys.stderr, flush=True)

namespaces = []
for name in pilot_names:
    members = [rel for rel in inventory if any(rel.startswith(area + '/' + name + '/')
               for area in ('data', 'output', 'sdv trained model'))]
    canonical_runs = [r for r in records if r['path'].startswith('output/' + name + '/')]
    namespaces.append({'namespace': name, 'files': len(members),
                       'logical_bytes': sum(inventory[p]['bytes'] for p in members),
                       'run_records': len(canonical_runs),
                       'externally_referenced_files': [p for p in members if any(
                           not ref.split(':', 1)[-1].startswith(tuple(area + '/' + name + '/'
                               for area in ('data', 'output', 'sdv trained model')))
                           for ref in consumers(p))],
                       'source_mentions': namespace_mentions[name]})

result = {'checked_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
          'root': str(ROOT), 'scope': list(AREAS), 'files': len(inventory),
          'json_files_inspected': len(json_paths), 'json_errors': json_errors,
          'runtime_links': links, 'pilot_namespaces': namespaces,
          'records': records, 'cross_namespace_record_references': cross_refs,
          'duplicate_groups': hash_groups, 'hashed_bytes': hash_bytes,
          'directory_references': {key: sorted(value) for key, value in directory_refs.items()},
          'missing_references': {key: sorted(value) for key, value in missing_refs.items()},
          'inventory': {rel: {**row, 'consumers': consumers(rel)} for rel, row in inventory.items()},
          'processes': subprocess.check_output(['ps', '-u', 'thuy', '-o', 'pid,ppid,etimes,pcpu,args'], text=True),
          'tmux': subprocess.check_output(['tmux', 'list-panes', '-a', '-F',
                                          '#S:#I.#P pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'], text=True)}
print(json.dumps(result, separators=(',', ':')))
