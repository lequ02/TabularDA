"""Inspect local report-archive contents and hashes; no datasets or models are loaded."""
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('D:/SummerResearch')
BASE = ROOT / '.cache/remote_intrusion'
archives = []
members = defaultdict(list)
canonical_hashes = {}
for path in sorted(BASE.rglob('*.zip')):
    archive_path = path.relative_to(ROOT).as_posix()
    entries = []
    with zipfile.ZipFile(path) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            digest = hashlib.sha256()
            with source.open(entry) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            row = {'member': entry.filename, 'bytes': entry.file_size,
                   'compressed_bytes': entry.compress_size, 'sha256': digest.hexdigest()}
            members[(entry.filename, row['sha256'])].append(archive_path)
            canonical = ROOT / entry.filename
            # Only inspect report artifacts. Never resolve an archive path outside the workspace.
            if not canonical.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Archive member escapes workspace: ' + entry.filename)
            if canonical.is_file():
                if entry.filename not in canonical_hashes:
                    saved_hash = hashlib.sha256()
                    with canonical.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            saved_hash.update(block)
                    canonical_hashes[entry.filename] = saved_hash.hexdigest()
                row['matches_current_file_bytes'] = row['sha256'] == canonical_hashes[entry.filename]
            else:
                row['matches_current_file_bytes'] = False
            entries.append(row)
    archives.append({'path': archive_path, 'bytes': path.stat().st_size, 'entries': entries})
    print('Verified archive: ' + archive_path, file=sys.stderr, flush=True)

for archive in archives:
    for entry in archive['entries']:
        alternatives = [p for p in members[(entry['member'], entry['sha256'])] if p != archive['path']]
        entry['identical_member_in_other_archives'] = alternatives
    unique = [entry for entry in archive['entries'] if not entry['identical_member_in_other_archives']
              and not entry['matches_current_file_bytes']]
    unique_records = [e for e in unique if e['member'].endswith('.run.json')]
    archive['unique_to_archive_entries'] = len(unique)
    archive['unique_to_archive_record_versions'] = [e['member'] for e in unique_records]
    archive['unique_to_archive_uncompressed_bytes'] = sum(e['bytes'] for e in unique)
    archive['unique_to_archive_compressed_bytes'] = sum(e['compressed_bytes'] for e in unique)

result = {'scope': 'Local report ZIP contents and current artifact byte hashes only',
          'archives': archives,
          'note': 'A redundant member is not a redundant archive. Preserve unique snapshot metadata, historical record versions, original path dependencies, and at least one copy of each member.'}
(ROOT / 'audit/experiment_cleanup_backup_archive_audit_2026_10_10.json').write_text(
    json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps([{'path': a['path'], 'bytes': a['bytes'],
                  'unique_entries': a['unique_to_archive_entries'],
                  'unique_record_versions': len(a['unique_to_archive_record_versions']),
                  'unique_compressed_bytes': a['unique_to_archive_compressed_bytes']}
                 for a in archives], indent=2))
