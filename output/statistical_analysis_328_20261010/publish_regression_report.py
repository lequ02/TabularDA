"""Publish only this report's files, preserving concurrent artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('D:/SummerResearch')
HERE=Path(__file__).parent
STAGE=ROOT/'tmp/pdfs/statistical_328_explicit/deliverables'
OUT=ROOT/'output/statistical_analysis_328_20261010'
PDF=ROOT/'output/pdf/statistical_analysis_328_20261010'
BACKUP=ROOT/'.cache/statistical_328_before_regression_anova_20261010_145303'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.regression.tmp')
    tmp.write_bytes(payload)
    tmp.replace(path)
def staged(path):
    return STAGE/'report'/path.relative_to(OUT) if path.is_relative_to(OUT) else STAGE/'pdf'/path.name
def saved(path):
    return BACKUP/'report'/path.relative_to(OUT) if path.is_relative_to(OUT) else BACKUP/'pdf'/path.name

preserved=OUT/'real_data_328_csv_replication.zip'
preserved_sha=sha(preserved)
assert preserved_sha==sha(BACKUP/'report'/preserved.name)
validation=json.loads((HERE/'validation.json').read_text())
assert validation['visual_review']=='passed'
for name in ['validation.json','verify_explicit.py','publish_regression_report.py']:
    atomic(STAGE/'report'/name,(HERE/name).read_bytes())
manifest=json.loads((HERE/'deliverable_manifest.json').read_text())
for name in ['validation.json','verify_explicit.py','publish_regression_report.py']:
    manifest['files'][str((OUT/name).relative_to(ROOT))]=sha(STAGE/'report'/name)
atomic(HERE/'deliverable_manifest.json',json.dumps(manifest,indent=2).encode())
package=STAGE/'report/real_data_328_analysis_package.zip'
temporary=package.with_suffix('.regression.tmp.zip')
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
    for relative,digest in manifest['files'].items():
        path=ROOT/relative;source=staged(path)
        assert sha(source)==digest
        member=str(path.relative_to(OUT)).replace('\\','/') if path.is_relative_to(OUT) else path.name
        archive.write(source,member)
    archive.write(HERE/'deliverable_manifest.json','deliverable_manifest.json')
temporary.replace(package)
# Inspect every destination before mutating any existing file.
for relative in manifest['files']:
    path=ROOT/relative
    if path.exists():assert saved(path).exists() and sha(path)==sha(saved(path)),f'Concurrent change: {path}'
assert sha(OUT/package.name)==sha(BACKUP/'report'/package.name)
changed=[]
for relative,digest in manifest['files'].items():
    path=ROOT/relative
    if not path.exists() or sha(path)!=digest:
        atomic(path,staged(path).read_bytes());changed.append(relative)
    assert sha(path)==digest
atomic(OUT/package.name,package.read_bytes())
assert sha(preserved)==preserved_sha
with zipfile.ZipFile(OUT/package.name) as archive:
    assert len(archive.namelist())==len(set(archive.namelist())) and archive.testzip() is None
publication=dict(changed_report_files=changed,package_sha256=sha(OUT/package.name),
    preserved_csv_replication_sha256=preserved_sha,backup=str(BACKUP),
    source_or_experiment_changes=False)
atomic(HERE/'regression_publication.json',json.dumps(publication,indent=2).encode())
print(json.dumps(dict(changed_report_files=len(changed),package_sha256=publication['package_sha256'],
                     preserved_csv_replication=True),indent=2))
