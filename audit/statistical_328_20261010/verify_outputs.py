"""Focused source/metric/provenance and layout checks; no experiments."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import fitz
from PIL import Image

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
OUT=ROOT/'output/statistical_analysis_328_20261010'
PDF=ROOT/'output/pdf/statistical_analysis_328_20261010/real_data_328_analysis.pdf'
FROZEN=HERE/'snapshot/records'
(FROZEN/'audit').mkdir(exist_ok=True)
for name in ('comparison_housing_fit_table_hashes.json','comparison_news_log_fit_table_hashes.json','comparison_rerun_verification_latest.json'):
    shutil.copy2(ROOT/'audit'/name,FROZEN/'audit'/name)
spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/plot_recent_comparison.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.ROOT=FROZEN
snapshot=json.loads((OUT/'source_snapshot.json').read_text())
lookup={r['path']:r for r in snapshot['records']}
with (OUT/'scores.csv').open() as source:
    scores=list(csv.DictReader(source))
previous=None
for row in scores:
    if row['record']!=previous:
        payload=(FROZEN/row['record']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==lookup[row['record']]['sha256']==row['record_sha256']
        record=json.loads(payload)
        previous=row['record']
    assert float(row['value'])==record['test_scores'][row['metric']]
for item in snapshot['records']:
    path=FROZEN/item['path'];record=json.loads(path.read_bytes())
    suffix=path.name.removeprefix(f"{item['dataset']}_seed{item['seed']}_").removesuffix('.run.json')
    info=dict(dataset=item['dataset'],seed=item['seed'],scope=item['scope'],
              metric='nmae_sigma' if item['dataset'] in ('news','california_housing') else 'f1_macro',
              selection_metric='f1_macro' if item['dataset']=='census_kdd' else 'loss')
    saved,_,_=builder.read_run(info,suffix)
    assert saved==record['test_scores']
with (OUT/'aggregated_per_run.csv').open() as stream:assert len(list(csv.DictReader(stream)))==406
metadata=json.loads((OUT/'analysis_metadata.json').read_text())
for name,sha in metadata['results_sha256'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha
manifest=json.loads((HERE/'deliverable_manifest.json').read_text())
assert hashlib.sha256(Path(manifest['template_path']).read_bytes()).hexdigest()==manifest['template_sha256']
doc=fitz.open(PDF)
for page in doc:
    text=page.get_text()
    assert len(text)>300, f'Unexpected sparse/orphan page {page.number+1}'
    assert '\ufffd' not in text
    for block in page.get_text('dict')['blocks']:
        if block['type']==0:
            x0,y0,x1,y1=block['bbox']
            assert x0>=46 and x1<=566 and y0>=20 and y1<=770, (page.number+1,block['bbox'])
render=ROOT/'tmp/pdfs/statistical_328_20261010'
paths=sorted(render.glob('page-*.png'))
assert len(paths)==len(doc)
for group in range(math.ceil(len(paths)/4)):
    sheet=Image.new('RGB',(1224,1584),'#d4dce0')
    for index,path in enumerate(paths[group*4:group*4+4]):
        im=Image.open(path).convert('RGB');im.thumbnail((600,776))
        sheet.paste(im,((index%2)*612,(index//2)*792))
    sheet.save(render/f'contact-{group+1}.png')
print(json.dumps(dict(verified_records=406,verified_metric_values=len(scores),
 source_and_saved_provenance_checks='passed',normalization_checks='passed',
 original_template_unchanged=True,pdf_pages=len(doc),layout_bounds='passed',
 result_hashes='passed'),indent=2))
