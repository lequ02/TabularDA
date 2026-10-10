"""Verify unchanged scores, requested table notation and final report layout."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile

import pymupdf

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
OUT=ROOT/'output/statistical_analysis_328_20261010'
PDF=ROOT/'output/pdf/statistical_analysis_328_20261010/real_data_328_analysis.pdf'
manifest=json.loads((HERE/'deliverable_manifest.json').read_text())
metadata=json.loads((OUT/'analysis_metadata.json').read_text())
for name,sha in metadata['results_sha256'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha
for name,sha in manifest['files'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256(Path(manifest['template_path']).read_bytes()).hexdigest()==manifest['template_sha256']

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.rows=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.rows=[]
        elif tag=='tr':self.row=[]
        elif tag in ('td','th'):self.cell={'text':'','bold':False}
        elif tag=='b' and self.cell is not None:self.cell['bold']=True
    def handle_data(self,data):
        if self.cell is not None:self.cell['text']+=data
    def handle_endtag(self,tag):
        if tag in ('td','th'):self.row.append(self.cell);self.cell=None
        elif tag=='tr':self.rows.append(self.row);self.row=None
        elif tag=='table':self.tables.append(self.rows);self.rows=None

parser=Tables();parser.feed((OUT/'real_data_328_analysis.html').read_text(encoding='utf-8'))
checked=0
for table in parser.tables:
    headers=[c['text'] for c in table[0]]
    if 'Sig.' not in headers:continue
    pcolumn='Nominal p' if 'Nominal p' in headers else 'Holm p' if 'Holm p' in headers else 'Adjusted p'
    pi=headers.index(pcolumn);si=headers.index('Sig.')
    for row in table[1:]:
        if not row[pi]['text']:continue
        p=float(row[pi]['text'])
        expected='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '.' if p<.1 else ''
        assert row[si]['text']==expected,(row,pcolumn)
        assert all(c['bold']==(p<.05) for c in row),(row,pcolumn)
        checked+=1
doc=pymupdf.open(PDF)
assert len(doc)==len(manifest['sections'])==9
for page,title in zip(doc,manifest['sections']):
    text=page.get_text()
    assert title in text
    assert all(label in text[:1200] for label in ('Research question:','Method:','Finding:'))
    assert len(text)>300
    for block in page.get_text('blocks'):
        if block[6]==0:assert 46<=block[0]<=block[2]<=566 and 20<=block[1]<=block[3]<=770,(page.number,block[:4])
with zipfile.ZipFile(OUT/'real_data_328_analysis_package.zip') as archive:
    assert hashlib.sha256(archive.read(PDF.name)).hexdigest()==manifest['pdf_sha256']
    for name,sha in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
validation=json.loads((HERE/'validation.json').read_text())
validation.update(final_pdf_pages=9,sections_begin_question_method_finding=True,
                  statistical_values_unchanged=True,checked_significance_rows=checked,
                  bold_and_stars='passed',layout_bounds='passed',deliverable_hashes='passed',
                  visual_review='all nine revised pages inspected')
(HERE/'validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps(validation,indent=2))
