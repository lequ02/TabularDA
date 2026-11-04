"""Lightweight delivery verification; no experiment or statistical model runs."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
checks = json.loads((OUT / 'validation.json').read_text())
for name, digest in checks['output_hashes'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
for name in ['report.md', 'report.html']:
    text = (OUT / name).read_text(encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', text) if name.endswith('.md') else re.findall(r'(?:href|src)="([^"]+)"', text)
    for link in links:
        if not link.startswith('https:'):
            assert (OUT / link).is_file(), (name, link)
expected_rows = dict(primary_anova=4, generator_contrasts=12, two_factor_repeated_anova=12,
                     sensitivity_anova=32, dataset_effects=20, dataset_method_means=90,
                     independent_F_checks=12, regression_exploratory_anova=4,
                     regression_descriptive=36, labeler_contrasts=12)
for name, count in expected_rows.items():
    with (OUT / (name + '.csv')).open(newline='', encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == count, name
assert hashlib.sha256((ROOT / 'analyze.py').read_bytes()).hexdigest() == '717f4965a065bc279b1985a66a185150b4c48470a1d208900f9bb82a69560655'
print('PASS: all delivered hashes, report links, statistical table sizes, and remote script hash.')
