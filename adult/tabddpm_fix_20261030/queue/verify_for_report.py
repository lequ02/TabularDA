"""Hash/provenance evidence for the 52 Adult TabDDPM fix records (read-only)."""
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
REPO = Path('/home/thuy/Research/minh_data_synth/TabularDA')
S = REPO / '.cache/tabddpm_covertype_rerun_20261011/repo'
sys.path[:0] = [str(S / 'src'), str(S / 'scripts')]
from run_corrected_matrix import completed_run, methods_for
from modeling import constants
from modeling.run_record import source_sha256
NS = 'tabddpm_adult_500ep_fix_20261030_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = REPO / 'output' / NS
records, generators, files = [], {}, {}
for seed in (42, 43):
    for method in methods_for('adult', 'tabddpm'):
        for mode in ('synthetic', 'mix'):
            assert completed_run(out, 'adult', seed, mode, method, 'corrected')
            name = constants.run_name('adult', seed, mode, method)
            path = out / 'adult' / 'acc' / (name + '.run.json')
            record = json.loads(path.read_text())
            records.append(dict(path=f'output/{NS}/adult/acc/{path.name}', sha256=sha(path), dataset='adult',
                                seed=seed, mode=mode, method=method, source_sha256=record['source_sha256'],
                                selected_dev_epoch=record['selected_dev_epoch']))
            for suffix in ('.run.json', '.predictions.csv', '.epochs.csv', '.report.txt'):
                f = path.with_name(name + suffix)
                files[str(f.relative_to(REPO))] = sha(f)
            gp = record['generator_provenance_path']
            if gp not in generators:
                generators[gp] = dict(provenance=json.loads(Path(gp).read_text()), sha256=sha(gp),
                                      fit='full' if '_full.' in gp else 'xonly')
                files[str(Path(gp).relative_to(REPO))] = sha(gp)
            assert record['generator_provenance'] == generators[gp]['provenance']
files[f'output/{NS}/launch.json'] = sha(out / 'launch.json')
for gp, g in generators.items():
    p = g['provenance']['parameters']
    assert p['epochs'] == 500 and p['steps'] == 34500 and p['is_y_cond'] is False and g['provenance']['sample_size'] == 100000
evidence = dict(checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(), namespace=NS,
                source_sha256=source_sha256(), verified_completed=len(records) + 2, planned_total=54,
                new_records=records, generators=generators, files=files,
                reused_baselines=[f'output/tabddpm_500ep_20261010_v1/adult/acc/adult_seed{s}_real_original.run.json' for s in (42, 43)])
(REPO / '.cache/tabddpm_adult_rerun_20261030/report_evidence_20261031.json').write_text(json.dumps(evidence, indent=1))
print(len(records), len(generators), len(files), evidence['source_sha256'])
