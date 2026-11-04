"""Rebuild best-development-F1 reports with the verified Adult posterior-fix replays.

Uses the existing checkpoint report builder; preserves historical reports and scores.
Run with .venv/Scripts/python.exe comparisons/20261031_adult_tabddpm_fix/rebuild_valbest.py.
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'adult/tabddpm_fix_20261030'
EVIDENCE = ARCHIVE / 'valbest'
RECOVERY = ROOT / 'scripts/checkpoint_recovery'
OUT = HERE / 'valbest'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def root_relative(path):
    return path.relative_to(ROOT).as_posix()


if OUT.exists():
    raise FileExistsError('Choose a new output directory; existing reports are preserved')
verification = json.loads((EVIDENCE / 'verification.json').read_text())
for name, digest in verification['files'].items():
    assert sha(EVIDENCE / name) == digest, name
assert verification['completed'] == 28 and verification['worker_exits'] == [0] * 4
new = {}
for path in (EVIDENCE / 'jobs').glob('*/result.json'):
    result = json.loads(path.read_text())
    assert result['original_checkpoint_tensors_exact'] and result['test_ids_and_targets_match']
    old_relative = result['record_relative_path'].replace('tabddpm_adult_500ep_fix_20261030_v1',
                                                        'tabddpm_500ep_20261010_v1')
    assert old_relative not in new
    new[old_relative] = (path, result)
assert len(new) == 28

historical = read(ROOT / 'comparisons/scores_valbest.csv')
assert len(historical) == 602
scores, selected, changes = [], [], []
fixed_scores = {Path(r['record_relative_path']).name: r for r in read(ARCHIVE / 'scores.csv')}
for original in historical:
    row = original.copy()
    if original['record_relative_path'] in new:
        path, result = new[original['record_relative_path']]
        fixed = fixed_scores[Path(result['record_relative_path']).name]
        for key in ('record_relative_path', 'source_sha256', 'retained_status', 'caveats_json'):
            row[key] = fixed[key]
        row.update(selected_dev_epoch=result['target_epoch'], selection_metric=result['selection_metric'],
                   original_selected_dev_epoch=result['original_epoch'], original_selection_metric='loss',
                   same_checkpoint=result['same_checkpoint'], test_loss=result['test_loss'],
                   result_path=root_relative(path),
                   original_record_path=root_relative(ARCHIVE / 'records' / Path(result['record_relative_path']).name),
                   original_history_path=root_relative(ARCHIVE / 'records' / Path(result['record_relative_path']).name.replace('.run.json', '.epochs.csv')),
                   replay_history_path=root_relative(path.parent / 'replay_history.csv') if result['replayed_epochs'] else '')
        for field in row:
            if field.startswith('test_') and field != 'test_loss':
                row[field] = result['test_scores'].get(field.removeprefix('test_'), '')
        for remote, digest in result['artifact_sha256'].items():
            stored = EVIDENCE / Path(remote).relative_to(verification['remote_base'])
            assert sha(stored) == digest
            row['weights_path' if remote.endswith('.pth') else 'predictions_path'] = root_relative(stored)
        selected.append(row)
        change = {k: row[k] for k in ('seed', 'construction', 'labeler', 'mode')}
        change.update(before_epoch=original['selected_dev_epoch'], after_epoch=row['selected_dev_epoch'],
                      fixed_loss_epoch=result['original_epoch'], before_result=original['result_path'],
                      after_result=row['result_path'])
        for metric in ('f1_binary', 'f1_macro', 'accuracy'):
            before = float(original['test_' + metric]) * 100
            after = result['test_scores'][metric] * 100
            change.update({metric + '_before_percent': before, metric + '_after_percent': after,
                           metric + '_change_pp': after - before,
                           metric + '_fixed_loss_percent': result['original_test_scores'][metric] * 100})
        changes.append(change)
    scores.append(row)
assert len(selected) == 28 and len(scores) == 602
for row in scores:
    for field in ('result_path', 'original_record_path', 'original_history_path', 'weights_path', 'predictions_path'):
        assert (ROOT / row[field]).is_file(), row[field]

inventory = json.loads((RECOVERY / 'inventory.json').read_text())
for i, item in enumerate(inventory):
    if item['record_relative_path'] in new:
        _, result = new[item['record_relative_path']]
        inventory[i] = {**{k: result[k] for k in item if k in result},
                        'retained_record': next(r['original_record_path'] for r in selected
                                                if r['record_relative_path'] == result['record_relative_path']),
                        'report_cells': item['report_cells']}

with tempfile.TemporaryDirectory(prefix='adult_fix_valbest_', dir=ROOT / '.cache') as temporary:
    view = Path(temporary)
    lookup = {}
    for name in ('plot_recipe.py', 'files_index.json'):
        shutil.copy2(RECOVERY / name, view / name)
    (view / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    for mode in ('synthetic', 'mix'):
        rows = read(RECOVERY / (mode + '.csv'))
        for row in rows:
            for field, value in row.items():
                if field.endswith('_record') and value in new:
                    row[field] = new[value][1]['record_relative_path']
        write(view / (mode + '.csv'), rows)
    for row in scores:
        slug = row['record_relative_path'].replace('/', '__').removesuffix('.run.json')
        alias = view / 'jobs' / slug / 'result.json'
        alias.parent.mkdir(parents=True)
        source = ROOT / row['result_path']
        os.link(source, alias)
        lookup[str(alias.resolve())] = source
    code = (RECOVERY / 'build_reports.py').read_text(encoding='utf-8')
    replacements = {
        'Sources: G:/SummerResearch_Retained_20261013_2221.': f'Sources: {ROOT.as_posix()}.',
        'Caveats remain: Adult TabDDPM has an unresolved historical generator-source defect; MNIST12 TVAE retains the documented feature-collision caveat. These alternate evaluations do not clear either caveat.':
            'Adult TabDDPM uses the posterior-fix namespace tabddpm_adult_500ep_fix_20261030_v1, with 28 verified development-binary-F1 evaluations completed October 31, 2026. Historical pre-fix scores are preserved separately. MNIST12 TVAE retains the documented feature-collision caveat.',
        'Historical caveats retained: Adult TabDDPM generator source; MNIST12 TVAE feature collisions. Seeds may share the same holdout.':
            'Adult TabDDPM: posterior-fix rerun, best dev binary F1. MNIST12 TVAE: feature collisions. Seeds may share the same holdout.',
    }
    for before, after in replacements.items():
        assert before in code, before
        code = code.replace(before, after)
    builder = view / 'build_reports.py'
    builder.write_text(code, encoding='utf-8')
    subprocess.run([sys.executable, str(builder)], check=True,
                   env=dict(os.environ, MPLCONFIGDIR=str(view / 'matplotlib')))
    OUT.mkdir()
    for source in (view / 'reports').iterdir():
        if source.name == 'artifact_hashes.json':
            continue
        target = OUT / source.name
        if source.suffix != '.csv':
            shutil.copy2(source, target)
            continue
        rows = read(source)
        for row in rows:
            for field, value in row.items():
                if value and field != 'retained_record' and (field == 'result' or field.endswith('_record') or field.endswith('_d2_metadata')):
                    absolute = (view / value).resolve() if field == 'result' else (source.parent / value).resolve()
                    row[field] = Path(os.path.relpath(lookup[str(absolute)], OUT)).as_posix()
        write(target, rows)

write(OUT / 'scores_valbest.csv', scores)
write(ARCHIVE / 'scores_valbest.csv', selected)
write(OUT / 'adult_tabddpm_before_after_valf1.csv', changes)

# Confirm every displayed score against its result, and every unchanged cell against history.
checks = {}
for mode in ('synthetic', 'mix'):
    name = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '') + '_val_f1.csv'
    current = read(OUT / name)
    old = read(ROOT / 'comparisons' / (mode + '_valbest.csv'))
    assert len(current) == len(old)
    changed = []
    for row, previous in zip(current, old):
        assert row['configuration'] == previous['configuration']
        adult_fix = row['configuration'].startswith('TabDDPM')
        for field, value in row.items():
            if field.endswith(('_record', '_d2_metadata')):
                if value:
                    target = (OUT / value).resolve()
                    assert target.is_file()
                    if not (adult_fix and field.startswith('adult_')):
                        assert target == (ROOT / 'comparisons' / previous[field]).resolve()
                continue
            if adult_fix and field.startswith('adult_'):
                panel = 'adult_corrected_seed43' if 'seed43' in field else 'adult_corrected'
                result = json.loads((OUT / row[panel + '_record']).read_text())
                metric = 'f1_macro' if 'f1_macro' in field else 'f1_binary'
                assert abs(float(value) - result['test_scores'][metric] * 100) < 1e-7
                changed.append(field)
            else:
                assert value == previous[field], (mode, row['configuration'], field)
    assert len(changed) == 28
    checks[mode] = dict(verified_adult_metric_cells=28, all_other_cells_and_sources_unchanged=True)

summary = {m: {'mean_change_pp': sum(r[m + '_change_pp'] for r in changes)/28,
               'min_change_pp': min(r[m + '_change_pp'] for r in changes),
               'max_change_pp': max(r[m + '_change_pp'] for r in changes)}
           for m in ('f1_binary', 'f1_macro', 'accuracy')}
(OUT / 'validation.json').write_text(json.dumps(dict(checks=checks, summary=summary,
    completed=28, report_evaluations=602, unchanged_evaluations=574,
    last_completed_at_chicago=verification['last_completed_at_chicago'],
    archive_verification=root_relative(EVIDENCE / 'verification.json')), indent=2)+'\n')

# The existing before/after plotting layout, restricted to the seven selected constructions.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
order = [('full', 'generated'), ('full', 'rf'), ('full', 'xgb'), ('full', 'dnn'),
         ('xonly', 'rf'), ('xonly', 'xgb'), ('xonly', 'dnn')]
labels = ['Full · generated target', 'Full · RF', 'Full · XGB', 'Full · DNN',
          'X-only · RF', 'X-only · XGB', 'X-only · DNN']
fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
for ri, mode in enumerate(('synthetic', 'mix')):
    for ci, (metric, label, seed) in enumerate((m,l,s) for m,l in [('f1_binary','Binary F1'), ('f1_macro','Macro F1')] for s in (42,43)):
        ax = axes[ri, ci]
        entries = {(r['construction'],r['labeler']):r for r in changes if r['mode']==mode and int(r['seed'])==seed}
        for y, key in enumerate(order):
            r = entries[key]
            before, after = r[metric+'_before_percent'], r[metric+'_after_percent']
            ax.plot([before,after],[y,y],color='#2a78d6' if after>=before else '#d64545',lw=2)
            ax.scatter(before,y,s=45,facecolor='white',edgecolor='#777777',zorder=3)
            ax.scatter(after,y,s=45,color='#2a78d6',zorder=4)
            ax.annotate(f'{after-before:+.2f}',(max(before,after),y),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
        ax.set_title(f'{"Synthetic-only" if mode=="synthetic" else "Mix"} · {label} · {seed}',fontsize=10,loc='left')
        ax.set_xlabel(label+' (%)',fontsize=9)
        ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.margins(x=.3)
for col in (0,2):
    for ri in (0,1):
        limits = [axes[ri,c].get_xlim() for c in (col,col+1)]
        for c in (col,col+1):
            axes[ri,c].set_xlim(min(x[0] for x in limits),max(x[1] for x in limits))
for ax in axes[:,0]:
    ax.set_yticks(range(7),labels,fontsize=9)
axes[0,0].set_ylim(6.5,-.6)
fig.suptitle('Adult TabDDPM posterior fix: before vs after, both selected by best validation binary F1',x=.01,ha='left',fontsize=14)
fig.legend(handles=[Line2D([],[],marker='o',ls='none',markerfacecolor='white',markeredgecolor='#777777',label='Before: held pre-fix source'),
                    Line2D([],[],marker='o',ls='none',color='#2a78d6',label='After: posterior-fix source')],loc='upper left',bbox_to_anchor=(.01,.95),ncol=2,frameon=False)
fig.text(.01,.015,'28 selected evaluations; labels show after − before in percentage points. Same held-out test rows; seeds share a prepared split. NB/PCA-GMM are outside this replay scope.',fontsize=9)
fig.subplots_adjust(left=.13,right=.98,top=.84,bottom=.09,wspace=.12,hspace=.3)
fig.savefig(OUT/'adult_tabddpm_before_after_valf1.png',dpi=180,facecolor='white')
plt.close(fig)
lines = ['# Adult TabDDPM: best-validation-F1 before and after the posterior fix', '',
         'Both sides select the earliest maximum development binary F1 within their respective completed histories. '
         'The 28 evaluations cover generated targets and full/X-only RF/XGB/DNN, both seeds and training modes. '
         'The CSV also includes the corrected run’s loss-selected scores for comparison. NB/PCA-GMM were not replayed.', '',
         '![Before and after](adult_tabddpm_before_after_valf1.png)', '',
         '| Mode | Arm | Seed | Binary F1 before | after | Change (pp) | Macro F1 before | after | Change (pp) |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for mode in ('synthetic','mix'):
    for key,label in zip(order,labels):
        for seed in (42,43):
            r=next(r for r in changes if r['mode']==mode and (r['construction'],r['labeler'])==key and int(r['seed'])==seed)
            values=[f"{r[m+'_before_percent']:.2f} | {r[m+'_after_percent']:.2f} | {r[m+'_change_pp']:+.2f}" for m in ('f1_binary','f1_macro')]
            lines.append(f'| {mode} | {label} | {seed} | '+ ' | '.join(values)+' |')
(OUT/'adult_tabddpm_before_after_valf1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
hashes = {p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
(OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(dict(output=str(OUT), checks=checks, summary=summary),indent=2))
