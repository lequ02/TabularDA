"""Plot paired test-score differences from completed, verified checkpoints."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PROJECT = Path('D:/SummerResearch')
BASE = PROJECT / 'output/comparison_devf1_20261015_v1'
REPORTS = BASE / 'reports'
OUT = REPORTS / 'deltas'
OUT.mkdir(exist_ok=True)
CACHE = PROJECT / '.cache/comparison_devf1_20261015'
RETAINED = Path('G:/SummerResearch_Retained_20261013_2221')
recipe = runpy.run_path(str(CACHE / 'plot_recipe.py'))
PANELS = recipe['PANELS']
PLOT_PANELS = {p: v for p, v in PANELS.items() if v['metric'] != 'nmae_sigma'}
UNSCALED = recipe['UNSCALED_METRICS']
COLORS = recipe['METRIC_COLORS']
index = json.loads((CACHE / 'files_index.json').read_text())
result_paths = list((BASE / 'jobs').glob('*/result.json'))
results = {r['record_relative_path']: (p, r) for p in result_paths for r in [json.loads(p.read_text())]}
assert len(results) == 602
assert all(r['original_checkpoint_tensors_exact'] and r['test_ids_and_targets_match'] for _, r in results.values())


def read_d2(pair):
    original, (_, result) = pair
    relative = original.replace('.run.json', '.d2.json')
    entry = index[relative]
    path = RETAINED / entry['storage_path']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f'Retained D2 sidecar hash mismatch: {path}')
    record = json.loads(data)
    assert (record['run_record'], record['dataset'], record['seed'], record['selected_dev_epoch']) == (
        original, result['dataset'], result['seed'], result['original_epoch'])
    return original, {'value': record['test_scores']['d2_absolute_error'],
                      'retained_path': str(path), 'sha256': entry['sha256']}


regression = [pair for pair in results.items() if pair[1][1]['dataset'] in ('news', 'california_housing')]
with ThreadPoolExecutor(max_workers=8) as pool:
    old_d2 = dict(pool.map(read_d2, regression))
assert len(old_d2) == 172
print('Verified all 172 full-precision retained regression sidecars.', flush=True)


def fields(panel, info):
    metrics = [info['metric']]
    if info['metric'] == 'f1_binary':
        metrics.append('f1_macro')
    elif info['metric'] == 'r2':
        metrics.append('d2_absolute_error')
    return [(metric, panel + '_' + metric + ('' if metric in UNSCALED else '_percent')) for metric in metrics]


all_long = []
proof_records = set()
source_hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in result_paths}
for mode in ('synthetic', 'mix'):
    alt_name = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '') + '_val_f1.csv'
    with (REPORTS / alt_name).open(newline='', encoding='utf-8-sig') as stream:
        alternate = list(csv.DictReader(stream))
    with (CACHE / (mode + '.csv')).open(newline='', encoding='utf-8-sig') as stream:
        original = {r['configuration']: r for r in csv.DictReader(stream)}
    rows = [r for r in alternate if not r['configuration'].startswith('Paper ')]
    order = [r['configuration'] for r in rows]
    assert len(order) == 22
    assert [order[i] for i in (0, 1, 8, 15)] == [
        'Original data only', 'CTGAN, generated target', 'TVAE, generated target', 'TabDDPM, generated target']
    assert 'Real-only (TabDDPM run)' not in order
    plot = {}
    long_rows = []
    wide = []
    for row in rows:
        label = row['configuration']
        plot[label] = {}
        wide_row = {'configuration': label}
        for panel, info in PANELS.items():
            original_relative = original[label][panel + '_record']
            path, result = results[original_relative]
            assert (REPORTS / row[panel + '_record']).resolve() == path.resolve()
            proof_records.add(original_relative)
            assert result['dataset'] == info['dataset'] and result['seed'] == info['seed']
            plot[label][panel] = []
            for metric, column in fields(panel, info):
                scale = 1 if metric in UNSCALED else 100
                old = old_d2[original_relative]['value'] if metric == 'd2_absolute_error' else result['original_test_scores'][metric]
                new = result['test_scores'][metric]
                assert abs(float(original[label][column]) - old * scale) <= 1e-6
                assert abs(float(row[column]) - new * scale) < 1e-7
                delta = (new - old) * scale
                if result['dataset'] == 'census_kdd':
                    assert result['original_selection_metric'] == 'f1_macro'
                    comparator = 'strict maximum dev macro F1 minus original dev macro F1 selector'
                else:
                    assert result['original_selection_metric'] == 'loss'
                    comparator = 'maximum dev R2 minus minimum dev loss' if metric in UNSCALED else 'maximum dev F1 minus minimum dev loss'
                point = {
                    'mode': mode, 'dataset': info['dataset'], 'seed': info['seed'],
                    'configuration': label, 'generator': result['generator'], 'metric': metric,
                    'original_test_score_unscaled': old, 'alternative_test_score_unscaled': new,
                    'delta_unscaled': new - old, 'delta': delta,
                    'delta_units': 'unscaled' if metric in UNSCALED else 'percentage points',
                    'original_selection_metric': result['original_selection_metric'],
                    'alternative_selection_metric': result['selection_metric'],
                    'original_epoch': result['original_epoch'], 'alternative_epoch': result['target_epoch'],
                    'same_checkpoint': result['same_checkpoint'], 'comparison': comparator,
                    'original_record': original_relative,
                    'alternative_result': Path(os.path.relpath(path, OUT)).as_posix(),
                }
                long_rows.append(point)
                plot[label][panel].append(point)
                suffix = 'unscaled' if metric in UNSCALED else 'pp'
                wide_row[panel + '_delta_' + metric + '_' + suffix] = delta
        wide.append(wide_row)
    all_long.extend(long_rows)
    basename = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '') + '_checkpoint_delta'
    for name, data in ((basename + '.csv', wide), (basename + '_points.csv', long_rows)):
        with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(7, 2, figsize=(18.5, 44.8), sharey='row')
    pair_limits = {}
    for info in PLOT_PANELS.values():
        dataset = info['dataset']
        panel_names = [p for p, v in PLOT_PANELS.items() if v['dataset'] == dataset]
        maximum = max(abs(point['delta']) for label in order for p in panel_names for point in plot[label][p])
        minimum = .01 if info['metric'] in UNSCALED else 1.0
        pair_limits[dataset] = max(minimum, maximum * 1.25)
    markers = {'f1_binary': 'o', 'f1_macro': 'D', 'accuracy': 's', 'r2': 's', 'd2_absolute_error': 'v'}
    for position, (panel, info) in enumerate(PLOT_PANELS.items()):
        ax = axes.flat[position]
        dataset = info['dataset']
        limit = pair_limits[dataset]
        ax.axvspan(-limit, 0, color='#fff4f4', zorder=0)
        ax.axvspan(0, limit, color='#f1faf4', zorder=0)
        ax.axvline(0, color='#475569', linewidth=1.3, zorder=2)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(21.8, -.9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, symmetric=True))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e2e8f0', linewidth=.7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=9)
        ax.tick_params(axis='x', colors='#475569')
        title = info['title'].split(' · ')[0] + f" · seed {info['seed']}"
        ax.set_title(title, loc='left', fontsize=13.5, fontweight='bold', pad=16)
        if dataset == 'census_kdd':
            ax.text(0, 1.015, 'Original baseline already uses macro F1 (not loss)',
                    transform=ax.transAxes, fontsize=8.5, color='#92400e', va='bottom')
        elif dataset in ('news', 'california_housing'):
            ax.text(0, 1.015, 'R² and loss select the same checkpoints; all changes are zero',
                    transform=ax.transAxes, fontsize=8.5, color='#64748b', va='bottom')
        for divider in (.5, 1.5, 4.5, 7.5, 8.5, 11.5, 14.5, 15.5, 18.5):
            ax.axhline(divider, color='#dce3eb', linewidth=.75)
        for y, label in enumerate(order):
            points = plot[label][panel]
            paired = len(points) == 2
            for j, point in enumerate(points):
                metric = point['metric']
                generator = point['generator'] if point['generator'] in ('ctgan', 'tvae', 'tabddpm') else 'ctgan'
                color = COLORS[metric][generator]
                value = point['delta']
                # Round only the displayed zero; raw differences remain in CSV.
                shown = 0.0 if abs(value) < 1e-10 else value
                dy = (-.19 if j == 0 else .19) if paired else 0
                marker = '^' if y == 0 else markers[metric]
                generated = ', generated target' in label
                ax.scatter(shown, y + dy, s=60 if y == 0 else 45, marker=marker,
                           facecolors='none' if generated else color, edgecolors=color, linewidths=1.1, zorder=4)
                text = '0' if shown == 0 else f'{shown:+.4f}' if metric in UNSCALED else f'{shown:+.2f}'
                # Opposite label sides keep paired zero changes readable.
                left = j == 0 if paired else shown < 0
                ax.annotate(text, (shown, y + dy), xytext=(-5 if left else 5, 0), textcoords='offset points',
                            ha='right' if left else 'left', va='center', color=color, fontsize=7.6 if paired else 8.5,
                            fontweight='bold' if y == 0 else 'normal', zorder=5)
        if position % 2 == 0:
            ax.set_yticks(range(22), order)
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)
        axis_label = {'f1_binary': 'Δ test binary / macro F1 (percentage points)',
                      'f1_macro': 'Δ test macro F1 (percentage points)',
                      'accuracy': 'Δ test accuracy (percentage points)',
                      'r2': 'Δ test R² / D² absolute error (unscaled)'}[info['metric']]
        ax.set_xlabel(axis_label, fontsize=10, labelpad=8)

    title = 'Synthetic-only' if mode == 'synthetic' else 'Mix'
    fig.suptitle(title + ' · checkpoint-selection score changes', x=.035, y=.994,
                 ha='left', fontsize=17, fontweight='bold')
    fig.text(.035, .983, 'Δ = test score at best validation F1/R² − test score at original checkpoint · positive = higher test score',
             ha='left', fontsize=10.5, color='#475569')
    handles = [Line2D([], [], marker=marker, color=COLORS[metric][generator], linestyle='none',
                      label=f'{generator.upper()} · {label}')
               for metric, marker, label in (('f1_binary', 'o', 'binary F1'), ('f1_macro', 'D', 'macro F1'),
                                             ('accuracy', 's', 'accuracy / R²'), ('d2_absolute_error', 'v', 'D²'))
               for generator in ('ctgan', 'tvae', 'tabddpm')]
    handles += [Line2D([], [], marker='^', color='#64748b', linestyle='none', label='Original data'),
                Line2D([], [], marker='o', markerfacecolor='none', color='#64748b', linestyle='none', label='Generated target')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.985, .970), ncol=5, frameon=False, fontsize=9)
    fig.text(.035, .030, 'Selection: Adult binary F1; Census/Covertype/MNIST macro F1; News/Housing R². Best epoch within each original completed history.', fontsize=9, color='#475569')
    fig.text(.035, .020, 'Census exception: original checkpoints already used dev macro F1; those deltas compare strict maximum F1 with the original F1 selector.', fontsize=9, color='#475569')
    fig.text(.035, .010, 'Historical caveats retained: Adult TabDDPM generator source; MNIST12 TVAE feature collisions. Seeds may share a holdout.', fontsize=9, color='#475569')
    fig.subplots_adjust(left=.25, right=.985, top=.943, bottom=.055, wspace=.20, hspace=.31)
    fig.savefig(OUT / (basename + '.png'), dpi=220, facecolor='white')
    plt.close(fig)
    print(f'{mode}: {len(long_rows)} score differences; saved {basename}.png', flush=True)

assert len(proof_records) == 602
assert len(all_long) == 1144
assert sum(p['metric'] != 'nmae_sigma' for p in all_long) == 968
regression_points = [p for p in all_long if p['dataset'] in ('news', 'california_housing')]
assert all(p['same_checkpoint'] for p in regression_points)
assert max(abs(p['delta']) for p in regression_points) < 1e-10
summary = {
    'created_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'definition': 'alternative test score minus original test score; classification scaled to percentage points',
    'source_runs': len(proof_records), 'score_differences_in_csv': len(all_long),
    'plotted_score_differences': 968, 'methods_per_panel': 22, 'panels_per_figure': 14,
    'original_regression_d2_sidecars_hash_verified': len(old_d2),
    'census_exception': 'Original criterion already macro F1; these deltas are not F1-minus-loss comparisons',
    'unchanged_regression_checkpoints': True,
    'largest_absolute_regression_delta': max(abs(p['delta']) for p in regression_points),
    'source_result_sha256': source_hashes,
    'retained_d2_sources': old_d2,
}
(OUT / 'delta_provenance.json').write_text(json.dumps(summary, indent=2) + '\n')
(OUT / 'README.md').write_text('''# Checkpoint-selection difference plots

Each point subtracts the original reported test score from the test score selected by maximum real-development F1 (classification) or R² (regression), within the original completed training history. Positive values mean a higher test score. F1 and accuracy differences use **percentage points**, while R² and D² differences are **unscaled**. MNIST panels continue to show test accuracy, with checkpoints selected by development macro F1.

**Census KDD exception:** its original checkpoints already used development macro F1. Those panels compare strict maximum macro F1 with the original tolerance-based macro-F1 selector, and are not F1-versus-loss comparisons. No loss-selected Census test result is invented or substituted.

News and Housing use the same checkpoints under maximum development R² and minimum development loss, so their differences are zero. Original full-precision D² values come from 172 hash-verified retained sidecars; other original scores come from the saved original run metrics. Differences are computed before rounding. The small NMAE differences are retained in CSVs but are not added to the figure panels. Paper benchmark lines have no paired checkpoints and are omitted.

The figures preserve all 22 methods, generator grouping, both seeds, all displayed metrics, and the existing generator/metric colors. Original and alternative source paths, epochs, selector names, full-precision scores and differences are in each `_points.csv` file. Each wide CSV has one row per method. Figure annotation rounding leaves the stored differences unchanged.

Historical source and interpretation caveats remain: Adult TabDDPM generator-source defect; MNIST12 TVAE feature collisions; shared holdouts across seeds. Later table hashes alone do not certify historical training-time bytes. The completed recovery results include exact original checkpoint and train/development-history checks.

- [Synthetic difference plot](dataset_pipeline_comparison_checkpoint_delta.png)
- [Mix difference plot](dataset_pipeline_comparison_mix_checkpoint_delta.png)
- [Delta provenance](delta_provenance.json)
''', encoding='utf-8')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'}
(OUT / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('source_result_sha256', 'retained_d2_sources')}, indent=2))
