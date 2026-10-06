"""Render retrieved, saved histories; performs no training or evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

DEST = Path(__file__).resolve().parent
source = json.loads((DEST / 'source_histories.json').read_text(encoding='utf-8-sig'))
runs = {r['method']: r for r in source['runs']}
groups = [
    ('baselines', 'Real data and generated-target baselines', [
        ('real_original', 'Real training data only'),
        ('ctgan_full_generated_synthetic', 'CTGAN full table — generated targets'),
        ('tvae_full_generated_synthetic', 'TVAE full table — generated targets'),
    ]),
    ('ctgan_rf_xgb', 'CTGAN features relabeled by RF or XGB', [
        ('ctgan_full_rf_synthetic', 'CTGAN full-table features + RF'),
        ('ctgan_full_xgb_synthetic', 'CTGAN full-table features + XGB'),
        ('ctgan_xonly_rf_synthetic', 'CTGAN X-only features + RF'),
        ('ctgan_xonly_xgb_synthetic', 'CTGAN X-only features + XGB'),
    ]),
    ('dnn_relabeling', 'CTGAN and TVAE features relabeled by DNN', [
        ('ctgan_full_dnn_synthetic', 'CTGAN full-table features + DNN'),
        ('ctgan_xonly_dnn_synthetic', 'CTGAN X-only features + DNN'),
        ('tvae_full_dnn_synthetic', 'TVAE full-table features + DNN'),
        ('tvae_xonly_dnn_synthetic', 'TVAE X-only features + DNN'),
    ]),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#334155', 'xtick.color': '#475569',
                     'ytick.color': '#475569', 'axes.titleweight': 'bold'})
colors = ['#2563eb', '#ea580c']
loss_max = max(max(h['train_loss'], h['dev_loss']) for r in runs.values() for h in r['history'])
summary, combined = [], []
for slug, group_title, methods in groups:
    fig, axes = plt.subplots(len(methods), 2, figsize=(13, 3.1 * len(methods) + 1.3), squeeze=False)
    fig.suptitle('Census KDD · seed 42\n' + group_title, x=0.065, y=0.985, ha='left', fontsize=18, fontweight='bold')
    for row, (method, title) in enumerate(methods):
        r = runs[method]
        df = pd.DataFrame(r['history'])
        chosen = df.loc[df.global_round.eq(r['selected_epoch'])].iloc[0]
        if method == 'real_original':
            prevalence = 'Real training positives: 6.09%'
        else:
            counts = r['synthetic_label_counts']
            prevalence = f"Synthetic training positives: {counts.get('1', 0):,} / {sum(counts.values()):,} ({counts.get('1', 0) / sum(counts.values()):.2%})"
        for col, (metric, label) in enumerate([('loss', 'Binary cross-entropy loss'), ('f1_binary', 'Binary F1 (%)')]):
            ax = axes[row, col]
            factor = 100 if metric == 'f1_binary' else 1
            for split, color in zip(['train', 'dev'], colors):
                ax.plot(df.global_round, df[f'{split}_{metric}'] * factor, color=color, lw=1.8,
                        linestyle='-' if split == 'train' else '--')
                ax.scatter(r['selected_epoch'], chosen[f'{split}_{metric}'] * factor,
                           color=color, s=40, zorder=4, edgecolors='white', linewidth=0.8)
            ax.axvline(r['selected_epoch'], color='#64748b', lw=1, ls=':')
            ax.set_xlim(0, 50)
            ax.set_ylim((0, 100) if factor == 100 else (0, loss_max * 1.07))
            ax.grid(alpha=0.18)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(title if col == 0 else f"Selected checkpoint: epoch {r['selected_epoch']}", loc='left', fontsize=12,
                         pad=25 if col == 0 else 8)
            if col == 0:
                ax.text(0, 1.02, prevalence, transform=ax.transAxes, fontsize=9, color='#64748b')
        combined.append(df.assign(method=method, selected_checkpoint=df.global_round.eq(r['selected_epoch'])))
        final = df.iloc[-1]
        summary.append({'configuration': title, 'selected_epoch': r['selected_epoch'],
                        'selected_train_loss': chosen.train_loss, 'selected_val_loss': chosen.dev_loss,
                        'selected_train_f1': chosen.train_f1_binary, 'selected_val_f1': chosen.dev_f1_binary,
                        'final_epoch': int(final.global_round), 'final_train_loss': final.train_loss,
                        'final_val_loss': final.dev_loss, 'final_train_f1': final.train_f1_binary,
                        'final_val_f1': final.dev_f1_binary})
    fig.legend(handles=[Line2D([0], [0], color=colors[0], lw=2, label='Training'),
                        Line2D([0], [0], color=colors[1], lw=2, ls='--', label='Real validation'),
                        Line2D([0], [0], color='#64748b', lw=1, ls=':', marker='o', markersize=4,
                               label='Selected checkpoint')], loc='upper right', bbox_to_anchor=(0.97, 0.98), frameon=False)
    fig.text(0.065, 0.018, 'Selection: minimum real-validation loss. F1 uses the fixed 0.5 threshold. No test metrics shown.',
             color='#64748b', fontsize=10)
    fig.tight_layout(rect=[0.02, 0.04, 0.99, 0.9 if len(methods) == 3 else 0.92], h_pad=2)
    fig.savefig(DEST / f'{slug}.png', dpi=150, facecolor='white')
    fig.savefig(DEST / f'{slug}.svg', facecolor='white')
    plt.close(fig)
pd.concat(combined, ignore_index=True).to_csv(DEST / 'histories.csv', index=False)
pd.DataFrame(summary).to_csv(DEST / 'checkpoint_summary.csv', index=False)
print(f"Rendered {len(groups)} figures from {len(runs)} saved histories.")
