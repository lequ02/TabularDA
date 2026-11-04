"""Run only remotely. Dataset is the independent unit; seed runs are averaged."""
import hashlib
import html
import itertools
import json
import platform
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
REG = ['news', 'california_housing']
GEN = ['ctgan', 'tvae', 'tabddpm']
LABEL = ['rf', 'xgb', 'dnn']
NAMES = dict(adult='Adult', census_kdd='Census KDD', covertype='Covertype', mnist12='MNIST12', mnist28='MNIST28', news='News', california_housing='Housing')
raw = pd.read_csv(ROOT / 'inputs/scores.csv')
manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
for name, expected_hash in manifest['files'].items():
    assert hashlib.sha256((ROOT / 'inputs' / name).read_bytes()).hexdigest() == expected_hash
assert len(raw) == 1204
experimental = raw[raw['mode'].isin(['synthetic', 'mix'])].copy()
for selection, part in experimental.groupby('selection'):
    assert len(part) == 588
    actual = set(part[['dataset', 'seed', 'generator', 'construction', 'labeler', 'mode']].itertuples(index=False, name=None))
    expected = {(d, s, g, c, l, m) for d, s, g, m in itertools.product(CLASS + REG, [42, 43], GEN, ['synthetic', 'mix'])
                for c, l in [('full', 'generated')] + list(itertools.product(['full', 'xonly'], LABEL))}
    assert actual == expected and len(actual) == len(part)

def cells(part, metric, datasets):
    sub = part[part.dataset.isin(datasets)].copy()
    assert sub[metric].notna().all()
    sub['treatment'] = np.where(sub.labeler.eq('generated'), 'generated', 'hybrid_' + sub.construction)
    return sub.groupby(['dataset', 'generator', 'treatment'], as_index=False)[metric].mean().rename(columns={metric: 'score'})

def paired(left, right):
    left, right = np.asarray(left), np.asarray(right)
    diff = left - right
    n = len(diff)
    assert n >= 2 and np.isfinite(diff).all() and diff.std(ddof=1) > 0
    t = stats.ttest_rel(left, right)
    f = float(t.statistic ** 2)
    p = float(stats.f.sf(f, 1, n - 1))
    np.testing.assert_allclose(p, t.pvalue, rtol=1e-10)
    sem = stats.sem(diff)
    # Exact sign-flip check over dataset-level differences, two-sided.
    flips = np.asarray(list(itertools.product([-1, 1], repeat=n)))
    exact_p = np.mean(np.abs(flips @ diff / n) >= abs(diff.mean()) - 1e-14)
    return dict(n_datasets=n, left_mean=left.mean(), right_mean=right.mean(), difference=diff.mean(),
                ci_low=diff.mean()-stats.t.ppf(.975, n-1)*sem, ci_high=diff.mean()+stats.t.ppf(.975, n-1)*sem,
                F=f, df_num=1, df_den=n-1, p=p, partial_eta_sq=f/(f+n-1),
                exact_signflip_p=exact_p,
                shapiro_difference_p=float(stats.shapiro(diff).pvalue) if n >= 3 else np.nan)

def contrasts(table, datasets, by_generator=False):
    result, effects = [], []
    for gen in GEN if by_generator else ['all']:
        sub = table if gen == 'all' else table[table.generator.eq(gen)]
        wide = sub.groupby(['dataset', 'treatment']).score.mean().unstack().reindex(datasets)
        for question, left, right in [('hybrid_minus_generated', (wide.hybrid_full+wide.hybrid_xonly)/2, wide.generated),
                                      ('full_minus_xonly', wide.hybrid_full, wide.hybrid_xonly)]:
            result.append(dict(generator=gen, question=question, **paired(left, right)))
            effects.extend(dict(generator=gen, question=question, dataset=d, left=l, right=r, difference=l-r)
                           for d, l, r in zip(datasets, left, right))
    return result, effects

def add_holm(frame, keys=None):
    result = frame.copy()
    if keys:
        for _, indices in result.groupby(keys).groups.items():
            result.loc[indices, 'p_holm'] = multipletests(result.loc[indices, 'p'], method='holm')[1]
    else:
        result['p_holm'] = multipletests(result.p, method='holm')[1]
    return result

def repeated_models(table, mode):
    rows = []
    checks = []
    for question in ['hybrid_minus_generated', 'full_minus_xonly']:
        values = []
        for d, g in itertools.product(CLASS, GEN):
            w = table[(table.dataset.eq(d)) & (table.generator.eq(g))].set_index('treatment').score
            pair = [w.generated, (w.hybrid_full+w.hybrid_xonly)/2] if question == 'hybrid_minus_generated' else [w.hybrid_xonly, w.hybrid_full]
            values.extend(dict(dataset=d, generator=g, condition=str(i), score=v) for i, v in enumerate(pair))
        frame = pd.DataFrame(values)
        fitted = AnovaRM(frame, 'score', 'dataset', within=['generator', 'condition']).fit().anova_table
        array = frame.set_index(['dataset', 'generator', 'condition']).score.reindex(pd.MultiIndex.from_product([CLASS, GEN, ['0', '1']])).to_numpy().reshape(5, 3, 2)
        for term, row in fitted.iterrows():
            if term == 'condition':
                direct = paired(array[:, :, 1].mean(1), array[:, :, 0].mean(1))['F']
                epsilon = 1.0
            else:
                u = array.mean(2) if term == 'generator' else array[:, :, 1]-array[:, :, 0]
                centered_means = u.mean(0)-u.mean()
                err = u-u.mean(0)-u.mean(1)[:, None]+u.mean()
                direct = (len(u)*sum(centered_means**2)/2)/(sum((err**2).ravel())/8)
                h = np.eye(3)-np.ones((3, 3))/3
                cov = h @ np.cov(u, rowvar=False) @ h
                epsilon = float(np.clip(np.trace(cov)**2/(2*np.trace(cov@cov)), .5, 1))
            np.testing.assert_allclose(row['F Value'], direct, rtol=1e-9, atol=1e-10)
            checks.append(dict(mode=mode, question=question, term=term, statsmodels_F=row['F Value'], independent_F=direct))
            f = row['F Value']; df1 = row['Num DF']; df2 = row['Den DF']
            rows.append(dict(mode=mode, question=question, term=term, F=f, df_num=df1, df_den=df2,
                             p_uncorrected=row['Pr > F'], GG_epsilon=epsilon,
                             GG_df_num=df1*epsilon, GG_df_den=df2*epsilon,
                             p=float(stats.f.sf(f, df1*epsilon, df2*epsilon)),
                             partial_eta_sq=f*df1/(f*df1+df2)))
    return rows, checks

primary, secondary, effects, methodmeans, omnibus, independent, sensitivity, regression, regdesc, labeler_rows = ([] for _ in range(10))
for selection in ['valbest', 'original']:
    for mode in ['synthetic', 'mix']:
        part = experimental[experimental.selection.eq(selection) & experimental['mode'].eq(mode)]
        table = cells(part, 'f1_macro', CLASS)
        for cohort, tasks in [('all_five', CLASS), ('exclude_adult', CLASS[1:]),
                              ('exclude_mnist12', [d for d in CLASS if d != 'mnist12']),
                              ('exclude_adult_and_mnist12', [d for d in CLASS if d not in ['adult', 'mnist12']])]:
            r, e = contrasts(table[table.dataset.isin(tasks)], tasks)
            sensitivity.extend(dict(selection=selection, mode=mode, cohort=cohort, **x) for x in r)
            if selection == 'valbest' and cohort == 'all_five':
                primary.extend(dict(mode=mode, **x) for x in r)
                effects.extend(dict(mode=mode, **x) for x in e)
        if selection == 'valbest':
            r, _ = contrasts(table, CLASS, True)
            secondary.extend(dict(mode=mode, **x) for x in r)
            r, c = repeated_models(table, mode); omnibus.extend(r); independent.extend(c)
            methodmeans.extend(dict(mode=mode, **x) for x in table.to_dict('records'))
            for labeler in LABEL:
                labelpart = part[part.labeler.isin(['generated', labeler])]
                r, _ = contrasts(cells(labelpart, 'f1_macro', CLASS), CLASS)
                labeler_rows.extend(dict(mode=mode, labeler=labeler, **x) for x in r)
            regtable = cells(part, 'r2', REG)
            r, _ = contrasts(regtable, REG)
            regression.extend(dict(mode=mode, **x) for x in r)
            for dataset in REG:
                ds = part[part.dataset.eq(dataset)].copy()
                ds['treatment'] = np.where(ds.labeler.eq('generated'), 'generated', 'hybrid_' + ds.construction)
                for metric in ['r2', 'nmae_sigma', 'd2_absolute_error']:
                    for treatment, group in ds.groupby('treatment'):
                        assert group[metric].notna().all()
                        regdesc.append(dict(mode=mode, dataset=dataset, metric=metric, treatment=treatment, mean=group[metric].mean()))
primary = add_holm(pd.DataFrame(primary))
secondary = add_holm(pd.DataFrame(secondary))
omnibus = add_holm(pd.DataFrame(omnibus))
sensitivity = add_holm(pd.DataFrame(sensitivity), ['selection', 'cohort'])
regression = add_holm(pd.DataFrame(regression))
labeler_rows = add_holm(pd.DataFrame(labeler_rows))
effects = pd.DataFrame(effects); means = pd.DataFrame(methodmeans)
for name, frame in [('primary_anova', primary), ('generator_contrasts', secondary), ('two_factor_repeated_anova', omnibus),
                    ('sensitivity_anova', sensitivity), ('dataset_effects', effects), ('dataset_method_means', means),
                    ('independent_F_checks', pd.DataFrame(independent)), ('regression_exploratory_anova', regression),
                    ('regression_descriptive', pd.DataFrame(regdesc)), ('labeler_contrasts', labeler_rows)]:
    frame.to_csv(OUT / (name + '.csv'), index=False)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col')
for i, mode in enumerate(['synthetic', 'mix']):
    for j, question in enumerate(['hybrid_minus_generated', 'full_minus_xonly']):
        ax = axes[i, j]
        vals = effects[(effects['mode'].eq(mode)) & (effects.question.eq(question))].set_index('dataset').loc[CLASS]
        p = primary[(primary['mode'].eq(mode)) & (primary.question.eq(question))].iloc[0]
        ax.axvline(0, color='#999999', lw=.8)
        ax.scatter(100*vals.difference, np.arange(5), color='#247B91', s=35, zorder=3)
        ax.errorbar(100*p.difference, 5.5, xerr=[[100*(p.difference-p.ci_low)], [100*(p.ci_high-p.difference)]],
                    fmt='D', color='#A64A27', capsize=4, label='Mean and unadjusted 95% CI')
        ax.set_yticks(list(range(5))+[5.5], [NAMES[d] for d in CLASS]+['Task mean'])
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.18)
        ax.set_title(f'{mode.capitalize()}: ' + ('hybrid − generated Y' if j == 0 else 'full − X only') + f'\nHolm p = {p.p_holm:.3g}')
        ax.set_xlabel('Macro F1 difference (percentage points)')
fig.suptitle('G: archive • validation-best results • five classification datasets', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(OUT/'effects.png', dpi=180); plt.close(fig)

def pretty(frame, columns):
    view = frame[columns].copy()
    for column in view.select_dtypes(include='number'):
        view[column] = view[column].map(lambda x: f'{x:.5g}')
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in view.itertuples(index=False, name=None))
    return '\n'.join(lines)

display = primary.copy()
for col in ['left_mean', 'right_mean', 'difference', 'ci_low', 'ci_high']:
    display[col] *= 100
text = f'''# ANOVA of the retained G: research results

Prepared {datetime.now(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p %Z')}.

## Main findings

The latest validation-best results are the primary analysis. The two requested comparisons use five classification datasets and macro F1 throughout. Positive differences favor hybrids for the first comparison, and full-table generation for the second. Results below are percentage points; means equally weight datasets, seeds, generators and hybrid labelers.

{pretty(display, ['mode','question','left_mean','right_mean','difference','ci_low','ci_high','F','df_num','df_den','p','p_holm'])}

**Conclusion:** Neither requested comparison is significant at familywise α=0.05 after Holm correction. Hybrids improve mean macro F1 by 7.36 percentage points in synthetic-only training and 4.73 points in mix. The synthetic-only effect is nominally p=0.0490, but its adjusted p=0.196 does not pass the planned family. Full-table minus X-only differences are +0.46 and +0.15 points, with wide intervals covering both directions. These results do not establish equivalence.

CTGAN shows the largest descriptive hybrid benefit: +20.57 points (synthetic) and +10.14 (mix); TVAE gains +4.32 and +3.84, while TabDDPM changes −2.83 and +0.20. No generator-specific comparison passes Holm correction. The pooled gain should therefore not be interpreted as a demonstrated benefit for each generator.

The confidence intervals are unadjusted 95% intervals. Holm correction covers the four primary tests together (two questions × two training modes). A nonsignificant result does not establish equivalence. See [the effect plot](effects.png) for dataset variation.

## Questions and statistical design

1. **CTGAN / TVAE / TabDDPM versus hybrids.** For each generator, compare its full-table generated-target baseline with the equally weighted mean of RF, XGB and DNN hybrids across both full and X-only constructions. Generated-target baselines enter once, rather than being duplicated for each labeler.
2. **Full features versus X only.** Among hybrids, compare a generator fitted on (X,Y), discarding generated Y and relabeling X, with a separate generator fitted only on X and then relabeled. Both constructions have the same downstream predictor inputs. This tests inclusion of Y in generator fitting, not removal of downstream feature columns.

The independent unit is a dataset. Seeds 42/43 are averaged within each configuration before inference because they can share heldout rows. Labelers and generators are averaged with equal weights for primary contrasts. Datasets are Adult, Census KDD, Covertype, MNIST12 and MNIST28. Synthetic-only and mix are analyzed separately; mix concatenates all real training rows and all synthetic rows.

For each question, the primary test is a two-level repeated-measures ANOVA on dataset means: score ~ dataset + condition, with F(1,4). Independently verified as the square of the paired t statistic. This tests the mean effect across these five tasks; the 420 classification runs are not 420 independent datasets. Macro F1 is used for every classification task, including MNIST; original comparison figures may display other metrics. The validation-best checkpoint selection metrics remain those saved in the archive (Adult binary F1, other classification macro F1, regression R²).

## Generator-specific comparisons

These secondary tests retain pairing across datasets. Holm correction covers all 12 tests together. Differences below are raw macro F1 units; multiply by 100 for percentage points.

{pretty(secondary, ['mode','generator','question','difference','F','df_num','df_den','p','p_holm'])}

## Two-factor repeated-measures ANOVA

For each question, generator (three levels) × condition (two levels) are within-dataset factors. The condition effect tests the average requested contrast; the interaction tests whether that contrast differs by generator. Labelers and seeds are averaged before fitting. Generator effects compare average scores across the two conditions and should not be read as a baseline-only generator ranking.

Greenhouse–Geisser epsilon and adjusted degrees of freedom are computed from the dataset-level covariance of centered generator means (generator effect) or centered condition differences (interaction). The two-level condition has epsilon=1. The p column uses corrected degrees of freedom; Holm then covers all 12 omnibus terms. Statsmodels does not supply sphericity corrections itself, so this correction is explicit. All 12 F values were independently reconstructed from sums of squares.

{pretty(omnibus, ['mode','question','term','F','GG_epsilon','GG_df_num','GG_df_den','p','p_holm'])}

## Regression

News and California Housing remain separate from classification. R² is unscaled, with higher better. The following paired ANOVAs average the same configurations but have only two dataset units, hence F(1,1). They are exploratory and cannot support a stable general claim about regression. No two-factor regression model is fitted with only two datasets.

{pretty(regression, ['mode','question','left_mean','right_mean','difference','F','df_num','df_den','p','p_holm'])}

Saved R², NMAEσ and D² means by dataset and construction are in [regression_descriptive.csv](regression_descriptive.csv). NMAEσ is read from saved scores, lower is better; its test-target normalization metadata was verified without refitting. D² is read only where saved in validation-best results. Metrics are never pooled into a common outcome.

## Sensitivity checks and limitations

The original-checkpoint analysis gives similar hybrid benefits (+7.48 synthetic, +4.46 mix points), with no adjusted significance. Full-versus-X-only estimates change direction across protocols and task exclusions. None of the four primary contrasts passes Holm correction in any prespecified exclusion cohort. These sensitivity results support reporting a descriptive hybrid gain and inconclusive evidence about full versus X-only generation.

- [sensitivity_anova.csv](sensitivity_anova.csv) repeats the four primary tests using original checkpoints, excluding Adult, excluding MNIST12, and excluding both. Adult TabDDPM retains its historical generator-source bug caveat; MNIST12 TVAE retains its feature-collision caveat. Entire datasets are excluded for these checks to preserve a balanced, matched three-generator comparison. These exclusions reduce the independent sample to four or three tasks.
- [labeler_contrasts.csv](labeler_contrasts.csv) reports each RF/XGB/DNN hybrid contrast and full-versus-X-only effect, averaged over generators; Holm covers 12 tests. [dataset_method_means.csv](dataset_method_means.csv) preserves generator and construction means within each dataset.
- Five tasks provide low power. Task effects may be heterogeneous and normality of task-level differences is uncertain; Shapiro p-values and exact sign-flip sensitivity p-values are saved with paired tests. Exact two-sided sign-flip p-values with five tasks cannot be below 0.0625. Their sign-exchangeability assumption is additional, not guaranteed by these experiments.
- All inference is conditional on these datasets, saved histories, generator settings, labelers and shared data splits. Seeds do not represent newly sampled datasets. No best labeler or best test epoch is selected for analysis. No claim of equivalence or full experiment-matrix completion follows.
- Later input hashes do not establish training-time bytes. Original archive caveats and source records remain authoritative. Eight early validation-best records did not preserve execution-source hashes, as documented in the archive.

## Sources, coverage and reproducibility

Read-only source archive: `{manifest['archive']}`. The current frozen selection contains 602 validation-best records: 420 classification generated/hybrid evaluations, 168 regression generated/hybrid evaluations, and 14 real-only references. Real-only references are retained in frozen inputs but excluded from these requested contrasts. Credit and bonus NB/PCA-GMM arms are outside this balanced validation-best selection; they are not substituted from older protocols.

The original-checkpoint sensitivity uses the 602 matching original records. All 1,204 small JSON records were hashed. {manifest['verified_score_values']} nonempty validation-best score cells were verified against the archived result JSONs, along with saved checkpoint-tensor checks, test IDs/targets checks, and News normalization. The complete grid was verified before fitting; there is no imputation. The archive and existing reports were not modified. Numerical analysis ran remotely using one CPU thread; no experiments were trained or resumed.

Frozen input hashes and provenance: [inputs/manifest.json](../inputs/manifest.json), [record hashes](../inputs/record_hashes.csv), [archive guide](../inputs/valbest.md). Reproducible analysis: [analyze.py](../analyze.py). Environment and output hashes: [validation.json](validation.json).

Remote execution namespace: `.cache/anova_gdrive_20261019_0009/` under the research repository, with `analysis.log`. Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python analyze.py`. This is a short score-analysis job and uses no GPU.

Method references: [Statsmodels repeated-measures ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html), [Holm adjustment](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html), [SciPy paired t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
'''
(OUT/'report.md').write_text(text, encoding='utf-8')
# Render the report's limited Markdown vocabulary with the standard library.
def inline(value):
    value = html.escape(value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', value)

blocks = []
for block in text.strip().split('\n\n'):
    if block.startswith('|'):
        lines = block.splitlines()
        header = '<tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in lines[0].strip('|').split('|'))+'</tr>'
        body = ''.join('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>' for line in lines[2:])
        blocks.append('<table>'+header+body+'</table>')
    elif block.startswith('## '):
        blocks.append('<h2>'+inline(block[3:])+'</h2>')
    elif block.startswith('# '):
        blocks.append('<h1>'+inline(block[2:])+'</h1>')
    elif block.startswith('- '):
        blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in block.splitlines())+'</ul>')
    else:
        blocks.append('<p>'+inline(block)+'</p>')
document = '<!doctype html><meta charset="utf-8"><title>G drive ANOVA</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1180px;margin:42px auto;padding:0 24px;color:#203040}h1,h2{line-height:1.25}h2{margin-top:2em}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}th,td{padding:7px 10px;border-bottom:1px solid #d9e0e5;text-align:right;white-space:nowrap}th{background:#edf3f6}th:first-child,td:first-child{text-align:left}a{color:#166783}code{font-size:13px}img{max-width:100%}</style>' + ''.join(blocks)
document = document.replace('<a href="effects.png">the effect plot</a>', '<a href="effects.png">the effect plot</a><p><img src="effects.png" alt="Dataset paired effects and task mean intervals"></p>')
(OUT/'report.html').write_text(document, encoding='utf-8')
validation = dict(completed_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__),
                  frozen_input_hashes_verified=True, complete_grid=True, valbest_records=602, original_records=602,
                  independent_omnibus_F_checks=len(independent), paired_F_equals_t_squared=True,
                  primary_tests=4, primary_holm_family=4, secondary_generator_holm_family=12,
                  classification_independent_units=5, regression_independent_units=2,
                  output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != 'validation.json'})
(OUT/'validation.json').write_text(json.dumps(validation, indent=2))
print(primary.to_string(index=False))
print('\nGENERATOR CONTRASTS\n'+secondary[['mode','generator','question','difference','p','p_holm']].to_string(index=False))
print('\nREGRESSION\n'+regression.to_string(index=False))
print('\nCompleted:', validation['completed_at_chicago'])
