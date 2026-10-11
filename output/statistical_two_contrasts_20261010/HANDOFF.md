# Statistical redo: two research contrasts

Requested scope: Adult, Census KDD, Covertype, MNIST12, MNIST28, News and
California Housing; CTGAN/TVAE; RF/XGB/DNN; seeds 42/43. Synthetic-only and mix
remain separate. The two questions are relabeled versus generated targets and
has_y=1 versus has_y=0 among relabeled runs.

The current focused report is
[the six-page PDF](../../output/pdf/statistical_two_contrasts_20261010/real_data_two_contrasts.pdf),
with [editable Markdown](../../output/statistical_two_contrasts_20261010/analysis.md),
[HTML](../../output/statistical_two_contrasts_20261010/analysis.html) and the
[reproducible package](../../output/statistical_two_contrasts_20261010/two_contrast_analysis_package.zip).
Previous 328 deliverables remain unchanged and retain their historical four-
contrast families. The new report does not overwrite them.

## Method and results

Use macro F1 for all five classification tasks. Average seeds, generators and
labelers within each task before paired inference on five task differences
(two-sided t, df=4). Keep generated baselines matched to dataset/seed/generator;
baseline reuse is not independent replication. Holm2 covers the two questions
within one training regime. Holm4 covers claims across both regimes.

| Training | Comparison | Macro-F1 change (points) | Raw p | Holm2 p | Holm4 p |
| --- | --- | --- | --- | --- | --- |
| Synthetic-only | Relabeled - generated | +11.6319 | .034977 | .069953 | .139907 |
| Synthetic-only | has_y=1 - 0 | +.3573 | .457451 | .457451 | .914902 |
| Mix | Relabeled - generated | +6.7729 | .056598 | .113197 | .169795 |
| Mix | has_y=1 - 0 | +.1380 | .718236 | .718236 | .914902 |

Exact sign-flip and exact Wilcoxon relabeling p-values are .0625 for both regimes.
The prior approximate Wilcoxon p=.0431 is not the exact five-task result. The
report includes pointwise 95% t intervals and Bonferroni simultaneous 95%
intervals for the stated two- and four-test families; these are not Holm intervals.

News/Housing contrasts are separate descriptive native-unit differences for
R2, NMAE and D2. Housing improves all three; News improves R2 but worsens NMAE
and D2. No mixed-metric seven-task score is formed. Eighteen categorical Type II
ANOVA models were refitted as nominal exploratory factor checks; all 202 F
statistics were independently reproduced by restricted least squares.

Two questions were selected after prior analysis, so this remains exploratory.
No one-sided conversion, newly claimed preregistration or extra baseline-specific
significance tests were introduced. Five selected tasks, related MNIST tasks,
shared heldout partitions and documented MNIST12 TVAE feature collisions limit
task-generalization inference. No detected has_y effect is not equivalence.

## Provenance and execution

All 406 records and 116 D2 sidecars matched remote hashes at October 10, 2026,
11:35:57 p.m. Chicago. All 3,770 frozen source metrics were compared with saved
run records. News/Housing saved test normalization was checked; normalization
and D2 denominators were not recomputed.

Remote analysis namespace:
`/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_two_contrasts_20261010`.
Command: `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -`.
Local launcher: `remote_io.py`; local log: `remote_analysis.log`.
Input/output hashes and exact command are in `deployment.json`.

The four active TabDDPM queues, processes, resource use and latest logs were
inspected before this isolated one-thread refit. No experiment was launched,
interrupted, retrained, redeployed or modified. No statistics ran locally.

## Validation

`verify_report.py` checks frozen input hashes, 1,056 exported matched score
values, 44 task contrasts, 88 seed contrasts, four headline tests, eighteen model
ranks, all 202 independently reproduced F statistics, report table contents and
PDF page bounds. All six rendered pages were visually reviewed. Every file in
the previous deliverable manifest still matches its saved hash.

Rebuild the numerical analysis remotely with `remote_io.py`; it refuses to
overwrite a differing remote input or script. Build presentation locally with
`build_report.py`, render with Poppler, review every page, then run
`verify_report.py --reviewed` to verify and package the deliverables. Do not use
that flag until visual review has passed.
