"""Prepare lightweight planning artifacts from the read-only remote inventory."""
import hashlib
import json
from pathlib import Path

ROOT = Path('D:/SummerResearch')
AUDIT = ROOT / 'audit/experiment_plan_20261007'
OUT = ROOT / 'output/experiment_plan_20261007'
OUT.mkdir(parents=True, exist_ok=True)
inventory = json.loads((AUDIT / 'inventory_parallel_recheck.json').read_text(encoding='utf-8-sig'))
supplement = json.loads((AUDIT / 'supplement.json').read_text(encoding='utf-8-sig'))
groups = inventory['groups']
historical = [g for g in groups if g['dataset'] != 'california_housing']
assert sum(g['completed_selected'] for g in historical) == 296
assert sum(len(g['missing']) for g in historical) == 168
primary = [g for g in groups if g['dataset'] in ('adult', 'census_kdd', 'covertype', 'mnist12')]
assert sum(len(g['missing']) for g in primary) == 53
for g in groups:
    assert g['completed_selected'] + len(g['missing']) == 29
    assert len(set(g['complete']) | {r['run_id'] for r in g['missing']}) == 29
    assert len({r['run_id'] for r in g['missing']}) == len(g['missing'])

names = {'adult': 'Adult', 'census_kdd': 'Census KDD (weighted)', 'credit': 'Credit fraud', 'covertype': 'Covertype', 'intrusion': 'Intrusion', 'mnist12': 'MNIST12', 'mnist28': 'MNIST28', 'news': 'News', 'california_housing': 'California Housing'}
summary = ['| Dataset | Seed 42 completed / 29 | Seed 43 completed / 29 | Missing | Proposed action |', '| --- | ---: | ---: | ---: | --- |']
actions = {'adult': 'Keep completed results', 'census_kdd': 'Finish 38 downstream evaluations', 'credit': 'Archive outside primary comparison', 'covertype': 'Keep completed results', 'intrusion': 'Defer 57 evaluations and unfinished generation', 'mnist12': 'Finish 15; main digit task for economy', 'mnist28': 'Keep seed 42 as resolution sensitivity; defer seed 43', 'news': 'Keep seed 42 as disclosed regression limitation; defer seed 43', 'california_housing': 'New regression study; not an interrupted run'}
for dataset, name in names.items():
    a, b = [g for g in groups if g['dataset'] == dataset]
    summary.append(f"| {name} | {a['completed_selected']} | {b['completed_selected']} | {len(a['missing']) + len(b['missing'])} | {actions[dataset]} |")

details = ['# Every unfinished selected real-data configuration', '', f"Checked remotely: {inventory['checked_at_chicago']} (Chicago).", '', 'Scope: original real-only student, CTGAN/TVAE generated targets, and RF/XGB/DNN full-table and X-only hybrids. Both synthetic-only and mixed training are included. Each dataset/seed has 29 configurations: one real-only + two generators × (one baseline + six hybrids) × two modes. NB and PCA/GMM are excluded.', '', 'Among the eight historical datasets, 296 configurations have final records, a selected development epoch, and existing predictions/weights; 168 lack that completion evidence. California Housing adds 58 proposed configurations with no production results, listed separately below. Presence checks do not independently rescore every prediction or rehash large checkpoints.', '', 'A missing record is not necessarily an interrupted training run. Log existence establishes an attempted command, not how far it progressed. Empty logs do not establish substantive training. Saved generator fits/tables can exist even when downstream training never started.', '']
details += summary
for g in groups:
    if not g['missing']:
        continue
    details += ['', f"## {names[g['dataset']]} — seed {g['seed']}", '', f"Namespace: `{g['namespace']}`. Completed {g['completed_selected']}/29; missing {len(g['missing'])}.", '', '| Run ID | Evidence of attempt | Input table present |', '| --- | --- | --- |']
    for row in g['missing']:
        run = row['run_id']
        state = {'attempted_no_completed_record': 'Log exists; no final record', 'no_attempt_log_in_selected_namespace': 'No matching attempt log found', 'record_incomplete': 'Final record/artifacts incomplete'}[row['state']]
        if row['state'] == 'attempted_no_completed_record' and not row['log_tail']:
            state = 'Empty log; no substantive progress established'
        table = 'Real split' if run.endswith('real_original') else 'Yes' if g['tables_present_for_run'][run] else 'No'
        details.append(f'| `{run}` | {state} | {table} |')
details += ['', '## Completed simulated and pilot collections', '', '- `simulated_all_20260930`: 378/378 production rows, seven datasets × two seeds × 27 methods. RF/XGB/DNN plus baselines are 210/210 rows. No unfinished production cells in this collection; the CSV hash matches the statistical report snapshot.', '- `new_simulated_eval`: older, separate single-process simulation collection, 54 rows for two seeds. Do not add it to the seven-dataset benchmark as independent evidence without a separate provenance/design comparison.', '- `new_simulated_smoke`, `simulated_all_smoke`, `simulated_all_verified`, `simulated_insurance_smoke`: seed-7, one-epoch checks, not production results or a production backlog.', '- Census weighted pilot: three completed runs; their validated imported records are part of the weighted Census matrix, not extra independent observations.', '- Credit weighted pilot: all three runs completed October 6 at 12:34 p.m. Chicago; no remaining pilot arm.', '- Tab-DDPM: no production run records or saved production bundles found. Implementation/smoke work is not a half-completed production comparison.', '', 'Detailed log tails, source record hashes, parameters, partial artifacts and queue/process evidence are saved in `audit/experiment_plan_20261007/inventory.json` and `supplement.json`.']
(OUT / 'unfinished_experiments.md').write_text('\n'.join(details) + '\n', encoding='utf-8')

plan = r'''# Next experiments after the statistical review

Revised October 7, 2026, after a fresh remote check at 2:02 p.m. Chicago. The user excluded additional controls and requested three parallel tmux queues. This is a proposal and read-only audit; no training, generation, restart, deployment, deletion or experiment-output change was performed.

**Recommendation: finish the 53 recoverable core evaluations in three parallel tmux queues, then test on new independent datasets. Add a small Tab-DDPM extension afterward.** No extra teacher/resampling controls are included in the plan. Do not restore every historical queue merely to complete the original matrix.

The existing results support predictive utility gains against original CTGAN/TVAE, especially CTGAN. They do not establish universal superiority, superiority to the real-only student, or improved distribution fidelity. Full versus X-only hybrid differences are small on average, so expanding the inclusion factor is a lower priority than controls and broader task coverage.

## What the statistics change about the next queue

- RF/XGB/DNN hybrids gain about 9.6 macro-F1 points on the four matched real tasks and 5.3 accuracy points on seven simulated tasks. These are family averages, not ensembles or selection of the best labeler per test set.
- Simulated approach ANOVA survives correction; the real approach's corrected p is 0.0589 in a balanced ANOVA with only three datasets/five dataset-seed units. Missing cells are restricting that analysis substantially.
- Including Y during generator fitting has no demonstrated average advantage: full minus X-only is approximately +0.08 real macro-F1 points and +0.12 simulated accuracy points. This does not establish equivalence. Keep a paired inclusion ablation, but use full-table relabeling as the simplest main construction.
- The current four real task means give a minimum two-sided exact sign-flip p of 0.125, even if all four improve. More method rows or training seeds cannot change that task count. Adding regression results does not enlarge a classification macro-F1 test.
- The immediate claim remains hybrid-versus-original-generator utility under matched evaluation. The user has excluded additional controls from the next-run scope.

These are exploratory findings from the frozen October 6 snapshot. The RF/XGB/DNN restriction and a DNN primary arm were chosen after examining results; prospective follow-up choices must be disclosed as such.

## Completion inventory

COUNTS_TABLE

Counts include the real-only baseline and both downstream training modes. The historical RF/XGB/DNN scope is 464 evaluations across eight datasets/two seeds, with 296 complete and 168 incomplete. These are not the 816 all-labeler historical matrix counts. California Housing is an additional planned dataset; its 58 selected evaluations are not interrupted historical results.

Adult, Credit and Covertype are complete at both seeds in the requested family. MNIST12, MNIST28 and News seed 42 are complete. Intrusion has only the seed-43 real-only reference. Older unweighted Census cannot fill gaps in the weighted comparison.

## First batch: 53 missing core evaluations, no new generator fits

Use Adult, weighted Census KDD, Covertype and **MNIST12** as the economical complete classification benchmark, seeds 42/43. Preserve MNIST28 seed 42 as a resolution sensitivity check. The two digit resolutions share a source and do not count as independent datasets. This scope choice is based on cost and saved artifacts; retain the earlier MNIST28 statistical results and disclose the change. If MNIST28 must remain the main task, the corresponding completion batch is **67**, including 29 seed-43 evaluations and four new generator fits instead of MNIST12's 15 evaluations.

1. **Weighted Census seed 42: nine evaluations.** TVAE full+RF, full+XGB and full+DNN under synthetic and mix training (six); TVAE X-only+XGB mix (one); TVAE X-only+DNN synthetic and mix (two). CTGAN is complete. Both TVAE generated-target baselines and X-only RF are complete.
2. **Weighted Census seed 43: all 29 evaluations.** One weighted real-only reference; each generator's original-target baseline and RF/XGB/DNN full/X-only hybrids, under both modes. All four generator fits and all required synthetic tables are already present with provenance. Revalidate their saved input hashes and protocol before reuse; no generator refitting is needed solely because the downstream queue stopped.
3. **MNIST12 seed 43: 15 evaluations.** One real-only baseline plus all 14 TVAE evaluations. All 14 CTGAN evaluations are complete, including full+DNN mix. All four generator fits are saved. TVAE full+RF and full+DNN tables remain unfinished; generate those labels on the saved full-table features, reusing the saved RF predictor where compatible and completing the DNN labeler as necessary. Do not refit or resample the full generator simply to recover those tables. Its full+XGB and X-only RF/XGB/DNN tables already exist.

This yields 232 real evaluation records across four datasets/two seeds (224 synthetic/mix cells plus eight real-only references). It improves balanced factorial coverage but still represents four tasks on fixed held-out splits. Do not equate completion with independent external replication.

The useful earliest subset is the six Census seed-42 full-table TVAE relabeling evaluations: they complete the decisive generated-target versus relabeled-target comparison for that seed. Finish the remaining inclusion cells rather than declaring equivalence from a nonsignificant effect.

## Three parallel tmux queues for the first batch

| Proposed session | Assigned work | Missing downstream evaluations |
| --- | --- | ---: |
| `finish-census-synthetic` | Census seeds 42/43, missing synthetic-only arms; weighted seed-43 real-only reference | 19 |
| `finish-census-mix` | Census seeds 42/43, missing mixed arms | 19 |
| `finish-mnist12-43` | Finish TVAE full RF/DNN relabeling tables, then MNIST12 seed-43 real-only and all TVAE evaluations | 15 |

All three sessions share the single GTX 1080 Ti on CUDA device 0. Run one GPU job per session; cap CPU/BLAS threads at two per worker. This is a plausible starting concurrency for the comparatively small downstream networks and can hide CPU/loading downtime; three sessions do not guarantee higher throughput. Inspect actual utilization, combined RAM/VRAM and progress after startup. If contention lowers throughput or available memory becomes inadequate, drain a worker at a configuration boundary and continue the remaining queue without changing scientific settings. Do not interrupt completed or running configurations merely to rebalance job counts.

Each configuration belongs to exactly one session. The MNIST preparation stage belongs only to the third session; Census tables are read-only shared inputs. Preserve old failure logs and use new per-attempt log paths. Recheck valid completion records before each job, skip completed configurations, and surface actual failures without automatic retries. Independent sessions can continue their own jobs if another session fails.

The accompanying `parallel_queues.json` contains all 53 unique run IDs and structured downstream command arguments. These are launch plans, not already running jobs. MNIST's table preparation and provenance checks must be completed before its dependent jobs. Census uses the weighted single-configuration entry point, not the broad 106-job all-labeler `--resume` queue.

## Use the completed simulation to explain the mechanism

The seven-dataset production benchmark is complete: 378 rows, or 210 rows after restricting to RF/XGB/DNN and the three references per block. Do not restart it.

Using saved tables, assess generated versus RF/XGB/DNN targets against the oracle on the **same generated full-table X**: Bayes-label agreement and oracle expected target correctness, P(Y=assigned label | X). Keep this distinct from downstream Bayes agreement. For zero-probability feature configurations, report their frequency separately; an oracle conditional is undefined there, not an invitation to invent one.

Pair these diagnostics with support violations, L_syn, L_test and downstream utility. Hard relabeling may improve decision information while removing genuine target noise; it cannot repair absent feature modes. The existing Insurance support violations and utility/fidelity tradeoff must remain visible. This work needs remote scoring, not new generator fits, and should write a new diagnostic artifact rather than overwrite the benchmark.

A probabilistic-target/noise-preserving experiment is a later option if the paper specifically claims better joint fidelity. It is unnecessary for the first predictive-utility follow-up.

## Independent datasets next; additional seeds afterward

For the classification claim, add **two compact, independent tabular tasks** before spending heavily on more model seeds. Suitable prospective choices are UCI Default of Credit Card Clients (30,000 rows, 23 features; exclude ID and specify categorical roles) and UCI Spambase (4,601 rows, 57 features). Default prediction is different from the current fraud Credit task. Verify minority support and feature-group split feasibility before evaluating methods; do not select tasks based on observed hybrid gains. [UCI Default](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients), [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase).

For each new dataset, use seeds 42/43 and synthetic-only evaluation first: one real-only reference + two generators × (one generated-target baseline + RF/XGB/DNN full-table hybrids + RF/XGB/DNN X-only hybrids) = **15 evaluations per seed, 30 per dataset, 60 total**, and four generator fits per seed/eight per dataset. Keep 500 epochs, batch size 500 and 100,000 rows. Put mixed training second if augmentation is a central claim. Freeze metrics and a DNN full-table primary contrast before obtaining these new test results; report all RF/XGB/DNN results and the separately identified X-only ablation. New-dataset preprocessing/evaluator support needs a reviewed implementation; it is not already production-ready.

Two additional datasets improve task coverage, not a guaranteed significant result. Six task means have a minimum raw exact p of 0.03125; multiplicity and task dependence can still prevent significance. Report effects and uncertainty rather than designing for a threshold. Existing exploratory datasets and genuinely new follow-up data must remain distinguishable.

California Housing is useful **if the paper includes regression**: its source support was validated in an isolated staging directory, but no production fits or results exist. A matching full RF/XGB/DNN two-mode design would add 58 evaluations and eight generator fits across two seeds. Analyze regression separately, preserving both R2 and saved normalized MAE. News's current normalized-MAE deterioration is a limitation, not a reason to discard it and retain only favorable regression results. Prioritize new classification tasks above Housing for a classification-generalization claim.

After task coverage, allocate extra repeats to frozen primary arms. Extra downstream seeds on fixed generated tables estimate student-training variability; new generator seeds estimate pipeline variability; neither creates a new dataset or held-out split. More independent data splits require a separate prospective split protocol. A seed-42 generator can support a separately named fixed-generator downstream-seed sensitivity, but cannot fill a missing seed-43 generator fit in the current matrix. Pair comparisons on saved predictions and respect feature-group dependence when estimating holdout uncertainty. No count of rows/methods substitutes for task-level evidence.

## Tab-DDPM: yes, a bounded extension after core completion

No production Tab-DDPM run records or saved production model bundles were found. The adapter/smoke work is implementation evidence, not utility evidence.

Start on Adult and Covertype, seeds 42/43, with native class-conditioned Tab-DDPM. Retain its assigned conditioning labels for the native baseline, then replace targets with RF/XGB/DNN predictions on **exactly the same sampled X**. With synthetic-only evaluation, this is **16 new student runs**: two datasets × two seeds × four constructions. Real-only student references already exist. Fit one diffusion model per dataset/seed (four fits). Add mix only if testing augmentation; that adds another 16 student runs. [Official configuration](https://github.com/yandex-research/tab-ddpm/blob/main/CONFIG_DESCRIPTION.md) explicitly uses target conditioning for classification; [original paper](https://proceedings.mlr.press/v202/kotelnikov23a.html).

The current unconditional joint adapter answers a different, useful matched-factorization question; it is not a reproduction of the native classification baseline. It needs adaptation/validation for native conditioning before that study. If using the existing unconditional adapter first, label it explicitly and do not claim superiority to published/native Tab-DDPM. Preserve training/development-only model selection and the same downstream dataset protocol across generators.

Do not add an X-only diffusion fit in the first extension unless portability of the inclusion factor is part of the claim. Freeze optimizer-step, sampling and resource budgets from training/development evidence; diffusion timesteps and optimizer steps are different, and neither is directly equivalent to CTGAN's 500 epochs. Compare wall time/storage as well as utility. No automatic full 436-evaluation extension.

## What to leave unfinished

- **Credit production matrix is complete**, and the three weighted pilots also finished. Archive it outside primary inference because the test set has only ten positives. Weighting fixed all-negative real-only and X-only predictions but does not increase independent evaluation support. A new fraud split is a new protocol, not a retrospective replacement of this test set.
- **Intrusion:** defer. Seed-42 CTGAN's old log reached about 296/500 epochs after more than 70 hours, but no completed generator fit/provenance or synthetic table is saved. TVAE also failed, including a documented disk-full download error. Seed 43 has the real-only student and attempted generators, but no completed generator fits. Do not assume training can resume from the last log epoch; optimizer/RNG/progress checkpoints would need verification. There are 57 selected downstream evaluations missing across both seeds.
- **MNIST28 seed 43:** defer if MNIST12 is the main digit benchmark. No completed fits, tables or student records were found; empty logs are not partial fit evidence. Its 29 evaluations and four generator fits are a new substantive block.
- **News seed 43:** defer pending the regression claim and diagnosis. Seed 42 is complete. No seed-43 fits, tables or final records were found; 29 selected evaluations remain.
- **NB/PCA-GMM:** retain archived evidence but omit new runs, as requested. The earlier all-labeler completion target is not the new stopping rule.

## Operational prerequisites and restart limitations

At the fresh 2:02 p.m. check, no research training/generation process was active. Existing tmux panes were dead or displaying logs/resources; their session names do not establish a live job. Storage had changed since the previous inspection: the root filesystem was now 50% used, with approximately 438 GB available. Around 27 GiB RAM was available; GPU use was approximately 244 MiB of 11,264 MiB. The current inventory reconfirmed the saved fits/tables and all completed records' referenced weights/predictions. This audit did not perform any cleanup or establish what caused the storage change.

Current storage headroom no longer blocks this proposed completion batch; estimate its declared artifact growth and preserve completed evidence. Several historical logs explicitly report `No space left on device`; other failures contain truncated import traces. Verify the actual failing command instead of presuming a broken environment or reinstalling packages. Installed versions were read as torch 2.5.1, SDV 1.18.0, CTGAN 0.10.2, scikit-learn 1.5.2, NumPy 1.26.4 and pandas 2.2.3; no training compatibility test was run for this plan.

The existing classifiers `--resume` skips verified completed records; it does not resume a half-trained optimizer. Census `--resume` still schedules the full 106-arm all-labeler matrix and opens attempt logs exclusively, so existing failed log files can block a restart. Prepare a targeted RF/XGB/DNN queue with preserved old logs/new attempt logs. Reuse a selected checkpoint to finish evaluation only after its provenance, checkpoint-selection history and finalization path are verified. A lone `.weights.pth` does not establish a complete resumable training state.

Keep every dataset's original protocol and immutable completed records. Weighted Census remains its distinct namespace and matches weighted reference/arms; do not replace missing cells with unweighted scores. Maintain the original budgets, feature-group partitions, training-fitted transforms and saved News normalization. All execution belongs on the research server. Inspect live jobs again immediately before any future launch.

## Writing and stopping criterion

Write methods, known-distribution reasoning, observed utility results, and limitations now. The target claim is: **supervised target replacement improves predictive utility relative to generated targets under the evaluated protocols, with limits set by teacher quality and generated-feature coverage**. More specific generator and dataset differences are part of the result. Distributional fidelity and superiority over real-only training require their own evidence.

The first executable follow-up is **53 completion evaluations in three sessions**, with no extra controls or new CTGAN/TVAE fits, although the missing MNIST12 DNN labeler may need training. Saved-simulation diagnostics can be a separate later analysis. The following coverage batch adds 60 synthetic-only evaluations and **16 generator fits total** (two datasets × two seeds × four fits). The bounded native Tab-DDPM extension adds 16 student evaluations/four fits later. These are staged proposals, not an authorized launch or a requirement to run every stage before writing.

Reassess the claim after core completion and new-dataset results, keeping failures and non-improvements visible. Completion of a defensible, scoped study is the stopping rule; statistical significance or victory on every method is not.

Evidence: `audit/experiment_plan_20261007/inventory_parallel_recheck.json`, the earlier `inventory.json` and `supplement.json`, the frozen `output/statistical_review_20261006/statistical_review.md`, `audit/COMPARISON_HANDOFF.md`, `audit/CREDIT_WEIGHTED_PILOT_2026_10_06.md`, and `audit/california_housing_support_validation_2026_10_04.json`. Every missing selected run ID is listed in the accompanying `unfinished_experiments.md`.
'''
plan = plan.replace('COUNTS_TABLE', '\n'.join(summary))
(OUT / 'next_experiments.md').write_text(plan, encoding='utf-8')
queues = [{'session': 'finish-census-synthetic', 'jobs': []}, {'session': 'finish-census-mix', 'jobs': []}, {'session': 'finish-mnist12-43', 'prerequisites': ['Verify the saved TVAE full generator and baseline table against training/provenance.', 'Finish only missing TVAE full RF and DNN relabeling tables; preserve the existing RF predictor and reuse it if compatible. No generator refit or resampling.', 'Verify all tables, predictors and quality metadata required by downstream records.'], 'jobs': []}]
remote_root = '/home/thuy/Research/minh_data_synth/TabularDA'
python = '/home/thuy/miniconda3/envs/env/bin/python'
for g in primary:
    for row in g['missing']:
        run_id = row['run_id']
        original = run_id.endswith('real_original')
        mode = 'original' if original else run_id.rsplit('_', 1)[1]
        method = None
        if not original:
            generator, fit, label = run_id.removeprefix(f"{g['dataset']}_seed{g['seed']}_").rsplit('_', 1)[0].split('_')
            method = generator if label == 'generated' else ('tvae_' if generator == 'tvae' else '') + ('compare_' if fit == 'full' else '') + label
        environment = {'CORRECTED_RUN_NAMESPACE': 'corrected_v2', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '0'}
        if g['dataset'] == 'census_kdd':
            queue = queues[1 if mode == 'mix' else 0]
            command = [python, '-u', 'scripts/run_census_weighted_matrix.py', '--seed', str(g['seed']), '--mode', mode]
            if method is not None:
                command += ['--method', method]
            cwd = remote_root
        else:
            queue = queues[2]
            command = [python, '-u', '-m', 'modeling', '--dataset-name', g['dataset'], '--train-option', mode, '--test-option', 'original', '--validation', '0.2', '--batchsize', '128', '--lr', '0.001', '--global-round', '100', '--patience', '30', '--early-stop-crit', 'loss', '--seed', str(g['seed'])]
            if method is not None:
                command += ['--augment-option', method]
            cwd = remote_root + '/src'
        queue['jobs'].append({'run_id': run_id, 'dataset': g['dataset'], 'seed': g['seed'], 'train_option': mode, 'augment_option': method, 'output_namespace': g['namespace'], 'cwd': cwd, 'environment': environment, 'argv': command, 'new_log_path': f".cache/core_completion_20261007/{queue['session']}/{run_id}.attempt1.log", 'execution_policy': 'Check and skip verified completed records; preserve earlier logs/artifacts; stop this session on failure; no automatic retry.'})
assert [len(q['jobs']) for q in queues] == [19, 19, 15]
run_ids = [j['run_id'] for q in queues for j in q['jobs']]
assert len(run_ids) == len(set(run_ids)) == 53
assert set(run_ids) == {r['run_id'] for g in primary for r in g['missing']}
(OUT / 'parallel_queues.json').write_text(json.dumps({'status': 'planned_not_launched', 'checked_at_chicago': inventory['checked_at_chicago'], 'controls': False, 'gpu': 'one GTX 1080 Ti, CUDA 0 shared by three sessions', 'queues': queues}, indent=2) + '\n', encoding='utf-8')
manifest = {'checked_at_chicago': inventory['checked_at_chicago'], 'historical_selected_complete': 296, 'historical_selected_missing': 168, 'first_completion_batch': 53, 'control_student_runs': 0, 'parallel_sessions': 3, 'sha256': {}}
for path in (AUDIT / 'inventory.json', AUDIT / 'inventory_parallel_recheck.json', AUDIT / 'supplement.json', OUT / 'next_experiments.md', OUT / 'unfinished_experiments.md', OUT / 'parallel_queues.json'):
    manifest['sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('Wrote plan and exhaustive unfinished-run list; checked 296 complete / 168 missing historical selected runs and 53 core completion jobs.')
