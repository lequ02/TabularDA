"""Prepare the full retained slow/fast schedule; does not launch experiments."""
import hashlib
import json
from pathlib import Path

ROOT = Path('D:/SummerResearch')
AUDIT = ROOT / 'audit/experiment_plan_20261007'
OUT = ROOT / 'output/experiment_plan_20261007'
inventory = json.loads((AUDIT / 'inventory_full_schedule.json').read_text(encoding='utf-8-sig'))
groups = inventory['groups']
retained = [g for g in groups if g['dataset'] != 'intrusion']
assert sum(g['completed_selected'] for g in retained) == 295
assert sum(len(g['missing']) for g in retained) == 169
lanes = [
    {'session': 'research-slow', 'role': 'Long generation and the MNIST28 seed-43 pipeline', 'blocks': [('mnist28', 43)]},
    {'session': 'research-fast-1', 'role': 'Close weighted Census, then Housing seed 42', 'blocks': [('census_kdd', 42), ('census_kdd', 43), ('california_housing', 42)]},
    {'session': 'research-fast-2', 'role': 'Close MNIST12, then News seed 43 and Housing seed 43', 'blocks': [('mnist12', 43), ('news', 43), ('california_housing', 43)]},
]
python = '/home/thuy/miniconda3/envs/env/bin/python'
remote = '/home/thuy/Research/minh_data_synth/TabularDA'
blocks = []
jobs = []
for lane in lanes:
    lane['jobs'] = []
    for dataset, seed in lane['blocks']:
        g = next(g for g in groups if (g['dataset'], g['seed']) == (dataset, seed))
        needed_fits = [{'generator': f['generator'], 'fit': f['fit']} for f in g['fits'] if not (f['model_present'] and f['provenance_present'])]
        block = {'dataset': dataset, 'seed': seed, 'default_session': lane['session'], 'missing_evaluations': len(g['missing']), 'input_namespace': 'corrected_v2', 'output_namespace': g['namespace'], 'new_generator_fits': needed_fits, 'split_manifest_present': g['split_manifest_present'], 'prerequisites': []}
        if dataset == 'census_kdd':
            block['prerequisites'] = ['Validate existing full/X-only generator provenance and all RF/XGB/DNN input tables against training hashes.', 'Keep the weighted loss/macro-F1-selection protocol and its separate output namespace. No new generator or labeler fits are scheduled.']
        elif dataset == 'mnist12':
            block['prerequisites'] = ['Validate all four saved 500-epoch fits and existing input tables.', 'Finish only TVAE full-table RF and DNN relabeling tables on saved full-table X. Preserve/reuse the saved RF predictor if compatible; complete the missing DNN labeler.']
        else:
            block['prerequisites'] = ['Prepare and save the original protocol train/dev/test partitions and training-fitted preprocessing once for this dataset/seed; preserve fixed split state 42.', 'Fit CTGAN-full, TVAE-full, CTGAN-X-only and TVAE-X-only at 500 epochs, batch 500, CUDA; sample 100000 rows per source.', 'Full-table generated-target baselines retain Y; full hybrids share exactly those saved X rows. Label each source with RF/XGB/DNN only; save required quality/predictor/provenance metadata.', 'Publish readiness per table, not only after the whole generation pipeline completes. Real-only evaluation needs prepared real partitions, not a generator fit.']
        blocks.append(block)
        for row in g['missing']:
            run_id = row['run_id']
            original = run_id.endswith('real_original')
            mode = 'original' if original else run_id.rsplit('_', 1)[1]
            generator = fit = label = None
            if original:
                method = None
            else:
                generator, fit, label = run_id.removeprefix(f'{dataset}_seed{seed}_').rsplit('_', 1)[0].split('_')
                method = generator if label == 'generated' else ('tvae_' if generator == 'tvae' else '') + ('compare_' if fit == 'full' else '') + label
            if dataset == 'census_kdd':
                argv = [python, '-u', 'scripts/run_census_weighted_matrix.py', '--seed', str(seed), '--mode', mode]
                cwd = remote
                if method is not None:
                    argv += ['--method', method]
            else:
                argv = [python, '-u', '-m', 'modeling', '--dataset-name', dataset, '--train-option', mode, '--test-option', 'original', '--validation', '0.2', '--batchsize', '128', '--lr', '0.001', '--global-round', '100', '--patience', '30', '--early-stop-crit', 'loss', '--seed', str(seed)]
                cwd = remote + '/src'
                if method is not None:
                    argv += ['--augment-option', method]
            closure_order = {'real': 0, 'generated': 1, 'dnn': 2, 'rf': 3, 'xgb': 4}
            job = {'run_id': run_id, 'default_session': lane['session'], 'dataset': dataset, 'seed': seed, 'generator': generator, 'fit': fit, 'labeler': label, 'train_option': mode, 'augment_option': method, 'input_namespace': 'corrected_v2', 'output_namespace': g['namespace'], 'ready_condition': 'Verified prepared real partitions' if original else 'Verified exact input table plus generator/predictor/provenance/quality dependencies', 'completion_priority': closure_order[label or 'real'], 'cwd': cwd, 'environment': {'CORRECTED_RUN_NAMESPACE': 'corrected_v2', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '0'}, 'argv': argv, 'new_log_path': f'.cache/full_completion_20261007/{run_id}.attempt1.log'}
            jobs.append(job)
            lane['jobs'].append(job)

ids = [j['run_id'] for j in jobs]
assert len(ids) == len(set(ids)) == 169
assert set(ids) == {r['run_id'] for g in retained for r in g['missing']}
assert [len(l['jobs']) for l in lanes] == [29, 67, 73]
assert sum(len(b['new_generator_fits']) for b in blocks) == 16
assert all(j['dataset'] != 'intrusion' for j in jobs)
assert all(j['labeler'] in (None, 'generated', 'rf', 'xgb', 'dnn') for j in jobs)
schedule = {'status': 'planned_not_launched', 'checked_at_chicago': inventory['checked_at_chicago'], 'scope': 'Finish CTGAN/TVAE and California Housing only; Intrusion dropped; no controls/NB/PCA-GMM; new classification datasets and Tab-DDPM deferred.', 'required_downstream_runs': 169, 'required_new_generator_fits': 16, 'retained_completion_target': 464, 'retained_current_complete': 295, 'blocks': blocks, 'queues': lanes, 'scheduling_policy': ['Start all three sessions together: MNIST28 generation, weighted Census seed-42 completion, and MNIST12 seed-43 completion.', 'Use the listed block order as the default schedule. Keep one long high-dimensional generator fit active at most; News and Housing are lower-dimensional pipelines, not zero-cost work.', 'Close full per-dataset/seed factorial blocks (both generators, both input sources, all three labelers, both training modes) before spreading evaluations across later blocks.', 'Release each verified generated/relabeled table immediately so its two downstream modes need not wait for unrelated labelers or generator fits.', 'If a fast worker exhausts its default ready jobs, it may claim an unstarted ready evaluation or preparation block from another lane. Atomically record the new owner; never duplicate an active or completed job.', 'In particular, fast workers can consume ready MNIST28 student evaluations while the slow worker continues the next generator fit.', 'No concurrent writes to one preparation/fit/sample/labeling output. Preparation has one owner per dataset/seed.', 'Skip only provenance-validated completed records. Preserve previous logs and use new attempt logs. Failures surface directly; no automatic retries or scientific-setting changes.', 'After each completed dataset/seed block, build a new coverage/results snapshot and refresh the scoped statistics. Preserve the October 6 analysis snapshot.', 'Check actual combined GPU/RAM use and throughput after startup. Maintain the 1-slow/2-fast arrangement only while it fits; adjust queued work at configuration boundaries, keeping scientific budgets fixed.']}
(OUT / 'parallel_queues.json').write_text(json.dumps(schedule, indent=2) + '\n', encoding='utf-8')

plan = '''# Full retained experiment schedule: one slow lane, two fast lanes

Revised October 7, 2026. Inventory frozen at CHECK_TIME Chicago. This replaces the earlier 53-run schedule. The user specified all remaining CTGAN/TVAE and California Housing experiments, excluded additional controls, dropped Intrusion, and deferred new datasets and Tab-DDPM. No jobs were launched by this planning task.

**Schedule all 169 missing downstream evaluations, plus the 16 necessary new generator fits, in three persistent tmux sessions.** Start the long MNIST28 seed-43 pipeline immediately while two faster lanes close the most valuable matched blocks. Do not defer MNIST28, News or Housing outside the schedule.

| Session | Default order | Downstream evaluations | New generator fits |
| --- | --- | ---: | ---: |
| `research-slow` | MNIST28 seed 43: full and X-only CTGAN/TVAE, RF/XGB/DNN, synthetic and mix, plus real-only | 29 | 4 |
| `research-fast-1` | Weighted Census seed 42 (9), then seed 43 (29), then Housing seed 42 (29) | 67 | 4 |
| `research-fast-2` | Finish MNIST12 seed 43 (15), then News seed 43 (29), then Housing seed 43 (29) | 73 | 8 |
| **Total** | **All retained unfinished blocks** | **169** | **16** |

“Fast” means faster than the 784-feature MNIST28 generator pipeline, not a measured runtime promise. Census/MNIST12 reuse saved fits; News has 58 features and Housing eight. News/Housing still need production generation and relabeling. Run them sequentially within their assigned lanes; do not start every fit at once. The initial three concurrent workloads are MNIST28 generation, Census evaluation and MNIST12 relabeling/evaluation.

## Complete backlog and artifact reuse

| Dataset / seed | Missing evaluations | What is already saved | Required preparation |
| --- | ---: | --- | --- |
| Weighted Census / 42 | 9 | Four production fits and all needed tables | Validate dependencies; train missing TVAE downstream arms |
| Weighted Census / 43 | 29 | Four production fits and all RF/XGB/DNN tables | Validate dependencies; train weighted real-only and all downstream arms |
| MNIST12 / 43 | 15 | Four production fits; all CTGAN evaluations; most TVAE tables | Finish TVAE full RF/DNN tables, then real-only and all 14 TVAE evaluations |
| MNIST28 / 43 | 29 | No prepared manifest, production fit or synthetic tables found | Prepare original partitions; four fits; sample/relabel; all evaluations |
| News / 43 | 29 | No prepared manifest, production fit or synthetic tables found | Prepare original partitions; four fits; sample/relabel; all evaluations |
| Housing / 42 | 29 | No production fits/results | Prepare original partitions; four fits; sample/relabel; all evaluations |
| Housing / 43 | 29 | No production fits/results | Prepare original partitions; four fits; sample/relabel; all evaluations |

Each 29-evaluation block is one real-only reference plus two generators × seven constructions × two training modes. The seven constructions are one full-table generated-target baseline, three full-table hybrids, and three X-only hybrids. Do not invent an X-only original-generator target baseline or duplicate the real-only reference across generators.

Adult, Covertype and Credit are complete at both seeds in RF/XGB/DNN scope. Both digit versions and News seed 42 are complete. Preserve these results without rerunning. Credit has no remaining jobs; its small positive holdout remains a statistical limitation. The seven-dataset simulated benchmark is complete (378 production rows; 210 in the selected family), and both three-arm weighted pilots completed. No simulated rerun is scheduled. Intrusion's 57 missing selected evaluations are removed from the launch list; its artifacts and failure history remain preserved.

Including Housing and excluding Intrusion, the retained CTGAN/TVAE RF/XGB/DNN scope contains 464 evaluations: 295 complete and 169 pending. Those counts include Credit's completed evidence and both resolutions, without claiming the resolutions are independent datasets. They are distinct from the old 816/890 all-labeler plans.

## Priority is statistical closure, not equal job counts

1. **Start MNIST28 seed 43 now in the slow lane.** Its long fits would otherwise determine the finish date. Prepare its real partitions once. Prioritize full CTGAN and full TVAE so the generated-target versus relabeling comparison can start; complete both X-only fits as well. Keep 500 epochs, batch 500, CUDA and 100,000 synthetic rows.
2. **Close weighted Census seed 42 first in fast lane 1.** Its nine missing cells are TVAE full RF/XGB/DNN under both modes (six), X-only XGB mix (one), and X-only DNN under both modes (two). Within this block, finish each labeler's full/X-only and synthetic/mix comparisons together. Do not fill weighted gaps with old unweighted scores.
3. **Close MNIST12 seed 43 in fast lane 2.** Finish only the two missing TVAE full relabeling tables on saved features. Reuse the saved RF predictor if compatible; complete the missing DNN labeler. The 15 evaluations are real-only and all 14 TVAE arms. No new CTGAN/TVAE fits are needed.
4. **Complete weighted Census seed 43 before Housing in fast lane 1.** All inputs are already saved. Prioritize the real-only reference and generator baselines, then complete matched RF/XGB/DNN constructions under both modes. This supplies another repeat for a currently incomplete primary ANOVA task.
5. **Complete News seed 43, then both Housing seeds.** News completes an existing regression repeat; Housing adds a separate regression task. Freeze Housing reporting before its first test evaluation and retain both regression results, including any negative effects. Keep regression separate from classification analysis.

Finishing Census gives the current primary classification ANOVA Adult/Covertype seeds 42/43, Census seeds 42/43 and MNIST28 seed 42: four datasets/seven dataset-seed units, 196 synthetic/mix scores. MNIST28 seed 43 then produces a balanced four-dataset/two-seed ANOVA with **224 scores**, excluding eight real-only references. Seed-within-dataset error df increases from two in the current analysis to four after full completion. This addresses repeat coverage, not a claim of universal superiority.

MNIST12 becomes a complete resolution sensitivity replacing MNIST28 in the same task slot; do not count them as independent tasks. News and Housing supply two regression datasets/two seeds; use a separate analysis, with R2 available as a common persisted utility metric. Retain News's saved normalized MAE and its interpretation. Do not mix classification F1 with regression scores or pick metrics from favorable new test results.

## Keep the two faster workers busy as tables become ready

The lane table is the default allocation, not an instruction to leave finished workers idle. Generator/table preparation and downstream evaluation have separate readiness conditions:

- Publish each verified baseline/relabeling table when it is complete. Its two downstream modes can run while another generator/source is still fitting. Do not wait for the entire MNIST28 generation pipeline to finish.
- A fast worker can take an unstarted ready MNIST28 evaluation while the slow worker advances to the next fit. A free worker can also take a not-yet-started Housing block from the other fast lane. Update its ownership before running it.
- Transfer only unstarted work. Every configuration and every preparation output has one recorded owner. Do not run competing fits/labelers against the same output or duplicate a running evaluation.
- When preparation runs out, all three sessions can drain remaining ready evaluations. Three permanent session names do not require three permanent dataset partitions.

Initial statistical priority remains Census seed 42, MNIST12 seed 43 and Census seed 43. Among later ready jobs, finish a dataset/seed comparison block rather than scatter results across many incomplete blocks. Generator baselines and real-only references precede their comparisons; all RF/XGB/DNN and both training modes remain scheduled.

## Execution preparation

The current broad generation/matrix entry points still include NB/PCA-GMM by default. Before launching, prepare a narrowly scoped RF/XGB/DNN-only driver using the existing generator/labeler routines, and separate fit/table readiness from classifier scheduling. This is necessary to execute the requested scope, not permission to change scientific settings. The JSON manifest supplies all 169 downstream command arguments and every block's prerequisites; it is not itself an executable queue runner.

Use existing production algorithms, hyperparameters, splits and quality gates. Sample each source once and preserve identical features across its labelers, with original generated targets and relabeled targets paired on full-table X. Verify current remote Housing integration before its first block; source files exist remotely, but their presence alone does not certify every integration point. News seed-43 run records must save test NMAE and its normalization metadata as required by the repository.

Census uses the existing weighted single-configuration entry point; its broad `--resume` would schedule excluded labelers and collide with existing exclusive attempt logs. Preserve prior logs and completed outputs; give each attempt a new log. Classifier `--resume` means skip verified completed records, not continue a half-trained optimizer. Reuse checkpoints for evaluation/finalization only with valid provenance and development-selection history. Do not substitute seed-42 generators for missing seed-43 fits.

All three workers share the one GTX 1080 Ti. Start with one job per session and two CPU/BLAS threads per worker; check measured combined RAM/VRAM and completed-work throughput. New fits, particularly MNIST28, have unmeasured peak memory in this concurrent arrangement. If three workers do not fit or contention slows progress, let queued work wait at configuration boundaries rather than change batch size, epochs, sample count or evaluation settings. Errors surface with their actual cause; no automatic retries/fallback models.

At the 2:11 p.m. resource check, no research jobs were active, about 27 GiB RAM and 438 GB disk were available, and GPU memory use was about 236 MiB of 11,264 MiB. Recheck immediately before the actual launch. No runtime estimate is asserted from dataset size alone.

## Reporting and completion

After each closed dataset/seed block, refresh completion coverage and verified comparison tables. Rebuild statistics from a new dated snapshot when Census/MNIST28 coverage changes and when both regression tasks are complete; preserve the October 6 snapshot. Use completed run records and the existing results validation logic, never maximum test scores across epochs.

The existing results builder's full-matrix gate expects all old labelers/datasets. Its expected-run scope must be explicitly narrowed to the retained RF/XGB/DNN matrix when aggregating this plan; do not weaken prediction/source-ID/provenance checks or report old all-labeler completion. Maintain weighted Census and the seed-42 MNIST28/News namespaces when selecting authoritative records.

Completion means all **169** scheduled downstream records and their referenced predictions/weights validate, plus the required generator/table provenance, and the retained selected coverage reaches **464/464**. A failed labeler quality gate remains a disclosed failed cell with its actual cause; scientific settings are not changed to manufacture completion. Intrusion is dropped; Tab-DDPM, new classification datasets, additional controls and extra seeds are outside this schedule.

The source inventory is `audit/experiment_plan_20261007/inventory_full_schedule.json`; the full machine-readable plan is `parallel_queues.json`, and every scheduled missing run ID is listed in `unfinished_experiments.md`. Rebuild these current artifacts with `audit/experiment_plan_20261007/build_full_schedule.py`; the earlier smaller builder is historical.
'''.replace('CHECK_TIME', inventory['checked_at_chicago'])
(OUT / 'next_experiments.md').write_text(plan, encoding='utf-8')

details = ['# All remaining experiments in the retained launch scope', '', f"Checked remotely: {inventory['checked_at_chicago']} Chicago.", '', '**169 downstream evaluations; 16 new generator fits. Intrusion dropped; no controls, NB/PCA-GMM, new classification datasets or Tab-DDPM.**', '', 'Each configuration is synthetic-only, mixed, or the unique real-only reference. The command list and preparation dependencies are in `parallel_queues.json`. These are planned jobs, not active experiments.', '']
for b in blocks:
    g = next(g for g in groups if (g['dataset'], g['seed']) == (b['dataset'], b['seed']))
    details += [f"## {b['dataset']} — seed {b['seed']}", '', f"Default session: `{b['default_session']}`. Missing evaluations: {b['missing_evaluations']}; new generator fits: {len(b['new_generator_fits'])}.", '', '| Missing run ID | Existing input table |', '| --- | --- |']
    for r in g['missing']:
        name = r['run_id']
        available = 'Real partitions' if name.endswith('real_original') and g['split_manifest_present'] else 'Preparation required' if name.endswith('real_original') else 'Present; validate dependencies' if g['tables_present_for_run'][name] else 'Preparation required'
        details.append(f'| `{name}` | {available} |')
    details += ['']
details += ['## Already complete or excluded', '', '- Adult, Covertype and Credit: both seeds complete; no reruns.', '- MNIST12, MNIST28 and News: seed 42 complete; preserve authoritative namespaces.', '- Known-distribution simulated production benchmark: complete; no rerun.', '- Weighted Credit/Census pilots: completed; no remaining pilot arms.', '- Intrusion: 57 missing selected evaluations excluded from this launch; evidence retained.', '- New classification datasets and Tab-DDPM: deferred by the user.']
(OUT / 'unfinished_experiments.md').write_text('\n'.join(details) + '\n', encoding='utf-8')
manifest = {'status': 'planned_not_launched', 'checked_at_chicago': inventory['checked_at_chicago'], 'required_downstream_runs': 169, 'required_new_generator_fits': 16, 'parallel_sessions': 3, 'controls': False, 'intrusion': 'dropped', 'extensions': 'deferred', 'sha256': {}}
for path in (AUDIT / 'inventory_full_schedule.json', AUDIT / 'build_full_schedule.py', OUT / 'next_experiments.md', OUT / 'unfinished_experiments.md', OUT / 'parallel_queues.json'):
    manifest['sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('Prepared full scope: 169 unique missing evaluations, 16 new generator fits; slow/fast default assignment 29/67/73. Nothing launched.')
