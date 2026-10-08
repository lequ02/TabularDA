# Full retained experiment schedule: one slow lane, two fast lanes

Revised October 7, 2026. Inventory frozen at 2026-10-07T14:13:55.717409-05:00 Chicago. This replaces the earlier 53-run schedule. The user specified all remaining CTGAN/TVAE and California Housing experiments, excluded additional controls, dropped Intrusion, and deferred new datasets and Tab-DDPM. All three sessions were launched at 2:44 p.m. Chicago on October 7; see [launch_handoff.md](launch_handoff.md) for verified progress, commands and logs. The backlog tables below describe the frozen pre-launch inventory, not current remaining counts.

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

The broad generation/matrix entry points include NB/PCA-GMM by default. Launch therefore uses the isolated RF/XGB/DNN-only `experiment_task.py` helper and dependency-aware `queue_runner.py` under `.cache/full_completion_20261007/`, calling the existing production routines. Production source files were not replaced. The JSON manifest supplies all 169 downstream command arguments and every block's prerequisites; the deployed copy is an immutable planning input, not live queue status.

Use existing production algorithms, hyperparameters, splits and quality gates. Sample each source once and preserve identical features across its labelers, with original generated targets and relabeled targets paired on full-table X. Verify current remote Housing integration before its first block; source files exist remotely, but their presence alone does not certify every integration point. News seed-43 run records must save test NMAE and its normalization metadata as required by the repository.

Census uses the existing weighted single-configuration entry point; its broad `--resume` would schedule excluded labelers and collide with existing exclusive attempt logs. Preserve prior logs and completed outputs; give each attempt a new log. Classifier `--resume` means skip verified completed records, not continue a half-trained optimizer. Reuse checkpoints for evaluation/finalization only with valid provenance and development-selection history. Do not substitute seed-42 generators for missing seed-43 fits.

All three workers share the one GTX 1080 Ti. Start with one job per session and two CPU/BLAS threads per worker; check measured combined RAM/VRAM and completed-work throughput. New fits, particularly MNIST28, have unmeasured peak memory in this concurrent arrangement. If three workers do not fit or contention slows progress, let queued work wait at configuration boundaries rather than change batch size, epochs, sample count or evaluation settings. Errors surface with their actual cause; no automatic retries/fallback models.

At the 2:11 p.m. resource check, no research jobs were active, about 27 GiB RAM and 438 GB disk were available, and GPU memory use was about 236 MiB of 11,264 MiB. Recheck immediately before the actual launch. No runtime estimate is asserted from dataset size alone.

## Reporting and completion

After each closed dataset/seed block, refresh completion coverage and verified comparison tables. Rebuild statistics from a new dated snapshot when Census/MNIST28 coverage changes and when both regression tasks are complete; preserve the October 6 snapshot. Use completed run records and the existing results validation logic, never maximum test scores across epochs.

The existing results builder's full-matrix gate expects all old labelers/datasets. Its expected-run scope must be explicitly narrowed to the retained RF/XGB/DNN matrix when aggregating this plan; do not weaken prediction/source-ID/provenance checks or report old all-labeler completion. Maintain weighted Census and the seed-42 MNIST28/News namespaces when selecting authoritative records.

Completion means all **169** scheduled downstream records and their referenced predictions/weights validate, plus the required generator/table provenance, and the retained selected coverage reaches **464/464**. A failed labeler quality gate remains a disclosed failed cell with its actual cause; scientific settings are not changed to manufacture completion. Intrusion is dropped; Tab-DDPM, new classification datasets, additional controls and extra seeds are outside this schedule.

The source inventory is `audit/experiment_plan_20261007/inventory_full_schedule.json`; the full machine-readable plan is `parallel_queues.json`, and every initially missing run ID is listed in `unfinished_experiments.md`. The builder `audit/experiment_plan_20261007/build_full_schedule.py` records the pre-launch plan; rerunning it does not inspect live queue state and can overwrite launch annotations. Use the remote `state.json` and completed run records for progress; the earlier smaller builder is historical.
