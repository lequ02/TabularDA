# TabDDPM experiment launch

Launched **October 10, 2026, at 10:53:05 p.m. Chicago** on `thuy@10.24.10.133`, in `/home/thuy/Research/minh_data_synth/TabularDA`, using `/home/thuy/miniconda3/envs/env/bin/python`.

Namespace: **`tabddpm_500ep_20261010_v1`** under `data/`, `sdv trained model/`, and `output/`. Only TabDDPM is running; experiment seeds are **42 and 43**. The agreed design is in [the experiment plan](TAB_DDPM_EXPERIMENT_PLAN_2026_10_10.md).

## Persistent queues

Each tmux session owns one independent Python process. Within each dataset, seed 42 precedes seed 43. The workers run concurrently on the same GTX 1080 Ti.

| Session | Dataset order | Planned downstream records | Initial parent PID |
| --- | --- | ---: | ---: |
| `tabddpm-1` | MNIST28 | 54 | 1227477 |
| `tabddpm-2` | Covertype | 54 | 1227484 |
| `tabddpm-3` | California Housing → News → Census KDD | 130 | 1227491 |
| `tabddpm-4` | Adult → MNIST12 | 108 | 1227498 |
| Total | Seven datasets, two seeds | **346** | |

Worker logs are `/home/thuy/Research/minh_data_synth/TabularDA/output/tabddpm_500ep_20261010_v1/logs/workers/tabddpm-{1,2,3,4}.log`. Detailed logs are under `logs/<dataset>/seed_<seed>/`. The remote `output/tabddpm_500ep_20261010_v1/launch.json` saves exact commands, queue ownership, source hashes, settings, and validation evidence.

The in-session code calls the maintained public runner:

```python
for dataset in ["adult", "mnist12"]:  # worker 4
    for seed in (42, 43):
        run_experiment(seed=seed, datasets=[dataset], generators=("tabddpm",),
                       generator_epochs=500, namespace="tabddpm_500ep_20261010_v1")
```

Each dataset/seed runs a fresh real-only baseline, full/X-only generation and all labeler tables, then synthetic-only/mix evaluations. Generated-target and RF/XGB/DNN evaluations precede supplementary NB/PCA-GMM evaluations. The pilot remains Adult, Housing, and MNIST12, both seeds: **90 primary downstream records**. First look remains Adult/Housing seed 42: **30 primary records**.

## Frozen settings and source changes

Generator settings are 500 complete epochs, batch 500, CUDA, 1,000 diffusion timesteps, sampling batch 500, and 100,000 synthetic rows per source. The fit computes `steps = epochs * len(train_loader)`, including the final short batch, and records epochs, training rows, updates per epoch, and actual total updates. The existing architecture, normalization, optimizer, learning-rate schedule, final checkpoint, and ancestral sampler are retained.

The full study plans **28 generator fits, 166 synthetic tables, and 346 downstream records** including fourteen originals. This is a planned inventory, not a completion claim. Generation uses the agreed unconditional joint/X-only constructions; native target-conditional TabDDPM remains a separate follow-up.

Only four production Python files required additional changes:

- `src/synthesize_data/tabddpm.py`: epoch budgets; shared prepared-split verification; News full-table log targets with one inverse transform on export; log-aware labeler calls; saved feature/generator hashes; protection against overwriting partial artifacts.
- `src/synthesize_data/main.py`: mutually exclusive TabDDPM epoch/step CLI arguments and corrected News support.
- `scripts/run_corrected_matrix.py`: generator selection and epoch argument in `run_experiment`; corrected News routing; primary TabDDPM evaluation order. Existing CTGAN/TVAE defaults are preserved.
- `src/synthesize_data/synthesizer.py`: retain the no-Faker check for SDV models while allowing actual TabDDPM checkpoint loading.

The existing `audit/tabddpm_smoke.py` was extended with an epoch/update check and current explicit matrix counts. No production launcher module, additional dependency, fallback, retry, or error-swallowing path was added. Launch/validation tooling is kept in ignored `.cache/tabddpm_launch_20261010/`.

The previously consolidated corrections were already deployed and validated by the other chat. They were preserved, including weighted Census, News log-target modeling/labelers, repaired MNIST heads, numerical PCA routing, categorical NB handling, fixed grouped partitions, train-only transformations, and saved regression normalization. The only substantive local/remote difference in the modeling entry point was remote support for Intrusion, outside this run; it was preserved.

Remote originals for the five transferred files are preserved in `.cache/tabddpm_launch_20261010/originals/`. Transfers and dependency hashes were checked before replacement. The main source hash recorded at launch is:

`f5daf5ff979a57d5e7c538e37e2e0a26d7fe92f44312309c700f59e61f87351b`

Do not deploy changes to main source while these workers are active. Completed records save the source and artifact hashes for strict classifier-only resume.

## Verification

All work involving fits or sampling was executed remotely. Local work was source inspection, edits, syntax parsing, and handoff preparation.

- Existing TabDDPM checks passed on CPU and CUDA, covering mixed/categorical/numerical inputs, target-free X-only fits, seed differences, save/reload, labeler integration, and epochs versus updates. Two epochs on 65 rows with batch 32 correctly produce six updates.
- The integration check exercised actual full/X-only generation and all applicable labelers on temporary Adult/News fixtures: **22 tables**, fifteen rows each. It verified log-target fit hashes, raw-share exports, identical features within each source, feature hashes, quality records, overwrite rejection, primary ordering, the 346-run command inventory, and preserved CTGAN/TVAE defaults.
- All **eight existing repair tests** passed.
- Four concurrent resource probes passed using production input widths, batch 500, 1,000 diffusion timesteps, and sampling batch 500. These used two fit updates and 500 samples each, strictly as resource checks. Peak framework-reserved CUDA memory was 26–70 MiB per probe; this excludes CUDA context overhead and does not predict full-data RAM or total runtime.
- All fourteen source split manifests, source IDs, six real partition files per dataset/seed, and SHA-256 hashes were verified before copying. Copies were verified again. MNIST12/MNIST28 source IDs match. Existing real partitions were not regenerated.

Prepared inputs came from `corrected_v2` for Adult/Census KDD/Covertype, `mnist_head_fixed_20261009` for both MNIST resolutions, `news_log_v1` for News, and `housing_no_faker_20261009` for Housing. Across inspected historical namespaces, each dataset had matching split IDs and raw training hashes.

The first integration fixture used perfectly separable classification labels and failed the existing DNN convergence requirement. Only the temporary fixture was changed to include noise; the rerun passed with production quality gates unchanged. Both logs are retained. An automatic approval-review usage-limit failure delayed a helper transfer; the subsequent normal approval retry succeeded before launch.

## Status and follow-up

At **10:55:52 p.m. Chicago**, all four tmux panes and their GPU children remained live. Housing seed 42 had completed its real-only baseline with development-selected epoch 18; its full-table TabDDPM fit had reached **10,400 / 15,000 updates** with finite losses. MNIST28, Adult, and Covertype were training their seed-42 originals at epochs 26, 47, and 4 respectively. There was **one verified completed downstream record out of 346 planned**, and no completed TabDDPM comparison yet. GPU memory use was 1,051 MiB of 11,264 MiB and available RAM was 24,953 MiB. The initial sampled current swap-in/swap-out rates were zero. These figures are dated launch snapshots.

Do not restart a whole queue after a failure: each worker stops on the real exception and preserves completed artifacts. Check live processes, panes, latest logs, and provenance before any recovery. Classifier-only resume is available after generation is complete and source/artifact hashes match. Incomplete generation requires explicit recovery that preserves completed fits; there is no automatic training resume.

After verifying all 346 records are complete, aggregate once from the remote repository:

```bash
/home/thuy/miniconda3/envs/env/bin/python scripts/build_corrected_results.py \
  --runs output/tabddpm_500ep_20261010_v1 \
  --out output/tabddpm_500ep_20261010_v1/results \
  --generators tabddpm \
  --datasets adult census_kdd covertype mnist12 mnist28 news california_housing \
  --seeds 42 43
```

Interim analysis must use only completed records and show missing configurations. Use saved News/Housing normalization and separate metrics; do not select the best labeler or maximum test epoch. A launch or a live pane is not completion evidence.

Local and remote ignored evidence directories are `.cache/tabddpm_launch_20261010/`, containing deployment, validation, resource, prepared-input, launch, and status JSON records plus focused-check logs. Exact live output paths are recorded in `launch.json`.

## Status check: October 11, 2026, 11:00 a.m. Chicago

Read-only inspection found **195 / 346 completed downstream records**: ten originals and 185 synthetic/mix evaluations. Completion records have development-selected checkpoints and existing prediction/weight artifacts; source hashes match the launch. There are no unexpected configurations or missing required artifact/normalization fields in this inventory. This status check did not rehash every large artifact or restart any worker.

| Worker | Status | Completed / planned | Current work or stop point |
| --- | --- | ---: | --- |
| `tabddpm-1` | Completed | 54 / 54 | All MNIST28 configurations, seeds 42/43; queue finished at approximately 10:53 a.m. |
| `tabddpm-2` | Failed | 1 / 54 | Covertype seed-42 original completed; full-table generator failed at 1:15:59 a.m. |
| `tabddpm-3` | Running | 70 / 130 | News seed-43 X-only DNN synthetic-only evaluation, epoch 27 at inspection. |
| `tabddpm-4` | Running | 70 / 108 | MNIST12 seed-42 X-only Gaussian NB mix evaluation, epoch 7 at inspection. |

Completed entire dataset/seed inventories: Adult 42/43, Housing 42/43, MNIST28 42/43, and News 42. News 43 has 13/19 records; MNIST12 42 has 16/27. Census KDD 42/43, MNIST12 43, and Covertype 43 have not started. Covertype 42 has only its original baseline.

The Covertype error is **`ValueError: Nonfinite Tab-DDPM loss at step 70869`**, during a 514,000-update full-table fit. The failed queue stopped; no Covertype generator checkpoint or synthetic table was saved in this namespace. Its seed-43 task remains blocked behind the failure. The numerical cause has not been diagnosed by this status check. No automatic restart, budget change, or source edit was made.

There are **18 completed 500-epoch generator fits out of 28 planned**, all with final loss-history steps matching their recorded budgets and saved samples, and **101 saved 100,000-row synthetic tables out of 166 planned**. The primary pilot has **75/90 records**; the thirty-record Adult/Housing seed-42 first look is complete. MNIST12 seed 43 supplies the remaining fifteen primary pilot records.

Two GPU training children were confirmed live. Initial resource snapshot at 10:58 a.m.: GPU memory 683 MiB used, RAM 27,574 MiB available, and zero sampled current swap-in/swap-out. The frozen main source still matches the launch hash. Snapshot evidence is `.cache/tabddpm_launch_20261010/progress_20261011.json`, with detailed worker traces and per-dataset/seed missing configurations.

A separate observed quality concern is News seed-43 DNN relabeling: saved quality records show maximum targets of approximately `1.74e13` shares for full-table relabeling and `1.01e14` for X-only relabeling. The running X-only DNN downstream log has very large finite training losses. This is not an execution failure in the current snapshot, but the exported target ranges need review before interpreting utility. This status check did not diagnose or change the method.

## Covertype diagnosis — October 11, 11:31 a.m. Chicago

A one-line categorical-posterior numerical fix is in the main local source and passed isolated remote CPU/CUDA checks, real-Covertype gradient checks for both seeds/sources, and all diffusion timesteps. See [the diagnosis and restart requirements](TAB_DDPM_COVERTYPE_DIAGNOSIS_2026_10_11.md). It has not replaced the frozen remote production source; workers 3/4 remain active and Covertype has not been restarted. The failed fit has no resumable intermediate checkpoint.

## Covertype rerun — October 11, 11:51 a.m. Chicago

Seeds 42 and 43 were relaunched in parallel as `tabddpm-covertype-fix42` and `tabddpm-covertype-fix43`, using the corrected frozen source snapshot and namespace `tabddpm_covertype_500ep_fix_20261011_v1`. At 11:52 a.m., seed 42 had reached 100/514,000 generator updates with finite loss, and seed 43 was training its real-only baseline. The verified original seed-42 baseline is reused. Main remote source and workers 3/4 remain unchanged. See [rerun settings, provenance, queues, and logs](TAB_DDPM_COVERTYPE_RERUN_2026_10_11.md).
