# News launch overlap review

Read-only server inspection on October 8, 2026, at 6:04 p.m. Chicago, with queue/log recheck and namespace-lock validation at 6:08 p.m. Evidence: `audit/news_log_launch_conflicts_2026_10_08.json`.

## Current work and duplicate-result assessment

- The only active experiment process is PID 388124: MNIST28, seed 43, CTGAN features-only fit/sample, under `research-fast-1`. Its live log reached epoch 165/500; the progress display estimated roughly eleven hours for the remaining generator fit, which is an estimate rather than a finish-time guarantee.
- `research-fast-2` and `research-slow` remain live queue workers with no current task. They can automatically claim jobs when dependencies become ready; they are not dedicated free launch slots.
- Queue state has 232 completed tasks, one running task, and nine pending tasks. All pending tasks are MNIST28 seed-43 CTGAN X-only: three relabelers and six downstream evaluations. All 46 News tasks in this existing queue are marked complete; no queued or active News rerun was found.
- The new `news_log_v1` data/model/output namespaces do not exist remotely. No new-protocol run records exist. The three historical pilot records remain in their separate namespace and are explicitly excluded from reuse.

One launch of the new protocol in its new namespace would therefore not duplicate an active News experiment or overwrite existing corrected/pilot outputs. Shared source data and X-only artifacts are read and copied into the separate namespace with hash checks.

## Conflicts that remain relevant

1. **Hardware sharing:** the server has one GTX 1080 Ti (11,264 MiB). At inspection, 844 MiB was allocated, GPU utilization was 9%, available RAM was 26,060 MiB, and load average was about 1.1. These readings show memory headroom, not a reservation of GPU or CPU throughput. The News runner would compete with the active MNIST generator and the nine tasks the other workers will automatically start afterward. It does not participate in the old queue's task ownership/scheduling.
2. **Shared source deployment:** the new entry point/helper are absent from the production server paths, and the eight modified existing source files differ from their new local versions. Only isolated validation copies have been transferred. Launch requires deployment first. Replacing shared `src/` files while existing queues are live can cause upcoming tasks to import a different source version from the active job; run records hash source files at completion. Although raw-path defaults are preserved, deploying into the active tree should not be treated as isolated execution.
3. **Concurrent launches into one namespace:** review found that completed-artifact resume checks alone did not prevent two live News runners from starting the same unfinished configuration. A small standard-library `fcntl` namespace lock was added locally and verified remotely using no-training stubs. A second runner now fails before preparation; sequential invocations work after the first exits. This patch has not been deployed to production paths.

At the time of this prelaunch audit, nothing was launched, interrupted, or changed in the existing queue. The recommendation to wait was superseded by the user's explicit instruction to launch concurrently.

## Authorized concurrent launch, 6:18 p.m. Chicago

Launched one News worker in `news-log-v1` after another live process/resource check and successful isolated preflight. The new code runs from `.cache/news_log_runtime_20261008/`, with backups and verified transferred hashes; canonical source files remained unchanged. The namespace lock is active. This worker is limited to CPU cores 6 and 7 with two-thread numerical libraries and shares GPU 0. Epochs, sample counts, model settings, and quality gates remain unchanged. MNIST28 continues in its original queue; its PID and session were preserved. No pilot records are reused.

Exact launch evidence is in `audit/news_log_launch_2026_10_08.json`; live health evidence is in `audit/news_log_launch_health_2026_10_08.json`. The full command, log, namespace, and resume instructions are recorded in `audit/NEWS_LOG_RERUN_2026_10_08.md`. The namespace absence and undeployed lock statements above describe the prelaunch state.
