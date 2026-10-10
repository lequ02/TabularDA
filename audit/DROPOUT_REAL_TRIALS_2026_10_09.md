# Housing and News real-only dropout trials

**Completed: 8/8 trial fits**, with all four workers finishing successfully by **12:01:59 a.m. Chicago on October 10, 2026**. Completion was verified at 12:41 a.m. against actual run records and their saved completion hashes, exact trial source hashes, existing weights/predictions, finite epoch statistics, and minimum-development-loss checkpoint selection. Baseline source tables, records, and weights remain preserved. See [verified results](dropout_real_results_2026_10_10.json).

| Dataset | Dropout | Test R², seed 42 | Test R², seed 43 | Mean development MSE |
|---|---:|---:|---:|---:|
| Housing | 0 | 0.4773 | 0.4323 | 0.715335 |
| Housing | 0.1 | 0.5159 | 0.4687 | 0.679573 |
| Housing | 0.4, existing baseline | 0.5246 | 0.5003 | 0.654893 |
| News | 0 | -0.0108 | 0.0091 | 68,472,179 |
| News | 0.1 | -0.0165 | 0.0110 | 67,951,916 |
| News | 0.4, existing baseline | 0.0048 | -0.0628 | 69,171,251 |

**Reducing dropout alone did not improve Housing.** News dropout 0.1 had the lowest average development MSE, but its test R² remains near zero.

Lower mean development MSE favors 0.4 for Housing and 0.1 for News among these three candidates. These results do not isolate architecture, BatchNorm, ordered batching, target objective, or training-update effects. The full matrices and main comparison reports are unchanged. The launch-time and interim notes below remain historical snapshots.

Four persistent workers launched on October 9, 2026, at **11:55:55 p.m. Chicago**, following the user's instruction to try dropout 0 and 0.1 for Housing and News in four parallel training workers. Each worker runs experiment seeds 42 and 43 sequentially: **eight planned real-only downstream fits**. Launch does not establish completion.

| Dataset | Dropout in all four hidden layers | tmux session | Output namespace |
|---|---:|---|---|
| California Housing | 0 | `housing-dropout-p0` | `housing_dropout_p0_20261009_v1` |
| California Housing | 0.1 | `housing-dropout-p01` | `housing_dropout_p01_20261009_v1` |
| News | 0 | `news-dropout-p0` | `news_dropout_p0_20261009_v1` |
| News | 0.1 | `news-dropout-p01` | `news_dropout_p01_20261009_v1` |

Each trial freezes the source used by its current report baseline. Housing copies `.cache/housing_no_faker_runtime_20261009/src`; News copies `.cache/news_log_runtime_20261008/src`. Each snapshot differs from its baseline in exactly four source substitutions: `nn.Dropout(0.4)` becomes `nn.Dropout(0.0)` or `nn.Dropout(0.1)`. The original model source is preserved as `model_news.original.py`. Canonical source files and other workers are unchanged.

Housing retains raw targets, BatchNorm, its existing architecture, and its original loader order. News retains log-target training, no BatchNorm, inverse-log predictions in shares, and raw-share development MSE for checkpoint selection. **Training remains unshuffled in all trials** to isolate dropout. All retain real-training feature scaling, batch size 128, learning rate 0.001, at most 100 epochs, patience 30, and development-loss checkpoint selection. CPU math libraries use one thread per worker; all four use GPU 0.

The six prepared real partitions and split manifest for each seed are copied into its new namespace and verified byte-for-byte against the baseline and manifest. Housing has 14,861 training, 1,651 development, and 4,128 test rows. News has 28,543 training, 3,172 development, and 7,929 test rows. These are existing splits, not newly sampled partitions. No synthetic data, generator, or upstream labeler is trained in this ablation.

Remote root: `/home/thuy/Research/minh_data_synth/TabularDA`. Python: `/home/thuy/miniconda3/envs/env/bin/python`. Each runtime is `.cache/<namespace>_runtime`; output is `output/<namespace>`. Its `config.json` saves exact commands, environment, source/input hashes, log paths, expected records, and model configuration. Its `worker_status.json` tracks the current seed and verified completed records. Worker logs are `worker.log`; training logs are `logs/seed_42.log` and `logs/seed_43.log`. No automatic experiment retries or resume occurs; inspect a failed/interrupted worker before taking further action.

Group launch evidence is `output/dropout_real_20261009_v1/launch.json` remotely and [dropout_real_launch_2026_10_09.json](dropout_real_launch_2026_10_09.json) locally. [Health evidence](dropout_real_health_2026_10_09.json) checks actual processes, GPU allocation, finite epoch statistics, input/source hashes, and preservation of baseline records and weights. The [worker source](dropout_real_trials_20261009/worker.py) is preserved locally.

An earlier preparation-only namespace, `dropout_real_20261009`, and its Housing p0 source draft were abandoned before any training launch after a constructor check exposed Housing's older interface. They contain no completed experiments and must not be treated as trial outputs. The four launched workers use the `v1` namespaces listed above.

These trials are separate from the main comparison reports. Compare dropout 0/0.1 with the current dropout-0.4 baseline using **mean real-development MSE across both seeds**; disclose all candidates and each seed. Test R²/NMAE are final descriptive outcomes, not the selection criterion. A later adoption requires applying the chosen downstream protocol consistently to the synthetic-only and mixed arms. This dropout-only ablation does not match optimizer-update budgets across real and synthetic training.

At **11:57:13 p.m. Chicago**, all four workers were actively fitting on GPU with finite saved development metrics and no tracebacks. Housing dropout 0 had completed seed 42 and advanced to seed 43; Housing dropout 0.1 was at epoch 42 of seed 42. News dropout 0/0.1 were at epochs 24/23 of seed 42. One of eight planned records was complete. GPU usage was 1,700/11,264 MiB with 59% instantaneous utilization; the separate MNIST jobs also held GPU contexts. Baseline records, weights, all prepared inputs, and frozen trial source hashes passed preservation checks.

The News dropout-0 epoch-24 training-set evaluation had raw-share MSE approximately `6.67e25` despite development MSE approximately `6.83e7`. These are actual finite inverse-log prediction errors, not the optimizer's log-target training objective. Preserve these results and investigate them if they persist in the selected checkpoint; they do not justify switching models, clipping predictions, changing selection, or choosing settings from test scores. The launch health snapshot is an interim observation, not a completed comparison.
