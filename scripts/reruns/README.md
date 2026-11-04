# Targeted rerun launchers (provenance)

Byte-for-byte copies, verified by SHA-256, of the launchers and launch records that
produced the rerun namespaces. They ran from frozen runtime copies on the research
server and keep those original absolute paths. To replicate the experiments, use
`scripts/run_corrected_matrix.py` as described in `REPRODUCE.md`; current `src/`
already contains each fix these reruns introduced.

| Folder | Output namespace | Purpose (audit note) |
| --- | --- | --- |
| `housing_no_faker_runtime_20261009` | `housing_no_faker_20261009` | Housing refit without SDV Faker transformers (`audit/HOUSING_FAKER_RERUN_2026_10_09.md`) |
| `mnist_head_rerun_plan_20261009` | `mnist_head_fixed_20261009` | MNIST12/28 rerun with the corrected output head (`audit/mnist_head_*_2026_10_09.json`) |
| `news_log_runtime_20261008` | `news_log_v1` | News log-target worker for `scripts/run_news_log_experiment.py` (`audit/NEWS_LOG_RERUN_2026_10_08.md`) |
| `tabddpm_launch_20261010` | `tabddpm_500ep_20261010_v1` | TabDDPM 500-epoch launch preparation and validation (`audit/TAB_DDPM_LAUNCH_2026_10_10.md`) |
| `tabddpm_covertype_rerun_20261011` | `tabddpm_covertype_500ep_fix_20261011_v1` | Covertype TabDDPM after the posterior fix (`audit/TAB_DDPM_COVERTYPE_RERUN_2026_10_11.md`) |
| `tabddpm_census_kdd_rerun_20261011` | `tabddpm_census_kdd_500ep_fix_20261011_v1` | Census KDD TabDDPM after the posterior fix (`audit/TAB_DDPM_CENSUS_KDD_RERUN_2026_10_11.md`) |

The Adult TabDDPM posterior-fix rerun (October 30) and its validation-best replay are
in `adult/tabddpm_fix_20261030/`.
