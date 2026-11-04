# Working research folder

To replicate the experiments from this repository alone, follow
[REPRODUCE.md](REPRODUCE.md). The rest of this file describes the authors'
working environment.

The main workspace is `D:/SummerResearch`. The complete original workspace is
`G:/SummerResearch_Full`. Production experiments continue on
`thuy@10.24.10.133` in `/home/thuy/Research/minh_data_synth/TabularDA`, using
`/home/thuy/miniconda3/envs/env`.

## Code and environments

`src/` contains the maintained preparation, synthesis, labeling and downstream
models. `scripts/` contains the runners, resume tools and report builders.
`simulated_eval/` is the separate known-distribution benchmark, including its
BIF oracles. `source/` preserves historical source versions; use the maintained
root source for future changes. The current editable paper is in
`papers/aaai27_relabeling`, and the October 19 analysis remains under
`output/statistical_anova_gdrive_20261019`.

The local interpreter is `.venv/Scripts/python.exe`. It supports metadata,
plotting, tables and reporting. `requirements-local.lock.txt` pins its installed
packages. `requirements.txt` and `audit/remote_environment_20261019.json`
describe the full experiment dependencies and the verified server environment.
The old 5 GiB environment and complete original Git history are preserved in
the comprehensive G copy.

To create the reporting environment in a fresh copy of this folder, run
`.\scripts\setup_local_env.ps1` from PowerShell. It installs the exact local
lock at the final path and preserves any existing `.venv`.

Git uses independent filtered metadata, the original remotes, both local
branches, and `backup` tracking `origin/backup`. The original local commit and
uncommitted source changes are preserved. The remote backup branch was ahead
at migration time; the swap does not merge it or publish changes. Large
organized artifacts and the local environment are ignored. Historical file
contents may be fetched on demand; the complete original history also remains
on G.

## Common checks and reports

From this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/workspace.py check
.\.venv\Scripts\python.exe scripts/workspace.py remote-help
.\.venv\Scripts\python.exe scripts/workspace.py reports --selection valbest --out .cache/valbest_preview
```

The report utility also accepts `--selection original` or `--selection delta`.
Choose a new output directory; it preserves existing figures and records.
It creates a temporary view for the existing builders, then writes portable
artifact references. Original-report rebuilding reads the larger saved run
records; validation-best rebuilding uses the small completed result records.
The main experiment runner remains `scripts/run_corrected_matrix.py` on the
server. Follow `AGENTS.md` to inspect live jobs/resources, select an explicit
output namespace and launch through a persistent queue or tmux session.

## Results and exact inputs

Each dataset retains its existing flat `full/`, `xonly/`, `bonus/` and real-only
files, with `scores.csv` and (where covered) `scores_valbest.csv`.

- Latest original-selection reports (October 31, Adult TabDDPM posterior-fix rerun): [comparisons/20261031_adult_tabddpm_fix](comparisons/20261031_adult_tabddpm_fix/README.md).
- Historical October 13 [synthetic plot](comparisons/synthetic.png) and [mix plot](comparisons/mix.png).
- Latest [validation-best results and interpretation](comparisons/20261031_adult_tabddpm_fix/valbest/README.md), refreshed October 31 with 28 posterior-fix Adult TabDDPM evaluations.
- [All original scores](comparisons/scores.csv), [latest validation-best scores](comparisons/20261031_adult_tabddpm_fix/valbest/scores_valbest.csv), and [historical validation-best archive](comparisons/valbest.md).
- [Artifact locations/hashes](comparisons/files.csv) and [validation-best reuse map](comparisons/valbest_artifacts.csv).

The original archive contains 1,034 matrix configurations and 14 comparison
baseline references. The separate validation-best selection contains 602
completed evaluations from seven datasets and seeds 42/43. These are different
selections, not additive claims of a complete new matrix. Historical Adult TabDDPM
keeps its recorded source caveat; the latest selected reports use the posterior-fix rerun.
MNIST12 TVAE retains its recorded collision caveat.

`workspace.json` records the active roots. Exact corrected real/synthetic tables
remain on the server; their expected hashes are in
`scripts/checkpoint_recovery/input_hashes.json`. The migration verified the
presence of all 719 CSV paths in that comparison inventory. Original workers
still verify hashes before using them. Old D CSVs are not substitutes for
those corrected tables. This working folder uses the existing remote execution
model; it is not an offline copy of every experiment input. The full G copy is
the complete original D workspace, not an implied backup of server-only files.

Historical run/result JSONs remain byte-for-byte evidence and may contain their
original execution paths. Use the artifact index to locate their files here.
The remote credential remains at the ignored `.cache/remote_intrusion/server_key`
path; it is not part of code commits or report exports.
