# Synthetic tabular benchmark report

The report contains the data-generating models, proofs, evaluation definitions, published CTGAN comparisons, archived mixture replication results, and the completed seven-dataset experiment.

## Files

- `synthetic_benchmark_report.pdf`: compiled report.
- `synthetic_benchmark_report.tex`: editable LaTeX source.
- `figures/`: seven vector PDF figures referenced by the source.
- `data/per_run.csv`: all 378 dataset–seed–method evaluations.
- `data/summary.csv`: the saved summary statistics.
- `data/config.json`: generation and evaluation settings, package versions, and the likelihood-rescoring timestamp.
- `data/archived_paper_baselines.csv`: the 42 selected records from the earlier paper-style benchmark; these are separate from the fresh mixed-data experiment.

## Compilation

From this directory, run `pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape synthetic_benchmark_report.tex` twice. The source uses standard LaTeX packages and includes its bibliography directly. Keep the `figures` directory beside the source.

BN likelihood scores apply `log(p + 1e-8)` once to the complete joint probability. Mixed-data likelihood scores are unadjusted log densities. Exact BN support violations are retained as a separate diagnostic. Tables report means over seeds 42 and 43; figures also show both individual seed values.
