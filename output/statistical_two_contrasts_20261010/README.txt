Current statistical redo: two contrasts, seven requested tasks, RF/XGB/DNN.
Main report: real_data_two_contrasts.pdf in output/pdf/statistical_two_contrasts_20261010.
Editable report: analysis.md and analysis.html.

Classification: macro F1 for Adult/Census KDD/Covertype/MNIST12/MNIST28.
Exactly two questions per regime: relabel_minus_generated and has_y1_minus_has_y0.
Two-sided dataset-level paired t (five task means, df=4), exact sensitivity checks.
p_raw_holm2 controls the two-test family within a regime.
p_raw_holm4 controls all four classification tests across both regimes.
ci1 is pointwise 95%; ci2/ci4 are Bonferroni simultaneous 95%, not Holm intervals.
News and Housing have separate descriptive R2/NMAE/D2 contrasts in native units.
Positive benefit means improvement; negative native NMAE difference means improvement.
Exploratory ANOVA p-values are nominal and do not add adjusted headline claims.

All 406 selected records and 116 D2 sidecars were reverified; no experiments changed.
Frozen inputs, table outputs, source hashes, remote environment and rebuild scripts
are included. analyze.py must run only on the research server. remote_io.py uses the
dedicated credential in .cache without putting credentials in this package.
Reports were prepared locally; numerical analysis was performed remotely.
No training, generation or experiment tests ran locally.

The two-question scope was selected after earlier analyses; this is exploratory.
Shared test rows, related MNIST tasks, two seeds and the MNIST12 TVAE collision
caveat remain. See HANDOFF.md, analysis_metadata.json and validation.json.
Previous statistical_analysis_328_20261010 deliverables remain unchanged.
