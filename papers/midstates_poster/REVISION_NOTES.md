# Midstates poster revisions

## Versions

- `ffts_midstates_poster.pptx`: hand-edited original (3 slides); reviewed in `ffts_midstates_poster.pdf`.
- `ffts_midstates_poster_v3.pptx`: `amend_poster.py` output, then hand-edited (slides 2-3 removed);
  reviewed in `ffts_midstates_poster_v3.pdf`. (`_v2.pptx` was an intermediate draft.)
- `ffts_midstates_poster_v4.pptx`: `make_v4.py` applied to the hand-edited v3 (`figures_v3/`).
- `ffts_midstates_poster_v5.pptx`: `make_v5.py`, also from the hand-edited v3 (it includes
  the v4 changes), with `figures_v4/` from `build_poster.py --figures figures_v4`. The scripts refuse
  to overwrite an existing output, because the PPTX files get hand-edited.
- `ffts_midstates_poster_v6.pptx`: `make_v6.py`, from the hand-edited v3 (includes v4/v5), then hand-edited
  (the "ysyn is generated/assigned" lines removed); reviewed in `ffts_midstates_poster_v6.pdf`.
- `ffts_midstates_poster_v7.pptx` (current): `make_v7.py` applied to the hand-edited v6, with `figures_v5/`.

## Round 1 (comments in ffts_midstates_poster.pdf)

- Method renamed from FFTS ("factorized feature-target synthesis") to **Learned Target Assignment (LTA)**;
  title "Learned Target Assignment for Synthetic Tabular Data", matching the AAAI draft's framing.
- Notation: bold **X** for the feature matrix, bold lowercase **y** for the target, in text and figures.
  Scalar coordinates (X0, X1 in Ring) and single label values (y = 1) stay non-bold.

## Round 2 (comments in ffts_midstates_poster_v3.pdf, plus advisor's concept-vs-implementation note)

- Generator and labeler are the symbols *G* and *L* (was "Generator" and *h*); Bayes rule *h** in Ring
  is unchanged.
- Method shows concepts only: the two parallel rows (standard, LTA) are split into explicit stages,
  "1. Fit to real data" and "2. Generate", and the labeling step is its own box,
  ***y***syn = *L*(***X***syn), with the arrow from ***X***syn. This keeps the parallel comparison and
  adopts the explicit, partitioned flow of the 2025 poster (its steps: generate Xsyn with a generator
  trained on real X; train a predictor on real (X, y); use it to assign ysyn). Only that poster's text
  survives locally (`G:/SummerResearch_Full/audit/reconstruction/Poster_final_text.txt`); the PDF was
  not found.
- Implementation details moved to Evaluation: the specific G and L, 100,000 rows, ablation, seeds/runs.
  Baselines added (real data only; standard synthesis with generated y). The training-rows-only /
  development-split sentence was cut as excessive; the method bullet keeps "fitted to real training
  rows only".
- "Only useful" softened; "dependency of y on X" wording in the motivation.

## Round 3 (user feedback on v4, Nov 2, 2026)

- Diagram wording: G "learns the distribution of (X, y)" / "of X" (not p(.)); L "learns y | X";
  "any number of rows" removed (implementation detail).
- "Real" is now "original" everywhere: text, diagram, table, and figure titles/legends.
- Headline table replaces the results paragraph above the central figure: rows standard (generated y),
  LTA, original data only; columns synthetic only / original + synthetic for classification macro F1
  (mean over 5 tasks x 3 generators x 2 seeds; LTA also over RF/XGB/DNN) and Housing R2. Computed in
  `make_v5.py` from the same hashed scores file and cells as the figure; synthetic-only LTA minus
  generated = +7.2 points, matching the removed paragraph.
- Evaluation rewritten without "utility test"/"baselines" jargon (Scoring, Training sets, Ablation).
- Caveats bullet removed from Takeaways.
- Metric check: the poster uses macro F1 for every classification task, as in the pre-registered
  analysis (`output/statistical_anova_gdrive_20261019/analyze.py`). The comparison panels and AAAI
  table use binary F1 for Adult/Census KDD and accuracy for MNIST. User chose macro F1 everywhere.

## Round 4 (user feedback on v5, Nov 2, 2026)

- Headline table split by generator (CTGAN, TVAE, TabDDPM rows; generated y vs. LTA columns within
  synthetic only / original + synthetic, classification macro F1 and Housing R2; original-only row).
  The v5 average (LTA original + synthetic 79.7 < original only 80.2) was pulled down by TabDDPM on
  MNIST. Per generator, LTA classification macro F1: CTGAN 80.1 / 82.2, TVAE 80.2 / 82.0,
  TabDDPM 59.6 / 74.9 (synthetic only / original + synthetic) vs. original only 80.2.
  Per task, LTA original + synthetic is still below original only on MNIST12/28 for every generator;
  the average gain comes mostly from Covertype, where the original-only network is weak (52.6).
- References section added at the bottom of the left column (CTGAN/TVAE, TabDDPM, RF, XGBoost; from
  `papers/aaai27_relabeling/references.bib`). Central figures moved down 0.35 in for the taller table.

## Round 5 (comments in ffts_midstates_poster_v6.pdf, Nov 3, 2026)

- "Generated y" is now **Baseline** (table, figure legend, caption); the method row is "Baseline: joint synthesis".
- Table: "Regression (Housing): R2" header; Average row (mean of the three generators). A paragraph
  between table and figure gives LTA vs. baseline (+7.2 / +4.6 macro F1 points; Housing R2 +0.09 / +0.09)
  and LTA vs. original data (CTGAN/TVAE 80.1/80.2 alone, 82.2/82.0 added, vs. 80.2; Housing >= 0.78 vs. 0.51).
- Caption: the "fall short on 5 of 5 / News |R2|" sentence removed.
- Evaluation starts with the baselines and splits Classification / Regression; set at 27 pt.
- Takeaways -> Conclusion: gain over baseline, base G matters, fitting G to y does not matter, future work
  on regression. TabDDPM MNIST explanation removed.
- Simulated section: Ring equations (P(S = +1 | X) = sigmoid(2 X0 X1), h* = 1[X1 + (1 + 0.75 S) X0 > 0],
  10% flips); box defines both bars (LTA minus baseline over CTGAN/TVAE x RF/XGB/DNN x 2 seeds;
  accuracy; L_test = (1/n) sum log q(z_i)).
- References shortened to one line each.
- Follow-up: the table's regression columns average Housing and News (header "Regression: R2";
  original data 0.24, LTA 0.39-0.40, baseline 0.31-0.39); `make_v6.headline_numbers` takes the regression
  datasets as an argument (default unchanged, Housing only).

## Known gap

`build_poster.py`'s PPTX section still writes the original FFTS/X-Y text; the hand-edited PPTX plus
the amend scripts are the source for the current poster. Only its figure labels are kept current.
