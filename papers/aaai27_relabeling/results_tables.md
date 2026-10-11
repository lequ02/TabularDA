# Draft numerical exhibits

Descriptive summaries of frozen results. See outline.md for comparison definitions, sources and limitations.

| Construction | Fit input | Feature source | Target source |
|---|---|---|---|
| Joint / generated | (X,Y) | Shared joint X | Generated Y |
| Joint / predicted | (X,Y) | Same joint X | Labeler h(X) |
| Features / predicted | X | Separate X-only fit | Labeler h(X) |

Table 1. Three target-assignment constructions. Only joint/generated versus joint/predicted holds the generated features fixed.

| Task | Real | S: gen. | S: joint | S: X-only | M: gen. | M: joint | M: X-only |
|---|---|---|---|---|---|---|---|
| Adult | 79.14 | 74.35 | 77.54 | 78.23 | 76.32 | 77.65 | 78.40 |
| Census KDD | 75.63 | 71.13 | 74.48 | 74.33 | 72.14 | 76.91 | 76.97 |
| Covertype | 50.47 | 44.09 | 56.78 | 54.91 | 49.27 | 63.62 | 62.18 |
| MNIST12* | 95.65 | 70.61 | 92.50 | 91.88 | 82.06 | 94.13 | 94.06 |
| MNIST28 | 97.97 | 74.42 | 92.36 | 92.53 | 94.63 | 96.32 | 96.33 |

Table 2. Macro F1 (%). S = synthetic-only, M = real plus synthetic. Real means use 2 seeds; generated-target means use 2 generators x 2 seeds; each relabeled-source mean uses 2 generators x 2 seeds x 3 labelers (RF/XGB/DNN). No best-labeler selection. * MNIST12 TVAE has documented heldout feature collisions.

| Task | Train | Metric | Real | Generated | Joint | X-only |
|---|---|---|---|---|---|---|
| News | Syn. | R2 | -0.0290 | -0.0235 | -0.0004 | 0.0027 |
| News | Syn. | NMAE | 0.2686 | 0.2546 | 0.2587 | 0.2597 |
| News | Mix | R2 | -0.0290 | -0.0196 | -0.0008 | 0.0002 |
| News | Mix | NMAE | 0.2686 | 0.2494 | 0.2544 | 0.2551 |
| Housing | Syn. | R2 | 0.5125 | 0.6730 | 0.7930 | 0.7946 |
| Housing | Syn. | NMAE | 0.5507 | 0.4146 | 0.3022 | 0.3049 |
| Housing | Mix | R2 | 0.5125 | 0.6676 | 0.7872 | 0.7905 |
| Housing | Mix | NMAE | 0.5507 | 0.4137 | 0.3052 | 0.3103 |

Table 3. Original-scale regression evaluation: R2 is higher-is-better and saved NMAE_sigma is lower-is-better. Joint and X-only use predicted targets; averages follow Table 2. The same real-only baseline is repeated for both regimes.

| Task | Gen. acc. | Delta acc. | Delta h* | Delta L_syn | Delta L_test |
|---|---|---|---|---|---|
| Gaussian | 82.06 | +7.29 | +9.09 | +0.4081 | -0.0195 |
| Grid | 86.27 | +3.61 | +4.39 | +0.2823 | -0.1906 |
| Ring | 87.49 | +2.14 | +2.70 | +0.4261 | -0.0581 |
| Asia | 85.48 | +0.22 | +0.41 | +0.2354 | -0.7728 |
| Alarm | 66.17 | +14.72 | +22.02 | +0.4146 | +0.0009 |
| Child | 82.42 | +5.35 | +7.92 | +0.3816 | +0.0563 |
| Insurance | 90.31 | +4.24 | +4.78 | +0.1158 | +0.0615 |

Table 4. Generated-target accuracy (%) and matched joint-relabeling changes. Accuracy and h* agreement changes are percentage points; density changes are native scores within each task. RF/XGB/DNN, 2 generators, 2 seeds; 12 matched changes per task. L_test is a density-refit proxy; BN likelihoods use the full-joint epsilon convention.

## Existing exploratory classification inference

These comparisons average BOTH relabeled feature sources against the generated baseline. Their p-values do not test only fixed-feature target replacement.

| Training | Contrast | Macro-F1 change (pp) | Raw p | Holm2 p | Holm4 p |
|---|---|---:|---:|---:|---:|
| synthetic | relabel_minus_generated | +11.632 | 0.034977 | 0.069953 | 0.139907 |
| synthetic | has_y1_minus_has_y0 | +0.357 | 0.457451 | 0.457451 | 0.914902 |
| mix | relabel_minus_generated | +6.773 | 0.056598 | 0.113197 | 0.169795 |
| mix | has_y1_minus_has_y0 | +0.138 | 0.718236 | 0.718236 | 0.914902 |

Five selected task means; two-sided exploratory inference. Exact raw sign-flip p=.0625 for relabeling in both regimes. No new tests were performed.
