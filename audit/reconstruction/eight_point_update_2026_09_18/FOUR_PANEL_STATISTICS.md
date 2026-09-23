# Four-panel comparison statistics

Maximum macro-F1; four tasks, with Adult counted once. Joint TVAE Covertype is set to the user-specified 0.49.

## Comparisons

Gain is hybrid minus e2e, except feature generator input, where it is Features + target minus Features only. The two pooled panels show eight task/generator observations per group. For inference, CTGAN/TVAE are averaged within task: paired t tests use four task differences (df = 3), since generator results on the same task are related. Holm adjustment covers all four comparisons.

| Comparison | Reference mean | Comparison mean | Gain | Relative gain | 95% CI | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| End-to-end vs hybrid | 0.6602 | 0.8013 | +0.1411 | +21.37% | [-0.0109, 0.2931] | 0.0598 | 0.2392 |
| Feature generator input | 0.8028 | 0.7998 | -0.0030 | -0.37% | [-0.0077, 0.0017] | 0.1379 | 0.2930 |
| CTGAN-based | 0.5462 | 0.7900 | +0.2438 | +44.63% | [-0.0823, 0.5698] | 0.0977 | 0.2930 |
| TVAE-based | 0.7742 | 0.8126 | +0.0384 | +4.96% | [-0.0209, 0.0978] | 0.1313 | 0.2930 |

## Both generators contribute to the pooled comparison

For every task, e2e = (CTGAN_e2e + TVAE_e2e) / 2 and hybrid = (CTGAN-hybrid + TVAE-hybrid) / 2. Each generator-specific hybrid mean averages RF/XGB and both generator-training input settings.

| Task | CTGAN_e2e | TVAE_e2e | Pooled e2e | CTGAN-hybrid | TVAE-hybrid | Pooled hybrid |
| --- | --- | --- | --- | --- | --- | --- |
| Adult | 0.7743 | 0.7461 | 0.7602 | 0.7740 | 0.7748 | 0.7744 |
| Covertype | 0.4835 | 0.4900 | 0.4868 | 0.6334 | 0.5835 | 0.6085 |
| MNIST12 | 0.4816 | 0.9248 | 0.7032 | 0.8791 | 0.9374 | 0.9083 |
| MNIST28 | 0.4454 | 0.9356 | 0.6905 | 0.8734 | 0.9547 | 0.9140 |

The 0.49 substitution is an assumption, not a measured result or leakage correction. None of the four comparisons is significant at 0.05; no result establishes leakage-free superiority. See [full notes](FOUR_PANEL_COMPARISON_NOTES.md) for limitations.

