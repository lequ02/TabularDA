# Historical modeling implementations

The active corrected experiment code lives in `src/modeling`. These directories
preserve the earlier Huy, Sabina, Thuy, and root modeling implementations,
their notebooks, checkpoints, and recorded outputs. They are historical
artifacts and are not inputs to the corrected experiment runner.

| Archive path | Original path | Purpose |
| --- | --- | --- |
| `huy/` | `src/modeling_huy/` | Earlier Adult, Census, and News experiments |
| `sabina/` | `src/modeling_sabina/` | Earlier tabular and MNIST experiments |
| `thuy/` | `src/modeling_thuy/` | Earlier source layout for the corrected runner, plus historical outputs |
| `root/` | selected root `src/` files | Original single-script modeling pipeline |

The corrected data and new results remain under `data/corrected_v2` and
`output/corrected_v2`.

Historical CSV files ignored by the repository's `*.csv` rule remain in the
local archive; they are not included in this Git commit.
