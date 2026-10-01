# SDGym

Synthetic Data Gym: A framework to benchmark the performance of synthetic data generators for
non-temporal tabular data.

## CTGAN paper simulated benchmark

To run **all paper baselines and all labeled CTGAN/TVAE method variants** on all
seven simulated datasets, activate the repository environment and run from
`SDGym-research`:

```powershell
python -m synthetic_data_benchmark.run_simulated_methods prepare --seeds 42 43
python -m synthetic_data_benchmark.run_simulated_methods run --seeds 42 43 --methods all --device cuda
```

On a multi-GPU host, select the device with `--gpu-index 0` (or another CUDA
index). CTGAN, TVAE, and DNN labeler fits use that GPU. Fits run one at a time
so several large models do not compete for the same GPU memory.

The device flags are optional; without them the runner uses CUDA when available
and CPU otherwise. Use `--device cpu` to force CPU or `--device cuda` to require
CUDA. For a small selection, for example:

```powershell
python -m synthetic_data_benchmark.run_simulated_methods prepare --datasets grid asia --seeds 42
python -m synthetic_data_benchmark.run_simulated_methods run --datasets grid asia --seeds 42 --methods ctgan tvae ctgan-xgb ctgan-rf tvae-nb --device cpu
```

The `prepare` command samples and saves **one oracle train/test split per
dataset and seed**. The `run` command reads those CSV files. It fits CTGAN or
TVAE once for each required full-table or X-only source, saves that sample as a
CSV, and reuses that saved file across labelers. `--methods all` includes
`identity`, paper `ctgan` and
`tvae`, full-table labeled baselines, and X-only and full-feature relabeling
with GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, and DNN. `ctgan-nb` and
`tvae-nb` select GaussianNB for the continuous mixtures and CategoricalNB for
the categorical Bayesian networks. The default sample sizes are 10,000 rows
each for train, test, and synthetic, with 300 generator epochs. To regenerate
an existing saved sample, use a new `--root` directory.

The seven paper simulations have no target labels. The runner keeps their
`L_syn` and `L_test` results under the `paper` benchmark. Hybrid labelers use
a separate `labeled_extension`: Grid/GridR/Ring use the fixed boundary
`feature_1 > 1.5 * feature_0 + 0.8`; Bayesian networks use `dysp`, `BP`,
`Disease`, and `Accident` as targets. A fresh oracle dev sample supports DNN
checkpoint selection. A fixed random forest trained on each synthetic table
is evaluated on the untouched real test table, yielding accuracy and macro F1.
Mixture likelihoods score the two generated features only; Bayesian likelihoods
score the full joint table. These labeled scores are **not paper Table 2
scores**. Results are saved in `data/simulated_paper/simulated_methods_per_run.csv`
and `simulated_methods_summary.csv`, with separate `benchmark` values.

The paper-style simulated benchmark is implemented in
`synthetic_data_benchmark/simulated_benchmark.py`. It covers Grid, GridR, Ring,
Asia, Alarm, Child, and Insurance. Each prepared dataset has independent oracle
train and test samples. `L_syn` is mean synthetic-row log likelihood under the
fixed oracle. `L_test` is mean real-test-row log likelihood under an oracle of the
same structure refitted on synthetic rows. Mixture refits use the oracle component
count and diagonal covariance; Bayesian refits retain the oracle graph and learn
its conditional probabilities from synthetic rows. For Bayesian probabilities,
the evaluator adds `1e-8` to each joint row probability before taking logarithms, as in
the historical evaluator.

Run these commands from `SDGym-research` with the repository-root requirements
installed. Preparation and evaluation are separate so every method receives the
same saved training and test tables. Repeat for each dataset and seed. The defaults
are 10,000 train rows, 10,000 test rows, 10,000 synthetic rows, and 300 epochs
with batch size 500 for CTGAN and TVAE.
Use `--device cpu` on a machine without CUDA.

This is a paper-style benchmark using `ctgan==0.10.2`, not an exact execution
of the 2019 implementation. Modern CTGAN uses discriminator weight decay
`1e-6`; the historical implementation used zero. Modern preprocessing uses
RDT and learns categorical ordering from the training frame. GM evaluation
uses five seeded GMM initializations, which can affect `L_test` independently
of generator quality. GridR centers and sampled train/test tables also depend
on the preparation seed. Compare per-dataset runs and repeated seeds before
attributing differences from Table 2 to a method improvement.

Saved generator samples now require JSON recording their training table,
settings and sample hash. Cached likelihood and labeled results also verify
their split and oracle hashes. Older caches without the required metadata are
rejected; use a fresh `--root` for corrected runs.
RF and XGB labelers now receive the experiment seed explicitly. CategoricalNB
preserves binary indicators rather than quantile-binning rare 0/1 features into
a constant. These labeler fixes affect the labeled extension, not paper rows.

```powershell
python -m synthetic_data_benchmark.simulated_benchmark prepare --dataset grid --seed 42
python -m synthetic_data_benchmark.simulated_benchmark evaluate --dataset grid --seed 42 --method identity
python -m synthetic_data_benchmark.simulated_benchmark evaluate --dataset grid --seed 42 --method ctgan
python -m synthetic_data_benchmark.simulated_benchmark evaluate --dataset grid --seed 42 --method tvae
```

For a new method, fit it using only
`data/simulated_paper/seed_42/grid/train.csv`, write a CSV with the same columns
and 10,000 generated rows, then score it against the saved oracle and test table:

```powershell
python -m synthetic_data_benchmark.simulated_benchmark evaluate --dataset grid --seed 42 --method my_method --synthetic path/to/my_method.csv
```

After all seven datasets have been evaluated for the requested methods and seeds,
create the paper-style GM and BN family averages:

```powershell
python -m synthetic_data_benchmark.simulated_benchmark summarize --methods identity ctgan tvae my_method --seeds 42 43
```

The command writes `per_run.csv` and `summary.csv` under `data/simulated_paper`.
It requires every requested method/dataset/seed result; an incomplete comparison
raises an error. Existing `simulated_label` tables are a separate label-generation
experiment and are not part of the paper's seven oracle-likelihood tests.

The Bayesian oracle BIF files in `synthetic_data_benchmark/oracles` came from the
[Bayesian Network Repository](https://www.bnlearn.com/bnrepository/), cited by
the [CTGAN paper](https://proceedings.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan.pdf).
The repository lists its content under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).

# Getting started

## Installation

To install `SDGym` you only need to fork the repository, clone it and install its requirements

```
git clone git@github.com:$YOUR_USERNAME/SDGym.git
cd SDGym/
pip install -r requirements.txt
```

After installing the requirements, we just need to build the models that are not in python

```
sudo apt install build-essential
cd privbayes
make
```

## Data requirements

### Input Format

The input for all the synthesizers included in `SDGym` is a couple of files:

- An `npz` file containing two tables, `train` and `test`, where each is a `numpy.ndarray`.
All continuous columns are stored as is, while categorical and ordinal columns are stored
using integers, although the dtype will be float because numpy does not support mixed types.

- A `json` file containing the metadata for the dataset, that is, information about the columns,
like the max and minimum values on continuous columns or the mapping from integer to string in
categorical columns.

```
[
    {
        'name': None or str
        'type': 'Ordinal' or 'Categorical' or 'Continuous'

        # if Ordinal or Categorical
        'size': integer
        'i2s': list of str

        # if Continuous
        'min': float
        'max': float
    },
    ...
]

```

### Output Format

The results from `SDGym` are stored in the `output` folder with the following structure:

```
output
   __results__
       $MODEL.json    # Raw scores for model $MODEL
       ...

   __summaries__
      result.csv    # Table summary of the results
      barchart_$MODEL    # Bar chart for model $MODEL
      ...
```


### Demo Datasets

`SDGym` includes a few datasets to use for development or demonstration purposes. These datasets
have been preprocessed to be ready to use with `SDGym`, following the requirements specified in
the [Input Format](#input-format) section.

These datasets can be downloaded from [here](https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip).
After downloading them, you just need to unzip their contents into a folder named `data` at the
root of `SDGym`.

You can also execute the following commands from the root of the repository:
```
curl https://s3.amazonaws.com/sdgym/SDGymBenchmarkData.zip -o data.zip
mkdir data
unzip data.zip -d data/
```

Have below the list of included datasets and their original source:

- MINIST28: Use flatten 28\*28 pixels into 784 binary columns with an extra label column.
- MINIST12: Reshape 28\*28 pixels into 12\*12 binary columns with an extra label column.
- Credit: Kaggle credit card fraud dataset. https://www.kaggle.com/mlg-ulb/creditcardfraud
- Adult: Adult Dataset. https://archive.ics.uci.edu/ml/datasets/adult
- Census: KDD Census dataset https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)
- News: Online News Popularity Dataset (Regression) https://archive.ics.uci.edu/ml/datasets/online+news+popularity
- Covertype: Covertype Dataset (8 continuous + 40 binary + 1 multi) https://archive.ics.uci.edu/ml/datasets/Covertype
- Intrusion: network intrusion detector kdd99 https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data

### Simulated data

- Bivariate

    - Gaussian Grid (Grid): Gaussian Mixtures arranged in a grid.

    - Gaussian Grid Random (Gridr): Gaussian Mixtures arranged in a grid plus random offset.

    - Gaussian Ring (Ring): Gaussian Mixtures arranged in a ring.


    <table>
    <tr>
    <td>
    <img src="https://i.imgur.com/BLe701k.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/3UkHCUp.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/rPwzsiz.jpg" width="100%">
    </td>
    </tr>
    </table>

- Multivariate Structured Data: Generate samples from some pre-specified common causal structures. The Bayesian networks are from [bnlearn](http://www.bnlearn.com/bnrepository/).
    <table>
    <tr>
    <th>Asia</th>
    <th>Alarm</th>
    </tr>
    <tr>
    <td>
    <figure>
    <img src="https://i.imgur.com/iu7s4Jl.png" width="200" height="200">
    </figure>
    </td>
    <td>
    <figure>
    <img src="https://i.imgur.com/2iPs87b.png" width="200" height="200">
    </figure>
    </td>
    </tr>

    <tr>
    <th>Child</th>
    <th>Insurance</th>
    </tr>

    <tr>
    <td>
    <figure>
    <img src="https://i.imgur.com/Bj9jH9N.png" width="200" height="200">
    </figure>
    </td>
    <td>
    <figure>
    <img src="https://i.imgur.com/a6pc8OC.png" width="200" height="200">
    </figure>
    </td>
    </tr>
    </table>

## Quickstart

After installing the requirements and preparing the datasets, you only need to run the following
commands to evaluate a synthesizer:

```
python3 -m launcher SYNTHESIZER
```

* `SYNTHESIZER`: Name of the synthesizer you want to evaluate.

  Available synthesizers: 

  - identity
  - uniform, independent
  - clbn, privbn
  - medgan, veegan, tablegan
  - tvae, tgan

Optional arguments:

* `--datasets`: A list of datasets to evaluate the synthesizer with.
  If the argument is not present or the datasets are not specified it defaults to all datasets.

  Available datasets: [ asia, alarm, child,
insurance, grid, gridr, ring, adult, credit, census, news, covtype, intrusion, mnist12, mnist28]

* `--force`: Whether or not overwrite results.
* `--repeat`(int): Number of copies to generate for each dataset.


## Results

<table>
<tr>
<td>
<img src="https://i.imgur.com/9BT1pzE.jpg" width="100%">
</td>
<tr>
    <td>
    <img src="https://i.imgur.com/D74EDT2.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/UBAP7CP.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/1ClIq3I.jpg" width="100%">
    </td>
</tr>
<tr>
    <td>
    <img src="https://i.imgur.com/CIIgD9s.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/TuRqkeN.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/wJnA8Jo.jpg" width="100%">
    </td>
</tr>
<tr>
    <td>
    <img src="https://i.imgur.com/eq2hIgI.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/DkLLgE8.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/Vuzpe3o.jpg" width="100%">
    </td>
</tr>

<tr>
    <td>
    <img src="https://i.imgur.com/fJnwI4f.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/ZqFJWVH.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/kAI6lHD.jpg" width="100%">
    </td>
</tr>

<tr>
    <td>
    <img src="https://i.imgur.com/v5EESVL.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/DEjIJ8u.jpg" width="100%">
    </td>
    <td>
    <img src="https://i.imgur.com/CZ5JJ47.jpg" width="100%">
    </td>
</tr>

</table>

