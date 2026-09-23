# from synthesize_data.create_synthetic_data import adult_old
from synthesizer import *
import sys
import os
import pandas as pd
import argparse
from create_synthetic_data import news, census, covertype, intrusion, credit, adult, mnist28, mnist12, census_kdd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from datasets import load_adult, load_news, load_census, load_covertype, load_intrusion

def single_run(dataset=None, seed=None, generator=None):
  datasets = [
      adult.CreateSyntheticDataAdult,
      census_kdd.CreateSyntheticDataCensusKdd,
      credit.CreateSyntheticDataCredit,
      covertype.CreateSyntheticDataCovertype,
      intrusion.CreateSyntheticDataIntrusion,
      mnist12.CreateSyntheticDataMnist12,
      mnist28.CreateSyntheticDataMnist28,
      news.CreateSyntheticDataNews,
  ]
  available = {factory.__name__.removeprefix('CreateSyntheticData'): factory for factory in datasets}
  aliases = {'adult': 'Adult', 'census_kdd': 'CensusKdd', 'credit': 'Credit',
             'covertype': 'Covertype', 'intrusion': 'Intrusion', 'mnist12': 'Mnist12',
             'mnist28': 'Mnist28', 'news': 'News'}
  if dataset is not None:
    canonical = aliases.get(dataset.lower())
    if canonical is None:
      raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(aliases)}")
    datasets = [available[canonical]]
  seeds = (seed,) if seed is not None else (42, 43, 44)
  generators = (generator.upper(),) if generator is not None else ('CTGAN', 'TVAE')
  if any(name not in {'CTGAN', 'TVAE'} for name in generators):
    raise ValueError("generator must be 'CTGAN' or 'TVAE'")
  for run_seed in seeds:
    for feature_synthesizer in generators:
      for dataset_factory in datasets:
        dataset_factory(feature_synthesizer=feature_synthesizer, seed=run_seed).create_synthetic_data()



  # pass




def main():
  # adult.create_synthetic_data_adult()
  # news.create_synthetic_data_news()
  # census.create_synthetic_data_census()
  # covertype.create_synthetic_data_covertype()
  # intrusion.create_synthetic_data_intrusion()
  # credit.create_synthetic_data_credit()
  pass


def create_synthetic_simulated():
  simulated_names = ['grid', 'gridr', 'ring']
  for name in simulated_names:

    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_data(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='gaussianNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_SDV_gaussian_100k.csv', verbose=True,
                              npz_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_300_300.npz')


    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='categoricalNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_categorical/{name}_SDV_categorical_100k.csv', verbose=True,
                              npz_file_name=f'../SDGym-research/data/SDV_categorical/{name}_300_300.npz')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate corrected synthetic datasets.')
  parser.add_argument('--dataset', choices=['adult', 'census_kdd', 'credit', 'covertype', 'intrusion', 'mnist12', 'mnist28', 'news'])
  parser.add_argument('--seed', type=int, choices=[42, 43, 44])
  parser.add_argument('--generator', choices=['CTGAN', 'TVAE'])
  args = parser.parse_args()
  single_run(dataset=args.dataset, seed=args.seed, generator=args.generator)
