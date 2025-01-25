# from synthesize_data.create_synthetic_data import adult_old
from synthesizer import *
import sys
import os
import pandas as pd
from create_synthetic_data import news, census, covertype, intrusion, credit, adult, mnist28, mnist12

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from datasets import load_adult, load_news, load_census, load_covertype, load_intrusion

def single_run():
  # adult.CreateSyntheticDataAdult().create_synthetic_data()
  # census.CreateSyntheticDataCensus().create_synthetic_data()
  # news.create_synthetic_data_news()
  # covertype.create_synthetic_data_covertype()
  # intrusion.create_synthetic_data_intrusion()
  # credit.create_synthetic_data_credit()
  # mnist28.CreateSyntheticDataMnist28().create_synthetic_data()
  # mnist12.CreateSyntheticDataMnist12().create_synthetic_data()



  adult.CreateSyntheticDataAdult().create_synthetic_data_pca_gmm()
  census.CreateSyntheticDataCensus().create_synthetic_data_pca_gmm()
  covertype.CreateSyntheticDataCovertype().create_synthetic_data_pca_gmm()

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
  # main()
  print("RUN THIS SCRIPT FROM /SRC/")
  print("the command should be python synthesize_data/main.py")
  single_run()