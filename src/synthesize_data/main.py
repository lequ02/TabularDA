from synthesizer import *
# from synthesizer import *
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_adult, load_news, load_census, load_covertype

def single_run():
  create_synthetic_data_covertype()

def main():
  create_synthetic_data_census()
  create_synthetic_data_adult()
  create_synthetic_data_news()

def create_synthetic_data_covertype():
  target_name = 'Cover_Type'
  categorical_columns = []

  x_original, y_original = load_covertype()
  x_original = pd.concat([x_original, y_original], axis=1)
  synthesize_covertype_sdv = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/covertype/covertype_synthesizer.pkl',
                            csv_file_name='../data/covertype/covertype_sdv_100k.csv', verbose=True,
                            # show_network=True
                            )

  x_original, y_original = load_covertype()
  synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/covertype/covertype_synthesizer_onlyX.pkl',
                            csv_file_name='../data/covertype/covertype_sdv_gaussian_100k.csv', verbose=True,
                            # show_network=True
                            )

  x_original, y_original = load_covertype()
  synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/covertype/covertype_synthesizer_onlyX.pkl',
                            csv_file_name='../data/covertype/covertype_sdv_categorical_100k.csv', verbose=True,
                            # show_network=True
                            )


def create_synthetic_data_census():
  # synthesize data for the census dataset
  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_census_sdv = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=1000_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/census/census_synthesizer.pkl',
                            csv_file_name='../data/census/onehot_census_sdv_1mil.csv', verbose=True,
                            show_network=True)


  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  # x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_census_sdv_gaussian_1mil = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=1000_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/census/census_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_census_sdv_gaussian_1mil.csv', verbose=True,
                            show_network=True)


  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  # x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_census_sdv_categorical_1mil = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=1000_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/census/census_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_census_sdv_categorical_1mil.csv', verbose=True,
                            show_network=True)

def create_synthetic_data_adult():
  # synthesize data for the adult dataset
  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_adult_sdv = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_100k.csv', verbose=True,
                            show_network=True)


  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  # x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_adult_sdv_gaussian_100k = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_gaussian_100k.csv', verbose=True,
                            show_network=True)


  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  # x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_adult_sdv_categorical_100k = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_categorical_100k.csv', verbose=True,
                            show_network=True)

def create_synthetic_data_news():
  # synthesize data for the news dataset 
  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer.pkl',
                            csv_file_name='../data/news/news_sdv_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_gaussian_100k = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                            csv_file_name='../data/news/news_sdv_gaussian_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_categorical_100k = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                            csv_file_name='../data/news/news_sdv_categorical_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_BN_MLE = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='BN_MLE',
                             target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                             csv_file_name='../data/news/news_BN_BE.csv', BN_filename='../data/news/news_BN_MLE_model.pkl', verbose=True,
                             show_network=True)


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
  single_run()