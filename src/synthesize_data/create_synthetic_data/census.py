import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
import pandas as pd

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