import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test

def create_synthetic_data_adult():
  # synthesize data for the adult dataset
  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  data = pd.concat([x_original, y_original], axis=1)
  print(data)
  xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name, test_size=0.2, random_state=42)
  xytrain = pd.concat([xtrain, ytrain], axis=1)
  print(xtrain)
  print(ytrain)
  print(xytrain)
  synthesize_adult_sdv = synthesize_data(xytrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_100k.csv', verbose=True,
                            show_network=True)



  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  data = pd.concat([x_original, y_original], axis=1)
  xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name, test_size=0.2, random_state=42)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_adult_sdv_gaussian_100k = synthesize_data(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_gaussian_100k.csv', verbose=True,
                            show_network=True)


  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  data = pd.concat([x_original, y_original], axis=1)
  xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name, test_size=0.2, random_state=42)
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
  synthesize_adult_sdv_categorical_100k = synthesize_from_trained_model(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/adult/adult_synthesizer_onlyX.pkl',
                            csv_file_name='../data/census/onehot_adult_sdv_categorical_100k.csv', verbose=True,
                            show_network=True)