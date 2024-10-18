import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot

def create_synthetic_data_census():
  """
  create train-test data for census dataset
  synthesize data using SDV only, SDV with GaussianNB, and SDV with CategoricalNB
  """
  ds_name = 'census'
  # synthesize data for the census dataset
  paths = {'synthesizer_dir': f'../sdv trained model/{ds_name}/',
            'data_dir': f'../data/{ds_name}/',

            'train_csv': f'{ds_name}_train.csv',
            'test_csv': f'{ds_name}_test.csv',

            'sdv_only_synthesizer': f'{ds_name}_synthesizer.pkl',
            'sdv_only_csv': f'onehot_{ds_name}_sdv_100k.csv',

            'sdv_gaussian_synthesizer': f'{ds_name}_synthesizer_onlyX.pkl',
            'sdv_gaussian_csv': f'onehot_{ds_name}_sdv_gaussian_100k.csv',

            'sdv_categorical_synthesizer': f'{ds_name}_synthesizer_onlyX.pkl',
            'sdv_categorical_csv': f'onehot_{ds_name}_sdv_categorical_100k.csv'
            }

  # save train-test data to csv files
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = prepare_train_test_census(paths['data_dir']+paths['train_csv'], paths['data_dir']+paths['test_csv'])

  # sdv only
  xytrain = pd.concat([xtrain, ytrain], axis=1)
  synthesize_census_sdv = synthesize_data(xytrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_only_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_only_csv'], verbose=True,
                            # show_network=True
                            )


  # sdv gaussian
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_census_data(paths['data_dir']+paths['train_csv'], paths['data_dir']+paths['test_csv'],
                                                                                    target_name=target_name, categorical_columns=categorical_columns)
  synthesize_census_sdv_gaussian_100k = synthesize_data(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_gaussian_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_gaussian_csv'], verbose=True,
                            # show_network=True
                            )

  # sdv categorical
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_census_data(paths['data_dir']+paths['train_csv'], paths['data_dir']+paths['test_csv'],
                                                                                    target_name=target_name, categorical_columns=categorical_columns)
  synthesize_census_sdv_categorical_100k = synthesize_from_trained_model(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_categorical_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_categorical_csv'], verbose=True,
                            # show_network=True
                            )
  


def prepare_train_test_census(save_train_as, save_test_as):
  """
  map the y value to 0 and 1
  train-test split the census data
  handle missing values
  save the train-test data to csv
  train-test csv files are NOT one-hot encoded
  """
  # process census data
  x_original, y_original = load_census()
  target_name = y_original.columns[0]
  y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
  categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']

  # train test split
  data = pd.concat([x_original, y_original], axis=1)
  xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=target_name, test_size=0.2, random_state=42)

  # handle missing values
  xtrain, ytrain = handle_missing_values.handle_missing_values(xtrain, ytrain, target_name=target_name, strategy='drop')
  xtest, ytest = handle_missing_values.handle_missing_values(xtest, ytest, target_name=target_name, strategy='drop')

  # save train-test to csv
  train_set = pd.concat([xtrain, ytrain], axis=1)
  test_set = pd.concat([xtest, ytest], axis=1)
  check_directory.check_directory(save_train_as) # check if the directory exists, if not create it
  check_directory.check_directory(save_test_as)
  train_set.to_csv(save_train_as, index=False)
  test_set.to_csv(save_test_as, index=False)

  return xtrain, xtest, ytrain, ytest, target_name, categorical_columns


def read_census_data(train_csv, test_csv, target_name, categorical_columns):
  """
  read train and test data from csv files
  """
  # train_csv="..\\data\\census\\census_train.csv"
  # test_csv="..\\data\\census\\census_test.csv"
  # target_name='income'
  # categorical_columns=['workclass', 'education', 'marital-status', 'occupation',
  #                     'relationship', 'race', 'sex', 'native-country']

  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_train_test_csv.read_train_test_csv(train_csv, test_csv,
   target_name=target_name, categorical_columns=categorical_columns)

  return xtrain, xtest, ytrain, ytest, target_name, categorical_columns
