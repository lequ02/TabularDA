import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot


def create_synthetic_data_news():
  """
  create train-test data for news dataset
  synthesize data using SDV only, SDV with GaussianNB, and SDV with CategoricalNB
  """
  ds_name = 'news'
  # synthesize data for the news dataset
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
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = prepare_train_test_news(paths['data_dir']+paths['train_csv'], paths['data_dir']+paths['test_csv'])

  # sdv only
  xytrain = pd.concat([xtrain, ytrain], axis=1)
  synthesize_news_sdv = synthesize_data(xytrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_only_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_only_csv'], verbose=True,
                            # show_network=True
                            )

  # sdv gaussian
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_news_data()
  synthesize_news_sdv_gaussian_100k = synthesize_data(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_gaussian_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_gaussian_csv'], verbose=True,
                            # show_network=True
                            )

  # sdv categorical
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_news_data()
  synthesize_news_sdv_categorical_100k = synthesize_from_trained_model(xtrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name= paths['synthesizer_dir']+paths['sdv_categorical_synthesizer'],
                            csv_file_name= paths['data_dir']+paths['sdv_categorical_csv'], verbose=True,
                            # show_network=True
                            )


def prepare_train_test_news(save_train_as, save_test_as):
  """
  map the y value to 0 and 1
  train-test split the news data
  handle missing values
  save the train-test data to csv
  train-test csv files are NOT one-hot encoded
  """
  # process data
  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = []

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


def read_news_data():
  """
  read train and test data from csv files
  """
  train_csv="..\\data\\news\\news_train.csv"
  test_csv="..\\data\\news\\news_test.csv"
  target_name=' shares'
  categorical_columns=[]

  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = read_train_test_csv.read_train_test_csv(train_csv, test_csv,
   target_name=target_name, categorical_columns=categorical_columns)

  return xtrain, xtest, ytrain, ytest, target_name, categorical_columns