import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test


def process_covertype_data():
    # process covertype data
    x_original, y_original = load_covertype()
    target_name = y_original.columns[0]
    categorical_columns = []
    
    # train test split
    data = pd.concat([x_original, y_original], axis=1)
    xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=target_name, test_size=0.2, random_state=42)
    return xtrain, xtest, ytrain, ytest, target_name, categorical_columns


def create_synthetic_data_covertype():
  paths = {'synthesizer_dir': '../sdv trained model/covertype/',
            'data_dir': '../data/covertype/',

            'sdv_only_synthesizer': 'covertype_synthesizer.pkl',
            'sdv_only_csv': 'covertype_sdv_100k.csv',

            'sdv_gaussian_synthesizer': 'covertype_synthesizer_onlyX.pkl',
            'sdv_gaussian_csv': 'covertype_sdv_gaussian_100k.csv',

            'sdv_categorical_synthesizer': 'covertype_synthesizer_onlyX.pkl',
            'sdv_categorical_csv': 'covertype_sdv_categorical_100k.csv'
            }

    # sdv only
  xtrain, xtest, ytrain, ytest, target_name, categorical_columns = process_covertype_data()
  xytrain = pd.concat([xtrain, ytrain], axis=1)
  synthesize_covertype_sdv = synthesize_data(xytrain, ytrain, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name=paths['synthesizer_dir']+paths['sdv_only_synthesizer'],
                            csv_file_name=paths['data_dir']+paths['sdv_only_csv'], verbose=True,
                            # show_network=True
                            )

    # sdv gaussian
  x_original, y_original = load_covertype()
  synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name=paths['synthesizer_dir']+paths['sdv_gaussian_synthesizer'],
                            csv_file_name=paths['data_dir']+paths['sdv_gaussian_csv'], verbose=True,
                            # show_network=True
                            )

    # sdv categorical
  x_original, y_original = load_covertype()
  synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name=paths['synthesizer_dir']+paths['sdv_categorical_synthesizer'],
                            csv_file_name=paths['data_dir']+paths['sdv_categorical_csv'], verbose=True,
                            # show_network=True
                            )