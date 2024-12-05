import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot


class CreateSyntheticData:
    def __init__(self, ds_name, load_data_func, target_name, categorical_columns, 
                 sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2):
        self.ds_name = ds_name
        self.load_data_func = load_data_func
        self.target_name = target_name
        self.categorical_columns = categorical_columns
        self.sample_size_to_synthesize = sample_size_to_synthesize
        self.missing_values_strategy = missing_values_strategy
        self.test_size = test_size
        self.paths = {
            'synthesizer_dir': f'../sdv trained model/{ds_name}/',
            'data_dir': f'../data/{ds_name}/',
            'train_csv': f'{ds_name}_train.csv',
            'test_csv': f'{ds_name}_test.csv',
            'train_csv_onehot': f'onehot_{ds_name}_train.csv',
            'test_csv_onehot': f'onehot_{ds_name}_test.csv',
            'sdv_only_synthesizer': f'{ds_name}_synthesizer.pkl',
            'sdv_only_csv': f'onehot_{ds_name}_sdv_100k.csv',
            'sdv_gaussian_synthesizer': f'{ds_name}_synthesizer_onlyX.pkl',
            'sdv_gaussian_csv': f'onehot_{ds_name}_sdv_gaussian_100k.csv',
            'sdv_categorical_synthesizer': f'{ds_name}_synthesizer_onlyX.pkl',
            'sdv_categorical_csv': f'onehot_{ds_name}_sdv_categorical_100k.csv'
        }

    def create_synthetic_data(self):
        self.create_synthetic_data_sdv_only()
        self.create_synthetic_data_sdv_gaussian()
        self.create_synthetic_data_sdv_categorical()

    def create_synthetic_data_sdv_only(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.prepare_train_test(self.missing_values_strategy, self.test_size)
        # need this line or the xy data will be double one-hot encoded. dont know why
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_data(xtrain, ytrain, categorical_columns, 'sdv_only', '')

    def create_synthetic_data_sdv_gaussian(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_data(xtrain, ytrain, categorical_columns, 'sdv_gaussian', 'gaussianNB')

    def create_synthetic_data_sdv_categorical(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_data(xtrain, ytrain, categorical_columns, 'sdv_categorical', 'categoricalNB')

    def prepare_train_test(self, missing_values_strategy='drop', test_size=0.2):
        """
        only for the first time, prepare data, train-test split data, handle missing values, and save to csv
        after that, read the data from the csv files
        """
        x_original, y_original = self.load_data_func()
        data = pd.concat([x_original, y_original], axis=1)
        xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=self.target_name, test_size=test_size, random_state=42)
        xtrain, ytrain = handle_missing_values.handle_missing_values(xtrain, ytrain, target_name=self.target_name, strategy=missing_values_strategy)
        xtest, ytest = handle_missing_values.handle_missing_values(xtest, ytest, target_name=self.target_name, strategy=missing_values_strategy)
        # save data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['train_csv'], self.paths['test_csv'])
        # save onehot encoded data to csv
        xtrain_onehot, xtest_onehot = onehot.onehot(xtrain, xtest, self.categorical_columns)
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['train_csv_onehot'], self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns

    def read_data(self):
        return read_train_test_csv.read_train_test_csv(self.paths['data_dir'] + self.paths['train_csv'], self.paths['data_dir'] + self.paths['test_csv'],
                                                       target_name=self.target_name, categorical_columns=self.categorical_columns)

    def synthesize_data(self, xtrain, ytrain, categorical_columns, synth_type, target_synthesizer):
        xytrain = pd.concat([xtrain, ytrain], axis=1)
        synthesize_data(xytrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                        target_name=self.target_name, synthesizer_file_name=self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer'],
                        csv_file_name=self.paths['data_dir'] + self.paths[f'{synth_type}_csv'], verbose=True)

    def save_to_csv(self, xtrain, ytrain, xtest, ytest, train_csv, test_csv):
        train_set = pd.concat([xtrain, ytrain], axis=1)
        test_set = pd.concat([xtest, ytest], axis=1)
        check_directory.check_directory(train_csv)
        check_directory.check_directory(test_csv)
        train_set.to_csv(train_csv, index=False)
        test_set.to_csv(test_csv, index=False)
        print(f"Data saved to csv at {train_csv} and {test_csv}")


