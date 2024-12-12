import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from synthesizer import *
# # from onehot import onehot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
# from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot


class CreateSyntheticDataAdult(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'adult'
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        super().__init__(ds_name, load_adult, 'income', categorical_columns=categorical_columns,                         
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2)
        

    def prepare_train_test(self, categorical_columns, missing_values_strategy='drop', test_size=0.2):
        """
        map the y value to 0 and 1
        train-test split the adult data
        handle missing values
        save the train-test data to csv
        train-test csv files are NOT one-hot encoded
        """
        # process adult data
        x_original, y_original = load_adult()
        target_name = y_original.columns[0]
        print("Mapping y value to 0 and 1")
        y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                                    'relationship', 'race', 'sex', 'native-country']


        x_original, y_original = self.load_data_func()
        data = pd.concat([x_original, y_original], axis=1)
        # xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=self.target_name, test_size=test_size, random_state=42, stratify=y_original, categorical_columns=categorical_columns)
        # xtrain, ytrain = handle_missing_values.handle_missing_values(xtrain, ytrain, target_name=self.target_name, strategy=missing_values_strategy)
        # xtest, ytest = handle_missing_values.handle_missing_values(xtest, ytest, target_name=self.target_name, strategy=missing_values_strategy)
        
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy)
        
        # print("\n\n paths:", self.paths)
        # print(self.paths['train_csv'])
        # print(self.paths['test_csv'])
        # save data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded data to csv
        # xtrain_onehot, xtest_onehot = onehot.onehot(xtrain, xtest, self.categorical_columns)
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns