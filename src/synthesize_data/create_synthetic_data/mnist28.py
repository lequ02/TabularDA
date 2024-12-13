import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_mnist28
import numpy as np


class CreateSyntheticDataMnist28(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'mnist28'
        super().__init__(ds_name, load_mnist28, 'label', categorical_columns=[],
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2)
        
    def binarize(self, data):
        """
        binarize the mnist28 data
        """
        df = data.copy()
        df = np.where(df > 0, 1, 0)
        df = pd.DataFrame(df)
        return df

        
    def prepare_train_test(self):
        """
        train-test split the mnist28 data
        handle missing values
        save the train-test data to csv
        train-test csv files are NOT one-hot encoded
        """
        # process mnist28 data
        x_original, y_original = load_mnist28()
        # binarize the mnist28 data
        x_binarized = self.binarize(x_original)
        data = pd.concat([x_binarized, y_original], axis=1)
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy)
        # save train, test data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded train, test data to csv
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns