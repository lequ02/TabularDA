import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype


class CreateSyntheticDataAdult(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN'):
        ds_name = 'adult'
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        super().__init__(ds_name, load_adult, 'income', categorical_columns=categorical_columns, features_synthesizer=feature_synthesizer,                    
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2)
        

    def prepare_train_test(self):
        """
        map the y value to 0 and 1
        train-test split the adult data
        handle missing values
        save the train-test data to csv
        train-test csv files are NOT one-hot encoded
        """
        # process adult data
        x_original, y_original = self.load_data_func()
        print("Mapping y value to 0 and 1")
        y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
        data = pd.concat([x_original, y_original], axis=1)
    
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy)

        # save data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded data to csv
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns