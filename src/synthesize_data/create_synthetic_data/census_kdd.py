import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype, load_census_kdd


class CreateSyntheticDataCensusKdd(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'census_kdd'
        categorical_columns = ['ACLSWKR', 'AHGA', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE',
       'AREORGN', 'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'FILESTAT',
       'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL', 'MIGSAME', 'PARENT',
       'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'VETQVA']

        super().__init__(ds_name, load_census_kdd, 'income', categorical_columns=categorical_columns,
                         sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=10000)
        
    def prepare_train_test(self):
        """
        map the y value to 0 and 1
        train-test split the census data
        handle missing values
        save the train-test data to csv
        train-test csv files are NOT one-hot encoded
        """
        # process census data
        x_original, y_original = self.load_data_func()
        print("Mapping y value to 0 and 1")
        y_original['income'] = y_original['income'].map({'-50000': 0,  '50000+.': 1})

        # drop columns with 99696/199523 missing values
        x_original = x_original.drop(columns=['MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSUN'] )

        data = pd.concat([x_original, y_original], axis=1)
        # drop rows with missing values
        # ends up with 190561/199523 rows
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy) 
        # save train, test data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded train, test data to csv
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns
                               