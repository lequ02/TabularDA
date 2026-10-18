import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype

class CreateSyntheticDataCensus(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN', seed=42, output_root=None):
        ds_name = 'census'
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
        super().__init__(ds_name, load_census, 'income', categorical_columns=categorical_columns, features_synthesizer=feature_synthesizer,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, seed=seed, output_root=output_root)
        
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
        y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})

        data = pd.concat([x_original, y_original], axis=1)
        splits = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy)
        xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot = splits
        self.save_train_dev_test(xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot)
        self.save_split_manifest()
        return xtrain, xdev, xtest, ytrain, ydev, ytest, self.target_name, self.categorical_columns
