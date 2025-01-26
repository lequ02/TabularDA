import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype, load_credit

class CreateSyntheticDataNews(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'news'
        categorical_columns = []
        super().__init__(ds_name, load_news, ' shares', categorical_columns=categorical_columns,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, is_classification=False)
        
    # def create_synthetic_data_pca_gmm(self):
    #     xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
    #     self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, 'sdv_pca_gmm', 'pca_gmm', is_classification=False)
        