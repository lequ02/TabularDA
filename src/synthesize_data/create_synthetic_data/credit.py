import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype, load_credit

class CreateSyntheticDataCredit(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN'):
        ds_name = 'credit'
        categorical_columns = []
        super().__init__(ds_name, load_credit, 'Class', categorical_columns=categorical_columns, features_synthesizer=feature_synthesizer,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=10000)
        