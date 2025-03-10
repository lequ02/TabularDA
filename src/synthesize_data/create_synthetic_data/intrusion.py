import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_intrusion

class CreateSyntheticDataIntrusion(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN'):
        ds_name = 'intrusion'
        target_name = 'target'
        categorical_columns=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
        super().__init__(ds_name, load_intrusion, target_name, categorical_columns=categorical_columns, features_synthesizer=feature_synthesizer,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2)
        