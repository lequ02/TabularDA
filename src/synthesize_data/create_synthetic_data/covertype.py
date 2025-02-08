import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype

class CreateSyntheticDataCovertype(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'covertype'

        ## sdv, gaussian, categorical originally generated with using empty categorical_columns [] 
        categorical_columns = []

    #     categorical_columns = ['Wilderness_Area1', 'Soil_Type1',
    #    'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
    #    'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
    #    'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
    #    'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
    #    'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
    #    'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
    #    'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
    #    'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
    #    'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
    #    'Soil_Type40', 'Wilderness_Area2', 'Wilderness_Area3',
    #    'Wilderness_Area4']

        numerical_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                            'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                            'Horizontal_Distance_To_Fire_Points']
        

        # originally used test_size=0.2
        super().__init__(ds_name, load_covertype, 'Cover_Type', categorical_columns=categorical_columns,
                        numerical_cols_pca_gmm=numerical_columns,
                        sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, is_classification=True)