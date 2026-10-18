from create_synthetic_data import CreateSyntheticData
from datasets import load_california_housing


class CreateSyntheticDataCaliforniaHousing(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer='CTGAN', seed=42, output_root=None):
        numerical_columns = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude',
        ]
        super().__init__(
            'california_housing', load_california_housing, 'MedHouseVal',
            categorical_columns=[], features_synthesizer=feature_synthesizer,
            numerical_cols_pca_gmm=numerical_columns,
            sample_size_to_synthesize=100_000, missing_values_strategy='drop',
            test_size=0.2, is_classification=False, seed=seed,
            output_root=output_root,
        )
