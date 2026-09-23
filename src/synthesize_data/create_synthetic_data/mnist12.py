import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_mnist28
import numpy as np
from PIL import Image
from synthesizer import *


class CreateSyntheticDataMnist12(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN', seed=42, output_root=None):
        ds_name = 'mnist12'
        categorical_columns = []
        numerical_columns_pca_gmm = []
        super().__init__(ds_name, load_mnist28, 'label', categorical_columns=[], numerical_cols_pca_gmm=numerical_columns_pca_gmm, features_synthesizer=feature_synthesizer,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=10000, seed=seed, output_root=output_root)
        
    def binarize(self, data):
        """
        binarize the mnist28 data
        """
        df = data.copy()
        df = np.where(df > 0, 1, 0)
        df = pd.DataFrame(df)
        return df
    
    def resize(self, df_row):
        """
        resize the mnist28 to mnist12
        """
        # Convert to numpy array
        original_image = df_row.to_numpy()
        original_image = original_image.reshape(28, 28)
        # Convert to uint8 because PIL can't handle 64-bit integer
        original_image = original_image.astype(np.uint8)
        pil_bilinear = np.array(Image.fromarray(original_image).resize((12, 12)))
        return pil_bilinear.flatten()

        
    def prepare_train_test(self):
        """
        train-test split the mnist12 data
        handle missing values
        save the train-test data to csv
        train-test csv files are NOT one-hot encoded
        """
        # process mnist28 data
        x_original, y_original = load_mnist28()
        source = pd.concat([x_original, y_original], axis=1)
        x_binarized_source = self.binarize(x_original)
        x_12_source = x_binarized_source.apply(self.resize, axis=1, result_type='expand')
        x_12_source = pd.DataFrame(x_12_source.values.tolist()).astype(np.uint8)
        groups = pd.util.hash_pandas_object(x_12_source, index=False).to_numpy()
        train_source, dev_source, test_source = self.split_source_data(source, self.test_size, groups=groups)
        transformed = []
        for frame in (train_source, dev_source, test_source):
            x_binarized = self.binarize(frame.drop(columns=[self.target_name]))
            x_resized = x_binarized.apply(self.resize, axis=1, result_type='expand')
            x_resized = pd.DataFrame(x_resized.values.tolist())
            transformed.append(pd.concat([x_resized, frame[[self.target_name]].reset_index(drop=True)], axis=1))
        splits = self.transform_source_splits(transformed[0], transformed[1], transformed[2], self.missing_values_strategy)
        xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot = splits
        self.save_train_dev_test(xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot)
        self.save_split_manifest()
        return xtrain, xdev, xtest, ytrain, ydev, ytest, self.target_name, self.categorical_columns
    
    def synthesize_categorical_pca_gmm_from_trained_model(self):
        """
        synthesize MNIST12 data from a trained model and pca_gmm. Assuming all the columns follow categorical distribution but don't use onehot.
        """
        # if target_synthesizer != 'pca_gmm':
        #     raise ValueError("This function is only intended to run pca_gmm for MNIST12 with categorical distribution.")
        print("Warning: This function is only intended to run pca_gmm for MNIST12 with categorical distribution.")

        synth_type, target_synthesizer = 'sdv_pca_gmm', 'pca_gmm'

        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()

        
        csv_file_name = self.paths['data_dir'] + f'onehot_{self.ds_name}_sdv_pca_gmm_cat_100k.csv'

        # load synthesizer
        synthesizer_file_name = self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer']
        synthesizer = load_synthesizer(synthesizer_file_name)
        # synthesize x'
        import torch
        torch.manual_seed(self.seed)
        x_synthesized = synthesizer.sample(num_rows=self.sample_size_to_synthesize)

        pca_gmm = PCA_GMM(xtrain, ytrain, x_synthesized, numerical_cols = [],
                        pca_n_components=0.99, gmm_n_components=10, verbose=True,
                        target_name = self.target_name, filename=csv_file_name, is_classification=self.is_classification)
        _, synthesized_data = pca_gmm.fit()

        # save synthesized data to csv
        check_directory(csv_file_name) # create directory if not exist
        synthesized_data.to_csv(csv_file_name, index=False)
        print(f"Successfully synthesized X and y data with {target_synthesizer}")
        return synthesized_data

