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
    def __init__(self):
        ds_name = 'mnist12'
        categorical_columns = []
        numerical_columns_pca_gmm = []
        super().__init__(ds_name, load_mnist28, 'label', categorical_columns=[], numerical_cols_pca_gmm=numerical_columns_pca_gmm,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=10000)
        
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
        # binarize the mnist28 data
        x_binarized = self.binarize(x_original)
        # resize the mnist28 data to mnist12
        x_resized = x_binarized.apply(self.resize, axis=1, result_type='expand')
        
        # Convert to DataFrame for easier manipulation
        x_resized = pd.DataFrame(x_resized.values.tolist())

        data = pd.concat([x_resized, y_original], axis=1)
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, test_size=self.test_size, missing_values_strategy=self.missing_values_strategy)
        # save train, test data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded train, test data to csv
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns
    
    def synthesize_categorical_pca_gmm_from_trained_model(self):
        """
        synthesize MNIST12 data from a trained model and pca_gmm. Assuming all the columns follow categorical distribution but don't use onehot.
        """
        # if target_synthesizer != 'pca_gmm':
        #     raise ValueError("This function is only intended to run pca_gmm for MNIST12 with categorical distribution.")
        print("Warning: This function is only intended to run pca_gmm for MNIST12 with categorical distribution.")

        synth_type, target_synthesizer = 'sdv_pca_gmm', 'pca_gmm'

        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()

        
        csv_file_name = self.paths['data_dir'] + f'onehot_{self.ds_name}_sdv_pca_gmm_cat_100k.csv'

        # load synthesizer
        synthesizer_file_name = self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer']
        synthesizer = load_synthesizer(synthesizer_file_name)
        # synthesize x'
        x_synthesized = synthesizer.sample(self.sample_size_to_synthesize)

        pca_gmm = PCA_GMM(xtrain, ytrain, x_synthesized, numerical_cols = [],
                        pca_n_components=0.99, gmm_n_components=10, verbose=True,
                        target_name = self.target_name, filename=csv_file_name, is_classification=self.is_classification)
        _, synthesized_data = pca_gmm.fit()

        # save synthesized data to csv
        check_directory(csv_file_name) # create directory if not exist
        synthesized_data.to_csv(csv_file_name, index=False)
        print(f"Successfully synthesized X and y data with {target_synthesizer}")
        return synthesized_data

