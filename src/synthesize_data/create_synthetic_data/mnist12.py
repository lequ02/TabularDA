import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_mnist28
import numpy as np
from PIL import Image


class CreateSyntheticDataMnist12(CreateSyntheticData.CreateSyntheticData):
    def __init__(self):
        ds_name = 'mnist12'
        super().__init__(ds_name, load_mnist28, 'label', categorical_columns=[],
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2)
        
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