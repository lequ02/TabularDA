import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Constants
import constants

class data_loader:
    def __init__(self, dataset_name, batch_size, multi_y = True, problem_type = 'classification'):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.train_columns = None  # Store train columns for alignment
        self.multi_y = multi_y
        self.problem_type = problem_type

    def drop_index_col(self, df):
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        return df
    
    def load_train_augment_data(self, train_option, augment_option, mix_ratio = -1, n_sample = -1, validation = 0):
        # if train_option not in ['original', 'synthetic', 'mix'] or augment_option not in [None, 'ctgan', 'categorical', 'gaussian', 'pca_gmm', 'pca_gmm_cat', 'pca_gmm_num']:
        #     raise ValueError("Invalid train_option or augment_option")
        if train_option == 'original':
            train_df = pd.read_csv(self.paths['train_original'])
            train_df = self.drop_index_col(train_df)
        elif train_option == 'synthetic':
            train_df = pd.read_csv(self.paths[train_option][augment_option])
            train_df = train_df[sorted(train_df.columns)]
            train_df = self.drop_index_col(train_df)
        else:  # 'mix' option
            original_df = pd.read_csv(self.paths['train_original'])
            original_df = self.drop_index_col(original_df)
            # sort original and synthetic alphabetically so the columns are in the same order
            original_df = original_df.reindex(sorted(original_df.columns), axis=1)

            print("\n\noriginal columns")
            print(original_df.columns)

            # print(self.paths)
            # print(self.paths['synthetic'])
            # print(self.paths['synthetic'][augment_option])

            synthetic_df = pd.read_csv(self.paths['synthetic'][augment_option])
            synthetic_df = self.drop_index_col(synthetic_df)
            # sort original and synthetic alphabetically so the columns are in the same order
            synthetic_df = synthetic_df.reindex(sorted(synthetic_df.columns), axis=1)

            print("\n\nsynthetic columns")
            print(synthetic_df.columns)


            train_df = self.concat(original_df, synthetic_df, axis = 0, concat_ratio = mix_ratio, n_sample = n_sample) #pd.concat([original_df, synthetic_df], axis=0)
        
        # sort train and test alphabetically so the columns are in the same order
        train_df = train_df.reindex(sorted(train_df.columns), axis=1)

        # train_df.to_csv("dummy_train_df.csv", index=False)

        print("\n\ntrain columns")
        print(train_df.columns)


        # print(f"Train dataset columns:\n{train_df.columns}")
        self.train_columns = train_df.columns  # Store columns for test data alignment
        
        if self.problem_type == 'classification':
            stratify_column = self.paths['target_name']
        elif self.problem_type == 'regression':
            stratify_column = None
        else:
            ValueError(f"This {self.problem_type} is not supported!")


        train_df, dev_df = self.split_data(train_df, stratify_column, validation)
        return self._load_data_in_batches(train_df), self._load_data_in_batches(dev_df)
        
    def load_test_data(self):
        # Load test data
        # print("\n\n path", self.paths['test'])
        test_df = pd.read_csv(self.paths['test'])
        
        # print(f"Test dataset preview:\n{test_df.head()}")  # Debugging
        # print(test_df.head(2))


        if self.train_columns is None:
            raise ValueError("Training data must be loaded before test data to align columns.")

        # Align test columns with training columns
        missing_cols = set(self.train_columns) - set(test_df.columns)
        print(missing_cols)
        for col in missing_cols:
            test_df[col] = 0  

        # sort train and test alphabetically so the columns are in the same order
        test_df = test_df.reindex(sorted(test_df.columns), axis=1)
        
        # Drop extra columns
        extra_cols = set(test_df.columns) - set(self.train_columns)
        
        if extra_cols:
            test_df = test_df.drop(columns=extra_cols)

        # print(f"Test dataset columns after alignment:\n{test_df.columns}")
        # print(f"Test dataset shape after alignment: {test_df.shape}")
            
        # test_df = test_df[sorted(test_df.columns)]
        print("\n\ntest columns")
        print(test_df.columns)

        # test_df.to_csv("dummy_test_df.csv", index=False)
        

        y = test_df[self.paths['target_name']]
        X = test_df.drop(self.paths['target_name'], axis=1)


        # Standardize features
        X = self._standardize(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        
        if (self.multi_y != True):
                y_tensor = torch.tensor(y.values, dtype=torch.float)
        else:
                y_tensor = torch.tensor(y.values, dtype=torch.long)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def _load_data_in_batches(self, df):
        

        y = df[self.paths['target_name']]
        X = df.drop(self.paths['target_name'], axis=1)
        X = self._standardize(X)
        return self._distribute_in_batches(X.values, y.values)
        
    def _standardize(self, df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
    def _distribute_in_batches(self, X, y):
        print("Type X", type(X))
        print("Type y", type(y))
        print("Shape 0 X", len(X))
        print("Shape 1 X", len(X[0]))
        
        num_batch = int(len(X) / self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X[start:end], dtype=torch.float)
            
            if (self.multi_y != True):
                batch_y = torch.tensor(y[start:end], dtype=torch.float)
            else:
                batch_y = torch.tensor(y[start:end], dtype=torch.long)
            
            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), batch_size=self.batch_size)

    def split_data(self, df, stratify_column = None, validation = 0):
        
        if validation > 1:
            validation = int(validation)
            
        if stratify_column != None:
            return train_test_split(df, test_size = validation, 
            stratify = df[stratify_column], random_state=42)
        else:
            return train_test_split(df, test_size = validation, random_state=42)

    def concat(self, df1, df2, axis = 0, concat_ratio = -1, n_sample = -1):
        #concat_ratio: percentage of df1 => 1-concat_ratio  = percentage of df2
        #df1 is for original data and df2 is for augmented data
        
        if (n_sample == -1) or (concat_ratio == -1): #take all original samples and all augmented samples
            return pd.concat([df1, df2], axis = 0, join = 'inner')
        
        df1_sample, df2_sample = df1.shape[0], df2.shape[0]

        if n_sample <= 0:
            ValueError("Number of sample must be an integer greater than 0 or equal to -1")

        if (concat_ratio >= 0) and (concat_ratio<=1):

            df1_sample = int(n_sample * concat_ratio)
            
        elif int(concat_ratio) == concat_ratio:
            df1_sample = concat_ratio
        else:
            ValueError("Concat_ratio must be a positive integer or in the range [0..1]!")
        
        df2_sample = n_sample - df1_sample

        if self.problem_type == 'classification':
            stratify_column = self.paths['target_name']
        elif self.problem_type == 'regression':
            stratify_column = None
        else:
            ValueError(f"This {self.problem_type} is not supported!")

        if (df1_sample <df1.shape[0]):
            _, df1 = self.split_data(df1, stratify_column, df1_sample)
        if df1_sample < df2.shape[0]:
            _, df2 = self.split_data(df2, stratify_column, df2_sample)
        
        return pd.concat([df1, df2], axis=0, join = 'inner')

            
#        return pd.concat([df1.sample(n = df1_sample, random_state = 42, axis = 'index'), 
#            df2.sample(n = df2_sample, random_state=42, axis = 'index', )], axis=0)








import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np


class DataLoaderMNIST:
    def __init__(self, dataset_name, batch_size, multi_y=True, problem_type="classification"):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.train_columns = None  # Store train columns for alignment
        self.multi_y = multi_y
        self.problem_type = problem_type

    def drop_index_col(self, df):
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        return df

    def load_train_augment_data(self, train_option, augment_option, mix_ratio=-1, n_sample=-1, validation=0):
        if train_option == "original":
            train_df = pd.read_csv(self.paths["train_original"])
            train_df = self.drop_index_col(train_df)
        elif train_option == "synthetic":
            train_df = pd.read_csv(self.paths[train_option][augment_option])
            train_df = self.drop_index_col(train_df)
        else:  # 'mix' option
            # original_df = pd.read_csv(self.paths["train_original"])
            # original_df = self.drop_index_col(original_df)

            raise NotImplementedError("Mixing not implemented for MNIST.")

        # Sort columns alphabetically and store for test alignment
        target_col = self.paths["target_name"]
        self.train_columns = sorted(train_df.columns.drop(target_col))
        train_df = train_df[[target_col] + self.train_columns]

        # Separate features and labels
        y = train_df[target_col].values
        X = train_df.drop(columns=[target_col]).values
        print(f"Shape of X after loading: {X.shape}")

        # Check dimensions of X
        if X.shape[1] != 784:
            raise ValueError(f"Expected 784 features for MNIST but got {X.shape[1]} features")

        # Split into training and validation sets
        train_X, val_X, train_y, val_y = self.split_data(X, y, validation=validation)

        # Return DataLoaders for training and validation
        train_loader = self._load_data_in_batches(train_X, train_y)
        val_loader = self._load_data_in_batches(val_X, val_y)
        return train_loader, val_loader


    def load_test_data(self):
        """
        Load test data, align with training data columns, and return a DataLoader.
        """
        # Load test data
        test_df = pd.read_csv(self.paths["test"])
        test_df = self.drop_index_col(test_df)

        # Check if training columns are defined
        if self.train_columns is None:
            raise ValueError("Training data must be loaded before test data to align columns.")

        # Align test columns with training columns
        target_col = self.paths["target_name"]
        test_features = sorted(self.train_columns)  # Ensure the same order as training data

        # Add missing columns as 0
        for col in test_features:
            if col not in test_df.columns:
                test_df[col] = 0

        # Drop extra columns not in training data
        test_df = test_df[[target_col] + test_features]

        # Separate features and labels
        y = test_df[target_col].values
        X = test_df.drop(columns=[target_col]).values

        # Check dimensions of X
        if X.shape[1] != 784:
            raise ValueError(f"Unexpected number of features: {X.shape[1]}. Expected 784 for MNIST.")

        # Return DataLoader
        return self._load_data_in_batches(X, y, shuffle=False)


    def _load_data_in_batches(self, X, y, shuffle=True):
        # Reshape X to [batch_size, channels, height, width] for MNIST
        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Ensure the input has 784 features (28x28)
        if num_features != 784:
            raise ValueError(f"Unexpected number of features: {num_features}. Expected 784 for MNIST.")

        # # Normalize the data to [0, 1] range if it isn't already
        # X = X / 255.0 if X.max() > 1.0 else X

        # Reshape into [N, 1, 28, 28] - Note the order of dimensions
        X = X.reshape(num_samples, 784)  # First ensure it's flat
        X = X.reshape(num_samples, 1, 28, 28)  # Then reshape to correct dimensions

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if not self.multi_y:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)



    def split_data(self, X, y, validation=0, stratify_column=None):
        """
        Splits the dataset into training and validation sets.
        :param X: Features (numpy array or pandas DataFrame).
        :param y: Labels (numpy array or pandas Series).
        :param validation: Fraction or number of samples to use for validation.
        :param stratify_column: Column to use for stratification.
        :return: train_X, val_X, train_y, val_y
        """
        if validation > 1:  # If validation is given as an absolute number
            validation = int(validation)
        elif 0 < validation <= 1:  # If validation is a fraction
            validation = validation
        else:
            raise ValueError("Validation must be a positive integer or a float in the range (0, 1].")

        if stratify_column is not None:
            stratify = y
        else:
            stratify = None

        # Split into train and validation sets
        train_X, val_X, train_y, val_y = train_test_split(
            X, y, test_size=validation, stratify=stratify, random_state=42
        )
        return train_X, val_X, train_y, val_y


    def concat(self, df1, df2, axis=0, concat_ratio=-1, n_sample=-1):
        raise NotImplementedError("Concat functionality is not required for MNIST.")

