import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler

# Constants
import constants

class data_loader:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.train_columns = None  # Store train columns for alignment

    def load_train_augment_data(self, train_option, augment_option):
        if train_option not in ['original', 'synthetic', 'mix'] or augment_option not in [None, 'ctgan', 'categorical', 'gaussian']:
            raise ValueError("Invalid train_option or augment_option")
        if train_option == 'original':
            train_df = pd.read_csv(self.paths['train_original'])
        elif train_option == 'synthetic':
            train_df = pd.read_csv(self.paths[train_option][augment_option])
        else:  # 'mix' option
            original_df = pd.read_csv(self.paths['train_original'])
            synthetic_df = pd.read_csv(self.paths['synthetic'][augment_option])

            train_df = pd.concat([original_df, synthetic_df], axis=0)
        
        print(f"Train dataset columns:\n{train_df.columns}")
        self.train_columns = train_df.columns[:-1]  # Store columns for test data alignment
        return self._load_data_in_batches(train_df)
        
    def load_test_data(self):
        # Load test data
        test_df = pd.read_csv(self.paths['test'])
        print(f"Test dataset preview:\n{test_df.head()}")  # Debugging

        if self.train_columns is None:
            raise ValueError("Training data must be loaded before test data to align columns.")

        # Align test columns with training columns
        missing_cols = set(self.train_columns) - set(test_df.columns)
        for col in missing_cols:
            test_df[col] = 0  

        # Drop extra columns
        extra_cols = set(test_df.columns) - set(self.train_columns)
        if extra_cols:
            test_df = test_df.drop(columns=extra_cols)

        print(f"Test dataset columns after alignment:\n{test_df.columns}")
        print(f"Test dataset shape after alignment: {test_df.shape}")

        # Prepare features and labels
        X = test_df[self.train_columns]  
        y = test_df.iloc[:, -1]

        # Standardize features
        X = self._standardize(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        y_tensor = torch.tensor(y.values, dtype=torch.float)

        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def _load_data_in_batches(self, df):
        X = df.iloc[:, :-1] 
        y = df.iloc[:, -1] 
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
            batch_y = torch.tensor(y[start:end], dtype=torch.float)

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), batch_size=self.batch_size)



