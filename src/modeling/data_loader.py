  
#     def print_sample_data(self):
#         print("\nFirst 5 samples from training set:")
#         for i, (inputs, labels) in enumerate(self.train_data):
#             if i < 5:
#                 print(f"Sample {i+1}:")
#                 print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
#                 print(f"Label: {labels[0].item()}")
#             else:
#                 break

#         print("\nFirst 5 samples from test set:")
#         for i, (inputs, labels) in enumerate(self.test_data):
#             if i < 5:
#                 print(f"Sample {i+1}:")
#                 print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
#                 print(f"Label: {labels[0].item()}")
#             else:
#                 break

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
        # print(f"Available keys in IN_DATA_PATHS: {list(constants.IN_DATA_PATHS.keys())}")

        self.paths = constants.IN_DATA_PATHS[dataset_name]


    def load_train_augment_data(self, train_option, augment_option):
     
        if train_option not in ['original', 'synthetic', 'mix'] or augment_option not in [None, 'ctgan', 'categorical', 'gaussian']:
            raise ValueError("Invalid train_option or augment_option")
        if train_option == 'original':
            return self. _load_data_in_batches(pd.read_csv(self.paths['train_original']))
        
        if train_option == 'synthetic':
            return self._load_data_in_batches(pd.read_csv(self.paths[train_option][augment_option]))
        
        return self._load_data_in_batches(pd.concat([pd.read_csv(self.paths['train_original']), pd.read_csv(self.paths[train_option][augment_option])], axis=0))
    
        
    def load_test_data(self):
        return pd.read_csv(self.paths['test'])
        
    def _load_data_in_batches(self, ds):
        ds = self._standardize(ds)
        X = ds.iloc[:, :-1] 
        y = ds.iloc[:, -1] 
        
        return self._distribute_in_batches(X.values,y.values)
        
    def _standardize(self, df):
        scaler = StandardScaler()

        return pd.DataFrame(scaler.fit_transform(df))
        
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
    

#     def balance_classes(self, X, y):
#         class_counts = y.value_counts()
#         majority_class = class_counts.idxmax()
#         minority_class = class_counts.idxmin()

#         majority_count = int(class_counts[minority_class] * (76 / 24))
#         minority_count = class_counts[minority_class]

#         # Ensure majority_count does not exceed the available samples
#         available_majority_count = class_counts[majority_class]
#         if majority_count > available_majority_count:
#             majority_count = available_majority_count

#         majority_samples = X[y == majority_class].sample(majority_count, random_state=42).reset_index(drop=True)
#         minority_samples = X[y == minority_class].reset_index(drop=True)

#         X_balanced = pd.concat([majority_samples, minority_samples]).reset_index(drop=True)
#         y_balanced = pd.concat([pd.Series([majority_class] * majority_count), pd.Series([minority_class] * minority_count)]).reset_index(drop=True)

#         balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
#         return balanced_data

#     def print_sample_data(self):
#         print("\nFirst 5 samples from training set:")
#         for i, (inputs, labels) in enumerate(self.train_data):
#             if i < 5:
#                 print(f"Sample {i+1}:")
#                 print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
#                 print(f"Label: {labels[0].item()}")
#             else:
#                 break

#         print("\nFirst 5 samples from test set:")
#         for i, (inputs, labels) in enumerate(self.test_data):
#             if i < 5:
#                 print(f"Sample {i+1}:")
#                 print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
#                 print(f"Label: {labels[0].item()}")
#             else:
#                 break