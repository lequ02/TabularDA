# #BEGIN of contants
# #----
# #ADULT_FOLDER = "../data/adult/"
# #ADULT_TRAIN_ORIGINAL_FNAME = "adult_train.csv"
# #ADULT_TEST_ORIGINAL_FNAME = "adult_test.csv"
# #ADULT_CTGAN_FNAME = "onehot_adult_sdv_100k.csv"
# #ADULT_CTGAN_GAUSSIAN_FNAME = "onehot_adult_sdv_gaussian_100k.csv"
# #ADULT_CTGAN_CAT_FNAME = "onehot_adult_sdv_categorical_100k.csv"
# #----

# #END of contants 

# import pandas as pd
# import torch
# import random
# from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
# from commons.onehot import onehot
# # from datasets import load_adult, load_news, load_census
# from sklearn.preprocessing import StandardScaler

# class data_loader:
#     def __init__(self, dataset_name, train_option, test_option,
#                 batch_size, numerical_columns=[]):
#         self.dataset_name = dataset_name
#         self.batch_size = batch_size
#         self.numerical_columns = numerical_columns
#         self.train_option = train_option
#         self.test_option = test_option
  

#         # Load and print shapes before batching
#         train_data, test_data = self.load_datasets(self.train_option)
#         print(f"Total train samples: {train_data.shape[0]}")
#         print(f"Total test samples: {test_data.shape[0]}")

#         self.test_data = self.load_data_in_batches(test_data)
#         self.train_data = self.load_data_in_batches(train_data)

#     def load_data_in_batches(self, ds):
#         ds = self.standardize(ds, self.numerical_columns)
#         x = ds.iloc[:, :-1]  # all columns except the last one
#         y = ds.iloc[:, -1]  # the last column
        
#         return self.distribute_in_batches(x,y)
    
#     def distribute_in_batches(self, X, y):
#         num_batch = int(len(X) / self.batch_size)
#         batches = []

#         for i in range(num_batch):
#             start = i * self.batch_size
#             end = start + self.batch_size

#             batch_X = torch.tensor(X.iloc[start:end].values, dtype=torch.float)
#             batch_y = torch.tensor(y.iloc[start:end].values, dtype=torch.float)  # Assuming y is numerical for regression

#             batch = TensorDataset(batch_X, batch_y)
#             batches.append(batch)

#         return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=self.batch_size)

#     def load_datasets(self, option, augment_option=None):
#         if option == 'original':
#             ds_train, ds_test = self.load_clean_original_data()
#             print(f"Original dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
#             return ds_train, ds_test

#         elif option == 'synthetic':
#             if augment_option == 'ctgan':
#                 ds_train_augmented = self.load_ctgan_augmented_data()
#             elif augment_option == 'categorical':
#                 ds_train_augmented = self.load_categorical_augmented_data()
#             elif augment_option == 'gaussian':
#                 ds_train_augmented = self.load_gaussian_augmented_data()
#             return ds_train_augmented

#         elif option == 'mix':
#             train_original, test_original = self.load_clean_original_data()

#             if augment_option == 'ctgan':
#                 ds_train_augmented = self.load_ctgan_augmented_data()
#             elif augment_option == 'categorical':
#                 ds_train_augmented = self.load_categorical_augmented_data()
#             elif augment_option == 'gaussian':
#                 ds_train_augmented = self.load_gaussian_augmented_data()
            
#             train_mix = pd.concat([train_original, ds_train_augmented], axis = 0)
            
#             print(f"Mixed dataset - Train shape: {train_mix.shape}, Test shape: {test_original.shape}")
#             return train_mix, test_original
        
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

#     def load_clean_original_data(self):
#         if self.dataset_name == 'adult':
#             ds_train = pd.read_csv('../data/adult/adult_train.csv')
#             ds_test = pd.read_csv('../data/adult/adult_test.csv')
#             ds_train_one_hot, ds_test_one_hot = onehot(ds_train, ds_test, ['workclass', 'education', 'marital-status', 'occupation',
#                                                'relationship', 'race', 'sex', 'native-country'], verbose=False)
#             # x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1) ## Don't know why we need it Jibran?
#             print("Adult dataset columns:", ds_train_one_hot.columns)
#             return ds_train_one_hot, ds_test_one_hot

#         elif self.dataset_name == 'census':
#             ds_train = pd.read_csv('../data/census/census_train.csv')
#             ds_test = pd.read_csv('../data/census/census_test.csv')
#             ds_train_one_hot, ds_test_one_hot = onehot(ds_train, ds_test, ['workclass', 'education', 'marital-status', 'occupation',
#                                                'relationship', 'race', 'sex', 'native-country'], verbose=False)
#             # x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1) ## Don't know why we need it Jibran?
#             print("Consus dataset columns:", ds_train_one_hot.columns)
#             return ds_train_one_hot, ds_test_one_hot

#         elif self.dataset_name == 'news':
#             ds_train = pd.read_csv('../data/news/news_train.csv')
#             ds_test = pd.read_csv('../data/news/news_test.csv')
           
#             return ds_train_one_hot, ds_test_one_hot
        
#     def load_ctgan_augmented_data(self):
#         if self.dataset_name == 'adult':
#             ds_augmented = pd.read_csv('../data/adult/onehot_adult_sdv_100k.csv')
           
#             print("Adult dataset columns:", ds_augmented.columns)
#             return ds_augmented

#         elif self.dataset_name == 'census':
#             ds_augmented = pd.read_csv('../data/adult/onehot_census_sdv_100k.csv')
           
#             print("Census dataset columns:", ds_augmented.columns)
#             return ds_augmented

#         elif self.dataset_name == 'news':
#             ds_augmented = pd.read_csv('../data/adult/onehot_news_sdv_100k.csv')
           
#             print("News dataset columns:", ds_augmented.columns)
#             return ds_augmented
        
#     def load_categorical_augmented_data(self):
#         if self.dataset_name == 'adult':
#             ds_augmented = pd.read_csv('../data/adult/onehot_adult_sdv_categorical_100k.csv')
           
#             print("Adult dataset columns:", ds_augmented.columns)
#             return ds_augmented
        
#         ## HAVEN'T HAD THIS YET
#         # elif self.dataset_name == 'census':
#         #     ds_augmented = pd.read_csv('../data/adult/onehot_census_sdv_100k.csv')
           
#         #     print("Census dataset columns:", ds_augmented.columns)
#         #     return ds_augmented

#         elif self.dataset_name == 'news':
#             ds_augmented = pd.read_csv('../data/adult/onehot_news_sdv_categorical_100k.csv')
           
#             print("News dataset columns:", ds_augmented.columns)
#             return ds_augmented
    
#     def load_gaussian_augmented_data(self):
#         if self.dataset_name == 'adult':
#             ds_augmented = pd.read_csv('../data/adult/onehot_adult_sdv_gaussian_100k.csv')
           
#             print("Adult dataset columns:", ds_augmented.columns)
#             return ds_augmented

#         ## HAVEN'T HAD THIS YET
#         # elif self.dataset_name == 'census':
#         #     ds_augmented = pd.read_csv('../data/adult/onehot_census_sdv_100k.csv')
           
#         #     print("Census dataset columns:", ds_augmented.columns)
#         #     return ds_augmented

#         elif self.dataset_name == 'news':
#             ds_augmented = pd.read_csv('../data/adult/onehot_news_sdv_gaussian_100k.csv')
           
#             print("News dataset columns:", ds_augmented.columns)
#             return ds_augmented
        

#     def load_test_data(self):
#         _, ds_test = self.load_datasets(self.test_option)
#         return ds_test

#     def load_train_data(self):
#         if self.train_option == self.test_option:
#             ds_train, _ = self.load_datasets(self.train_option)
#         else:
#             ds1, ds2 = self.load_datasets(self.train_option)
#             ds_train = pd.concat([ds1, ds2], axis=0)
#         return ds_train

#     def standardize(self, df, numerical_cols):
#         scaler = StandardScaler()
#         df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
#         return df
    
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

    # def load_train_augment_data(self, train_option, augment_option):
    #     if train_option not in ['original', 'synthetic', 'mix'] or augment_option not in [None, 'ctgan', 'categorical', 'gaussian']:
    #         raise ValueError("Invalid train_option or augment_option")

    #     if train_option == 'original':
    #         df = pd.read_csv(self.paths['train_original'])
    #     elif train_option == 'synthetic':
    #         df = pd.read_csv(self.paths[train_option][augment_option])
    #     else:
    #         original_df = pd.read_csv(self.paths['train_original'])
    #         synthetic_df = pd.read_csv(self.paths[train_option][augment_option])
    #         df = pd.concat([original_df, synthetic_df], axis=0)

    #     df = self._standardize(df)
    #     X = df.iloc[:, :-1]
    #     y = df.iloc[:, -1]
    #     return TensorDataset(torch.tensor(X.values, dtype=torch.float), torch.tensor(y.values, dtype=torch.float))

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
            batch_y = torch.tensor(y[start:end], dtype=torch.float).unsqueeze(1)

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), batch_size=self.batch_size)
    
    # def _distribute_in_batches(self, X, y):
    # # Shuffle the data
    #     indices = torch.randperm(len(X))
    #     X = X.iloc[indices]
    #     y = y.iloc[indices]

    #     num_batch = int(len(X) / self.batch_size)
    #     batches = []

    #     for i in range(num_batch):
    #         start = i * self.batch_size
    #         end = start + self.batch_size

    #         batch_X = torch.tensor(X.iloc[start:end].values, dtype=torch.float)
    #         batch_y = torch.tensor(y.iloc[start:end].values, dtype=torch.float).unsqueeze(1)

    #         batch = TensorDataset(batch_X, batch_y)
    #         batches.append(batch)

    #     return DataLoader(ConcatDataset(batches), batch_size=self.batch_size)
    
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