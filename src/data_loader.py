import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from synthesize_data.onehot import onehot
from datasets import load_adult, load_news, load_census
from sklearn.preprocessing import StandardScaler

data_DIR = "./data" # run local

class data_loader:
    def __init__(self, file_name, dataset_name, train_option, test_option,
                 test_ratio, batch_size, numerical_columns = []):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.numerical_columns = numerical_columns
        self.file_name = file_name
        self.train_option = train_option
        self.test_option = test_option
        self.test_ratio = test_ratio
        
        # Load and print shapes before batching
        train_data, test_data = self.load_datasets(self.train_option)
        print(f"Total train samples: {train_data.shape[0]}")
        print(f"Total test samples: {test_data.shape[0]}")
        
        self.test_data = self.load_data_in_batches(test_data)
        self.train_data = self.load_data_in_batches(train_data)

    def load_data_in_batches(self, ds):
        ds = self.standardize(ds, self.numerical_columns)
        x = ds.iloc[:, :-1] # all columns except the last one
        y = ds.iloc[:, -1] # the last column
        return self.distribute_in_batches(x, y)

    def train_test_split(self, ds, test_size, random_seed=42):
        random.seed(random_seed)
        ds = ds.sample(frac=1, random_state=random_seed)
        if test_size < 1:
            test_size = int(len(ds) * test_size)
        ds_train = ds.iloc[test_size:]
        ds_test = ds.iloc[:test_size]
        return ds_train, ds_test

    def load_datasets(self, option):            
        if option == 'original':
            x, y = self.load_clean_ori_data()
            ds_ori = pd.concat([x, y], axis=1)
            ds_train, ds_test = self.train_test_split(ds_ori, self.test_ratio)
            print(f"Original dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test

        elif option == 'synthetic':
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_train, ds_test = self.train_test_split(ds_synth, self.test_ratio)
            print(f"Synthetic dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test

        elif option == 'mix':
            x_ori, y_ori = self.load_clean_ori_data()
            ds_ori = pd.concat([x_ori, y_ori], axis=1)
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_concat = pd.concat([ds_ori, ds_synth], axis=0)
            ds_train, ds_test = self.train_test_split(ds_concat, self.test_ratio)
            print(f"Mixed dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test

    def load_clean_ori_data(self):
        if self.dataset_name == 'adult':
            x, y = load_adult()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _ = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                               'relationship', 'race', 'sex', 'native-country'], verbose=False)
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            x_onehot = self.standardize(x_onehot, self.numerical_columns)
            return x_onehot, y

        elif self.dataset_name == 'census':
            x, y = load_census()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _ = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                               'relationship', 'race', 'sex', 'native-country'], verbose=False)
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            x_onehot = self.standardize(x_onehot, self.numerical_columns)
            return x_onehot, y

        elif self.dataset_name == 'news':
            xtest, ytest = load_news()  
            xtest = self.standardize(xtest, self.numerical_columns)
            return xtest, ytest

    def load_test_data(self):
        _, ds_test = self.load_datasets(self.test_option)
        return ds_test
    
    def load_train_data(self):
        if self.train_option == self.test_option:
            ds_train, _ = self.load_datasets(self.train_option)
        else:
            ds1, ds2 = self.load_datasets(self.train_option)
            ds_train = pd.concat([ds1, ds2], axis=0)
        return ds_train

    def standardize(self, df, numerical_cols):
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def distribute_in_batches(self, X, y, batch_size='default'):
        num_batch = int(len(X) / self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X[start:end].values, dtype=torch.float)
            batch_y = torch.tensor(y[start:end].values, dtype=torch.float)

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)

        if batch_size == 'default':
            batch_size = self.batch_size

        return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=batch_size)
    
    def print_sample_data(self):
        print("\nFirst 5 samples from training set:")
        for i, (inputs, labels) in enumerate(self.train_data):
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
                print(f"Label: {labels[0].item()}")
            else:
                break

        print("\nFirst 5 samples from test set:")
        for i, (inputs, labels) in enumerate(self.test_data):
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
                print(f"Label: {labels[0].item()}")
            else:
                break
