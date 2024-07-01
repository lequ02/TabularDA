import pandas as pd
# from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from synthesize_data.onehot import onehot
from datasets import load_adult, load_news, load_census
import random



data_DIR = "./data" # run local


#in: batch size, dataset name
#out: train_data (in batches), test_data
class data_loader:
    def __init__(self, file_name, dataset_name, train_option, test_option,
                        test_ratio, batch_size, numerical_columns = []):

        self.dataset_name = dataset_name # 'adult' or 'census', .... a folder name in the data_DIR
        self.batch_size = batch_size
        # self.dataset_type = dataset_type # 'original' or 'sdv' 'sdv_categorical' or 'sdv_gaussian'
        self.numerical_columns = numerical_columns
        self.file_name = file_name
        self.train_option = train_option
        self.test_option = test_option
        self.test_ratio = test_ratio
        self.test_data = self.load_data_in_batches(self.load_test_data())
        self.train_data = self.load_data_in_batches(self.load_train_data())

    def load_data_in_batches(self, ds):

        ds = self.normalize(ds, self.numerical_columns)
        x = ds.iloc[:, :-1] # all columns except the last one
        y = ds.iloc[:, -1] # the last column

        return self.distribute_in_batches(x, y)

    def train_test_split(self, ds, test_size, random_seed=42):
        random.seed(random_seed)
        # shuffle the data
        ds = ds.sample(frac=1, random_state=random_seed)
        if test_size<1:
            test_size = int(len(ds) * test_size)
        ds_train = ds.iloc[test_size:]
        ds_test = ds.iloc[:test_size]
        return ds_train, ds_test

    def load_datasets(self, option):            

        if option == 'original':
            x, y = self.load_clean_ori_data()
            ds_ori = pd.concat([x, y], axis=1)
            ds_train, ds_test = self.train_test_split(ds_ori, self.test_ratio)
            return ds_train, ds_test

        elif option == 'synthetic':
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_train, ds_test = self.train_test_split(ds_synth, self.test_ratio)
            return ds_train, ds_test

        elif option == 'mix':
            x_ori , y_ori = self.load_clean_ori_data()
            ds_ori = pd.concat([x_ori, y_ori], axis=1)
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_concat = pd.concat([ds_ori, ds_synth], axis=0)
            ds_train, ds_test = self.train_test_split(ds_concat, self.test_ratio)
            return ds_train, ds_test


    def load_clean_ori_data(self):

        if self.dataset_name == 'adult':
            x , y = load_adult()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _  = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                    'relationship', 'race', 'sex', 'native-country'], verbose=False)
            # sort columns so it matches the training data
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            x_onehot = self.normalize(x_onehot, self.numerical_columns)
            # x = pd.concat([x_onehot, y], axis=1)
            return x_onehot, y

        elif self.dataset_name == 'census':
            x , y = load_census()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _  = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                    'relationship', 'race', 'sex', 'native-country'], verbose=False)
            # sort columns so it matches the training data
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            x_onehot = self.normalize(x_onehot, self.numerical_columns)
            # x = pd.concat([x_onehot, y], axis=1)
            return x_onehot, y

        elif self.dataset_name == 'news':
            xtest, ytest = load_news()  
            xtest = self.normalize(xtest, self.numerical_columns)
            return xtest, ytest


    def load_test_data(self):
        _, ds_test = self.load_datasets(self.test_option)
        # print("test")
        # print(ds_test.shape)
        # print(ds_test)
        return ds_test
    
    def load_train_data(self):
        if self.train_option == self.test_option:
            ds_train, _ = self.load_datasets(self.train_option)
        else:
            ds1, ds2 = self.load_datasets(self.train_option)
            ds_train = pd.concat([ds1, ds2], axis=0)
        # print("train")
        # print(ds_train.shape)
        # print(ds_train)
        return ds_train

    # def get_train_data(self):
    #     if self.train_option=='original':
    #         # load the original data and remove the test data to get the training data
    #         x, y = self.load_clean_ori_data()
    #         xtest, ytest = self.load_test_data()
    #         xtrain = x[~x.index.isin(xtest)].dropna(how = 'all')
    #         ytrain = y[~y.isin(ytest)].dropna(how = 'all')
    #         return pd.concat([xtrain, ytrain], axis=1)

    #     elif self.train_option=='synthetic':
    #         ds = pd.read_csv(self.file_name)

    #     elif self.train_option=='mix':
    #         ds = pd.read_csv(self.file_name)
    #     return ds


    def normalize(self, df, numerical_cols):
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col]-mean)/std
        return df


    # def distribute_in_batches(self, X, y, batch_size = 'default'):
        
    #     num_batch = int(len(X) /self.batch_size)
    #     batches = []

    #     for i in range(num_batch):
    #         start = i * self.batch_size
    #         end = start + self.batch_size

    #         batch_X = torch.tensor(X[start:end].values) # convert to PyTorch tensor
    #         batch_y = torch.tensor(y[start:end].values) # convert to PyTorch tensor

    #         batch = TensorDataset(batch_X, batch_y)
    #         batches.append(batch)

    #     if batch_size == 'default':
    #         batch_size = self.batch_size
        
    #     return DataLoader(ConcatDataset(batches), shuffle=True, batch_size = batch_size)
    
    def distribute_in_batches(self, X, y, batch_size='default'):
        num_batch = int(len(X) / self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X[start:end].values, dtype=torch.float)  # Ensure tensor is float
            batch_y = torch.tensor(y[start:end].values, dtype=torch.float)  # Ensure tensor is float

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)

        if batch_size == 'default':
            batch_size = self.batch_size

        return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=batch_size)
    
    
    ################ Printing 5 samples from the training and testing batch ##################
    
    def print_sample_data(self):
        print("\nFirst 5 samples from training set:")
        for i, (inputs, labels) in enumerate(self.train_data):
            if i < 5:
                print(f"Sample {i+1}: Input shape: {inputs.shape}, Label: {labels[0].item()}")
            else:
                break

        print("\nFirst 5 samples from test set:")
        for i, (inputs, labels) in enumerate(self.test_data):
            if i < 5:
                print(f"Sample {i+1}: Input shape: {inputs.shape}, Label: {labels[0].item()}")
            else:
                break
