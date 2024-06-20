import pandas as pd
# from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from synthesize_data.onehot import onehot
from datasets import load_adult, load_news, load_census
import random



data_DIR = "../data" # run local


#in: batch size, dataset name
#out: train_data (in batches), test_data
class data_loader:
    def __init__(self, dataset_name, dataset_type, batch_size, numerical_columns = []):

        self.dataset_name = dataset_name # 'adult' or 'census', .... a folder name in the data_DIR
        self.batch_size = batch_size
        self.dataset_type = dataset_type # 'original' or 'sdv' 'sdv_categorical' or 'sdv_gaussian'
        self.numerical_columns = numerical_columns
        self.train_data = self.load_data_in_batches()
        self.test_data = self.load_test_data()

    def load_data_in_batches(self):
    
        trainds =  self.get_train_data()
        xtrain = trainds.iloc[:, :-1] # all columns except the last one
        xtrain = self.normalize(xtrain, self.numerical_columns)
        ytrain = trainds.iloc[:, -1] # the last column

        return self.distribute_in_batches(xtrain, ytrain)


    def load_test_data(self, random_seed=42):            

        if self.dataset_type == 'original':
            x, y = self.load_clean_ori_data()
            random.seed(seed)
            data = list(zip(x, y))
            random.shuffle(data)
            x, y = zip(*data)
            test_size = min(int(len(x) * 0.2), 10000)
            x_test = x[:test_size]
            y_test = y[:test_size]
            return x_test, y_test

        if self.dataset_type == 'original':
            xtest, ytest = self.load_clean_ori_data()


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
            return xtest, ytest
        
    def get_train_data(self):
        if self.dataset_type=='original':
            ds = load_clean_ori_data() - self.load_test_data()


        else: ds = pd.read_csv(f"{data_DIR}/{self.dataset_name}/onehot_{self.dataset_name}_{self.dataset_type}_100k.csv", index_col=0)
        return ds


    def normalize(self, df, numerical_cols):
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col]-mean)/std
        return df


    def distribute_in_batches(self, X, y):
        
        num_batch = int(len(X) /self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X[start:end].values) # convert to PyTorch tensor
            batch_y = torch.tensor(y[start:end].values) # convert to PyTorch tensor

            # batch_X = X[start:end]
            # batch_y = y[start:end]

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), shuffle=True, batch_size = self.batch_size)
