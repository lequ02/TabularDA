from tqdm import tqdm
import pandas as pd

import torch

import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData
from tqdm import tqdm
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData


data_DIR = "../data" # run local


#in: batch size, dataset name
#out: train_data,  test_data: in batches
class data_loader:
    def __init__(self, dataset_name, batch_size):

        self.dataset_name = dataset_name # 'mnist' or 'cifar10', .... a folder name in the data_DIR
        self.batch_size = batch_size
        
        self.train_data = self.load_data_in_batches(train=True)
        self.test_data = self.load_data_in_batches(train=False)
    
    def load_data_in_batches(self, train):
        dataset = self.get_dataset(train=train)
        X = dataset.data
        y = dataset.target

        return self.distribute_in_batches(X, y)
        
    def get_dataset(self, train):
        
        if self.dataset_name == "adult":
            adult_ds = pd.read_csv(f"{data_DIR}/adult/adult_sdv{}_100k.csv")
            return adult_ds
        
        elif self.dataset_name == "census":
            census_ds = pd.read_csv(f"{data_DIR}/census.csv")
            return census_ds
            
        elif self.dataset_name == "news":
            cifa100 = CIFAR100_truncated(data_DIR, train=train, download=True)
            return cifa100



    def distribute_in_batches(self, X, y):
        
        num_batch = int(len(X) /self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = X[start:end]
            batch_y = y[start:end]

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), shuffle=True, batch_size = self.batch_size)
