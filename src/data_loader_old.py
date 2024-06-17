from tqdm import tqdm

import torch

import numpy as np
import random
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData


#run in gg colab
#data_DIR = "/content/drive/MyDrive/eFL/data/"
data_DIR = "./data" # run local

#in: batch size, dataset name
#out: train_data,  test_data: in batches

class data_loader:
	def __init__(self, dataset_name, batch_size):

		self.dataset_name = dataset_name # 'mnist' or 'cifar10', .... a folder name in the data_DIR
		self.batch_size = batch_size

		transform = transforms.Compose([transforms.ToTensor()])
		
		self.train_data = self.load_data_in_batches(train=True, transform = transform)
		self.test_data = self.load_data_in_batches(train=False, transform = transform)
	
	def load_data_in_batches(self, train, transform):
		dataset = self.get_dataset(train=train, transform = transform)
		X = dataset.data
		y = dataset.target

		return self.distribute_in_batches(X, y)
		
	def get_dataset(self, train, transform):
	    
		if self.dataset_name == "mnist":
		    
		    mnist_ds = MNIST_truncated(data_DIR, train=train, download=True, transform=transform)
		    mnist_ds.data = self.normalize(mnist_ds.data)
		    return mnist_ds
		
		elif self.dataset_name == "femnist":
			femnist_ds = FEMNIST(data_DIR, train=train, transform=transform, download=True)
			femnist_ds.target = femnist_ds.target.long()
			return femnist_ds
		elif self.dataset_name == "cifar100":
			cifa100 = CIFAR100_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa100

		elif self.dataset_name == "cifar10":
			transform_train = transforms.Compose([
		    transforms.RandomCrop(32, padding=4),
		    transforms.RandomHorizontalFlip(),
		    transforms.ToTensor()
			])
			if train == True:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform_train, download=True)
			else:
				cifa10 = CIFAR10_truncated(data_DIR, train=train, transform=transform, download=True)
			return cifa10


	def normalize(self, x, mean=0.1307, std=0.3081):
		return (x-mean)/std

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
		
		



