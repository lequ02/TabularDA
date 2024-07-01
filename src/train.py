import random
import pandas as pd
import math
import os
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch import nn
from models import DNN_Adult  
from trainer import trainer
from data_loader import data_loader 
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class train:
    def __init__(self, dataset_name, data_path, batch_size, learning_rate, lr_decay,
                 num_epochs, w_dir, acc_dir, test_ratio, train_option, test_option,
                 numerical_columns, pre_trained_w_file=None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.num_epochs = num_epochs
        self.w_dir = w_dir
        self.acc_dir = acc_dir
        self.test_ratio = test_ratio
        self.train_option = train_option
        self.test_option = test_option
        self.numerical_columns = numerical_columns
        self.pre_trained_w_file = pre_trained_w_file

        self.data = data_loader(self.data_path, self.dataset_name, test_ratio=self.test_ratio,
                                train_option=self.train_option, test_option=self.test_option,
                                batch_size=self.batch_size, numerical_columns=self.numerical_columns)
        
        # Printing the  sizes of training and testing datasets
        print(f"Training dataset size: {len(self.data.train_data.dataset)}")
        print(f"Testing dataset size: {len(self.data.test_data.dataset)}")

        # Printing DataLoader objects
        print("train", self.data.train_data)
        print("test", self.data.test_data)

        self.data.print_sample_data()

        self.trainer = self.setup_trainer(pre_trained_w_file)

        self.w_file_name = f"{self.model_name}_lr{self.learning_rate}_dc{self.lr_decay}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        self.acc_file_name = f"{self.w_file_name}.acc.csv"

        print("Configuration: ")
        print(f"dataset, model: {dataset_name}, {self.model_name}")
        print(f"B, lr, lr_decay: {batch_size}, {learning_rate}, {lr_decay}")
        print(f"num_epochs: {num_epochs}")
        print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
        print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")


######################## Adult Dataset ################


    def setup_trainer(self, pre_trained_w_file):
        input_size = next(iter(self.data.train_data))[0].shape[1]
        print(f"input size shape: {input_size}")
        model = DNN_Adult(input_size=input_size).to(device)
        self.model_name = "DNN_Adult"

        if pre_trained_w_file is not None:
            print(f"Loading weight from {pre_trained_w_file}")
            model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file))

        mtrainer = trainer()
        mtrainer.model["model"] = copy.deepcopy(model)
        mtrainer.data = self.data.train_data

        summary(mtrainer.model['model'], (1, input_size))
        return mtrainer

    def training(self):
        print("==========================================================================================")
        print("Start training...")

        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.w_dir, exist_ok=True)

        save_at = [(i + 1) * 500 for i in range(int(self.num_epochs / 500))]

        # Updated CSV header to include train_option and test_option
        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            acc_file.write("global_round,train_loss,train_acc,test_loss,test_acc,train_option,test_option\n")

        lr = self.learning_rate

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            self.trainer.train(device, lr)

            train_loss, train_acc = self.trainer.train_stats(device)

            print(f"Training statistic: Accuracy {train_acc:.4f}%, Loss: {train_loss:.4f}")

            test_loss, test_acc = self.validate(load_weight=False)
            print(f"lr: {lr}")

            # Updated the  CSV row to include train_option and test_option
            with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                acc_file.write(f"{epoch+1},{train_loss},{train_acc},{test_loss},{test_acc},{self.train_option},{self.test_option}\n")

            if epoch + 1 in save_at:
                fmodel = f"{epoch + 1}_{self.w_file_name}"
                self.save_model(fmodel)

            lr *= self.lr_decay

        print("Finish training!")

    def validate(self, load_weight=False):
        print("Validation statistic...")

        if load_weight:
            self.trainer.model['model'].load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model['model'].eval()
        corrects, loss, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data.test_data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
                output = self.trainer.model['model'](X)
                pred = (output > 0.5).float()
                corrects += pred.eq(y).sum().item()
                loss += self.trainer.model["criterion"](output, y).item() * len(y)
                total += len(y)
        accuracy = 100 * corrects / total
        loss = loss / total

        print(f"Accuracy: {accuracy:.4f}%, Loss: {loss:.4f}")
        print("-------------------------------------------")

        return loss, accuracy

    def save_model(self, fmodel):
        print("Saving model...")
        torch.save(self.trainer.model['model'].state_dict(), self.w_dir + fmodel)
        print("Model saved!")