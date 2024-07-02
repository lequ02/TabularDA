# import random
# import pandas as pd
# import math
# import os
# import torch
# import copy
# import numpy as np
# from tqdm import tqdm
# from torch import nn
# from models import DNN_Adult  
# from trainer import trainer
# from data_loader import data_loader 
# from torchsummary import summary
# from sklearn.metrics import f1_score, classification_report  # Import the F1 score and classification report functions
# import matplotlib.pyplot as plt  # Import matplotlib for plotting

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(1)


# class train:
#     def __init__(self, dataset_name, data_path, batch_size, learning_rate, lr_decay,
#                  num_epochs, w_dir, acc_dir, test_ratio, train_option, test_option,
#                  numerical_columns, pre_trained_w_file=None):
#         self.dataset_name = dataset_name
#         self.data_path = data_path
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.lr_decay = lr_decay
#         self.num_epochs = num_epochs
#         self.w_dir = w_dir
#         self.acc_dir = acc_dir
#         self.test_ratio = test_ratio
#         self.train_option = train_option
#         self.test_option = test_option
#         self.numerical_columns = numerical_columns
#         self.pre_trained_w_file = pre_trained_w_file

#         self.data = data_loader(self.data_path, self.dataset_name, test_ratio=self.test_ratio,
#                                 train_option=self.train_option, test_option=self.test_option,
#                                 batch_size=self.batch_size, numerical_columns=self.numerical_columns)
        
#         # Print sizes of training and testing datasets
#         print(f"Training dataset size: {len(self.data.train_data.dataset)}")
#         print(f"Testing dataset size: {len(self.data.test_data.dataset)}")

#         # Print DataLoader objects
#         print("train", self.data.train_data)
#         print("test", self.data.test_data)

#         self.data.print_sample_data()

#         self.trainer = self.setup_trainer(pre_trained_w_file)

#         # Determine the data file name part based on the options
#         if self.train_option == 'original' and self.test_option == 'original':
#             data_file_name_part = 'original_adult_dataset'
#         else:
#             data_file_name_part = os.path.basename(self.data_path)

#         # Update file names to include train_option, test_option, and data file name part
#         self.w_file_name = f"{self.model_name}_train_{self.train_option}_test_{self.test_option}_data_{data_file_name_part}_lr{self.learning_rate}_dc{self.lr_decay}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
#         self.acc_file_name = f"{self.w_file_name}.acc.csv"
#         self.report_file_name = f"{self.w_file_name}.report.txt"

#         print("Configuration: ")
#         print(f"dataset, model: {dataset_name}, {self.model_name}")
#         print(f"B, lr, lr_decay: {batch_size}, {learning_rate}, {lr_decay}")
#         print(f"num_epochs: {num_epochs}")
#         print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
#         print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")

#     def setup_trainer(self, pre_trained_w_file):
#         input_size = next(iter(self.data.train_data))[0].shape[1]
#         print(f"input size shape: {input_size}")
#         model = DNN_Adult(input_size=input_size).to(device)
#         self.model_name = "DNN_Adult"

#         if pre_trained_w_file is not None:
#             print(f"Loading weight from {pre_trained_w_file}")
#             model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file))

#         mtrainer = trainer()
#         mtrainer.model["model"] = copy.deepcopy(model)
#         mtrainer.data = self.data.train_data

#         summary(mtrainer.model['model'], (1, input_size))
#         return mtrainer

#     def training(self):
#         print("==========================================================================================")
#         print("Start training...")

#         os.makedirs(self.acc_dir, exist_ok=True)
#         os.makedirs(self.w_dir, exist_ok=True)

#         save_at = [(i + 1) * 500 for i in range(int(self.num_epochs / 500))]

#         # Updated CSV header to include F1 scores
#         with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
#             acc_file.write("global_round,train_loss,train_acc,train_f1,test_loss,test_acc,test_f1\n")

#         lr = self.learning_rate

#         train_losses = []
#         test_losses = []

#         for epoch in range(self.num_epochs):
#             print(f"Epoch {epoch+1}/{self.num_epochs}")

#             self.trainer.train(device, lr)

#             train_loss, train_acc, train_f1 = self.train_stats(device)
#             train_losses.append(train_loss)

#             print(f"Training statistic: Accuracy {train_acc:.4f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")

#             test_loss, test_acc, test_f1 = self.validate(load_weight=False)
#             test_losses.append(test_loss)

#             print(f"lr: {lr}")

#             # Updated CSV row to include F1 scores
#             with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
#                 acc_file.write(f"{epoch+1},{train_loss},{train_acc},{train_f1},{test_loss},{test_acc},{test_f1}\n")

#             if epoch + 1 in save_at:
#                 fmodel = f"{epoch + 1}_{self.w_file_name}"
#                 self.save_model(fmodel)

#             lr *= self.lr_decay

#         print("Finish training!")

#         # Plot training and test losses
#         self.plot_losses(train_losses, test_losses)

#         # Generate and save classification report
#         self.save_classification_report()

#     def train_stats(self, device):
#         self.trainer.model['model'].eval()
#         corrects, loss, total = 0, 0, 0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for batch_idx, (X, y) in enumerate(self.data.train_data):
#                 X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
#                 output = self.trainer.model['model'](X)
#                 pred = (output > 0.5).float()
#                 corrects += pred.eq(y).sum().item()
#                 loss += self.trainer.model["criterion"](output, y).item() * len(y)
#                 total += len(y)
#                 all_preds.extend(pred.cpu().numpy())
#                 all_labels.extend(y.cpu().numpy())
#         accuracy = 100 * corrects / total
#         loss = loss / total
#         f1 = f1_score(all_labels, all_preds)

#         return loss, accuracy, f1

#     def validate(self, load_weight=False):
#         print("Validation statistic...")

#         if load_weight:
#             self.trainer.model['model'].load_state_dict(torch.load(self.w_dir + self.w_file_name))

#         self.trainer.model['model'].eval()
#         corrects, loss, total = 0, 0, 0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for batch_idx, (X, y) in enumerate(self.data.test_data):
#                 X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
#                 output = self.trainer.model['model'](X)
#                 pred = (output > 0.5).float()
#                 corrects += pred.eq(y).sum().item()
#                 loss += self.trainer.model["criterion"](output, y).item() * len(y)
#                 total += len(y)
#                 all_preds.extend(pred.cpu().numpy())
#                 all_labels.extend(y.cpu().numpy())
#         accuracy = 100 * corrects / total
#         loss = loss / total
#         f1 = f1_score(all_labels, all_preds)

#         print(f"Accuracy: {accuracy:.4f}%, Loss: {loss:.4f}, F1: {f1:.4f}")
#         print("-------------------------------------------")

#         return loss, accuracy, f1

#     def save_model(self, fmodel):
#         print("Saving model...")
#         torch.save(self.trainer.model['model'].state_dict(), self.w_dir + fmodel)
#         print("Model saved!")

#     def plot_losses(self, train_losses, test_losses):
#         plt.plot(train_losses, label='Training Loss')
#         plt.plot(test_losses, label='Test Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training and Test Loss')
#         plt.legend()
#         plt.savefig(self.acc_dir + self.w_file_name.replace('.weight.pth', '_loss_plot.png'))
#         plt.show()

#     def save_classification_report(self):
#         # Generate classification report for the test set
#         all_preds, all_labels = [], []
#         self.trainer.model['model'].eval()
#         with torch.no_grad():
#             for batch_idx, (X, y) in enumerate(self.data.test_data):
#                 X, y = X.to(device), y.to(device).float().unsqueeze(1)
#                 output = self.trainer.model['model'](X)
#                 pred = (output > 0.5).float()
#                 all_preds.extend(pred.cpu().numpy())
#                 all_labels.extend(y.cpu().numpy())
        
#         report = classification_report(all_labels, all_preds, target_names=['<=50K', '>50K'])
        
#         with open(self.acc_dir + self.report_file_name, 'w') as report_file:
#             report_file.write(report)
#         print("Classification report saved!")



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
from sklearn.metrics import f1_score, classification_report  # Import the F1 score and classification report functions
import matplotlib.pyplot as plt  # Import matplotlib for plotting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class train:
    def __init__(self, dataset_name, data_path, batch_size, learning_rate,
                 num_epochs, w_dir, acc_dir, test_ratio, train_option, test_option,
                 numerical_columns, pre_trained_w_file=None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        
        # Printing the sizes of training and testing datasets
        print(f"Training dataset size: {len(self.data.train_data.dataset)}")
        print(f"Testing dataset size: {len(self.data.test_data.dataset)}")

        print("train", self.data.train_data)
        print("test", self.data.test_data)

        self.data.print_sample_data()

        self.trainer = self.setup_trainer(pre_trained_w_file)

        if self.train_option == 'original' and self.test_option == 'original':
            data_file_name_part = 'original_adult_dataset'
        else:
            data_file_name_part = os.path.basename(self.data_path)

        self.w_file_name = f"{self.model_name}_train_{self.train_option}_test_{self.test_option}_data_{data_file_name_part}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        self.acc_file_name = f"{self.w_file_name}.acc.csv"
        self.report_file_name = f"{self.w_file_name}.report.txt"

        print("Configuration: ")
        print(f"dataset, model: {dataset_name}, {self.model_name}")
        print(f"B, lr: {batch_size}, {learning_rate}")
        print(f"num_epochs: {num_epochs}")
        print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
        print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")

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

        # Updating the CSV header to include F1 scores
        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            acc_file.write("global_round,train_loss,train_acc,train_f1,test_loss,test_acc,test_f1\n")

        lr = self.learning_rate

        train_losses = []
        test_losses = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            self.trainer.train(device, lr)

            train_loss, train_acc, train_f1 = self.train_stats(device)
            train_losses.append(train_loss)

            print(f"Training statistic: Accuracy {train_acc:.4f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")

            test_loss, test_acc, test_f1 = self.validate(load_weight=False)
            test_losses.append(test_loss)

            print(f"lr: {lr}")

            # Adding F1 scores to the csv file
            with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                acc_file.write(f"{epoch+1},{train_loss},{train_acc},{train_f1},{test_loss},{test_acc},{test_f1}\n")

            if epoch + 1 in save_at:
                fmodel = f"{epoch + 1}_{self.w_file_name}"
                self.save_model(fmodel)

        print("Finish training!")

        # Plot training and test losses
        self.plot_losses(train_losses, test_losses)

        # Generate and save classification report
        self.save_classification_report()

    def train_stats(self, device):
        self.trainer.model['model'].eval()
        corrects, loss, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data.train_data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
                output = self.trainer.model['model'](X)
                pred = (output > 0.5).float()
                corrects += pred.eq(y).sum().item()
                loss += self.trainer.model["criterion"](output, y).item() * len(y)
                total += len(y)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        accuracy = 100 * corrects / total
        loss = loss / total
        f1 = f1_score(all_labels, all_preds)

        return loss, accuracy, f1

    def validate(self, load_weight=False):
        print("Validation statistic...")

        if load_weight:
            self.trainer.model['model'].load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model['model'].eval()
        corrects, loss, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data.test_data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
                output = self.trainer.model['model'](X)
                pred = (output > 0.5).float()
                corrects += pred.eq(y).sum().item()
                loss += self.trainer.model["criterion"](output, y).item() * len(y)
                total += len(y)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        accuracy = 100 * corrects / total
        loss = loss / total
        f1 = f1_score(all_labels, all_preds)

        print(f"Accuracy: {accuracy:.4f}%, Loss: {loss:.4f}, F1: {f1:.4f}")
        print("-------------------------------------------")

        return loss, accuracy, f1

    def save_model(self, fmodel):
        print("Saving model...")
        torch.save(self.trainer.model['model'].state_dict(), self.w_dir + fmodel)
        print("Model saved!")

    def plot_losses(self, train_losses, test_losses):
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.savefig(self.acc_dir + self.w_file_name.replace('.weight.pth', '_loss_plot.png'))
        plt.show()

    def save_classification_report(self):
        all_preds, all_labels = [], []
        self.trainer.model['model'].eval()
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data.test_data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                output = self.trainer.model['model'](X)
                pred = (output > 0.5).float()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        report = classification_report(all_labels, all_preds, target_names=['<=50K', '>50K'])
        
        with open(self.acc_dir + self.report_file_name, 'w') as report_file:
            report_file.write(report)
        print("Classification report saved!")
