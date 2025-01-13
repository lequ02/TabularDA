from data_loader import data_loader
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import os
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torchsummary import summary
from sklearn.metrics import f1_score,precision_score, recall_score, classification_report, mean_squared_error, r2_score
from models_folder import model_mnist12, model_mnist28, model_intrusion
from adult_model import DNN_Adult, DNN_Adult_mix
from census_model import DNN_Census,DNN_Census_mix
from news_model import DNN_News
from trainer import trainer
import constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

out_data_path = constants.OUT_DATA_PATHS

class train:

    #in_data_path = "../../data/"
    
    def __init__(self, dataset_name, batch_size, learning_rate, 
                num_epochs, w_dir, acc_dir, train_option,
                test_option, patience=10, early_stop_criterion='loss', augment_option=None, pre_trained_w_file=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.w_dir = w_dir #out_data_path + "weight/"
        self.acc_dir = acc_dir #out_data_path + "acc/"

        self.train_option = train_option
        self.augment_option = augment_option
        self.test_option = test_option

        self.patience = patience
        self.early_stop_criterion = early_stop_criterion

        if self.dataset_name.lower() in ['mnist12', 'mnist28', 'intrusion', 'covertype']:
            self.f1_type = "micro"
        elif self.dataset_name.lower() in ['adult', 'census', 'credit']:
            self.f1_type = "micro"
        else: # news ds
            self.f1_type = None


        self.pre_trained_w_file = pre_trained_w_file

        # Initialize train and test data
        self.data_loader = data_loader(self.dataset_name, self.batch_size)
        self.train_data = self.data_loader.load_train_augment_data(self.train_option, self.augment_option)
        self.test_data = self.data_loader.load_test_data()

        # self.train_data.to_csv("dummy_train_data_mnist12.csv")
        # self.test_data.to_csv("dummy_test_data_mnist12.csv")
        # print("\n\n\n\n\n break now\n\n\n")

        print("----------------------------------------------------")
        print(f"Training dataset size: {len(self.train_data) * self.batch_size}")
        print(f"Testing dataset size: {len(self.test_data) * self.batch_size}")
        print("----------------------------------------------------")

        # Set up output configuration and trainer
        self.setup_output(pre_trained_w_file)

    def setup_output(self, pre_trained_w_file):
        # Initialize trainer
        self.trainer = self.setup_trainer(pre_trained_w_file)

        # Configure file names
        # if self.train_option == 'original':
        #     data_file_name_part = 'original_adult_dataset'
        # else:
        #     data_file_name_part = os.path.basename(self.data_path)
        # self.w_file_name = f"{self.model_name}_train_{self.train_option}_test_{self.test_option}_data_{data_file_name_part}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        if self.augment_option is None:
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_test_{self.test_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        else:
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_augment_{self.augment_option}_test_{self.test_option}_augment_{self.augment_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        self.acc_file_name = f"{self.w_file_name}.acc.csv"
        self.report_file_name = f"{self.w_file_name}.report.txt"
        
        

        print("Configuration: ")
        # print(f"dataset, CSV file, model: {self.dataset_name}, {csv_file_name}, {self.model_name}")
        print(f"dataset, CSV file, model: {self.dataset_name}, {self.model_name}")

        print(f"B, lr: {self.batch_size}, {self.learning_rate}")
        print(f"num_epochs: {self.num_epochs}")
        print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
        print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")

    def setup_trainer(self, pre_trained_w_file):
        input_size = next(iter(self.train_data))[0].shape[1]
        # input_size = self.train_data.iloc[:, :-1].shape[1]
        
        print("input_size is:", input_size)
        print(f"input size shape: {input_size}")

        if self.dataset_name.lower() == "adult":
            if self.train_option.lower() == "mix"  :
                model = DNN_Adult_mix(input_size=input_size).to(device)
                self.model_name = "DNN_Adult_mix"
            else:
                model = DNN_Adult(input_size=input_size).to(device)
                self.model_name = "DNN_Adult"
            criterion = nn.BCELoss()

        
        


        elif self.dataset_name.lower() == "census":
            if self.train_option.lower() == "mix"  :
                model = DNN_Census_mix(input_size=input_size).to(device)
                self.model_name = "DNN_Census_mix"
            else:

                model = DNN_Census(input_size=input_size).to(device)
                self.model_name = "DNN_Census"
            criterion = nn.BCELoss()
            
        elif self.dataset_name.lower() == "news":
            model = DNN_News(input_size=input_size).to(device)
            self.model_name = "DNN_News"
            criterion = nn.MSELoss()
        # elif self.dataset_name.lower() == "mnist12":
        #     model = model_mnist12.DNN_MNIST12(input_size=input_size).to(device)
        #     self.model_name = "DNN_MNIST12"
        #     criterion = nn.CrossEntropyLoss()  
        # elif self.dataset_name.lower() == "mnist28":
        #     model = model_mnist28.DNN_MNIST28(input_size=input_size).to(device)
        #     self.model_name = "DNN_MNIST28"
        #     criterion = nn.CrossEntropyLoss()
        # elif self.dataset_name.lower() == "intrusion":
        #     model = DNN_Intrusion(input_size=input_size).to(device)
        #     self.model_name = "DNN_Intrusion"
        #     criterion = nn.CrossEntropyLoss()
        # elif self.dataset_name.lower() == "covertype":
        #     model = DNN_Covertype(input_size=input_size).to(device)
        #     self.model_name = "DNN_Covertype"
        #     criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown dataset name")

        if pre_trained_w_file is not None:
            print(f"Loading weight from {pre_trained_w_file}")
            model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file))

        mtrainer = trainer(model, self.train_data, criterion, self.learning_rate, dataset_name=self.dataset_name, device=device)
        print("SELF.train_data", self.train_data)
        mtrainer.data = self.train_data

        summary(model, (input_size,))
        return mtrainer


    def training(self):
        print("==========================================================================================")
        print("Start training...")

        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.w_dir, exist_ok=True)

        save_at = [(i + 1) * 500 for i in range(int(self.num_epochs / 500))]

        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            if self.dataset_name.lower() in ["adult", "census"]:
                acc_file.write("global_round,train_loss,train_acc,train_f1_micro,test_loss,test_acc,test_f1_micro\n")
            else:
                acc_file.write("global_round,train_mse,train_r2,test_mse,test_r2\n")

        lr = self.learning_rate

        train_losses = []
        test_losses = []
        train_f1_scores = []
        test_f1_scores = []
        train_mse_scores = []
        test_mse_scores = []
        train_r2_scores = []
        test_r2_scores = []

        best_val_loss = float('inf')
        best_val_f1 = 0
    
        patience = self.patience  # Number of epochs to wait for improvement before stopping
        patience_counter = 0

        for epoch in range(self.num_epochs):
            print("-----------------------")
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            self.trainer.train(device, epochs=1)  # Train for 1 epoch at a time

            if self.dataset_name.lower() in ["adult", "census", 'mnist12', 'mnist28', 'intrusion']:
                train_loss, train_acc, train_f1 = self.train_stats_classification(device)
                train_losses.append(train_loss)
                train_f1_scores.append(train_f1)

                if train_acc is not None:
                    print(f"Training statistic: Accuracy {train_acc:.4f}%, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
                else:
                    print(f"Training statistic: Loss: {train_loss:.4f}, MSE: {train_f1:.4f}")

                test_loss, test_acc, test_f1 = self.validate_classification(load_weight=False)
                test_losses.append(test_loss)
                test_f1_scores.append(test_f1)

                with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                    acc_file.write(f"{epoch+1},{train_loss},{train_acc},{train_f1},{test_loss},{test_acc},{test_f1}\n")
            else:
                train_loss, train_mse, train_r2 = self.train_stats_regression(device)
                train_losses.append(train_loss)
                train_mse_scores.append(train_mse)
                train_r2_scores.append(train_r2)

                print(f"Training statistic: Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, R2: {train_r2:.4f}")

                test_loss, test_mse, test_r2 = self.validate_regression(load_weight=False)
                test_losses.append(test_loss)
                test_mse_scores.append(test_mse)
                test_r2_scores.append(test_r2)

                with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                    acc_file.write(f"{epoch+1}, {train_mse}, {train_r2}, {test_mse}, {test_r2}\n")

            print(f"lr: {lr}")
            # print("-------------------------------------------")

            if self.early_stop_criterion == 'loss':
                # Saving the model only if the current test loss is better
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    self.save_model(self.w_file_name)  # Save the best model
                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping (loss) triggered!")
                        break
                

            if self.early_stop_criterion == 'f1':
                if test_f1 > best_val_f1:
                    best_val_f1 = test_f1
                    self.save_model(self.w_file_name)  # Save the best model
                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping (f1) triggered!")
                        break


            if epoch + 1 in save_at:
                fmodel = f"{epoch + 1}_{self.w_file_name}"
                self.save_model(fmodel)

        # print("len", len(train_losses), len(test_losses), len(train_f1_scores), len(test_f1_scores))
        self.plot_loss_and_f1_curves(train_losses, test_losses, train_f1_scores, test_f1_scores)
        print("Finish training!")

    def train_stats_regression(self, device):
        # Set model to evaluation mode for statistics calculation
        self.trainer.model.eval()
        loss, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad(): # Don't compute gradients during evaluation
            for batch_idx, (X, y) in enumerate(self.train_data):
                # Move data to device and ensure correct shape
                X, y = X.to(device), y.to(device).float().reshape(-1, 1)
                
                output = self.trainer.model(X)

                # Store predictions and actual values
                all_preds.extend(output.cpu().numpy())
                # Calculate loss and accumulate
                loss += self.trainer.criterion(output, y).item() * len(y)
                total += len(y)
                all_labels.extend(y.cpu().numpy())
        
        loss = loss / total
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        return loss, mse, r2
    





    def validate_classification(self, load_weight=False):
        print("Validation statistic...")

        if load_weight:
            self.trainer.model.load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model.eval() # fixed this part
        
        corrects, loss, total = 0, 0, 0
        all_preds, all_labels = [], []

        # first_batch = next(iter(self.test_data))
        # print(f"First batch from test_data: {first_batch}")

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.test_data):
                #  multi-class classification
                if self.dataset_name.lower() == "mnist12" or self.dataset_name.lower() == "mnist28" or self.dataset_name.lower() == "intrusion":
                    X, y = X.to(device), y.long().to(device)

                    output = self.trainer.model(X)  
                    _, predicted = torch.max(output.data, 1)  # Changed prediction for multi-class
                    corrects += (predicted == y).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    loss += self.trainer.criterion(output, y).item() * len(y)
                    total += len(y)
                    all_labels.extend(y.cpu().numpy())

                    # precision = precision_score(all_labels, all_preds, average=self.f1_type)
                    # recall = recall_score(all_labels, all_preds, average=self.f1_type)
                    # loss = loss / total
                    # accuracy = 100 * corrects / total
                    # f1 = f1_score(all_labels, all_preds, average=self.f1_type)  # Added average=self.f1_type for multi-class


                else: # binary classification
                    X, y = X.to(device), y.to(device).float().unsqueeze(1)
                    
                    output = self.trainer.model(X)
                    # Convert probabilities to binary predictions (threshold = 0.5)
                    pred = (output > 0.5).float()
                    # print(f"Batch {batch_idx}: output={output}, pred={pred}, target-y={y}")
                
                    # Count correct predictions
                    corrects += pred.eq(y).sum().item()
                    all_preds.extend(pred.cpu().numpy())
                
                    loss += self.trainer.criterion(output, y).item() * len(y)
                    
                    total += len(y)
                    all_labels.extend(y.cpu().numpy())
            
                    # print("\n\nval preds:", np.unique(all_preds))
                    # print("\n\nval labels:", np.unique(all_labels))

        precision = precision_score(all_labels, all_preds, average=self.f1_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=self.f1_type, zero_division=0)
        loss = loss / total
        accuracy = 100 * corrects / total
        f1 = f1_score(all_labels, all_preds, average=self.f1_type)

        print(f"Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, Loss: {loss:.4f}, F1: {f1:.4f}")
      
        # print("-------------------------------------------")
        
        return loss, accuracy, f1
    

    
    def train_stats_classification(self, device):
        self.trainer.model.eval()
        
        corrects, loss, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.train_data):
                #  multi-class classification 
                if self.dataset_name.lower() == "mnist12" or self.dataset_name.lower() == "mnist28" or self.dataset_name.lower() == "intrusion":
                    X, y = X.to(device), y.long().to(device)  # Changed this line
                    
                    output = self.trainer.model(X)
                    _, predicted = torch.max(output.data, 1)  # Changed prediction for multi-class
                    corrects += (predicted == y).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    loss += self.trainer.criterion(output, y).item() * len(y)
                    total += len(y)
                    all_labels.extend(y.cpu().numpy())

                    # loss = loss / total
                    # accuracy = 100 * corrects / total
                    # f1 = f1_score(all_labels, all_preds, average=self.f1_type)  # Added average=self.f1_type for multi-class

                    
                else: # binary classification
                    X, y = X.to(device), y.to(device).float().unsqueeze(1)
                
                    output = self.trainer.model(X)
                    pred = (output > 0.5).float()
                    corrects += pred.eq(y).sum().item()
                    all_preds.extend(pred.cpu().numpy())
                    loss += self.trainer.criterion(output, y).item() * len(y)
                    total += len(y)
                    all_labels.extend(y.cpu().numpy())
                    # print("\n\ntrain preds:", np.unique(all_preds))
                    
        loss = loss / total
        accuracy = 100 * corrects / total
        f1 = f1_score(all_labels, all_preds, average=self.f1_type)

        return loss, accuracy, f1
    





    def validate_regression(self, load_weight=False):
        print("Validation statistic...")

        if load_weight:
            self.trainer.model.load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model.eval()
        
        with torch.no_grad():
            # Convert entire test DataFrame to tensors
            X = torch.FloatTensor(self.test_data.iloc[:, :-1].values).to(device)
            y = torch.FloatTensor(self.test_data.iloc[:, -1].values).reshape(-1, 1).to(device)
            
            # Get predictions for all test data at once
            output = self.trainer.model(X)
            
            # Calculate loss
            loss = self.trainer.criterion(output, y).item()
            
            # Convert to numpy for sklearn metrics
            predictions = output.cpu().numpy()
            actual = y.cpu().numpy()
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            r2 = r2_score(actual, predictions)

        print(f"Loss: {loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
        print("-------------------------------------------")

        return loss, mse, r2

    def save_model(self, fmodel):
        print("Saving model...")
        torch.save(self.trainer.model.state_dict(), self.w_dir + fmodel)
        print("Model saved!")
        print("-------------------------------------------")

    def plot_loss_and_f1_curves(self, train_losses, test_losses, train_f1_scores, test_f1_scores):
        """Plots and saves training/validation loss and F1 score curves to the model directory."""
        # Ensure the directory exists
        os.makedirs(self.acc_dir, exist_ok=True)
        # Determine the number of x-ticks to display automatically
        num_epochs = len(train_losses)
        max_ticks = min(num_epochs, 25)  # Maximum number of ticks to display
        step = max(1, num_epochs // max_ticks)
        x_ticks = range(0, num_epochs, step)

        # Plot Loss Curves
        loss_plot_file = os.path.join(self.acc_dir, f"{self.acc_file_name}_loss_curve.png")
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(test_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
        plt.savefig(loss_plot_file)
        plt.close()

        # Plot F1 Score Curves
        f1_plot_file = os.path.join(self.acc_dir, f"{self.acc_file_name}_f1_curve.png")
        plt.figure(figsize=(10, 6))
        plt.plot(train_f1_scores, label='Training F1 Score', color='green')
        plt.plot(test_f1_scores, label='Validation F1 Score', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score Curves')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
        plt.savefig(f1_plot_file)
        plt.close()