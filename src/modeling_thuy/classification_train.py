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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from models import DNN_Adult, DNN_Census
from models_folder import model_mnist12, model_mnist28, model_intrusion
from trainer import trainer
import constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

out_data_path = constants.OUT_DATA_PATHS

class train:

    #in_data_path = "../../data/"
    
    def __init__(self, dataset_name, train_option = None, augment_option = None, test_option = None, 
                        batch_size = 128, learning_rate = 0.001, num_epochs = 10, patience=10, early_stop_criterion='loss', 
                        eval_metrics = None, metric_to_plot = None,
                        pre_trained_w_file = None,
                        w_dir = None, acc_dir = None ):
        
        self.setup_data(dataset_name = dataset_name, train_option = train_option,
                        augment_option = augment_option, test_option = test_option, batch_size = batch_size)

        self.setup_fitting(num_epochs = num_epochs, learning_rate = learning_rate, patience=patience, early_stop_criterion=early_stop_criterion)
        self.setup_eval_metric(eval_metrics, metric_to_plot)
        self.setup_trainer(pre_trained_w_file = pre_trained_w_file)
        self.setup_output(w_dir = w_dir, acc_dir = acc_dir)

    def setup_data(self, dataset_name, train_option, augment_option, test_option, batch_size):

        print("=====Setting up data=====")
        
        if dataset_name.lower() in ['mnist12', 'mnist28', 'intrusion', 'covertype']: #more than 2 classes
            self.multi_y = True
        elif dataset_name.lower() in ['adult', 'census']: #binary
            self.multi_y = False

        self.dataset_name = dataset_name
        self.train_option = train_option
        self.augment_option = augment_option
        self.test_option = test_option
        self.batch_size = batch_size

        self.data_loader = data_loader(self.dataset_name, self.batch_size, multi_y = self.multi_y)
        self.train_data = self.data_loader.load_train_augment_data(self.train_option, self.augment_option)
        self.test_data = self.data_loader.load_test_data()

        print(f"Dataset name:{self.dataset_name}")
        print(f"Training dataset size: {len(self.train_data) * self.batch_size} samples")
        print(f"Testing dataset size: {len(self.test_data) * self.batch_size} samples")
        print("")

        
    def setup_fitting(self, num_epochs, learning_rate, patience, early_stop_criterion):
        print("=====Setting up parameters for training=====")
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience

        self.early_stop_criterion = early_stop_criterion
        
        print(f"num_epochs: {self.num_epochs}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"patience: {self.patience}")
        print(f"early_stop_criterion: {self.early_stop_criterion}")
    
    def setup_eval_metric(self, eval_metrics, metric_to_plot):
        #use all types of f1 and accuracy
        #set up a metric to plot when training is done

        self.eval_metrics = eval_metrics #{"accuracy": None, "f1": ['macro', 'micro', 'weighted']}
        self.metric_to_plot = metric_to_plot #"f1_micro"


    def setup_trainer(self, pre_trained_w_file):
        print("=====Setting up trainer=====")
        input_size = next(iter(self.train_data))[0].shape[1]
    
        if self.dataset_name.lower() == "adult":
            model = DNN_Adult(input_size=input_size).to(device)
            self.model_name = "DNN_Adult"
            #self.multi_y = False
            criterion = nn.BCELoss()
        elif self.dataset_name.lower() == "census":
            model = DNN_Census(input_size=input_size).to(device)
            self.model_name = "DNN_Census"
            #self.multi_y = False
            criterion = nn.BCELoss()
        elif self.dataset_name.lower() == "mnist12":
            model = model_mnist12.DNN_MNIST12(input_size=input_size).to(device)
            self.model_name = "DNN_MNIST12"
            #self.multi_y = True
            criterion = nn.CrossEntropyLoss()  
        elif self.dataset_name.lower() == "mnist28":
            model = model_mnist28.DNN_MNIST28(input_size=input_size).to(device)
            self.model_name = "DNN_MNIST28"
            #self.multi_y = True
            criterion = nn.CrossEntropyLoss()
        elif self.dataset_name.lower() == "intrusion":
            model = model_intrusion.DNN_Intrusion(input_size=input_size).to(device)
            self.model_name = "DNN_Intrusion"
            #self.multi_y = True
            criterion = nn.CrossEntropyLoss()
        elif self.dataset_name.lower() == "covertype":
            model = DNN_Covertype(input_size=input_size).to(device)
            self.model_name = "DNN_Covertype"
            #self.multi_y = True
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown dataset name")

        if pre_trained_w_file is not None:
            print(f"Loading weight from {pre_trained_w_file}")
            model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file))

        mtrainer = trainer(model, self.train_data, criterion, self.learning_rate, dataset_name=self.dataset_name, device=device, multi_y = self.multi_y)
        mtrainer.data = self.train_data

        print(f"input_size: {input_size}")
        print(f"model: {self.model_name}")
        
        summary(model, (input_size,))
        self.trainer = mtrainer

    def setup_output(self, w_dir, acc_dir):
        
        print("=====Setting up output files=====")
        
        self.w_dir = w_dir
        self.acc_dir = acc_dir

        os.makedirs(self.acc_dir, exist_ok=True)
        os.makedirs(self.w_dir, exist_ok=True)
        
        if self.augment_option is None:
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_test_{self.test_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        elif self.test_option == 'original':
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_augment_{self.augment_option}_test_{self.test_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        else:
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_augment_{self.augment_option}_test_{self.test_option}_augment_{self.augment_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        self.acc_file_name = f"{self.w_file_name}.acc.csv"
        self.report_file_name = f"{self.w_file_name}.report.txt"
                
        print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
        print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")
        print(f"report_file: {self.report_file_name}")

    def get_dict_of_eval_metrics(self):
        metrics = sorted(list(self.eval_metrics.keys()))
        ret_dict = {}
        for metric in metrics:
            if self.eval_metrics[metric]== None:
                ret_dict[metric] = []
            else:
                for mtype in self.eval_metrics[metric]:
                    ret_dict[metric + "_" + mtype] = []
        return ret_dict


    def training(self):
        print("==========================================================================================")
        print("Start training...")

        save_at = [(i + 1) * 500 for i in range(int(self.num_epochs / 500))]

        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            train_to_write = ",".join([f"train_{x}" for x in list(self.get_dict_of_eval_metrics().keys())])
            test_to_write = ",".join([f"test_{x}" for x in list(self.get_dict_of_eval_metrics().keys())])

            acc_file.write("global_round,train_loss," + train_to_write + ",test_loss," + test_to_write + "\n")
            
            #acc_file.write("global_round,train_loss,train_acc,train_f1_binary,train_f1_macro,train_f1_micro,train_f1_sample,train_f1_weighted,test_loss,test_acc,test_f1_binary,test_f1_macro,test_f1_micro,test_f1_sample,test_f1_weighted\n")
            
        lr = self.learning_rate

        train_losses, test_losses = [], []
        

        train_scores = self.get_dict_of_eval_metrics()
        test_scores = self.get_dict_of_eval_metrics()
        
        best_loss = float("inf")
        best_score = 0 
        
        patience_counter = 0

        for epoch in range(self.num_epochs):
            print("-----------------------")
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            self.trainer.train(device, epochs = 1)  # Train for 1 epoch at a time

            train_loss, train_score = self.validate(data = self.train_data, load_weight=False)
            test_loss, test_score = self.validate(data = self.test_data, load_weight=False)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            for stype in list(train_score.keys()):
                train_scores[stype].append(train_score[stype])
                test_scores[stype].append(test_score[stype])

            print(f"Training statistic: Loss: {train_loss:.4f}, train_score: {train_score}")
            print(f"Testing statistic: Loss: {test_loss:.4f}, test_score: {test_score}")
                         
            with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                train_score_to_write = ",".join(map(str, list(train_score.values())))
                test_score_to_write = ",".join(map(str, list(test_score.values())))
                
                acc_file.write(f"{epoch+1},{train_loss},{train_score_to_write},{test_loss},{test_score_to_write}\n")
        
            if self.patience == -1: #no early stopping
                if epoch + 1 in save_at:
                    self.save_model(self.w_file_name)
            else:
                stop, patience_counter, best_loss, best_score = self.check_early_stop(test_loss, test_score, best_loss, best_score, patience_counter)
                
                if stop == True:
                    print("Early stopping (loss) triggered!")
                    print("Early stopping by " + self.early_stop_criterion)
                    break
                

        self.plot_loss_and_f1_curves(train_losses, test_losses, train_scores[self.metric_to_plot], test_scores[self.metric_to_plot])
        print("Finish training!")

    
    def check_early_stop(self, loss, score, best_loss, best_score, patience_counter):
        tolerance = 1e-5
        need_stopping = False
        if self.early_stop_criterion == "loss":
            if best_loss > loss + tolerance:
                best_loss = loss
                self.save_model(self.w_file_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if (patience_counter > self.patience):
                    need_stopping = True          
        else:
            if best_score < score[self.early_stop_criterion] - tolerance:
                best_score = score[self.early_stop_criterion]
                self.save_model(self.w_file_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if (patience_counter > self.patience):
                    need_stopping = True
            
        return need_stopping, patience_counter, best_loss, best_score

    def validate(self, data, load_weight=False):
        
        if load_weight:
            self.trainer.model.load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model.eval() # fixed this part
        
        loss, total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data):
                if self.multi_y == True:
                    X, y = X.to(device), y.to(device)
                else:
                    X, y = X.to(device), y.to(device).unsqueeze(1)
                
                output = self.trainer.model(X)
                
                predicted = 0
                if self.multi_y == True:
                    _, predicted = torch.max(output.data, 1)  # Changed prediction for multi-class
                else:
                    predicted = (output > 0.5).float()
                    
                loss += self.trainer.criterion(output, y).item() * len(y)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total += len(y)
                        
        loss = loss / total
        
        return loss, self.compute_scores(all_labels, all_preds)

    
    def compute_scores(self, y, y_hat):
        ret_score = self.get_dict_of_eval_metrics()
        for metric in list(self.eval_metrics.keys()):
            if metric == "accuracy":
                ret_score[metric] = accuracy_score(y, y_hat)
            elif metric == "f1":
                for mtype in self.eval_metrics[metric]:
                    ret_score[metric + "_"+ mtype] = f1_score(y, y_hat, average = mtype, zero_division = 0)
            elif metric == 'recall':
                for mtype in self.eval_metrics[metric]:
                    ret_score[metric + "_"+ mtype] = recall_score(y, y_hat, average = mtype, zero_division = 0)
            elif metric == 'precision':
                for mtype in self.eval_metrics[metric]: 
                    ret_score[metric + "_"+ mtype] = precision_score(y, y_hat, average = mtype, zero_division = 0)
            else:
                ValueError("Can not recognize the metrics: " + metric)
        return ret_score

    
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
        plt.plot(train_losses, label='Training', color='blue')
        plt.plot(test_losses, label='Validation', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
        plt.savefig(loss_plot_file)
        plt.close()

        # Plot F1 Score Curves
        f1_plot_file = os.path.join(self.acc_dir, f"{self.acc_file_name}_{self.metric_to_plot}.png")
        plt.figure(figsize=(10, 6))
        plt.plot(train_f1_scores, label='Training', color='blue')
        plt.plot(test_f1_scores, label='Validation', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel(self.metric_to_plot)
        plt.title(f'Training and Validation {self.metric_to_plot}')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
        plt.savefig(f1_plot_file)
        plt.close()