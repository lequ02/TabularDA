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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error
# from models import DNN_News
from models_folder.model_news import DNN_News
from trainer import trainer
import constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

out_data_path = constants.OUT_DATA_PATHS

class train:

    #in_data_path = "../../data/"
    
    def __init__(self, dataset_name, train_option = None, augment_option = None, mix_ratio = -1, n_sample = -1, test_option = None, validation = 0.2,
                        batch_size = 128, learning_rate = 0.001, num_epochs = 10, patience=10, early_stop_criterion='loss', 
                        eval_metrics = None, metric_to_plot = None,
                        pre_trained_w_file = None,
                        w_dir = None, acc_dir = None ):
        
        self.setup_data(dataset_name = dataset_name, train_option = train_option,
                        augment_option = augment_option, mix_ratio = mix_ratio, n_sample = n_sample, test_option = test_option, validation = validation, batch_size = batch_size)

        self.setup_fitting(num_epochs = num_epochs, learning_rate = learning_rate, patience=patience, early_stop_criterion=early_stop_criterion)
        self.setup_eval_metric(eval_metrics, metric_to_plot)
        self.setup_trainer(pre_trained_w_file = pre_trained_w_file)
        self.setup_output(w_dir = w_dir, acc_dir = acc_dir)

    def setup_data(self, dataset_name, train_option, augment_option, mix_ratio, n_sample, test_option, validation, batch_size):

        
        print("=====Setting up data=====")
        
        self.multi_y = False

        self.dataset_name = dataset_name
        self.train_option = train_option
        self.augment_option = augment_option
        self.mix_ratio = mix_ratio
        self.n_sample = n_sample

        self.test_option = test_option
        self.batch_size = batch_size
        self.validation = validation

        self.data_loader = data_loader(self.dataset_name, self.batch_size, multi_y = self.multi_y, problem_type = 'regression')
        self.train_data, self.dev_data = self.data_loader.load_train_augment_data(self.train_option, self.augment_option, self.mix_ratio, self.n_sample, self.validation)
        
        self.test_data = self.data_loader.load_test_data()

        print(f"Dataset name:{self.dataset_name}")
        print(f"train_option:{self.train_option}")
        print(f"augment_option:{self.augment_option}")
        print(f"mix_ratio:{self.mix_ratio}")
        print(f"n_sample:{self.n_sample}")
        
        print(f"Dataset name:{self.dataset_name}")
        print(f"Training dataset size: {len(self.train_data) * self.batch_size} samples")
        print(f"Validation dataset size: {len(self.dev_data) * self.batch_size} samples")
        print(f"Testing dataset size: {len(self.test_data) * self.batch_size} samples")
        print("")

        
    def setup_fitting(self, num_epochs, learning_rate, patience, early_stop_criterion):
        print("=====Setting up parameters for training=====")
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience

        self.early_stop_criterion = early_stop_criterion.lower()
        
        print(f"num_epochs: {self.num_epochs}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"patience: {self.patience}")
        print(f"early_stop_criterion: {self.early_stop_criterion}")
    
    def setup_eval_metric(self, eval_metrics, metric_to_plot):
        #use all types of f1 and accuracy
        #set up a metric to plot when training is done

        print("=====Setting up metrics for evaluating and plotting=====")
        
        self.eval_metrics = [m.lower() for m in eval_metrics]
        self.metric_to_plot = metric_to_plot
        print(f"metrics for evaluating: {self.eval_metrics}")
        print(f"metric for plotting: {self.metric_to_plot}")


    def setup_trainer(self, pre_trained_w_file):
        print("=====Setting up trainer=====")
        input_size = next(iter(self.train_data))[0].shape[1]
        criterion = nn.MSELoss(reduction = 'mean')
    
        if self.dataset_name.lower() == "news":
            model = DNN_News(input_size=input_size).to(device)
            self.model_name = "DNN_News"
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
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_augment_{self.augment_option}_mix_ratio{self.mix_ratio}_n{self.n_sample}_test_{self.test_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        else:
            self.w_file_name = f"{self.model_name}_train_{self.train_option}_augment_{self.augment_option}_mix_ratio{self.mix_ratio}_n{self.n_sample}_test_{self.test_option}_augment_{self.augment_option}_lr{self.learning_rate}_B{self.batch_size}_G{self.num_epochs}.weight.pth"
        self.acc_file_name = f"{self.w_file_name}.acc.csv"
        self.report_file_name = f"{self.w_file_name}.report.txt"
                
        print(f"weight_dir, weight_file: {self.w_dir}, {self.w_file_name}")
        print(f"acc_dir, acc_file: {self.acc_dir}, {self.acc_file_name}")
        print(f"report_file: {self.report_file_name}")

    def get_dict_of_eval_metrics(self):
        ret_dict = {}
        for metric in self.eval_metrics:
            ret_dict[metric] = []
        return ret_dict


    def training(self):
        print("==========================================================================================")
        print("Start training...")

        z = 10
        save_at = [(i + 1) * z for i in range(int(self.num_epochs / z))] #save model at every 20 iteration, start from 1
        save_at = [0] + save_at

        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            train_to_write = ",".join([f"train_{x}" for x in self.eval_metrics])
            dev_to_write = ",".join([f"dev_{x}" for x in self.eval_metrics])
            test_to_write = ",".join([f"test_{x}" for x in self.eval_metrics])

            acc_file.write("global_round,train_loss," + train_to_write + ",dev_loss," + dev_to_write + ",test_loss," + test_to_write+"\n")
                        
        lr = self.learning_rate

        train_losses, dev_losses, test_losses = [], [], []
        

        train_scores = self.get_dict_of_eval_metrics()
        dev_scores = self.get_dict_of_eval_metrics()
        test_scores = self.get_dict_of_eval_metrics()
        
        #for early stopping

        best_error = float("inf") #any type of errors, including loss
        best_score = float("-inf")            #r2, bigger is better
        
        
        patience_counter = 0

        for epoch in range(self.num_epochs):
            print("-----------------------")
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            self.trainer.train(device, epochs = 1)  # Train for 1 epoch at a time

            train_loss, train_score = self.validate(data = self.train_data, load_weight=False)
            dev_loss, dev_score = self.validate(data = self.dev_data, load_weight=False)
            test_loss, test_score = self.validate(data = self.test_data, load_weight=False)

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            test_losses.append(test_loss)
            
            for metric in self.eval_metrics:
                train_scores[metric].append(train_score[metric])
                dev_scores[metric].append(dev_score[metric])
                test_scores[metric].append(test_score[metric])

            print(f"Training statistic: Loss: {train_loss:.4f}, train_score: {train_score}")
            print(f"Validation statistic: Loss: {dev_loss:.4f}, dev_score: {dev_score}")
                         
            with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                train_score_to_write = ",".join(map(str, list(train_score.values())))
                dev_score_to_write = ",".join(map(str, list(dev_score.values())))
                test_score_to_write = ",".join(map(str, list(test_score.values())))
                
                acc_file.write(f"{epoch+1},{train_loss},{train_score_to_write},{dev_loss},{dev_score_to_write},{test_loss},{test_score_to_write}\n")
        
            if self.patience == -1: #no early stopping
                if epoch in save_at:
                    self.save_model(self.w_file_name)
            else:
                stop, patience_counter, best_error, best_score = self.check_early_stop(dev_loss, dev_score, best_error, best_score, patience_counter)
                
                if stop == True:
                    print("Early stopping (loss) triggered!")
                    print("Early stopping by " + self.early_stop_criterion)
                    break
                
        # test_loss, test_score = self.validate(data = self.test_data, load_weight=True)
        # print(f"Testing statistic: loss: {test_loss}, scores: {test_score}")
        # self.plot_loss_and_score_curves(train_losses, dev_losses, train_scores[self.metric_to_plot], dev_scores[self.metric_to_plot])
        
        print(f"Testing statistic: loss: {test_loss}, scores: {test_score}")
        self.plot_loss_and_f1_curves(train_losses, train_scores[self.metric_to_plot], 
                                     dev_losses, dev_scores[self.metric_to_plot], 
                                     test_losses=test_losses, test_f1_scores=test_scores[self.metric_to_plot])

        print("Finish training!")

    
    def check_early_stop(self, loss, score, best_error, best_score, patience_counter):
        tolerance = 1e-5
        need_stopping = False
        if self.early_stop_criterion in ["mae", "mse", "rmse", "mape", "loss"]:
            if self.early_stop_criterion == "loss":
                error = loss
            else:
                error = score[self.early_stop_criterion]
            if (best_error > error + tolerance):
                best_error = error
                
                self.save_model(self.w_file_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if (patience_counter > self.patience):
                    need_stopping = True

        elif self.early_stop_criterion in ["r2"]:
            if best_score < score[self.early_stop_criterion] - tolerance:
                best_score = score[self.early_stop_criterion]
                self.save_model(self.w_file_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if (patience_counter > self.patience):
                    need_stopping = True
        else:
            ValueError(f"Cannot recognize the metric for early stopping {self.early_stop_criterion}")
        return need_stopping, patience_counter, best_error, best_score

    def validate(self, data, load_weight=False):
        
        if load_weight:
            self.trainer.model.load_state_dict(torch.load(self.w_dir + self.w_file_name))

        self.trainer.model.eval() # fixed this part
        
        loss, total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data):
                
                X, y = X.to(device), y.to(device).unsqueeze(1)
                
                output = self.trainer.model(X)
                
                    
                loss += self.trainer.criterion(output, y).item() * len(y)
                
                all_preds.extend(output.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total += len(y)
                        
        loss = loss / total
        
        return loss, self.compute_scores(all_labels, all_preds)

    
    def compute_scores(self, y, y_hat):
        ret_score = self.get_dict_of_eval_metrics()
        for metric in self.eval_metrics:
            if metric == "mse":
                ret_score[metric] = mean_squared_error(y, y_hat)
            elif metric == "rmse":
                ret_score[metric] = root_mean_squared_error(y, y_hat)
            elif metric == "mae":
                ret_score[metric] = mean_absolute_error(y, y_hat)
            elif metric == "r2":
                ret_score[metric] = r2_score(y, y_hat)
            elif metric == "mape":
                ret_score[metric] = mean_absolute_percentage_error(y, y_hat)
            else:
                ValueError(f"Cannot recognize the metric {metric}")
        return ret_score
    
    def save_model(self, fmodel):
        print("Saving model...")
        torch.save(self.trainer.model.state_dict(), self.w_dir + fmodel)
        print("Model saved!")
        print("-------------------------------------------")

    # def plot_loss_and_score_curves(self, train_losses, test_losses, train_scores, test_scores):
    #     """Plots and saves training/validation loss and score curves to the model directory."""
    #     # Ensure the directory exists
    #     os.makedirs(self.acc_dir, exist_ok=True)
    #     # Determine the number of x-ticks to display automatically
    #     num_epochs = len(train_losses)
    #     max_ticks = min(num_epochs, 25)  # Maximum number of ticks to display
    #     step = max(1, num_epochs // max_ticks)
    #     x_ticks = range(0, num_epochs, step)

    #     # Plot Loss Curves
    #     loss_plot_file = os.path.join(self.acc_dir, f"{self.acc_file_name}_loss_curve.png")
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(train_losses, label='Training', color='blue')
    #     plt.plot(test_losses, label='Validation', color='orange')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss Curves')
    #     plt.legend()
    #     plt.grid()
    #     plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
    #     plt.savefig(loss_plot_file)
    #     plt.close()

    #     # Plot Score Curves
    #     score_plot_file = os.path.join(self.acc_dir, f"{self.acc_file_name}_{self.metric_to_plot}.png")
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(train_scores, label='Training', color='blue')
    #     plt.plot(test_scores, label='Validation', color='orange')
    #     plt.xlabel('Epochs')
    #     plt.ylabel(self.metric_to_plot)
    #     plt.title(f'Training and Validation {self.metric_to_plot}')
    #     plt.legend()
    #     plt.grid()
    #     plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
    #     plt.savefig(score_plot_file)
    #     plt.close()



    def plot_loss_and_f1_curves(self, train_losses, train_f1_scores, val_losses, val_f1_scores, test_losses=None, test_f1_scores=None):
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
        plt.plot(val_losses, label='Validation', color='orange')
        if test_losses is not None:
            plt.plot(test_losses, label='Testing', color='red')
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
        plt.plot(val_f1_scores, label='Validation', color='orange')
        if test_f1_scores is not None:
            plt.plot(test_f1_scores, label='Testing', color='red')
        plt.xlabel('Epochs')
        plt.ylabel(self.metric_to_plot)
        plt.title(f'Training and Validation {self.metric_to_plot}')
        plt.legend()
        plt.grid()
        plt.xticks(x_ticks, [str(x) for x in x_ticks])  # Ensure x-labels are integers
        plt.savefig(f1_plot_file)
        plt.close()