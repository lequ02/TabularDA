import random
import pandas as pd
import math

import os
import torch
import copy
import numpy as np
from tqdm import tqdm


from torch import nn

from models import MLP2
from trainer import trainer
from data_loader import data_loader


from torchsummary import summary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class train:
    def __init__(self, dataset_name, batch_size, learning_rate, lr_decay, 
        num_epochs, w_dir, acc_dir, pre_trained_w_file = None):

        
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        
        self.data = data_loader(dataset_name, batch_size)
        self.trainer = self.setup_trainer(pre_trained_w_file)


        self.w_dir = w_dir
        self.acc_dir = acc_dir 

        self.w_file_name = self.model_name + "_lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay) + "_B" + str(self.batch_size) + "_G"+str(self.num_epochs) + ".weight.pth"
        self.acc_file_name = self.w_file_name + ".acc.csv"
        

        
        print ("Configuration: ")
        print("dataset, model: ", dataset_name, ", ", ", ", self.model_name)
        print("B, lr, lr_decay: ", batch_size, ", ", learning_rate, ", ", lr_decay)
        print("num_epochs :", num_epochs)

        
        print("weight_dir, weight_file: ", self.w_dir, self.w_file_name)
        print("acc_dir, acc_file: ", self.acc_dir, self.acc_file_name)


                     
    def setup_trainer(self, pre_trained_w_file):
        #mnist set up model and training data
        
        model = MLP2().to(device)
        self.model_name = "MLP2"
        
        if (pre_trained_w_file != None):
            print("Loading weight from " + pre_trained_w_file)
            model = load_state_dict(torch.load(self.w_dir + pre_trained_w_file)).to(device)
        
        mtrainer = trainer()
        mtrainer.model["model"] = copy.deepcopy(model) #self.copy_model(edge_server.model).to(device)
        mtrainer.data = self.data.train_data

        summary(mtrainer.model['model'], (1, 28, 28))
        return mtrainer


    def training(self):
        
        print("==========================================================================================")
        print("Start training...")

        if  not os.path.exists(self.acc_dir):
            os.makedirs(self.acc_dir)

        if  not os.path.exists(self.acc_dir):
            os.makedirs(self.acc_dir)
        
        save_at = []
        n_save_at = int(self.num_epochs / 500)

        for i in range(n_save_at):
            save_at.append( (i + 1) * 500 )
            
        with open(self.acc_dir + self.acc_file_name, 'w') as acc_file:
            acc_file.write("global_round,train_loss,train_acc,test_loss,test_acc\n")
        
        lr = self.learning_rate
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
    
            self.trainer.train(device, lr)            
        
            train_loss, train_acc = self.trainer.train_stats(device)
            
            print("Training statistic: ")
            print("Accuracy ","{0:.10}%".format(train_acc) , " Loss: ", f'{train_loss:.10}')
            
            # Validate new model
            test_loss, test_acc = self.validate(load_weight=False)
            print ("lr: ", lr)
            
            with open(self.acc_dir + self.acc_file_name, 'a') as acc_file:
                acc_file.write(str(epoch+1) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc)+ "\n")
                
            if epoch in save_at:
                fmodel = str(epoch) + "_" + self.w_file
                self.save_model(fmodel)

        print("Finish training!")
        acc_file.close()
        
                
        
    def validate(self, load_weight=False):

        print("Validation statistic...")

        if load_weight == True:
            self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

        self.trainer.model['model'].eval()
        corrects, loss = 0, 0
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data.test_data):
                X, y = X.to(device), y.to(device)
                output = self.trainer.model['model'](X)
                pred = output.argmax(dim=1)
                corrects += pred.eq(y.view_as(pred)).sum().item()

                loss += nn.CrossEntropyLoss()(output, y).item()


        total_test = len(self.data.test_data)*self.batch_size
        accuracy = 100*corrects/total_test
        loss = loss/len(self.data.test_data)

        print("Number of corrects: {}/{}".format(corrects, len(self.data.test_data)*self.batch_size))
        print("Accuracy, {}%".format(accuracy), " Loss: ", f'{loss:.3}')
        print("-------------------------------------------")

        return loss, accuracy 

    
    def save_model(self, fmodel):
        print("Saving model...")
        if not os.path.exists(self.w_dir):
            os.makedirs(self.w_dir)
        torch.save(self.model.state_dict(), self.w_dir + fmodel)
        print("Model saved!")