import random
import numpy as np
import copy
import torch
from torch import nn, optim
from models import DNN_Adult, DNN_Census, DNN_News 
from sklearn.metrics import mean_squared_error
import math


random.seed(1)

class trainer:
    def __init__(self):
        # Initialize model details and data
        self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
        self.data = []
        self.model_type = None

    def clear_model(self):
        del self.model["model"]
        self.model["model"] = None

    def set_model(self, model_type, input_size, hidden_sizes, output_size):
        # Set the model based on the type and initialize the criterion
        self.model_type = model_type
        if model_type == "adult":
            self.model["model"] = DNN_Adult(input_size, hidden_sizes, output_size)
            self.model["criterion"] = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        elif model_type == "census":
            self.model["model"] = DNN_Census(input_size, hidden_sizes, output_size)
            self.model["criterion"] = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        elif model_type == "news":
            self.model["model"] = DNN_News(input_size, hidden_sizes, output_size)
            self.model["criterion"] = nn.MSELoss()  # Mean Squared Error Loss for regression
        else:
            raise ValueError("Unknown model type")

    def train(self, device, lr, epochs):
        # Train the model
        self.model["optim"] = optim.Adam(self.model["model"].parameters(), lr=lr)  # Using Adam optimizer
        self.model["model"].to(device)
        self.model["model"].train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (X, y) in enumerate(self.data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)  # Ensure y is float and has correct shape

                self.model["optim"].zero_grad()
                output = self.model["model"].forward(X)

                loss = self.model["criterion"](output, y)
                epoch_loss += loss.item()

                loss.backward()
                self.model["optim"].step()

    def train_stats(self, device):
        # Evaluating the model and return statistics
        self.model["model"].eval()
        corrects, loss, total = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                output = self.model["model"](X)
                
                if self.model_type in ["adult", "census"]:
                    pred = (output > 0.5).float()  # Threshold at 0.5 for binary classification
                    corrects += pred.eq(y).sum().item()
                else:
                    pred = output  # For regression, no thresholding

                loss += self.model["criterion"](output, y).item() * len(y)
                total += len(y)

        if self.model_type in ["adult", "census"]:
            accuracy = 100 * corrects / total
            loss = loss / total
            return loss, accuracy
        else:
            loss = loss / total  # For regression
            return loss, None  # Return None for accuracy in regression

