import random
import numpy as np
import copy
import torch
from torch import nn, optim

random.seed(1)

class trainer:
    def __init__(self):
        self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
        self.data = []
        self.model["criterion"] = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification

    def clear_model(self):
        del self.model["model"]
        self.model["model"] = None

    def train(self, device, lr, epochs):
        self.model["optim"] = optim.Adam(self.model["model"].parameters(), lr=lr)  # Using Adam optimizer
        scheduler = optim.lr_scheduler.StepLR(self.model["optim"], step_size=10, gamma=0.1)  # Learning rate scheduler
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
            
            scheduler.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(self.data)}')

    def train_stats(self, device):
        self.model["model"].eval()
        corrects, loss, total = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.data):
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
                output = self.model["model"](X)
                pred = (output > 0.5).float()  # Threshold at 0.5 for binary classification

                corrects += pred.eq(y).sum().item()
                loss += self.model["criterion"](output, y).item() * len(y)
                total += len(y)

        accuracy = 100 * corrects / total
        loss = loss / total
        return loss, accuracy