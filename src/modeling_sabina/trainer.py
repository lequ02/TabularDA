
import torch
from torch import nn, optim

class trainer:
    def __init__(self, model, data, criterion, learning_rate, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.data = data 
        
    def clear_model(self):
        del self.model["model"]
        self.model["model"] = None

    def train(self, device, epochs):
        # Train the model
        self.model.to(device)
        self.model.train()
        # print("SELF.DATA IS", self.data)
        for batch_idx, (X, y) in enumerate(self.data):

            
            
            X, y = X.to(device), y.to(device).float().unsqueeze(1)

            # nan_columns = torch.isnan(X).any(dim=0)

            

            # # Print the columns with NaN values
            # print("Columns with NaN values:")
            # print(X[:, nan_columns])
            # print(nan_columns)
            
            
            self.optimizer.zero_grad()
            output = self.model.forward(X)


            # print(f"y : {y}")
            # print (f"output : {output}")

            

            loss = self.criterion(output, y)

           
            loss.backward()
            self.optimizer.step()


            
