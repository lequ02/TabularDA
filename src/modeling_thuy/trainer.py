
import torch
from torch import nn, optim

class trainer:
    def __init__(self, model, data, criterion, learning_rate, dataset_name='', device='cuda' if torch.cuda.is_available() else 'cpu', multi_y = True):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.data = data 
        self.dataset_name = dataset_name
        self.multi_y = multi_y
        
    def clear_model(self):
        del self.model["model"]
        self.model["model"] = None

    def train(self, device, epochs):
        self.model.to(device)
        self.model.train()
        
        for batch_idx, (X, y) in enumerate(self.data):
            if self.multi_y == True:
                X, y = X.to(device), y.to(device)
            else:
                X, y = X.to(device), y.to(device).unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model.forward(X)
            
            try:
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
            except RuntimeError as e:
                print(f"Error details: {str(e)}")
                print(f"Output first few values: {output[:5]}")
                print(f"Target first few values: {y[:5]}")
                raise e
    
    def train1(self, device, epochs):
        self.model.to(device)
        self.model.train()
        
        for batch_idx, (X, y) in enumerate(self.data):
            if self.dataset_name == 'mnist12' or self.dataset_name == 'mnist28' or self.dataset_name.lower() == "intrusion":
                ## Print types and devices before conversion
                # print(f"Before conversion - y dtype: {y.dtype}, device: {y.device}")
                
                # Convert to long and move to device
                y = y.long()
                X, y = X.to(device), y.to(device)
                
                # # Print types and devices after conversion
                # print(f"After conversion - y dtype: {y.dtype}, device: {y.device}")
            else:
                X, y = X.to(device), y.to(device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            output = self.model.forward(X)
            
            # Print detailed tensor information
            # print(f"Output dtype: {output.dtype}, device: {output.device}")
            # print(f"Target dtype: {y.dtype}, device: {y.device}")
            # print(f"Output shape: {output.shape}")
            # print(f"Target shape: {y.shape}")
            # print(f"Target unique values: {torch.unique(y)}")
            
            try:
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
            except RuntimeError as e:
                print(f"Error details: {str(e)}")
                print(f"Output first few values: {output[:5]}")
                print(f"Target first few values: {y[:5]}")
                raise e
