import torch
import torch.nn as nn

######################## Adult Dataset ################

class DNN_Adult(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1):
        super(DNN_Adult, self).__init__()
        
        # Input layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.act1 = nn.ReLU()
        
        # Hidden layers
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.act3 = nn.ReLU()
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[2], output_size)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid() if output_size == 1 else None

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x


