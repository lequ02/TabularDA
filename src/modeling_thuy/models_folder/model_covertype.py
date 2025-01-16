import torch
import torch.nn as nn





class DNN_Covertype(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], output_size=8):
        super(DNN_Covertype, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.7)
        
        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.7)
        
        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.7)
        
        # Output Layer
        self.output = nn.Linear(hidden_sizes[2], output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        x = self.output(x)
        # x = self.softmax(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self
