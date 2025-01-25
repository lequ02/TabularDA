import torch
import torch.nn as nn

class DNN_News(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 512, 256, 128], output_size=1):

        # add relu output
        # add beginning layer
        # redesign the hidden layers

        super(DNN_News, self).__init__()
        
        # Input layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)
        
        # Hidden layers
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.4)
        
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.4)
        
        # Output layer
        
        # can't use nn.ReLU() here
        
        # self.output = nn.ReLU(hidden_sizes[-1], output_size)
        # TypeError: ReLU.__init__() takes from 1 to 2 positional arguments but 3 were given

        # self.output = nn.ReLU()
        # ValueError: y_true and y_pred have different number of output (1!=64)

        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.drop1(self.bn1(self.act1(self.layer1(x))))
        x = self.drop2(self.bn2(self.act2(self.layer2(x))))
        x = self.drop3(self.bn3(self.act3(self.layer3(x))))
        x = self.drop4(self.bn4(self.act4(self.layer4(x))))
        x = self.output(x)
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self