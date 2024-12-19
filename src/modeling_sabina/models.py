import torch
import torch.nn as nn

######################## Adult Dataset ################

class DNN_Adult(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=1):
        super(DNN_Adult, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        
        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.2)
        
        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.LeakyReLU()
        self.drop3 = nn.Dropout(0.2)
        
        # Fourth Hidden Layer
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.act4 = nn.LeakyReLU()
        self.drop4 = nn.Dropout(0.2)

        # # Fifth Hidden Layer
        # self.layer5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        # self.bn5 = nn.BatchNorm1d(hidden_sizes[4])
        # self.act5 = nn.LeakyReLU()
        # self.drop5 = nn.Dropout(0.2)

        # # Sixth Hidden Layer
        # self.layer6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
        # self.bn6 = nn.BatchNorm1d(hidden_sizes[5])
        # self.act6 = nn.LeakyReLU()
        # self.drop6 = nn.Dropout(0.2)

        # Output Layer
        self.output = nn.Linear(hidden_sizes[3], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        x = self.drop4(self.act4(self.bn4(self.layer4(x))))
        # x = self.drop5(self.act5(self.bn5(self.layer5(x))))
        # x = self.drop6(self.act6(self.bn6(self.layer6(x))))
        x = self.output(x)
        x = self.sigmoid(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self

    
    
    
    
    
    
    
    
    
    
    
    
    
    ######################## Census Dataset ################
    

class DNN_Census(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=1):
        super(DNN_Census, self).__init__()
        
        # Input layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.7)
        
        # Hidden layers
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.7)
        
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.7)
        
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.7)

        
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[3], output_size)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid() if output_size == 1 else None

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        x = self.drop4(self.act4(self.bn4(self.layer4(x))))
        x = self.output(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self
    









    
    

######################## News Dataset ################


class DNN_News(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=1):
        super(DNN_News, self).__init__()
        
        # Input layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.7)
        
        # Hidden layers
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.7)
        
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.7)
        
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.7)
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[3], output_size)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        x = self.drop4(self.act4(self.bn4(self.layer4(x))))
        x = self.output(x)
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self
    
    
    
