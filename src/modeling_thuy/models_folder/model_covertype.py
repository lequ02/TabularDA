import torch
import torch.nn as nn





class DNN_Covertype(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 512, 512], output_size=8):
        super(DNN_Covertype, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        
        # Output Layer
        self.output = nn.Linear(hidden_sizes[2], output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop1(self.bn1(self.act1(self.layer1(x))))
        x = self.drop2(self.bn2(self.act2(self.layer2(x))))
        x = self.drop3(self.bn3(self.act3(self.layer3(x))))
        x = self.output(x)
        # x = self.softmax(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self


# class DNN_Adult_mix(nn.Module):
#     def __init__(self, input_size, hidden_sizes=[128, 256, 512, 256, 128], output_size=1):
#         super(DNN_Adult_mix, self).__init__()
        
#         # First Hidden Layer
#         self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#         self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
#         self.act1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.0)
        
#         # Second Hidden Layer
#         self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
#         self.act2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.2)
        
#         # Third Hidden Layer
#         self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.3)

# # fourth Hidden Layer
#         self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
#         self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
#         self.act4 = nn.ReLU()
#         self.drop4 = nn.Dropout(0.1)

# # fifth Hidden Layer
#         self.layer5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
#         self.bn5 = nn.BatchNorm1d(hidden_sizes[4])
#         self.act5 = nn.ReLU()
#         self.drop5 = nn.Dropout(0.0)

#         # Output Layer
#         self.output = nn.Linear(hidden_sizes[5], output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.drop1(self.act1(self.bn1(self.layer1(x))))
#         x = self.drop2(self.act2(self.bn2(self.layer2(x))))
#         x = self.drop3(self.act3(self.bn3(self.layer3(x))))
#         x = self.drop4(self.act4(self.bn4(self.layer4(x))))
#         x = self.drop5(self.act5(self.bn5(self.layer5(x))))
        
#         x = self.output(x)
#         x = self.sigmoid(x)  # output probabilities
#         return x
    
#     def train(self, mode=True):
#         super().train(mode)
#         return self











