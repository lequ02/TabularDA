import torch
from torch import nn, optim

class DNN_Intrusion(nn.Module):
    def __init__(self, input_size=126, hidden_sizes=[150, 250, 300, 250, 150], output_size=23):
        
        #23 classes

        super(DNN_Intrusion, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.05)

        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)

        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.15)

        # Fourth Hidden Layer
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.1)

        # Fifth Hidden Layer
        self.layer5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.05)


        # Output Layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        #self.softmax = nn.Softmax(dim = 1)    

    def forward(self, x):

        # # Make sure input is correctly shaped
        # if len(x.shape) == 3:  # If input is [batch, height, width]
        #     x = x.reshape(x.shape[0], -1)
        # elif len(x.shape) == 4:  # If input is [batch, channels, height, width]
        #     x = x.reshape(x.shape[0], -1)

        x = self.drop1(self.act1(self.layer1(x)))
        x = self.drop2(self.act2(self.layer2(x)))
        x = self.drop3(self.act3(self.layer3(x)))
        x = self.drop4(self.act4(self.layer4(x)))
        x = self.drop5(self.act5(self.layer5(x)))
        x = self.output(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self
