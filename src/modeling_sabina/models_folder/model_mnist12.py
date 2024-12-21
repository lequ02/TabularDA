import torch
import torch.nn as nn

######################## MNIST12 Dataset ################

class DNN_MNIST12(nn.Module):
    def __init__(self, input_size=144, hidden_sizes=[256], output_size=10):
        super(DNN_MNIST12, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        # self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.act1 = nn.ReLU()

        # Output Layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        # self.softmax = nn.Softmax()

    def forward(self, x):

        # Make sure input is correctly shaped
        if len(x.shape) == 3:  # If input is [batch, height, width]
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 4:  # If input is [batch, channels, height, width]
            x = x.reshape(x.shape[0], -1)

        x = self.layer1(x)
        x = self.act1(x)
        x = self.output(x)
        # x = self.softmax(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self

    
    