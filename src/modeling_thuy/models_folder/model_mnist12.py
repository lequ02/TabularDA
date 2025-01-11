import torch
import torch.nn as nn

######################## MNIST12 Dataset ################

class DNN_MNIST12(nn.Module):
    def __init__(self, input_size=144, hidden_sizes=[128, 256, 128], output_size=10):
        super(DNN_MNIST12, self).__init__()
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)


        # Output Layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        # self.softmax = nn.Softmax()    # don't use softmax because it's included in the loss function CrossEntropyLoss


    def forward(self, x):

        # # Make sure input is correctly shaped
        # if len(x.shape) == 3:  # If input is [batch, height, width]
        #     x = x.reshape(x.shape[0], -1)
        # elif len(x.shape) == 4:  # If input is [batch, channels, height, width]
        #     x = x.reshape(x.shape[0], -1)

        x = self.drop1(self.act1(self.layer1(x)))
        x = self.drop2(self.act2(self.layer2(x)))
        x = self.drop3(self.act3(self.layer3(x)))
        # x = self.softmax(x)  # output probabilities
        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self

    
    