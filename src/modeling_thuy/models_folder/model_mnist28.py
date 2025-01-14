import torch
import torch.nn as nn
import torch.nn.functional as F


######################## MNIST12 Dataset ################

#### MODEL 1 ####
# class DNN_MNIST28_1(nn.Module):
#     def __init__(self, input_size=784, hidden_sizes=[128, 256, 128], output_size=10):
#         super(DNN_MNIST28, self).__init__()
        
#         # First Hidden Layer
#         self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#         self.act1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.2)

#         # Second Hidden Layer
#         self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.act2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.2)

#         # Third Hidden Layer
#         self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.2)


#         # Output Layer
#         self.output = nn.Linear(hidden_sizes[-1], output_size)
#         # self.softmax = nn.Softmax()    # don't use softmax because it's included in the loss function CrossEntropyLoss


#     def forward(self, x):

#         x = self.drop1(self.act1(self.layer1(x)))
#         x = self.drop2(self.act2(self.layer2(x)))
#         x = self.drop3(self.act3(self.layer3(x)))
#         return x
    
#     def train(self, mode=True):
#         super().train(mode)
#         return self


# #### MODEL 2 #### more layers + dropout + batchnorm
# class DNN_MNIST28(nn.Module):
#     def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], output_size=10):
#         super(DNN_MNIST28, self).__init__()
        
#         # First Hidden Layer
#         self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#         self.act1 = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
#         self.drop1 = nn.Dropout(0.2)

#         # Second Hidden Layer
#         self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.act2 = nn.ReLU()
#         self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
#         self.drop2 = nn.Dropout(0.3)

#         # Third Hidden Layer
#         self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.act3 = nn.ReLU()
#         self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
#         self.drop3 = nn.Dropout(0.4)


#         # Output Layer
#         self.output = nn.Linear(hidden_sizes[-1], output_size)
#         # self.softmax = nn.Softmax()    # don't use softmax because it's included in the loss function CrossEntropyLoss


#     def forward(self, x):

#         # x = self.drop1(self.act1(self.layer1(x)))
#         # x = self.drop2(self.act2(self.layer2(x)))
#         # x = self.drop3(self.act3(self.layer3(x)))

#         x = self.drop1(self.act1(self.bn1(self.layer1(x))))
#         x = self.drop2(self.act2(self.bn2(self.layer2(x))))
#         x = self.drop3(self.act3(self.bn3(self.layer3(x))))

#         return x
    
#     def train(self, mode=True):
#         super().train(mode)
#         return self



# #### MODEL 3 #### 
# class DNN_MNIST28(nn.Module):
#     def __init__(self, input_size=784, hidden_sizes=[512, 1024, 256, 128], output_size=10):
#         super(DNN_MNIST28, self).__init__()
        
#         # First Hidden Layer
#         self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#         self.act1 = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
#         self.drop1 = nn.Dropout(0.2)

#         # Second Hidden Layer
#         self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.act2 = nn.ReLU()
#         self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
#         self.drop2 = nn.Dropout(0.3)

#         # Third Hidden Layer
#         self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.act3 = nn.ReLU()
#         self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
#         self.drop3 = nn.Dropout(0.4)

#         # Fourth Hidden Layer
#         self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
#         self.act4 = nn.ReLU()
#         self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
#         self.drop4 = nn.Dropout(0.5)


#         # Output Layer
#         self.output = nn.Linear(hidden_sizes[-1], output_size)
#         # self.softmax = nn.Softmax()    # don't use softmax because it's included in the loss function CrossEntropyLoss


#     def forward(self, x):


#         x = self.drop1(self.act1(self.bn1(self.layer1(x))))
#         x = self.drop2(self.act2(self.bn2(self.layer2(x))))
#         x = self.drop3(self.act3(self.bn3(self.layer3(x))))
#         x = self.drop4(self.act4(self.bn4(self.layer4(x))))

#         return x
    
#     def train(self, mode=True):
#         super().train(mode)
#         return self


#### MODEL 4 #### 
class DNN_MNIST28(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 1024, 256, 128], output_size=10):
        super(DNN_MNIST28, self).__init__()

    # input dropout and 0.5 dropout for hidden layers (Dropout paper, (Srivastava, Hinton, et al. 2014))

        # Input regularization
        self.input_drop = nn.Dropout(0.1)
        
        # First Hidden Layer
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.drop1 = nn.Dropout(0.5)

        # Second Hidden Layer
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.drop2 = nn.Dropout(0.5)

        # Third Hidden Layer
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.drop3 = nn.Dropout(0.5)

        # Fourth Hidden Layer
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.drop4 = nn.Dropout(0.5)


        # Output Layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        # self.softmax = nn.Softmax()    # don't use softmax because it's included in the loss function CrossEntropyLoss

        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.input_drop(x)

        x = self.drop1(self.bn1(self.act1(self.layer1(x))))
        x = self.drop2(self.bn2(self.act2(self.layer2(x))))
        x = self.drop3(self.bn3(self.act3(self.layer3(x))))
        x = self.drop4(self.bn4(self.act4(self.layer4(x))))

        return x
    
    def train(self, mode=True):
        super().train(mode)
        return self


    
# #### MODEL 0: CNN ####

# class DNN_MNIST28(nn.Module):
#     def __init__(self, input_size=784, hidden_sizes=[9216, 128], output_size=10):

#         super(DNN_MNIST28, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.fc2 = nn.Linear(hidden_sizes[1], output_size)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output
    
#     def train(self, mode=True):
#         super().train(mode)
#         return self
