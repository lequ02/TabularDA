import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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


## most results are run using this model
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

# class CNN_MNIST28(nn.Module):
#     def __init__(self, input_size=784, hidden_sizes=[9216, 128], output_size=10):

#         super(CNN_MNIST28, self).__init__()
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


# class EfficientNetWideSE(nn.Module):
#     def __init__(self, input_size=None, pretrained=True):
#         super(EfficientNetWideSE, self).__init__()
#         # Load the EfficientNet-WideSE model
#         # self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=pretrained)
#         self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)


#     def forward(self, x):
#         # Repeat the single channel 3 times to make it RGB
#         x = x.repeat(1, 3, 1, 1)  # This converts (batch_size, 1, 28, 28) to (batch_size, 3, 28, 28)
#         # Forward pass through the model
#         x = self.model(x)
#         return x

#     def train(self, mode=True):
#         super().train(mode)
#         return self




# # https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# class CNN_MNIST28(nn.Module):
#     def __init__(self, input_size=(1, 28, 28), hidden_size=100, output_size=10):
#         super(CNN_MNIST28, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.dropout = nn.Dropout(0.5)

#         # Compute the correct flattened size dynamically
#         self.flatten_size = self._get_flattened_size(input_size)

#         self.fc1 = nn.Linear(self.flatten_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#         # Apply He Initialization (Kaiming Uniform)
#         self._initialize_weights()

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)  # Flatten before passing into FC layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

#     def train(self, mode=True):
#         super().train(mode)
#         return self

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 init.zeros_(m.bias)

#     def _get_flattened_size(self, input_size):
#         """ Pass a dummy input through conv layers to compute the output size dynamically. """
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *input_size)  # Create a fake batch of size 1
#             x = F.relu(self.conv1(dummy_input))
#             x = self.pool(x)
#             x = F.relu(self.conv2(x))
#             x = F.relu(self.conv3(x))
#             x = self.pool(x)
#             return x.numel()  # Compute the total number of features