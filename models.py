## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # All the convolution layers
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # goes through pooling, so outputs (32, 110, 110)
        self.drop_c1 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on
        # 32 input image channel (conv1), 64 output channels/feature maps, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 5) # goes through pooling, so outputs (64, 53, 53)
        self.drop_c2 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on
        # 64 input image channel (conv2), 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3) # goes through pooling, so outputs (128, 25, 25)
        self.drop_c3 = nn.Dropout(p=0.2)
        # 128 input image channel (conv3), 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3) # goes through pooling, so outputs (256, 11, 11)
        self.drop_c4 = nn.Dropout(p=0.3)
        # 256 input image channel (conv2), 512 output channels/feature maps, 3x3 square convolution kernel
        self.conv5 = nn.Conv2d(256, 512, 3) # goes through pooling, so outputs (512, 4, 4)
        self.drop_c5 = nn.Dropout(p=0.4)
        
        # All the dense layers
        # 512 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(512*4*4, 2000)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2000, 1000)
        self.drop_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1000, 136)  # total number of keypoints: 68*2
        
        
        # Pooling is 2x2 across all feature levels
        self.pool2d = nn.MaxPool2d(2, 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # 5 conv + pooling layers with dropout probability increasing from 0.1 to 0.5
        x = self.pool2d(F.elu(self.conv1(x)))
        x = self.drop_c1(x)
        x = self.pool2d(F.elu(self.conv2(x)))
        x = self.drop_c2(x)
        x = self.pool2d(F.elu(self.conv3(x)))
        x = self.drop_c3(x)
        x = self.pool2d(F.elu(self.conv4(x)))
        x = self.drop_c4(x)
        x = self.pool2d(F.elu(self.conv5(x)))
        x = self.drop_c5(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # 3 linear layer with dropout in between
        x = F.elu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.elu(self.fc2(x))
        x = self.drop_fc2(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # All the convolution layers
        # 1 input image channel (rgb), 16 output channels/feature maps, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 3) # goes through pooling, so outputs (16, 111, 111)
#         self.drop_c1 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on, BAD, VERY BAD!
        # 16 input image channel (conv1), 32 output channels/feature maps, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(16, 32, 3) # goes through pooling, so outputs (32, 54, 54)
        self.drop_c2 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on
        # 32 input image channel (conv2), 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(32, 64, 3) # goes through pooling, so outputs (64, 26, 26)
        self.drop_c3 = nn.Dropout(p=0.2)
        # 64 input image channel (conv3), 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(64, 128, 3) # goes through pooling, so outputs (128, 12, 12)
        self.drop_c4 = nn.Dropout(p=0.3)
        # 128 input image channel (conv2), 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv5 = nn.Conv2d(128, 256, 3) # goes through pooling, so outputs (256, 5, 5)
        self.drop_c5 = nn.Dropout(p=0.4)
        
        # All the dense layers
        # 512 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.drop_fc2 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(1000, 136)  # total number of keypoints: 68*2
        
        
        # Pooling is 2x2 across all feature levels
        self.pool2d = nn.MaxPool2d(2, 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # 5 conv + pooling layers with dropout probability increasing from 0.1 to 0.5
        x = self.pool2d(F.elu(self.conv1(x)))
#         x = self.drop_c1(x)
        x = self.pool2d(F.elu(self.conv2(x)))
        x = self.drop_c2(x)
        x = self.pool2d(F.elu(self.conv3(x)))
        x = self.drop_c3(x)
        x = self.pool2d(F.elu(self.conv4(x)))
        x = self.drop_c4(x)
        x = self.pool2d(F.elu(self.conv5(x)))
        x = self.drop_c5(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # 3 linear layer with dropout in between
        x = F.elu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.elu(self.fc2(x))
        x = self.drop_fc2(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net3(nn.Module):
#     This is a network, that trains on color images

    def __init__(self):
        super(Net3, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # All the convolution layers
        # 3 input image channel (rgb), 16 output channels/feature maps, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, 3) # goes through pooling, so outputs (16, 111, 111)
        self.drop_c1 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on
        # 16 input image channel (conv1), 32 output channels/feature maps, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(16, 32, 3) # goes through pooling, so outputs (32, 54, 54)
        self.drop_c2 = nn.Dropout(p=0.1) # low probability as we don't want to lose features early on
        # 32 input image channel (conv2), 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(32, 64, 3) # goes through pooling, so outputs (64, 26, 26)
        self.drop_c3 = nn.Dropout(p=0.2)
        # 64 input image channel (conv3), 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(64, 128, 3) # goes through pooling, so outputs (128, 12, 12)
        self.drop_c4 = nn.Dropout(p=0.3)
        # 128 input image channel (conv2), 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv5 = nn.Conv2d(128, 256, 3) # goes through pooling, so outputs (256, 5, 5)
        self.drop_c5 = nn.Dropout(p=0.4)
        
        # All the dense layers
        # 512 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(256*5*5, 1000)
        self.drop_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.drop_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1000, 136)  # total number of keypoints: 68*2
        
        
        # Pooling is 2x2 across all feature levels
        self.pool2d = nn.MaxPool2d(2, 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # 5 conv + pooling layers with dropout probability increasing from 0.1 to 0.5
        x = self.pool2d(F.elu(self.conv1(x)))
        x = self.drop_c1(x)
        x = self.pool2d(F.elu(self.conv2(x)))
        x = self.drop_c2(x)
        x = self.pool2d(F.elu(self.conv3(x)))
        x = self.drop_c3(x)
        x = self.pool2d(F.elu(self.conv4(x)))
        x = self.drop_c4(x)
        x = self.pool2d(F.elu(self.conv5(x)))
        x = self.drop_c5(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # 3 linear layer with dropout in between
        x = F.elu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.elu(self.fc2(x))
        x = self.drop_fc2(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x