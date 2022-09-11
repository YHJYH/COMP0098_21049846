# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:59:57 2022

@author: Yuheng

input format for nn.Conv2d: nSamples x nChannels x Height x Width
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# shallow fully-connected network
class LeNet_300_100_raw(nn.Module):
    # source: https://github.com/TimDettmers/sparse_learning/blob/e19ff16c475462d50ad247ec2927113617662a4e/sparselearning/models.py#L201
    
    def __init__(self, num_classes, input_height, input_width, input_channel):
        
        super(LeNet_300_100_raw, self).__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        
        self.fc1 = nn.Linear(input_width * input_height * input_channel, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        
        x = x.view(-1, self.input_width * self.input_height * self.input_channel)
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        x4 = F.relu(x3)
        x5 = self.fc3(x4)
        x6 = F.log_softmax(x5, dim=1)
        
        """
        x = x.view(-1, self.input_width * self.input_height * self.input_channel)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        x4 = F.log_softmax(x3, dim=1)
        """
        feature_map = [x1, x3, x5]
        return (feature_map, x6)
        
   
# deep convolutional network
class LeNet_5_raw(nn.Module):
    # source: https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
    
    def __init__(self, num_classes, input_height, input_width, input_channel):
        
        super(LeNet_5_raw, self).__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        
        self.conv1 = nn.Conv2d(input_channel, 6, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding='valid')
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding='valid')
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        x1 = self.conv1(x)
       
        
        x2 = torch.tanh(x1)
        x3 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        x4 = torch.sigmoid(x3)
        
        x5 = self.conv2(x4)
        
        
        x6 = torch.tanh(x5)
        x7 = F.avg_pool2d(x6, kernel_size=2, stride=2)
        x8 = torch.sigmoid(x7)
        
        x9 = self.conv3(x8)
        
        x10 = torch.tanh(x9)
        x11 = torch.reshape(torch.flatten(x10), (-1, 120))
        
        x12 = self.fc1(x11)
        x13 = torch.tanh(x12)
        
        x14 = self.fc2(x13)
        
        
        x15 = F.log_softmax(x14, dim=1)
        """
        x1 = torch.tanh(self.conv1(x))
        x2 = torch.sigmoid(F.avg_pool2d(x1, kernel_size=2, stride=2))
        x3 = torch.tanh(self.conv2(x2))
        x4 = torch.sigmoid(F.avg_pool2d(x3, kernel_size=2, stride=2))
        x5 = torch.tanh(self.conv3(x4))
        x6 = torch.reshape(torch.flatten(x5), (-1, 120))
        x7 = torch.tanh(self.fc1(x6))
        x8 = F.log_softmax(self.fc2(x7), dim=1)
        """
        feature_map = [x1, x5, x9, x12, x14]
        
        
        return (feature_map, x15)
    
        
# deep convolutional network
class LeNet_5_N(nn.Module):
    # source: https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
    
    def __init__(self, num_classes, input_height, input_width, input_channel):
        
        super(LeNet_5_N, self).__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        
        self.conv1 = nn.Conv2d(input_channel, 6, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding='valid')
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding='valid')
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        
        x3 = F.avg_pool2d(x1, kernel_size=2, stride=2)
       
        '''
        x2 = torch.tanh(x1)
        x3 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        x4 = torch.sigmoid(x3)
        
        x5 = self.conv2(x4)
        '''
        x5 = self.conv2(x3)
        x7 = F.avg_pool2d(x5, kernel_size=2, stride=2)
        '''
        x6 = torch.tanh(x5)
        x7 = F.avg_pool2d(x6, kernel_size=2, stride=2)
        x8 = torch.sigmoid(x7)
        
        x9 = self.conv3(x8)
        '''
        x9 = self.conv3(x7)
        '''
        x10 = torch.tanh(x9)
        x11 = torch.reshape(torch.flatten(x10), (-1, 120))
        x12 = self.fc1(x11)
        x13 = torch.tanh(x12)
        
        x14 = self.fc2(x13)
        '''
        #print(x9.shape)
        x11 = x9.view(x9.size(0),-1)
        #x11 = torch.reshape(torch.flatten(x9), (-1, 120))
        x12 = self.fc1(x11)
        x14 = self.fc2(x12)
        
        x15 = F.log_softmax(x14, dim=1)
        """
        x1 = torch.tanh(self.conv1(x))
        x2 = torch.sigmoid(F.avg_pool2d(x1, kernel_size=2, stride=2))
        x3 = torch.tanh(self.conv2(x2))
        x4 = torch.sigmoid(F.avg_pool2d(x3, kernel_size=2, stride=2))
        x5 = torch.tanh(self.conv3(x4))
        x6 = torch.reshape(torch.flatten(x5), (-1, 120))
        x7 = torch.tanh(self.fc1(x6))
        x8 = F.log_softmax(self.fc2(x7), dim=1)
        """
        #feature_map = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]
        feature_map = [x1, x3, x5, x7, x9, x12, x14, x15]
        
        return (feature_map, x15)        
        
        
class VGG16(nn.Module):
    # source: https://iq.opengenus.org/vgg16/ and https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16%20is%20a%20convolution%20neural,vision%20model%20architecture%20till%20date.
    
    def __init__(self, num_classes, input_height, input_width, input_channel):
        
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.input_height = input_height 
        self.input_width = input_width 
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.fc1 = nn.Linear(512, 4096, bias=False)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.fc3 = nn.Linear(4096, 10, bias=False)
        
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        
        x2 = F.relu(x1)
        x3 = self.conv2(x2)
        
        x4 = F.relu(x3)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x6 = self.conv3(x5)
        
        x7 = F.relu(x6)
        x8 = self.conv4(x7)
        
        x9 = F.relu(x8)
        x10 = F.max_pool2d(x9, kernel_size=2, stride=2)
        x11 = self.conv5(x10)
        
        x12 = F.relu(x11)
        x13 = self.conv6(x12)
        
        x14 = F.relu(x13)
        x15 = self.conv7(x14)
        
        x16 = F.relu(x15)
        x17 = F.max_pool2d(x16, kernel_size=2, stride=2)
        x18 = self.conv8(x17)
        
        x19 = F.relu(x18)
        x20 = self.conv9(x19)
        
        x21 = F.relu(x20)
        x22 = self.conv10(x21)
        
        x23 = F.relu(x22)
        x24 = F.max_pool2d(x23, kernel_size=2, stride=2)
        x25 = self.conv11(x24)
        
        x26 = F.relu(x25)
        x27 = self.conv12(x26)
        
        x28 = F.relu(x27)
        x29 = self.conv13(x28)
        
        x30 = F.relu(x29)
        x31 = F.max_pool2d(x30, kernel_size=2, stride=2)
        x32 = torch.reshape(torch.flatten(x31), (-1, 512))
        x33 = self.fc1(x32)
        
        x34 = F.relu(x33)
        x35 = self.fc2(x34)
        
        x36 = F.relu(x35)
        x37 = self.fc3(x36)
        
        x38 = F.log_softmax(x37, dim=1)
        
        feature_map = [x1, x3, x6, x8, x11, x13, x15, x18, x20, x22, x25, x27, x29, x33, x35, x37]
        
        return (feature_map, x38)
    # total params: 33638218
        
    
class VGG16_N(nn.Module):
    # source: https://iq.opengenus.org/vgg16/ and https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c#:~:text=VGG16%20is%20a%20convolution%20neural,vision%20model%20architecture%20till%20date.
    
    def __init__(self, num_classes, input_height, input_width, input_channel):
        
        super(VGG16_N, self).__init__()
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.input_height = input_height 
        self.input_width = input_width 
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.fc1 = nn.Linear(512, 4096, bias=False)
        #self.fc1 = nn.Linear(512*2*2, 4096, bias=False)
        self.fc2 = nn.Linear(4096, 4096, bias=False)
        self.fc3 = nn.Linear(4096, 10, bias=False)
        
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        
        #x2 = F.relu(x1)
        #x3 = self.conv2(x2)
        
        x4 = F.relu(x1)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x6 = self.conv3(x5)
        
        #x7 = F.relu(x6)
        #x8 = self.conv4(x7)
        
        x9 = F.relu(x6)
        x10 = F.max_pool2d(x9, kernel_size=2, stride=2)
        x11 = self.conv5(x10)
        
        #x12 = F.relu(x11)
        #x13 = self.conv6(x12)
        
        #x14 = F.relu(x13)
        #x15 = self.conv7(x14)
        
        x16 = F.relu(x11)
        x17 = F.max_pool2d(x16, kernel_size=2, stride=2)
        x18 = self.conv8(x17)
        
        #x19 = F.relu(x18)
        #x20 = self.conv9(x19)
        
        #x21 = F.relu(x18)
        #x22 = self.conv10(x21)
        
        x23 = F.relu(x18)
        x24 = F.max_pool2d(x23, kernel_size=2, stride=2)
        x25 = self.conv11(x24)
        
        x26 = F.relu(x25)
        x27 = self.conv12(x26)
        
        #x28 = F.relu(x27)
        #x29 = self.conv13(x28)
        
        x30 = F.relu(x27)
        x31 = F.max_pool2d(x30, kernel_size=2, stride=2)
        x32 = torch.reshape(torch.flatten(x31), (-1, 512))
        x33 = self.fc1(x32)
        
        x34 = F.relu(x33)
        x35 = self.fc2(x34)
        
        #x36 = F.relu(x35)
        #x37 = self.fc3(x36)
        
        x38 = F.log_softmax(x35, dim=1)
        
        feature_map = [x1, x6, x11, x18, x25, x27, x33, x35]
        
        return (feature_map, x38)
