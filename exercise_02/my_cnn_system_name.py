# -*- coding: utf-8 -*-
import torch.nn as nn
from torch import Tensor
#############################################1 
#Create a class, extending the Module class of PyTorch
class MyCNNSystem(nn.Module):
    
    def __init__(self, kernel_size_1=5, kernel_size_2=5, channel_1=20, channel_2=20, dropout=0.2):   
        super().__init__()
#2. Make the init method to set up two layers of CNNs, each layer followed by a rectified linear
#unit, a batch normalization process, and a max pooling process. You should have the parameters
#for each layer to be defined using input argument to the init method. As an example you
#can see the provided code at Section 2.
#############################################2    
#Add a linear layer, that will act as your classifier, using output features (i.e. out features)
# equal to 1. 
        self.conv1   = nn.Conv2d(1, channel_1, kernel_size=kernel_size_1)
        self.relu1   = nn.ReLU()
        self.batch1  = nn.BatchNorm2d(100)
        self.maxp1   = nn.MaxPool2d(4)
        
        self.conv2   = nn.Conv2d(channel_1,channel_2, kernel_size=kernel_size_2)
        self.relu2   = nn.ReLU()
        self.batch2  = nn.BatchNorm2d(100)
        self.maxp2   = nn.MaxPool2d(4)
       
#############################################3
# Add a linear layer, that will act as your classifier, using output features (i.e. out features)
# equal to 1.
        self.linear  = nn.Linear(20, 1)    
                                     #using output features (i.e. out features) equal to 1.
                                     
#############################################4
# Add, where appropriate, dropout with a probability defined at the init method.
        self.dropout = nn.Dropout(dropout)   
        
#############################################5
# Implement the forward method, which will have as an input a PyTorch tensor, process it according to your DNN, and return the output of your linear layer.
        
    def forward(self, x) -> Tensor:

        h = self.conv1(x)
        h = self.relu1(x)
        h = self.batch1(x)
        h = self.maxp1(x)
        
        h = self.conv2(x)
        h = self.relu2(x)
        h = self.batch2(x)
        h = self.maxp2(x)
              
        
        y_hat = self.linear(h)     
        # Return the output
        return y_hat

DNN = MyCNNSystem(kernel_size_1=2, kernel_size_2=2, channel_1=20, channel_2=20, dropout=0.2)
print(DNN)
