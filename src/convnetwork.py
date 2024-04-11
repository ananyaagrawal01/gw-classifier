# Author: Ananya Agrawal
# Version: ver 1.0

# Importing necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

#Define The CNN Architecture
class ConvNet(nn.Module):
  # Model definition
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  #padding 0 , stride = 1
        h,w=findConv2dOutShape(21,201,self.conv1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        h,w=findConv2dOutShape(h,w,self.conv2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 47 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Relu activation
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 47 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc3(F.relu(self.fc2(x)))))
        x = self.sigmoid(self.fc4(x)).to(dtype=torch.float32)
        return x
    
def findConv2dOutShape(hin,win,conv,pool=2):
    # find how the input image dimensions change with every layer
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    print(kernel_size, stride, padding, dilation)

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool

    print(hout, wout, "\n")
    return int(hout),int(wout)    