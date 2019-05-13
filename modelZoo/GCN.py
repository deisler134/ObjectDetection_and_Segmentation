'''
Created on May 7, 2019

    create Global Convolutional Network module with boundary refine module

@author: deisler
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math

class GCN(nn.Module):
    def __init__(self,inplanes,planes,k=7): #planes=21 for classes in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(k,1), padding =((k-1)/2,0))
        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1,k), padding =(0,(k-1)/2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1,k), padding =((k-1)/2,0))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(k,1), padding =(0,(k-1)/2))
        
    def forward(self, x):
        out_l = self.conv_l1(x)
        out_l = self.conv_l2(out_l)
        
        out_r = self.conv_r1(x)
        out_r = self.conv_r2(out_r)
        
        out = out_l + out_r
        
        return out
    
class BR(nn.Module):
    def __init__(self, planes): #planes = 21 is equal to inplanes for classes in paper
        super(BR, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes,planes, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(planes,planes, kernel_size=3,padding=1)
    
    def forward(self,x):
        identity = x
#         x_res = self.bn(x)
#         x_res = self.relu(x_res)
        out = self.conv1(x)
#         x_res = self.bn(x_res)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = identity + out
        return out

    
    
    