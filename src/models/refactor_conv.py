"""
Custom made convolution - Conv2d_drop that performs targeted weight dropout before validation.
The weights of each convolution are passed to the function targeted_weight_dropout with user 
defined parameters drop_rate and targ_perc. Targeted weight dropout is performed only if the model is 
not undergoing training. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.targeted_dropout import targeted_weight_droput

class Conv2d_drop(nn._ConvNd):
    
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate, targ_perc, stride=1,
                 padding=0, dilation=1, groups=1, bias=True ):
        kernel_size = nn._pair(kernel_size)
        stride = nn._pair(stride)
        padding = nn._pair(padding)
        dilation = nn._pair(dilation)
        self.drop_rate = drop_rate
        self.targ_perc = targ_perc
        super(Conv2d_drop, self).__init__( in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, nn._pair(0), groups, bias)

    def forward(self, input):
        if(self.training):
            return F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        else:
#            print("Before", self.weight.size())
            self.weight = torch.nn.Parameter(targeted_weight_droput(self.weight, self.drop_rate, self.targ_perc, False))
#            print("After", self.weight.size())
            return F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)