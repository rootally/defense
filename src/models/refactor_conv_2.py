#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:58:51 2019

@author: anisha
"""

"""
Custom made convolution - Conv2d_drop that performs targeted weight dropout before validation.
The weights of each convolution are passed to the function targeted_weight_dropout with user 
defined parameters drop_rate and targ_perc. Targeted weight dropout is performed only if the model is 
not undergoing training. 
"""

import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from models.targeted_dropout import targeted_weight_droput
from torch._six import container_abcs
from itertools import repeat
    

    
def Conv2d_drop(in_channels, out_channels, kernel_size, drop_rate, targ_perc, stride=1,
             padding=0, dilation=1, groups=1, bias=True, flag = 1):
    x = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    if(flag == 1):
        x.weight = torch.nn.Parameter(targeted_weight_droput(x.weight, drop_rate, targ_perc, False))
    return x