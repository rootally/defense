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
from targeted_dropout import targeted_weight_droput
from torch._six import container_abcs
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class _ConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
    
class Conv2d_drop(_ConvNd):
    
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate, targ_perc, stride=1,
                 padding=0, dilation=1, groups=1, bias=True ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.drop_rate = drop_rate
        self.targ_perc = targ_perc
        super(Conv2d_drop, self).__init__( in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        if(self.training):
            return F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        else:
#            print("Before", self.weight.size())
            self.weight = torch.nn.Parameter(targeted_weight_droput(self.weight, self.drop_rate, self.targ_perc, False))
#            print("After", self.weight.size())
            return F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)