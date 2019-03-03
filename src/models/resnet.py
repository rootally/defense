"""Code based on official pytorch implementation of resnet models with 
nn.Conv2d replaced with custom made convolution named Conv2d_drop that performs 
targeted weight dropout before validation """

import torch
import torch.nn as nn
import torch.nn.functional as F
from targeted_dropout import targeted_weight_droput
from torch._six import container_abcs
from itertools import repeat
import math 

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
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, drop_rate, targ_perc, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_drop(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, drop_rate = drop_rate, targ_perc = targ_perc)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_drop(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, drop_rate = drop_rate, targ_perc = targ_perc)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_drop(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, drop_rate = drop_rate, targ_perc = targ_perc),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes,  drop_rate, targ_perc, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_drop(in_planes, planes, kernel_size=1, bias=False,drop_rate = drop_rate, targ_perc = targ_perc)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_drop(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, drop_rate = drop_rate , targ_perc = targ_perc)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_drop(planes, self.expansion*planes, kernel_size=1, bias=False, drop_rate = drop_rate, targ_perc = targ_perc)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_drop(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, drop_rate = drop_rate, targ_perc = targ_perc),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, drop_rate = 1.0, targ_perc= 0.01):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, drop_rate = drop_rate, targ_perc = targ_perc)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, drop_rate = drop_rate, targ_perc = targ_perc)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop_rate = drop_rate, targ_perc = targ_perc)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop_rate = drop_rate, targ_perc = targ_perc)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, drop_rate, targ_perc ):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate , targ_perc, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
##      self.weight = targeted dropout(pass self.weight) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    net.eval()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

