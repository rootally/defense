"""Code based on official pytorch implementation of resnet models with 
nn.Conv2d replaced with custom made convolution named Conv2d_drop that performs 
targeted weight dropout before validation """

import torch
import torch.nn as nn
import torch.nn.functional as F
from refactor_conv import Conv2d_drop 

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

