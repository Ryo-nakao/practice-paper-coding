import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f =  nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias))
        self.conv_g =  nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias))
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return torch.mul(self.tanh(self.conv_f(x)), self.sig(self.conv_g(x)))
    
    
class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                GatedConv1d(channel, channel, kernel_size=3, padding=0, dilation=3**i),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(4)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(4)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)