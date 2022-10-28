import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        self.input_channels = input_channels,
        self.output_channels = output_channels

        self.conv_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
            in_channels=self.input_channels[0],
            out_channels=self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1))
        )
        
        self.average_pooling = nn.AvgPool1d(2,padding=0, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.average_pooling(x)

        return x
    
    
class SpeakerEncoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_spk, frame_num):
        super(SpeakerEncoder, self).__init__()
        self.dim_spk = dim_spk
        self.frame_num = frame_num
        
        self.conv_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels=80,out_channels=32,kernel_size=3,stride=1,padding=1)),
        )
            
        downsamples = []
        for i in range(5):
            ds_layer = nn.Sequential(
                DownBlock(
                    input_channels=32*(2**i),
                    output_channels=512 if i== 4 else 32*(2**(i+1)),
                )
            )
            downsamples.append(ds_layer)
        self.downsamples = nn.ModuleList(downsamples)

        self.average_pooling_last = nn.AvgPool1d(int(self.frame_num/32),padding=0, stride=2) # ここの設計どうする？

        self.conv_mean = nn.utils.weight_norm(nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_spk,
            kernel_size=1,
            stride=1,
            padding=0
            ))
        self.conv_var = nn.utils.weight_norm(nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_spk,
            kernel_size=1,
            stride=1,
            padding=0
            ))

    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        for downsample in self.downsamples:
            x = F.leaky_relu(downsample(x), negative_slope=0.2)

        x=self.average_pooling_last(x)

        mu = F.leaky_relu(self.conv_mean(x),negative_slope=0.2)
        sigma =  F.leaky_relu(self.conv_var(x), negative_slope=0.2)
        eps = torch.randn_like(sigma)
        return eps * sigma + mu