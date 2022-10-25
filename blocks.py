import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F


class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return torch.mul(self.tanh(self.conv_f(x)), self.sig(self.conv_g(x)))


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, 
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        # self.output_width = output_width
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)

        return residual
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels,
        self.out_channels = out_channels

        convolutions = []
        dilations = [1,3,9,27] 
        for dilation in dilations:
            conv_layer = nn.Sequential(
                GatedResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size=3, 
                        stride=1,
                        padding=dilation,
                        dilation=dilation,
                        ),
                nn.InstanceNorm1d(num_features=out_channels, affine=True)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        return x
    

class DownBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownBlock, self).__init__()
        self.input_channels = input_channels,
        self.output_channels = output_channels

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
            in_channels=self.input_channels[0],
            out_channels=self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.InstanceNorm1d(num_features=self.output_channels, affine=True),
        )

        self.average_pooling = nn.AvgPool1d(2,padding=0, stride=2)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.average_pooling(x)

        return x