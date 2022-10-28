import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from res_stack import ResStack


class ContentEncoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_con):
        super(ContentEncoder, self).__init__()
        self.dim_con = dim_con
        
        self.conv_1 = nn.Sequential(
             nn.utils.weight_norm(nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,stride=1,padding=3)),
        )
        
        res_downsample = []
        for i in range(4):
            ds_layer = nn.Sequential(
                ResStack(
                    32*(2**i),
                ),
                nn.utils.weight_norm(nn.Conv1d(
                    in_channels=int(32*(2**i)),
                    out_channels=int(32*(2**i)*2),
                    kernel_size=4 if i in [0,1] else 16,
                    stride=2 if i in [0,1] else 8,
                    padding=1 if i in [0,1] else 4,
                    )),
            )
            res_downsample.append(ds_layer)
        self.res_downsample = nn.ModuleList(res_downsample)

        self.conv_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_con,
            kernel_size=7,
            stride=1,
            padding=3
            )),
        )

        self.conv_3 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
            in_channels=self.dim_con,
            out_channels=self.dim_con,
            kernel_size=7,
            stride=1,
            padding=3
            )),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)

        for resdown in self.res_downsample:
            x = F.relu(resdown(x))

        x = self.conv_2(x)
        x = F.gelu(x)
        x = self.conv_3(x)

        return x