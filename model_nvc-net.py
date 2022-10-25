import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from blocks import GatedResidualBlock, ResidualBlock


class ContentEncoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_con):
        super(ContentEncoder, self).__init__()
        self.dim_con = dim_con
        
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=7,stride=1,padding=3),
            nn.InstanceNorm1d(num_features=32, affine=True),
        )
        
        res_downsample = []
        for i in range(4):
            ds_layer = nn.Sequential(
                ResidualBlock(
                    in_channels=int(32*(2**i)),
                    out_channels=int(32*(2**i)),
                ),
                nn.Conv1d(
                    in_channels=int(32*(2**i)),
                    out_channels=int(32*(2**i)*2),
                    kernel_size=4 if i in [0,1] else 16,
                    stride=2 if i in [0,1] else 8,
                    padding=1 if i in [0,1] else 4,
                    ),
                nn.InstanceNorm1d(num_features=32*(2**i)*2, affine=True)
            )
            res_downsample.append(ds_layer)
        self.res_downsample = nn.ModuleList(res_downsample)

        self.conv_2 = nn.Sequential(
            nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_con,
            kernel_size=7,
            stride=1,
            padding=3
            ),
            nn.InstanceNorm1d(num_features=4, affine=True),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(
            in_channels=self.dim_con,
            out_channels=self.dim_con,
            kernel_size=7,
            stride=1,
            padding=3
            ),
            nn.InstanceNorm1d(num_features=4, affine=True),
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
    
    
class SpeakerEncoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_spk, frame_num):
        super(SpeakerEncoder, self).__init__()
        self.dim_spk = dim_spk
        self.frame_num = frame_num
        
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=80,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm1d(num_features=32, affine=True),
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

        self.conv_mean = nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_spk,
            kernel_size=1,
            stride=1,
            padding=0
            )
        self.conv_var = nn.Conv1d(
            in_channels=512,
            out_channels=self.dim_spk,
            kernel_size=1,
            stride=1,
            padding=0
            )

    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        for downsample in self.downsamples:
            x = F.leaky_relu(downsample(x), negative_slope=0.2)

        x=self.average_pooling_last(x)

        mu = F.leaky_relu(self.conv_mean(x),negative_slope=0.2)
        sigma =  F.leaky_relu(self.conv_var(x), negative_slope=0.2)

        return mu, sigma
    

class Generator(nn.Module):
    """Encoder module:
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=512,kernel_size=7,stride=1,padding=3), # dconをどこかで定義。
            nn.InstanceNorm1d(num_features=512, affine=True),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512,out_channels=512,kernel_size=7,stride=1,padding=3), # dconをどこかで定義。
            nn.InstanceNorm1d(num_features=512, affine=True),
        )
            
        upsamples = []
        for i in range(4):
            up_layer = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=int(512/(2**i)),
                    out_channels=int(512/(2**(i+1))),
                    kernel_size=16 if i in [0,1] else 4,
                    stride=8 if i in [0,1] else 2,
                    padding=4 if i in [0,1] else 1,
                ),
                nn.InstanceNorm1d(num_features=int(512/(2**(i+1))), affine=True),
                ResidualBlock(
                    in_channels=int(512/(2**(i+1))),
                    out_channels=int(512/(2**(i+1))),
                )
            )
            upsamples.append(up_layer)
        self.upsamples = nn.ModuleList(upsamples)

        self.conv_3 = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            )

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        for upsample in self.upsamples:
            x = F.gelu(upsample(x))

        x = self.conv_3(x)
        x = torch.tanh(x)

        return x

