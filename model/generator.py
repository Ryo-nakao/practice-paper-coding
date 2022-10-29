import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from res_stack import ResStack


class Generator(nn.Module):
    """Generator module:
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels=4,out_channels=512,kernel_size=7,stride=1,padding=3)), # dconをどこかで定義。
        )

        self.conv_2 = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels=512,out_channels=512,kernel_size=7,stride=1,padding=3)), 
        )
            
        upsamples = []
        emb_1_1=[]
        res_stacks = []
        for i in range(4):
            up_layer = nn.Sequential(
                nn.utils.weight_norm(nn.ConvTranspose1d(
                    in_channels=int(512/(2**i)),
                    out_channels=int(512/(2**(i+1))),
                    kernel_size=16 if i in [0,1] else 4,
                    stride=8 if i in [0,1] else 2,
                    padding=4 if i in [0,1] else 1,
                )),
                nn.LeakyReLU(0.2),
            )
            
            conv_1_1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=128,
            out_channels=int(512/(2**(i+1))),
            kernel_size=1,
            ))
            
            residual_stack = ResStack(
                    int(512/(2**(i+1))),
                )
            
            upsamples.append(up_layer)
            emb_1_1.append(conv_1_1)
            res_stacks.append(residual_stack)
        self.upsamples = nn.ModuleList(upsamples)
        self.emb_1_1 = nn.ModuleList(emb_1_1)
        self.res_stacks = nn.ModuleList(res_stacks)

        self.conv_3 = nn.utils.weight_norm(nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            ))

    def forward(self, x, emb_org):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)

        for upsample, emb11, res_stack in zip(self.upsamples, self.emb_1_1, self.res_stacks):
            x = upsample(x)
            emb = F.relu(emb11(emb_org))
            
            x = F.relu(res_stack(x+emb))

        x = self.conv_3(x)
        x = torch.tanh(x)

        return x