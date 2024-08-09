import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)

        return out * gate.sigmoid()


class conform_conv(nn.Module):
    def __init__(self, channels: int,
                 kernel_size: int = 31,

                 DropoutL=0.1,

                 bias: bool = True):
        super().__init__()
        self.act2 = nn.SiLU()
        self.act1 = GLU(1)

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0:
        #    it's a causal convolution, the input will be padded with
        #    `self.lorder` frames on the left in forward (causal conv impl).
        # else: it's a symmetrical convolution

        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(channels, channels,  kernel_size,
                                        stride=1,
                                        padding=padding,
                                        groups=channels,
                                        bias=bias)


        self.norm = nn.BatchNorm1d(channels)


        self.pointwise_conv2 = nn.Conv1d(channels,
                                         channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=bias)
        self.drop=nn.Dropout(DropoutL) if DropoutL>0. else nn.Identity()
    def forward(self,x):
        x=x.transpose(1,2)
        x=self.act1(self.pointwise_conv1(x))
        x=self.depthwise_conv (x)
        x=self.norm(x)
        x=self.act2(x)
        x=self.pointwise_conv2(x)
        return self.drop(x).transpose(1,2)

