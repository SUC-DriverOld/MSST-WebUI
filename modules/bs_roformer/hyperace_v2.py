from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from modules.bs_roformer.hyperace import DSConv, DS_C3k2, HyperACE, Decoder

# core backbone of pcunwa/BS-Roformer-HyperACE v2

class Backbone(nn.Module):
    def __init__(self, in_channels=256, base_channels=64, base_depth=3):
        super().__init__()
        c = base_channels
        c2 = base_channels
        c3 = 256
        c4 = 384
        c5 = 512
        c6 = 768

        self.stem = DSConv(in_channels, c2, k=3, s=(2, 1), p=1)
        
        self.p2 = nn.Sequential(
            DSConv(c2, c3, k=3, s=(2, 1), p=1),
            DS_C3k2(c3, c3, n=base_depth)
        )
        
        self.p3 = nn.Sequential(
            DSConv(c3, c4, k=3, s=(2, 1), p=1),
            DS_C3k2(c4, c4, n=base_depth*2)
        )
        
        self.p4 = nn.Sequential(
            DSConv(c4, c5, k=3, s=2, p=1),
            DS_C3k2(c5, c5, n=base_depth*2)
        )
        
        self.p5 = nn.Sequential(
            DSConv(c5, c6, k=3, s=2, p=1),
            DS_C3k2(c6, c6, n=base_depth)
        )
        
        self.out_channels = [c3, c4, c5, c6]

    def forward(self, x):
        x = self.stem(x)
        x2 = self.p2(x)
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        x5 = self.p5(x4)
        return [x2, x3, x4, x5]

class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn=4):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                nn.InstanceNorm2d(in_c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                nn.InstanceNorm2d(c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Linear(f, f // bn, bias=False),
                nn.InstanceNorm2d(c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                nn.InstanceNorm2d(c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x

class FreqPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale, f):
        super().__init__()
        self.scale = scale
        self.conv = DSConv(in_channels, out_channels * scale)
        self.out_conv = TFC_TDF(out_channels, out_channels, 2, f)
        
    def forward(self, x):
        x = self.conv(x)
        B, C_r, H, W = x.shape
        out_c = C_r // self.scale
        
        x = x.view(B, out_c, self.scale, H, W)
        
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, out_c, H, W * self.scale)
        
        return self.out_conv(x)

class ProgressiveUpsampleHead(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025, in_bands=62):
        super().__init__()
        self.target_bins = target_bins
        
        c = in_channels
        
        self.block1 = FreqPixelShuffle(c, c//2, scale=2, f=in_bands*2)
        self.block2 = FreqPixelShuffle(c//2, c//4, scale=2, f=in_bands*4)
        self.block3 = FreqPixelShuffle(c//4, c//8, scale=2, f=in_bands*8)
        self.block4 = FreqPixelShuffle(c//8, c//16, scale=2, f=in_bands*16)
        
        self.final_conv = nn.Conv2d(c//16, out_channels, kernel_size=3, stride=1, padding='same', bias=False)

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        if x.shape[-1] != self.target_bins:
            x = F.interpolate(x, size=(x.shape[2], self.target_bins), mode='bilinear', align_corners=False)
            
        x = self.final_conv(x)
        return x

class SegmModelHyperACE2(nn.Module):
    def __init__(self, in_bands=62, in_dim=256, out_bins=1025, out_channels=4,
                 base_channels=64, base_depth=2, 
                 num_hyperedges=32, num_heads=8):
        super().__init__()
        
        self.backbone = Backbone(in_channels=in_dim, base_channels=base_channels, base_depth=base_depth)
        enc_channels = self.backbone.out_channels
        c2, c3, c4, c5 = enc_channels
        
        hyperace_in_channels = enc_channels
        hyperace_out_channels = c4
        self.hyperace = HyperACE(
            hyperace_in_channels, hyperace_out_channels, 
            num_hyperedges, num_heads, k=2, l=1
        )
        
        decoder_channels = [c2, c3, c4, c5]
        self.decoder = Decoder(
            enc_channels, hyperace_out_channels, decoder_channels
        )

        self.upsample_head = ProgressiveUpsampleHead(
            in_channels=decoder_channels[0], 
            out_channels=out_channels,
            target_bins=out_bins,
            in_bands=in_bands
        )

    def forward(self, x):
        H, W = x.shape[2:]
        
        enc_feats = self.backbone(x)
        
        h_ace_feats = self.hyperace(enc_feats)
        
        dec_feat = self.decoder(enc_feats, h_ace_feats)
        
        feat_time_restored = F.interpolate(dec_feat, size=(H, dec_feat.shape[-1]), mode='bilinear', align_corners=False)
        
        out = self.upsample_head(feat_time_restored)
        
        return out
