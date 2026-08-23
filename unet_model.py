# unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return x

class SubdomainUNet(nn.Module):
    """
    U-Net mapping from (B, C_in, 8, 8) F_sub modes -> (B, C_out, 8, 8) U_sub modes.
    Uses 2 downsampling stages suited for 8x8 input resolutions.
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)           # 8x8
        self.enc2 = ConvBlock(base_ch, base_ch * 2)     # 4x4
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # 2x2
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2    = ConvBlock(base_ch * 4, base_ch * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1    = ConvBlock(base_ch * 2, base_ch)
        
        self.final_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # 8x8
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)  # 4x4
        
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)  # 2x2
        
        # Bottleneck (2x2)
        b = self.bottleneck(p2)
        
        # 4x4
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        # 8x8
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)