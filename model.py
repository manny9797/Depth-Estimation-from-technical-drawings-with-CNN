import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Squeeze-and-Excitation (SE) Block for Channel Attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# EESP Module for Multi-Scale Feature Fusion
class EESP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(EESP, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Atrous Spatial Pyramid Pooling (ASPP) Module for Global Context
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        dilation_rates = [1, 6, 12, 18]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for d in dilation_rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        global_feature = self.global_avg_pool(x)
        global_feature = F.interpolate(global_feature, size=size, mode='bilinear', align_corners=False)
        res.append(global_feature)
        x = torch.cat(res, dim=1)
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        return x

# Enhanced Pixel Shuffle Decoder (PSD) Module with integrated attention (SE)
class EnhancedPSDModule(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(EnhancedPSDModule, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # PixelShuffle reduces channels by (upscale_factor^2)
        self.conv = nn.Conv2d(in_channels // (upscale_factor ** 2), out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)  # Integrate channel attention

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(self.bn(x))
        x = self.se(x)
        return x

# Combined Depth Estimation Network
class DepthEstimationNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationNet, self).__init__()

        # Backbone encoder (ResNet-50) extraction
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B, 64, H/2, W/2)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)           # (B, 256, H/4, W/4)
        self.layer2 = resnet.layer2                                         # (B, 512, H/8, W/8)
        self.layer3 = resnet.layer3                                         # (B, 1024, H/16, W/16)
        self.layer4 = resnet.layer4                                         # (B, 2048, H/32, W/32)

        # ASPP module applied after the deepest encoder layer for enhanced global context
        self.aspp = ASPP(2048, 512)

        # EESP skip connections for multi-scale feature refinement
        self.esp1 = EESP(256, 256)
        self.esp2 = EESP(512, 512)
        self.esp3 = EESP(1024, 1024)

        # Decoder using Enhanced PSD Modules
        self.psd1 = EnhancedPSDModule(512, 1024, upscale_factor=2)  # Input from ASPP -> (B, 1024, H/16, W/16)
        self.psd2 = EnhancedPSDModule(1024, 512, upscale_factor=2)   # (B, 512, H/8, W/8)
        self.psd3 = EnhancedPSDModule(512, 256, upscale_factor=2)    # (B, 256, H/4, W/4)
        self.psd4 = EnhancedPSDModule(256, 128, upscale_factor=2)    # (B, 128, H/2, W/2)

        # Residual connections to further refine decoded features
        self.residual1 = nn.Conv2d(1024, 512, kernel_size=5, padding=2)
        self.residual2 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.residual3 = nn.Conv2d(256, 128, kernel_size=5, padding=2)

        # Final convolution to produce the depth prediction
        self.conv_final = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder forward pass
        x0 = self.layer0(x)  # (B, 64, H/2, W/2)
        x1 = self.layer1(x0)  # (B, 256, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 1024, H/16, W/16)
        x4 = self.layer4(x3)  # (B, 2048, H/32, W/32)

        # Enhance global context with ASPP
        aspp_out = self.aspp(x4)  # (B, 512, H/32, W/32)

        # Decoder: progressively upscale and fuse features with skip connections
        # PSD1: Upscale to H/16
        d1 = self.psd1(aspp_out)  # (B, 1024, H/16, W/16)
        d1 = d1 + F.interpolate(self.esp3(x3), size=d1.shape[2:], mode='bilinear', align_corners=False)

        # PSD2: Upscale to H/8
        d2 = self.psd2(d1)  # (B, 512, H/8, W/8)
        d2 = d2 + F.interpolate(self.esp2(x2), size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + F.interpolate(self.residual1(d1), size=d2.shape[2:], mode='bilinear', align_corners=False)

        # PSD3: Upscale to H/4
        d3 = self.psd3(d2)  # (B, 256, H/4, W/4)
        d3 = d3 + F.interpolate(self.esp1(x1), size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + F.interpolate(self.residual2(d2), size=d3.shape[2:], mode='bilinear', align_corners=False)

        # PSD4: Upscale to H/2
        d4 = self.psd4(d3)  # (B, 128, H/2, W/2)
        d4 = d4 + F.interpolate(self.residual3(d3), size=d4.shape[2:], mode='bilinear', align_corners=False)

        # Final depth map convolution and rescale to input size
        depth = self.conv_final(d4)
        depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear', align_corners=False)
        return depth

