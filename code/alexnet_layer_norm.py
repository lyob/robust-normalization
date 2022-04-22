'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from robustness.tools import custom_modules1

class LRN(nn.Module):
    def __init__(self, channel_size=1, spatial_size=1, alpha=1.0, beta=0.75, across_channel_spatial=True):
        super(LRN, self).__init__()
        self.across_channel_spatial = across_channel_spatial
        self.spatial_pad = int((spatial_size-1.0)/2)
        self.channel_pad = int((channel_size-1.0)/2)
        if self.across_channel_spatial:
            # AvgPool3d needs to have input shape (N, C, D, H, W)
            self.average=nn.AvgPool3d(kernel_size=(channel_size, spatial_size, spatial_size),
                    stride=1,
                    padding=(self.channel_pad, self.spatial_pad, self.spatial_pad))
        else: #if not, then only do LocalResponseNorm across spatial
            self.average=nn.AvgPool2d(kernel_size=spatial_size,
                    stride=1,
                    padding=self.spatial_pad)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.across_channel_spatial:
            div = x.pow(2).unsqueeze(1) #squeeze to fit the input shape with AvgPool3d
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(
        self, 
        normalize: str, 
        num_classes: int = 10, 
        ) -> None:
        super().__init__()

        self.num_channels = [64, 192, 384, 256, 256]

        self.bn1 = nn.BatchNorm2d(self.num_channels[0])
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])
        self.bn5 = nn.BatchNorm2d(self.num_channels[4])

        self.ln1 = nn.LayerNorm([self.num_channels[0], 7, 7])
        self.ln2 = nn.LayerNorm([self.num_channels[1], 10, 10])
        self.ln3 = nn.LayerNorm([self.num_channels[2], 10, 10])
        self.ln4 = nn.LayerNorm([self.num_channels[3], 10, 10])
        self.ln5 = nn.LayerNorm([self.num_channels[4], 10, 10])
        
        self.in1 = nn.InstanceNorm2d(self.num_channels[0], affine=True)
        self.in2 = nn.InstanceNorm2d(self.num_channels[1], affine=True)
        self.in3 = nn.InstanceNorm2d(self.num_channels[2], affine=True)
        self.in4 = nn.InstanceNorm2d(self.num_channels[3], affine=True)
        self.in5 = nn.InstanceNorm2d(self.num_channels[4], affine=True)
        
        self.gn1 = nn.GroupNorm(4, self.num_channels[0])
        self.gn2 = nn.GroupNorm(4, self.num_channels[1])
        self.gn3 = nn.GroupNorm(4, self.num_channels[2])
        self.gn4 = nn.GroupNorm(4, self.num_channels[3])
        self.gn5 = nn.GroupNorm(4, self.num_channels[4])

        self.lrn = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_channel = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_spatial = LRN(spatial_size=3, across_channel_spatial=False)
        self.lrn_both = LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)

        self.norm_dict1 = {'nn': nn.Identity(),'bn': self.bn1, 'ln': self.ln1, 
                           'in': self.in1, 'gn': self.gn1, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict2 = {'nn': nn.Identity(), 'bn': self.bn2, 'ln': self.ln2,
                           'in': self.in2, 'gn': self.gn2, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict3 = {'nn': nn.Identity(), 'bn': self.bn3, 'ln': self.ln3,
                           'in': self.in3, 'gn': self.gn3, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict4 = {'nn': nn.Identity(), 'bn': self.bn4, 'ln': self.ln4,
                           'in': self.in4, 'gn': self.gn4, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict5 = {'nn': nn.Identity(), 'bn': self.bn5, 'ln': self.ln5,
                           'in': self.in5, 'gn': self.gn5, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.normalize = normalize

        self.features = nn.Sequential(
            # first layer
            nn.Conv2d(3, self.num_channels[0], kernel_size=11, stride=4, padding=2),
            self.norm_dict1[self.normalize],
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # second layer
            nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=5, padding=2),
            self.norm_dict2[self.normalize],
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # third layer
            nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, padding=1),
            self.norm_dict3[self.normalize],
            nn.ReLU(inplace=False),

            # fourth layer
            nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=3, padding=1),
            self.norm_dict4[self.normalize],
            nn.ReLU(inplace=False),

            # fifth layer
            nn.Conv2d(self.num_channels[3], self.num_channels[4], kernel_size=3, padding=1),
            self.norm_dict5[self.normalize],
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels[4] * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x