
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import gabor_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x

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

class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(
            self, sf, theta, sigx, sigy, phase,
            k_exc=7,
            simple_channels=16, complex_channels=16, ksize=7, stride=2, input_size=28,
            norm_method='nn',
            norm_position='both'
        ):

        super().__init__()

        self.in_channels = 1

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU()
        self.complex = Identity()
        self.gabors = Identity()
        self.output = Identity()
        
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(20)
        self.ln1 = nn.LayerNorm([self.out_channels, 14, 14])
        self.ln2 = nn.LayerNorm([20, 10, 10])
        self.in1 = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(20, affine=True)
        self.gn1 = nn.GroupNorm(4, self.out_channels)
        self.gn2 = nn.GroupNorm(4, 20)
        self.lrn_channel = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_spatial = LRN(spatial_size=3, across_channel_spatial=False)
        self.lrn_both = LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)

        self.norm_dict = {
            'nn': nn.Identity(),
            'bn': self.bn1, 
            'ln': self.ln1, 
            'in': self.in1, 
            'gn': self.gn1, 
            'lrns': self.lrn_spatial,
            'lrnc': self.lrn_channel, 
            'lrnb': self.lrn_both
        }

        self.norm_method = norm_method
        self.norm_position = norm_position

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        # s_q1 = self.simple_conv_q1(x)
        # c = self.complex(torch.sqrt(0.00001 + s_q0[:, self.simple_channels:, :, :] ** 2 + s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        # s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        # s = self.simple(s_q0)
        # return self.gabors(self.k_exc * torch.cat((s, c), 1))
        return self.gabors(self.k_exc * s_q0)

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)

        # Normalization layer
        if (self.norm_position=='1' or self.norm_position=='both'):
            x = self.norm_dict[self.norm_method](x)
        else: 
            x = self.norm_dict['nn'](x)

        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.simple(x)
        return x