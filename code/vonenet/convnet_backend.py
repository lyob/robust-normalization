import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvNetBackEnd(nn.Module):
    def __init__(self, in_channels, norm_method=None, norm_position='both'):
        super(ConvNetBackEnd, self).__init__()

        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=500, out_features=10)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(20)
        # self.ln1 = nn.LayerNorm([in_channels, 14, 14])
        self.ln2 = nn.LayerNorm([20, 10, 10])
        # self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(20, affine=True)
        # self.gn1 = nn.GroupNorm(4, in_channels)
        self.gn2 = nn.GroupNorm(4, 20)

        self.lrn = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_channel = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_spatial = LRN(spatial_size=3, across_channel_spatial=False)
        self.lrn_both = LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)

        # self.norm_dict1 = {'nn': nn.Identity(),'bn': self.bn1, 'ln': self.ln1, 
        #                    'in': self.in1, 'gn': self.gn1, 'lrns': self.lrn_spatial,
        #                    'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict2 = {'nn': nn.Identity(), 'bn': self.bn2, 'ln': self.ln2,
                           'in': self.in2, 'gn': self.gn2, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_method = norm_method
        self.norm_position = norm_position


    def forward(self, x):
            # remove first layer
            # x = self.conv_1(x)
            # x = self.norm_dict1[self.norm_method](x)
            # x = self.relu(x)

            # same back-end, but remove norm layer
            x = self.conv_2(x)
            if (self.norm_position=='2' or self.norm_position=='both'):
                x = self.norm_dict2[self.norm_method](x)
            else:
                x = self.norm_dict2['nn'](x)
            x = self.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = torch.flatten(x, 1)
            
            # 3rd layer
            x = self.fc_1(x)
            return x