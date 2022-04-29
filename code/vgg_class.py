'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LRN(nn.Module):
    def __init__(self, channel_size=1, spatial_size=1, alpha=1.0, beta=0.75, across_channel_spatial=True):
        super(LRN, self).__init__()
        self.across_channel_spatial = across_channel_spatial
        self.spatial_pad = int((spatial_size-1.0)/2)
        self.channel_pad = int((channel_size-1.0)/2)
        if self.across_channel_spatial:
            # AvgPool3d needs to have input shape (N, C, D, H, W)
            self.average=nn.AvgPool3d(
                kernel_size=(channel_size, spatial_size, spatial_size),
                stride=1,
                padding=(self.channel_pad, 1, 1),
                # padding=(self.channel_pad, self.spatial_pad, self.spatial_pad),
                ceil_mode=False
            )
        else: #if not, then only do LocalResponseNorm across spatial
            self.average=nn.AvgPool2d(
                kernel_size=spatial_size,
                stride=1,
                padding=self.spatial_pad
            )
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

class LRN_both(nn.Module):
    def __init__(self, channel_size=1, spatial_size=1, alpha=1.0, beta=0.75):
        super().__init__()
        # kernel sizes
        self.channel_size = channel_size
        self.spatial_size = spatial_size
        
        # padding sizes
        self.spatial_pad = int((spatial_size  -1)/2)
        self.channel_pad = int((channel_size - 1)/2)
        
        # hyperparameters
        self.alpha = alpha
        self.beta = beta

        # AvgPool3d needs to have input shape (N, C, D, H, W)
        self.spatial_avg = nn.AvgPool2d(
            kernel_size=self.spatial_size,
            stride=1,
            padding=self.spatial_pad
        )

    def channel_avg(self, input):
        sizes = input.size()
        div = input.unsqueeze(1)
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = F.pad(div, (0, 0, 0, 0, self.channel_size//2, (self.channel_size - 1)//2))
        div = F.avg_pool3d(div, (self.channel_size, 1, 1), stride=1).squeeze(1)
        div = div.view(sizes)
        return div

    def forward(self, x):
        # since their AvgPool3d is buggy, we can stack 2 AvgPool2ds instead
        div = x.pow(2)
        div = self.spatial_avg(div)
        div = self.channel_avg(div)
        div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


cfg = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}

class VGG(nn.Module):
    def __init__(self, vgg_name, norm_method='nn', num_classes=10):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name], norm_method)
        self.classifier = nn.Linear(512, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data,0)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        out = self.features(x)
        latent = out.view(out.size(0), -1)
        out = self.classifier(latent)
        if with_latent:
            return out, latent
        return out

    def set_layer_norm(self, num_channels, norm_method, conv_layer_idx):
        gn_param = 16 # 32  # I made this the same as the paper that used VGG16, but Hang used 16 for Resnet and 4 for LeNet
        ln_params = [32,16,8,8,4,4,2,2]

        norm_dict = {
            'nn': nn.Identity(),
            'bn': nn.BatchNorm2d(num_channels),  # the expected number of channels here is the num of output channels of the conv layer preceding the norm method
            'gn': nn.GroupNorm(gn_param, num_channels), 
            'in': nn.InstanceNorm2d(num_channels),
            'ln': nn.LayerNorm((num_channels, ln_params[conv_layer_idx], ln_params[conv_layer_idx])),  # not sure if 7, 7 changes with layer?
            'lrnc': nn.LocalResponseNorm(5, alpha=0.001),
            'lrns': LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False),
            'lrnb': LRN_both(spatial_size=3, channel_size=5, alpha=0.001)
        }
        return norm_dict[norm_method]

    def _make_layers(self, cfg, norm_method):
        layers = []
        in_channels = 3
        conv_layer_idx = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # print(conv_layer_idx)
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    self.set_layer_norm(x, norm_method, conv_layer_idx),
                    nn.ReLU(inplace=False)
                ]
                in_channels = x
                conv_layer_idx += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)


vgg11 = VGG11
