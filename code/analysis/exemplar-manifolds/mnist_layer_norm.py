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



'''
start of refactor
'''
class BatchNorm(nn.Module):
    def __init__(self, layer, in_channels):
        super(BatchNorm, self).__init__()
        self.layer = layer
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn = {'1': self.bn1, '2': self.bn2}

    def forward(self, x):
        return self.bn[self.layer](x)

class LayerNorm(nn.Module):
    def __init__(self, layer, in_channels) -> None:
        super(LayerNorm, self).__init__()
        self.layer = layer
        self.in_channels = in_channels
        self.ln1 = nn.LayerNorm([in_channels, 14, 14])
        self.ln2 = nn.LayerNorm([20, 10, 10])
        self.ln = {'1': self.ln1, '2': self.ln2}

    def forward(self):
        return self.ln[self.layer]

class InstanceNorm(nn.Module):
    def __init__(self, layer, in_channels) -> None:
        super(InstanceNorm, self).__init__()
        self.layer = layer
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(20, affine=True)
        self.instance_norm = {'1': self.in1, '2': self.in2}

    def forward(self):
        return self.instance_norm[self.layer]

class GroupNorm(nn.Module):
    def __init__(self, layer, in_channels) -> None:
        super(GroupNorm, self).__init__()
        self.layer = layer
        self.gn1 = nn.GroupNorm(4, in_channels)
        self.gn2 = nn.GroupNorm(4, 20)
        self.gn = {'1': self.gn1, '2': self.gn2}

    def forward(self):
        return self.gn[self.layer]

class Normalization(nn.Module):
    def __init__(self, layer, norm_method, norm_position, in_channels):
        super(Normalization, self).__init__()
        self.layer = layer
        self.norm_method = norm_method
        self.norm_position = norm_position
        self.in_channels = in_channels

        self.norm_dict = {
            'nn': nn.Identity(),
            'bn': BatchNorm(layer=self.layer, in_channels=self.in_channels),
            'ln': LayerNorm(layer=self.layer, in_channels=self.in_channels),
            'in': InstanceNorm(layer=self.layer, in_channels=self.in_channels),
            'gn': GroupNorm(layer=self.layer, in_channels=self.in_channels),
            'lrns': LRN(spatial_size=3, across_channel_spatial=False),
            'lrnc': nn.LocalResponseNorm(5, alpha=0.001), 
            'lrnb': LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)
        }

        self.arch_selection = {
            '1': {'1': self.norm_dict[self.norm_method], '2': self.norm_dict['nn']},
            '2': {'1': self.norm_dict['nn'], '2': self.norm_dict[self.norm_method]},
            'both': {'1': self.norm_dict[self.norm_method], '2': self.norm_dict[self.norm_method]}
        }

    def forward(self, x):
        return self.arch_selection[self.norm_position][self.layer](x)

class Net(nn.Module):
    def __init__(self, conv_1, in_channels, norm_method=None, norm_position='both'):
        super(Net,self).__init__()
        self.conv_1 = conv_1
        self.in_channels = in_channels
        self.norm_method = norm_method
        self.norm_position = norm_position
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels= in_channels, out_channels=20, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=500, out_features=10)
        self.norm_1 = Normalization(layer='1', norm_method=self.norm_method, norm_position=self.norm_position, in_channels=self.in_channels)
        self.norm_2 = Normalization(layer='2', norm_method=self.norm_method, norm_position=self.norm_position, in_channels=self.in_channels)

    def forward(self, x):
        # first layer
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.relu(x)
        
        # second layer
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)

        # third layer
        x = self.fc_1(x)
        return x
'''
end of refactor
'''



class Net_both(nn.Module):
    def __init__(self, conv_1, in_channels, width_scale=1, fc_neurons=500, normalize=None):
        super(Net_both, self).__init__()
        self.width_scale = width_scale
        self.out_channels = int(20*self.width_scale)
        self.conv_1 = conv_1
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=5, stride=1)
        self.maxpool2d = nn.MaxPool2d(2, 2)
        self.fc_1 = nn.Linear(in_features=int(fc_neurons*self.width_scale), out_features=10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.ln1 = nn.LayerNorm([in_channels, 14, 14])
        self.ln2 = nn.LayerNorm([self.out_channels, 10, 10])
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.gn1 = nn.GroupNorm(4, in_channels)
        self.gn2 = nn.GroupNorm(4, self.out_channels)

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
        self.normalize = normalize

    def forward(self, x):
        # first layer
        x = self.conv_1(x)
        x = self.norm_dict1[self.normalize](x)
        x = self.relu1(x)
        
        # second layer
        x = self.conv_2(x)
        x = self.norm_dict2[self.normalize](x)
        x = self.relu2(x)
        # x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2d(x)
        x = torch.flatten(x, 1)

        # third layer
        x = self.fc_1(x)
        return x

class Net_1(nn.Module):
    def __init__(self, conv_1, in_channels, normalize=None):
        super(Net_1, self).__init__()
        self.conv_1 = conv_1
        self.conv_2 = nn.Conv2d(in_channels= in_channels, out_channels=20, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=500, out_features=10)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.ln1 = nn.LayerNorm([in_channels, 14, 14])
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.gn1 = nn.GroupNorm(4, in_channels)

        self.lrn = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_channel = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_spatial = LRN(spatial_size=3, across_channel_spatial=False)
        self.lrn_both = LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)

        self.norm_dict1 = {'nn': nn.Identity(),'bn': self.bn1, 'ln': self.ln1, 
                           'in': self.in1, 'gn': self.gn1, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.norm_dict2 = {'nn': nn.Identity()}
        self.normalize = normalize
        self.maxpool2d = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # first layer
        x = self.conv_1(x)
        x = self.norm_dict1[self.normalize](x)
        x = self.relu(x)
        
        # second layer
        x = self.conv_2(x)
        x = self.norm_dict2['nn'](x)
        x = self.relu(x)
        # x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2d(x)
        x = torch.flatten(x, 1)

        # third layer
        x = self.fc_1(x)
        return x

class Net_2(nn.Module):
    def __init__(self, conv_1, in_channels, normalize=None):
        super(Net_2, self).__init__()
        self.conv_1 = conv_1
        self.conv_2 = nn.Conv2d(in_channels= in_channels, out_channels=20, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=500, out_features=10)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(20)
        self.ln2 = nn.LayerNorm([20, 10, 10])
        self.in2 = nn.InstanceNorm2d(20, affine=True)
        self.gn2 = nn.GroupNorm(4, 20)

        self.lrn = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_channel = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrn_spatial = LRN(spatial_size=3, across_channel_spatial=False)
        self.lrn_both = LRN(spatial_size=3, channel_size=5, across_channel_spatial=True)

        self.norm_dict1 = {'nn': nn.Identity()}
        self.norm_dict2 = {'nn': nn.Identity(), 'bn': self.bn2, 'ln': self.ln2,
                           'in': self.in2, 'gn': self.gn2, 'lrns': self.lrn_spatial,
                           'lrnc': self.lrn_channel, 'lrnb': self.lrn_both}
        self.normalize = normalize
        self.maxpool2d = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # first layer
        x = self.conv_1(x)
        x = self.norm_dict1['nn'](x)
        x = self.relu(x)
        
        # second layer
        x = self.conv_2(x)
        x = self.norm_dict2[self.normalize](x)
        x = self.relu(x)
        # x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2d(x)
        x = torch.flatten(x, 1)

        # third layer
        x = self.fc_1(x)
        return x