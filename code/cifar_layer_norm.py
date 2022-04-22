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
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, size, stride=1, normalize='nn'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.ln1 = nn.LayerNorm((planes, size*1, size*1))
        else:
            self.ln1 = nn.LayerNorm((planes, size*1, size*1))

        self.ln2 = nn.LayerNorm((planes, size, size))
        self.in1 = nn.InstanceNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.gn1 = nn.GroupNorm(16, planes)
        self.gn2 = nn.GroupNorm(16, planes)
        self.lrnc1 = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrnc2 = nn.LocalResponseNorm(5, alpha=0.001)

        if size < 20:
            self.lrnb1 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
        else:
            self.lrnb1 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
        
        self.normalize = normalize
        self.norm_dict1 = {'nn': nn.Identity(), 'bn': self.bn1, 'ln': self.ln1, 'in': self.in1, 'gn': self.gn1,
                           'lrnc': self.lrnc1, 'lrns': self.lrns1, 'lrnb': self.lrnb1}
        self.norm_dict2 = {'nn': nn.Identity(), 'bn': self.bn2, 'ln': self.ln2, 'in': self.in2, 'gn': self.gn2,
                           'lrnc': self.lrnc2, 'lrns': self.lrns2, 'lrnb': self.lrnb2}
        self.fake_relu = custom_modules1.FakeReLUM()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if size < 15:
                self.lrnb_shortcut = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
                self.lrns_shortcut = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            else:
                self.lrnb_shortcut = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
                self.lrns_shortcut = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
                
            self.norm_shortcut = {'nn': nn.Identity(), 'bn': nn.BatchNorm2d(planes*self.expansion), 
            'ln': nn.LayerNorm((planes*self.expansion, size, size)), 'gn': nn.GroupNorm(16, planes*self.expansion),
            'in': nn.InstanceNorm2d(planes*self.expansion), 'lrnc': nn.LocalResponseNorm(5,alpha=0.001),
            'lrns': self.lrns_shortcut, 'lrnb': self.lrnb_shortcut}
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False), self.norm_shortcut[self.normalize])
        self.stride = stride
        self.size = size

    def forward(self, x, fake_relu=False):
        out = F.relu(self.norm_dict1[self.normalize](self.conv1(x)))
        out = self.norm_dict2[self.normalize](self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return self.fake_relu(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, size, stride=1, normalize='nn'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if stride == 1:
            self.ln1 = nn.LayerNorm((planes, size, size))
        else:
            self.ln1 = nn.LayerNorm((planes, size*2, size*2))
        self.ln2 = nn.LayerNorm((planes, size, size))
        self.ln3 = nn.LayerNorm((planes, size, size))
        
        self.in1 = nn.InstanceNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.in3 = nn.InstanceNorm2d(planes)
        self.gn1 = nn.GroupNorm(16, planes)
        self.gn2 = nn.GroupNorm(16, planes)
        self.gn2 = nn.GroupNorm(16, planes)
        
        self.lrnc1 = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrnc2 = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrnc3 = nn.LocalResponseNorm(5, alpha=0.001)
        
        if size < 15:
            self.lrnb1 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb3 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns3 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)

        else:
            self.lrnb1 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb3 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)

            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns3 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)

        self.size = size
        self.stride = stride
        self.normalize = normalize
        self.norm_dict1 = {'nn': nn.Identity(), 'bn': self.bn1, 'ln': self.ln1}
        self.norm_dict2 = {'nn': nn.Identity(), 'bn': self.bn2, 'ln': self.ln2}
        self.norm_dict3 = {'nn': nn.Identity(), 'bn': self.bn3, 'ln': self.ln3}
        

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if size < 15:
                self.lrnb_shortcut = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
                self.lrns_shortcut = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            else:
                self.lrnb_shortcut = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
                self.lrns_shortcut = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
                
            self.norm_shortcut = {'nn': nn.Identity(), 'bn': nn.BatchNorm2d(planes*self.expansion), 
            'ln': nn.LayerNorm((planes*self.expansion, size, size)), 'gn': nn.GroupNorm(16, planes*self.expansion),
            'in': nn.InstanceNorm2d(planes*self.expansion), 'lrnc': nn.LocalResponseNorm(5,alpha=0.001),
            'lrns': self.lrns_shortcut, 'lrnb': self.lrnb_shortcut}
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False), self.norm_shortcut[self.normalize])

        self.fake_relu = custom_modules1.FakeReLUM()

    def forward(self, x, fake_relu=False):
        out = F.relu(self.norm_dict1[self.normalize](self.conv1(x)))
        out = F.relu(self.norm_dict2[self.normalize](self.conv2(out)))
        out = self.norm_dict3[self.normalize](self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return self.fake_relu(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, input_size=32, num_classes=10, feat_scale=1, wm=1, normalize='nn'):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.ln1 = nn.LayerNorm((self.in_planes, input_size, input_size))
        self.ln2 = nn.LayerNorm((self.in_planes, input_size, input_size))
        self.in1 = nn.InstanceNorm2d(self.in_planes)
        self.in2 = nn.InstanceNorm2d(self.in_planes)
        self.gn1 = nn.GroupNorm(16, self.in_planes)
        self.gn2 = nn.GroupNorm(16, self.in_planes)
        self.lrnc1 = nn.LocalResponseNorm(5, alpha=0.001)
        self.lrnc2 = nn.LocalResponseNorm(5, alpha=0.001)

        if input_size < 15:
            self.lrnb1 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=3, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
        else:
            self.lrnb1 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrnb2 = LRN(spatial_size=5, channel_size=5, alpha=0.001, across_channel_spatial=True)
            self.lrns1 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
            self.lrns2 = LRN(spatial_size=3, alpha=0.001, across_channel_spatial=False)
        
        self.normalize = normalize
        self.norm_dict = {'nn': nn.Identity(), 'bn': self.bn1, 'ln': self.ln1, 'in': self.in1, 'gn': self.gn1,
                           'lrnc': self.lrnc1, 'lrns': self.lrns1, 'lrnb': self.lrnb1}
        self.input_size = input_size
        self.normalize = normalize
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.input_size = int(self.input_size / stride)
            layers.append(block(self.in_planes, planes, self.input_size, stride,
                                normalize=self.normalize))
            self.in_planes = planes * block.expansion
        return custom_modules1.SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, no_relu=False, fake_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.norm_dict[self.normalize](self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=5, **kwargs)

def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wd=.75, **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)