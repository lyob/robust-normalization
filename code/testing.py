#%%
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

#%%
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
        print('x size:', x.size())
        print('div size :', div.size())
        x = x.div(div)
        return x

#%%
'''Local response norm with 2 AvgNorm2D operations (channel wise and spatial wise)
'''
class LRN(nn.Module):
    def __init__(self, channel_size=1, spatial_size=1, alpha=1.0, beta=0.75):
        super(LRN, self).__init__()
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

        print('x size:', x.size())
        print('div size:', div.size())
        x = x.div(div)
        return x

#%%
'''compare the stacked 2d avg pools against the single avg pool 3d'''
spatial_size = 3
channel_size = 5

def channel_avg(input, kernel_size):
    sizes = input.size()
    div = input.unsqueeze(1)
    div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
    div = F.pad(div, (0, 0, 0, 0, (kernel_size)//2, (kernel_size - 1)//2))
    div = F.avg_pool3d(div, (kernel_size, 1, 1), stride=1).squeeze(1)
    div = div.view(sizes)
    return div

def avg_stacked_2d(input, spatial_size=3, channel_size=5):
    a = nn.AvgPool2d(
        kernel_size=spatial_size,
        stride=1,
        padding=1,
        ceil_mode=False
    )
    post_spatial = a(input)
    post_channel = channel_avg(post_spatial, channel_size)
    return post_spatial, post_channel

avg3d = nn.AvgPool3d(
    kernel_size=(channel_size, spatial_size, spatial_size),
    stride=1,
    padding=(2, 1, 1),
    ceil_mode=False
)

# the test input
input = torch.randn(1, 51, 4, 4)
input3d = input.unsqueeze(1)

# 1 avgpool3d op
post_one_3d = avg3d(input3d).squeeze(1)

# stacked
post_spatial, post_stacked_2d = avg_stacked_2d(input)

print('post_3d size:     ', tuple(post_one_3d.size()))
print('post_spatial size:', tuple(post_spatial.size()))
print('post 2d size:     ', tuple(post_stacked_2d.size()))

print('do the 2 methods produce the same result??:', torch.allclose(post_one_3d, post_stacked_2d, atol=1e-07))
print('stacked 2d ops===\n', post_stacked_2d[0][1])
print('one 3d op===\n', post_one_3d[0][1] )
print(torch.allclose(post_one_3d[0], post_stacked_2d[0], atol=1e-07))
#%% test the stacked 2D LRN on a 3x3 kernel on 2x2 HW

input = torch.randn(1, 512, 2, 2)
_, output = avg_stacked_2d(input, 3, 5)
print(output.size())


#%%
input = torch.randn(1,51,2,2)
op = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
print('div size', op(input).size())


#%% channel-wise averaging (taken from the F.local_response_norm implementation)
# the channel size is the second dimension
input = torch.randn(1, 51, 2, 2)
print('input===', input.size())
div = input.mul(input).unsqueeze(1)
size = 5
sizes = input.size()
print('sizes===', sizes)
div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)  # the -1 fills out the other dim size so that the tensor has the same data but a different shape
print('div.size()===', div.size())
div = F.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
print('post pad===', div.size())
div = F.avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
print('post avg3d===', div.size())
div = div.view(sizes)
print('view sizes===', div.size())

#%%
m = nn.AvgPool3d((5,2,2), stride=1, padding=(2,1,1))
input = torch.randn(1, 1, 512, 2, 2)
output = m(input)
print('input size:', tuple(input.size()))
print('kernel size:', m.kernel_size)
print('output size:', tuple(output.size()))

# %%
n = nn.AvgPool2d((3, 3), stride=1, padding=1)
input = torch.randn(512, 2, 2)
output = n(input)
print('input size:', tuple(input.size()))
print('kernel size:', n.kernel_size)
print('output size:', tuple(output.size()))

# %%
c = nn.Conv3d(3, 1, (3, 3, 3), stride=1, padding=1)
input = torch.randn(1, 3, 3, 2, 2)
output = c(input)
print('input size:', tuple(input.size()))
print('kernel size:', c.kernel_size)
print('output size:', tuple(output.size()))
