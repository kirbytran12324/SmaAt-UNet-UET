import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Flatten


# The DepthToSpace class is a PyTorch module that rearranges data from depth into blocks of spatial data.
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    # The forward method reshapes and rearranges the tensor.
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


# The SpaceToDepth class is a PyTorch module that rearranges blocks of spatial data into depth.
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    # The forward method reshapes and rearranges the tensor.
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


# The DepthwiseSeparableConv class is a type of convolutional layer that performs a depthwise convolution followed by a pointwise convolution.
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    # The forward method applies the depthwise and pointwise convolutions.
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# The DoubleDense class is a simple feed-forward network with two hidden layers.
class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels):
        super().__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    # The forward method applies the ReLU activation function after each layer except the last one.
    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out


# The DoubleDSConv class applies two depthwise separable convolutions in sequence, each followed by batch normalization and a ReLU activation function.
class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # The forward method applies the sequence of operations defined in the constructor.
    def forward(self, x):
        return self.double_ds_conv(x)


# The ChannelAttention class computes attention weights for each channel of the input.
class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super().__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    # The forward method computes the attention weights and applies them to the input.
    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


# The SpatialAttention class computes attention weights for each spatial location.
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    # The forward method computes the attention weights and applies them to the input.
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


# The CBAM class combines channel and spatial attention.
class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    # The forward method applies the channel and spatial attention to the input.
    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
