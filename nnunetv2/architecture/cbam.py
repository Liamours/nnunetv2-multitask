from typing import Sequence, Type

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from torch import nn
from torch.nn.modules.conv import _ConvNd


class ChannelAttention(nn.Module):
    def __init__(self, conv_op: Type[_ConvNd], channels: int, reduction: int = 16):
        super().__init__()
        spatial_dim = convert_conv_op_to_dim(conv_op)
        hidden_channels = max(1, channels // reduction)
        pool_op = nn.AdaptiveAvgPool2d if spatial_dim == 2 else nn.AdaptiveAvgPool3d
        self.avg_pool = pool_op(1)
        self.max_pool = pool_op(1)
        self.shared = nn.Sequential(
            conv_op(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            conv_op(hidden_channels, channels, 1, bias=False),
        )
        self.activation = nn.Sigmoid()
        self.channels = channels
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x))
        return x * self.activation(attention)

    def compute_conv_feature_map_size(self):
        return np.int64(2 * (self.hidden_channels + self.channels))


class SpatialAttention(nn.Module):
    def __init__(self, conv_op: Type[_ConvNd], kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CBAM spatial kernel size must be odd.")
        spatial_dim = convert_conv_op_to_dim(conv_op)
        if spatial_dim not in (2, 3):
            raise ValueError(f"CBAM only supports 2D/3D convolutions, got spatial_dim={spatial_dim}.")
        self.conv = conv_op(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_projection = torch.mean(x, dim=1, keepdim=True)
        max_projection = torch.max(x, dim=1, keepdim=True).values
        attention = self.activation(self.conv(torch.cat((avg_projection, max_projection), dim=1)))
        return x * attention

    def compute_conv_feature_map_size(self, spatial_size):
        return np.prod([1, *spatial_size], dtype=np.int64)


class CBAM(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(conv_op, channels, reduction)
        self.spatial_attention = SpatialAttention(conv_op, spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))

    def compute_conv_feature_map_size(self, spatial_size):
        return (
            self.channel_attention.compute_conv_feature_map_size()
            + self.spatial_attention.compute_conv_feature_map_size(spatial_size)
        )


class CBAMFeatureAdapter(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels_per_feature: Sequence[int],
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CBAM(conv_op, int(ch), reduction, spatial_kernel_size) for ch in channels_per_feature]
        )

    def forward(self, features):
        if isinstance(features, list):
            return [block(feature) for block, feature in zip(self.blocks, features)]
        return self.blocks[0](features)

    def compute_conv_feature_map_size(self, spatial_sizes):
        output = np.int64(0)
        for block, spatial_size in zip(self.blocks, spatial_sizes):
            output += block.compute_conv_feature_map_size(spatial_size)
        return output
