from typing import List, Tuple, Type, Union

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class FeatureUNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        output_channels = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            output_channels.append(input_features_skip)

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.output_channels = output_channels[::-1]

    def forward(self, skips):
        lres_input = skips[-1]
        decoder_features = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            decoder_features.append(x)
            lres_input = x
        decoder_features = decoder_features[::-1]
        if self.deep_supervision:
            return decoder_features
        return decoder_features[0]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class TaskSegmentationHead(nn.Module):
    def __init__(self, conv_op: Type[nn.Module], feature_channels: List[int], num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.seg_layers = nn.ModuleList([conv_op(ch, num_classes, 1, 1, 0, bias=True) for ch in feature_channels])

    def forward(self, decoder_features):
        if isinstance(decoder_features, list):
            return [seg_layer(feature) for seg_layer, feature in zip(self.seg_layers, decoder_features)]
        return self.seg_layers[0](decoder_features)

    def compute_conv_feature_map_size(self, spatial_sizes: List[List[int]]):
        output = np.int64(0)
        for seg_layer, spatial_size in zip(self.seg_layers, spatial_sizes):
            output += np.prod([seg_layer.out_channels, *spatial_size], dtype=np.int64)
        return output
