from typing import Dict, List, Tuple, Type, Union

import torch
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.architecture.cbam import CBAMFeatureAdapter
from nnunetv2.architecture.multitask_components import FeatureUNetDecoder, TaskSegmentationHead


class _BaseMultiTaskUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        multitask: dict,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        cbam: dict = None,
    ):
        super().__init__()
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )
        self.deep_supervision = deep_supervision
        self.multitask = multitask
        self.task_configs = self._normalize_tasks(multitask)
        self.task_names = [task["name"] for task in self.task_configs]
        self.cbam = self._normalize_cbam(cbam)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )
        encoder_cbam_channels = list(self.encoder.output_channels)
        if not self.cbam["bottleneck"]:
            encoder_cbam_channels = encoder_cbam_channels[:-1]
        self.encoder_attention = (
            CBAMFeatureAdapter(
                self.encoder.conv_op,
                encoder_cbam_channels,
                self.cbam["reduction"],
                self.cbam["spatial_kernel_size"],
            )
            if self.cbam["enabled"] and self.cbam["encoder"] and encoder_cbam_channels
            else None
        )
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder
        self.nonlin_first = nonlin_first
        self._decoder_kwargs = {
            "norm_op": norm_op,
            "norm_op_kwargs": norm_op_kwargs,
            "dropout_op": dropout_op,
            "dropout_op_kwargs": dropout_op_kwargs,
            "nonlin": nonlin,
            "nonlin_kwargs": nonlin_kwargs,
            "conv_bias": conv_bias,
            "cbam": self.cbam,
        }
        self.num_classes = num_classes
        self._build_variant_modules()

    @staticmethod
    def _normalize_tasks(multitask: dict) -> List[dict]:
        tasks = multitask.get("tasks", []) if multitask is not None else []
        if len(tasks) < 1:
            raise ValueError("Multitask architectures require at least one task.")
        normalized = []
        for idx, task in enumerate(tasks):
            normalized.append(
                {
                    "name": task.get("name", f"task{idx + 1}"),
                    "num_classes": int(task["num_classes"]),
                    "output_channels": int(task.get("output_channels", task["num_classes"])),
                    "loss_weight": float(task.get("loss_weight", 1.0)),
                }
            )
        return normalized

    @staticmethod
    def _normalize_cbam(cbam: dict) -> dict:
        cbam = cbam or {}
        reduction = cbam.get("reduction", cbam.get("reduction_ratio", 16))
        return {
            "enabled": bool(cbam.get("enabled", False)),
            "encoder": bool(cbam.get("encoder", True)),
            "decoder": bool(cbam.get("decoder", True)),
            "bottleneck": bool(cbam.get("bottleneck", True)),
            "reduction": int(reduction),
            "spatial_kernel_size": int(cbam.get("spatial_kernel_size", 7)),
        }

    def _apply_encoder_attention(self, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.encoder_attention is None:
            return skips
        if self.cbam["bottleneck"]:
            return self.encoder_attention(skips)
        attended_skips = self.encoder_attention(skips[:-1])
        return [*attended_skips, skips[-1]]

    def _make_decoder(self, stage_range: Tuple[int, int] = None):
        return FeatureUNetDecoder(
            self.encoder,
            self.n_conv_per_stage_decoder,
            self.deep_supervision,
            nonlin_first=self.nonlin_first,
            stage_range=stage_range,
            **self._decoder_kwargs,
        )

    def _make_head(self, num_classes: int, feature_channels: List[int]):
        return TaskSegmentationHead(self.encoder.conv_op, feature_channels, num_classes)

    def set_deep_supervision(self, enabled: bool):
        self.deep_supervision = enabled
        for decoder in self._iter_decoders():
            decoder.deep_supervision = enabled

    def _iter_decoders(self):
        raise NotImplementedError

    def deep_supervision_num_levels(self) -> int:
        raise NotImplementedError

    def _apply_heads(self, task_features: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        outputs = {}
        for task_name, features in task_features.items():
            outputs[task_name] = self.heads[task_name](features)
        return outputs

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        output = self.encoder.compute_conv_feature_map_size(input_size)
        encoder_spatial_sizes = []
        encoder_current_size = list(input_size)
        for stride in self.encoder.strides:
            encoder_current_size = [i // j for i, j in zip(encoder_current_size, stride)]
            encoder_spatial_sizes.append(encoder_current_size)
        if self.encoder_attention is not None:
            output += self.encoder_attention.compute_conv_feature_map_size(
                encoder_spatial_sizes if self.cbam["bottleneck"] else encoder_spatial_sizes[:-1]
            )
        spatial_sizes = []
        current_size = list(input_size)
        for stride in self.encoder.strides[:-1]:
            current_size = [i // j for i, j in zip(current_size, stride)]
            spatial_sizes.append(current_size)
        for decoder in self._iter_decoders():
            output += decoder.compute_conv_feature_map_size(input_size)
        for task_name in self.task_names:
            output += self.heads[task_name].compute_conv_feature_map_size(spatial_sizes)
        return output

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class MultiTaskDualHeadUNet(_BaseMultiTaskUNet):
    """Hickson et al. 2022, Fig. 1d, Late Fission: the whole decoder is shared, only the final
    per-task output head splits."""

    def _build_variant_modules(self):
        self.decoder = self._make_decoder()
        feature_channels = self.decoder.output_channels
        self.heads = nn.ModuleDict(
            {task["name"]: self._make_head(task["output_channels"], feature_channels) for task in self.task_configs}
        )

    def _iter_decoders(self):
        return [self.decoder]

    def deep_supervision_num_levels(self) -> int:
        return len(self.n_conv_per_stage_decoder)

    def forward(self, x):
        skips = self._apply_encoder_attention(self.encoder(x))
        shared_features = self.decoder(skips)
        return self._apply_heads({task_name: shared_features for task_name in self.task_names})


class MultiTaskDualDecoderUNet(_BaseMultiTaskUNet):
    """Hickson et al. 2022, Fig. 1a, Early Fission: every decoder stage is task-specific from
    right after the shared encoder."""

    def _build_variant_modules(self):
        self.decoders = nn.ModuleDict({task["name"]: self._make_decoder() for task in self.task_configs})
        feature_channels = next(iter(self.decoders.values())).output_channels
        self.heads = nn.ModuleDict(
            {task["name"]: self._make_head(task["output_channels"], feature_channels) for task in self.task_configs}
        )

    def _iter_decoders(self):
        return list(self.decoders.values())

    def deep_supervision_num_levels(self) -> int:
        return len(self.n_conv_per_stage_decoder)

    def forward(self, x):
        skips = self._apply_encoder_attention(self.encoder(x))
        task_features = {task_name: decoder(skips) for task_name, decoder in self.decoders.items()}
        return self._apply_heads(task_features)


class _PartialSharedDecoderMultiTaskUNet(_BaseMultiTaskUNet):
    """Shared base for the two intermediate fission points: one shared decoder "trunk" run once on
    the encoder skips, then one independent per-task decoder "tail" continuing from the trunk's
    output. Concrete subclasses only need to say how many stages the trunk covers.
    """

    def _trunk_stage_count(self) -> int:
        raise NotImplementedError

    def _build_variant_modules(self):
        n_stages = len(self.n_conv_per_stage_decoder)
        trunk_stages = self._trunk_stage_count()
        if not 0 < trunk_stages < n_stages:
            raise ValueError(
                f"{type(self).__name__} needs 0 < trunk stages < {n_stages} decoder stages, "
                f"got {trunk_stages}."
            )
        self.trunk = self._make_decoder(stage_range=(0, trunk_stages))
        self.tails = nn.ModuleDict(
            {task["name"]: self._make_decoder(stage_range=(trunk_stages, n_stages)) for task in self.task_configs}
        )
        feature_channels = next(iter(self.tails.values())).output_channels
        self.heads = nn.ModuleDict(
            {task["name"]: self._make_head(task["output_channels"], feature_channels) for task in self.task_configs}
        )

    def _iter_decoders(self):
        return [self.trunk, *self.tails.values()]

    def deep_supervision_num_levels(self) -> int:
        tail = next(iter(self.tails.values()))
        return tail.stage_range[1] - tail.stage_range[0]

    def forward(self, x):
        skips = self._apply_encoder_attention(self.encoder(x))
        trunk_out = self.trunk(skips)
        if isinstance(trunk_out, list):
            trunk_out = trunk_out[0]
        task_features = {task_name: tail(skips, initial_input=trunk_out) for task_name, tail in self.tails.items()}
        return self._apply_heads(task_features)


class MultiTaskEarlyMidUNet(_PartialSharedDecoderMultiTaskUNet):
    """Hickson et al. 2022, Fig. 1b, Early-Mid Fission: encoder plus the first (coarsest) decoder
    stage shared, the rest split per task."""

    def _trunk_stage_count(self) -> int:
        return 1


class MultiTaskMidUNet(_PartialSharedDecoderMultiTaskUNet):
    """Hickson et al. 2022, Fig. 1c, Mid Fission: encoder and every decoder stage except the last
    (finest) one shared. The paper's proposed / headline configuration."""

    def _trunk_stage_count(self) -> int:
        return len(self.n_conv_per_stage_decoder) - 1
