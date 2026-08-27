from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn

LAYER_NORM_EPS = 1e-6

FeaturePyramid = Sequence[torch.Tensor]


@dataclass(frozen=True)
class MiTSpec:
    name: str
    embed_dims: int
    num_stages: int
    num_layers: Tuple[int, int, int, int]
    num_heads: Tuple[int, int, int, int]
    patch_sizes: Tuple[int, int, int, int]
    strides: Tuple[int, int, int, int]
    sr_ratios: Tuple[int, int, int, int]
    mlp_ratio: int
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float

    @property
    def stage_channels(self) -> Tuple[int, int, int, int]:
        return tuple(self.embed_dims * head for head in self.num_heads)


MIT_B0 = MiTSpec(
    name="mit_b0", embed_dims=32, num_stages=4, num_layers=(2, 2, 2, 2), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

MIT_B1 = MiTSpec(
    name="mit_b1", embed_dims=64, num_stages=4, num_layers=(2, 2, 2, 2), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

MIT_B2 = MiTSpec(
    name="mit_b2", embed_dims=64, num_stages=4, num_layers=(3, 4, 6, 3), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

MIT_B3 = MiTSpec(
    name="mit_b3", embed_dims=64, num_stages=4, num_layers=(3, 4, 18, 3), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

MIT_B4 = MiTSpec(
    name="mit_b4", embed_dims=64, num_stages=4, num_layers=(3, 8, 27, 3), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

MIT_B5 = MiTSpec(
    name="mit_b5", embed_dims=64, num_stages=4, num_layers=(3, 6, 40, 3), num_heads=(1, 2, 5, 8),
    patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1), mlp_ratio=4,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)

SUPPORTED_MIT_SPECS = {
    MIT_B0.name: MIT_B0, MIT_B1.name: MIT_B1, MIT_B2.name: MIT_B2,
    MIT_B3.name: MIT_B3, MIT_B4.name: MIT_B4, MIT_B5.name: MIT_B5,
}


def get_mit_spec(name: str) -> MiTSpec:
    if name not in SUPPORTED_MIT_SPECS:
        raise ValueError(f"Unsupported variant={name}.")
    return SUPPORTED_MIT_SPECS[name]


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim, eps=LAYER_NORM_EPS)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        height, width = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (height, width)


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            height, width = spatial_shape
            x_kv = x.transpose(1, 2).reshape(batch_size, dim, height, width)
            x_kv = self.sr(x_kv).reshape(batch_size, dim, -1).transpose(1, 2)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        kv = self.kv(x_kv).reshape(batch_size, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        out = self.proj(out)
        return self.proj_drop(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        batch_size, _, channels = x.shape
        height, width = spatial_shape
        out = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = self.fc1(out)
        out = self.dwconv(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, sr_ratio: int, drop: float,
                 attn_drop: float, drop_path_prob: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attn = EfficientSelfAttention(dim=dim, num_heads=num_heads, sr_ratio=sr_ratio,
                                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = MixFFN(dim=dim, hidden_dim=dim * mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), spatial_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x), spatial_shape))
        return x


class PyramidBackbone(nn.Module, ABC):
    @property
    @abstractmethod
    def out_channels(self) -> Tuple[int, int, int, int]:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        pass


class MixVisionTransformer(PyramidBackbone):
    """From-scratch MiT backbone. Ported from repo/segformer_multitask/src/backbones.py so this
    trainer does not depend on that frozen, separate-venv repo at import time."""

    def __init__(self, spec: MiTSpec, in_channels: int = 3) -> None:
        super().__init__()
        self.spec = spec
        self.in_channels = in_channels

        drop_path_schedule = torch.linspace(0, spec.drop_path_rate, sum(spec.num_layers)).tolist()
        cursor = 0
        stage_in_channels = in_channels

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for stage_index in range(spec.num_stages):
            stage_dim = spec.stage_channels[stage_index]
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    in_channels=stage_in_channels,
                    embed_dim=stage_dim,
                    patch_size=spec.patch_sizes[stage_index],
                    stride=spec.strides[stage_index],
                )
            )

            stage_blocks = []
            for block_index in range(spec.num_layers[stage_index]):
                stage_blocks.append(
                    TransformerBlock(
                        dim=stage_dim,
                        num_heads=spec.num_heads[stage_index],
                        mlp_ratio=spec.mlp_ratio,
                        sr_ratio=spec.sr_ratios[stage_index],
                        drop=spec.drop_rate,
                        attn_drop=spec.attn_drop_rate,
                        drop_path_prob=drop_path_schedule[cursor + block_index],
                    )
                )
            self.blocks.append(nn.ModuleList(stage_blocks))
            self.norms.append(nn.LayerNorm(stage_dim, eps=LAYER_NORM_EPS))

            cursor += spec.num_layers[stage_index]
            stage_in_channels = stage_dim

        self._init_weights()

    @property
    def out_channels(self) -> Tuple[int, int, int, int]:
        return self.spec.stage_channels

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, mean=0.0, std=(2.0 / fan_out) ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        outputs = []
        for patch_embed, stage_blocks, norm in zip(self.patch_embeds, self.blocks, self.norms, strict=True):
            x, spatial_shape = patch_embed(x)
            for block in stage_blocks:
                x = block(x, spatial_shape)
            x = norm(x)
            height, width = spatial_shape
            x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
            outputs.append(x)
        return tuple(outputs)


class HuggingFacePretrainedMiT(PyramidBackbone):
    """PyramidBackbone wrapping an ImageNet-pretrained HuggingFace SegformerModel encoder.
    Fixed to 3-channel input by construction, the pretrained weights are sized for it. Ported
    from repo/segformer_multitask/src/backbones.py, see MixVisionTransformer's docstring."""

    def __init__(self, hf_name: str, spec: MiTSpec) -> None:
        super().__init__()
        try:
            from transformers import SegformerModel
        except ImportError as error:
            raise ImportError("transformers is required for a HuggingFace-pretrained backbone.") from error

        self.spec = spec
        self.hf_name = hf_name
        # use_safetensors=True: this repo's pinned torch==2.5.1 (pyproject.toml) is older than the
        # torch>=2.6 recent transformers releases require for pickle-format torch.load (CVE-2025-32434);
        # safetensors avoids torch.load entirely and nvidia/mit-b2 publishes a safetensors checkpoint.
        self.model, self.loading_info = SegformerModel.from_pretrained(
            hf_name, output_loading_info=True, use_safetensors=True
        )
        actual_channels = tuple(self.model.config.hidden_sizes)
        if actual_channels != spec.stage_channels:
            raise ValueError(
                f"Pretrained checkpoint {hf_name} has stage channels {actual_channels}, "
                f"expected {spec.stage_channels} for variant={spec.name}."
            )

    @property
    def out_channels(self) -> Tuple[int, int, int, int]:
        return self.spec.stage_channels

    def forward(self, x: torch.Tensor) -> FeaturePyramid:
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        return tuple(outputs.hidden_states)
