from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.architecture.segformer_backbone import FeaturePyramid


def _project_pyramid(features: FeaturePyramid, projections: nn.ModuleList) -> torch.Tensor:
    if len(features) != len(projections):
        raise ValueError(f"Expected {len(projections)} feature maps, got {len(features)}.")

    target_size = features[0].shape[2:]
    projected = []
    for feature, projection in zip(features, projections, strict=True):
        x = projection(feature)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        projected.append(x)
    return torch.cat(projected, dim=1)


class SegFormerDecoder(nn.Module):
    """Ported from repo/segformer_multitask/src/decoders.py, see that file for design notes this
    trainer does not repeat. Used by the dual_head (Late Fission) variant: one shared instance."""

    def __init__(self, in_channels: Tuple[int, int, int, int], embedding_dim: int = 256,
                 output_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim

        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embedding_dim, kernel_size=1) for channels in in_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: FeaturePyramid) -> torch.Tensor:
        fused = self.fuse(_project_pyramid(features, self.projections))
        return self.dropout(fused)


class SegFormerProjectionTrunk(nn.Module):
    """Shared per-scale projection stage, stopping short of fuse. Deferred to the dual_decoder/
    dual_fuse follow-up landing, kept here now so the module layout matches the plan up front."""

    def __init__(self, in_channels: Tuple[int, int, int, int], embedding_dim: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim * len(in_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embedding_dim, kernel_size=1) for channels in in_channels
        )

    def forward(self, features: FeaturePyramid) -> torch.Tensor:
        return _project_pyramid(features, self.projections)


class SegFormerFusionHead(nn.Module):
    """Task-specific fuse stage reading a shared SegFormerProjectionTrunk. Deferred to the
    dual_fuse follow-up landing, kept here now so the module layout matches the plan up front."""

    def __init__(self, in_channels: int, output_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim or in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fuse(projected))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        logits = self.classifier(x)
        if output_size is not None and logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


def to_paired_view_canvas(x: torch.Tensor, num_views: int) -> torch.Tensor:
    """nnU-Net hands the network `x` shaped (B, num_views, H, W): one raw single-channel view per
    input channel (anterior, posterior, ...). This project's SegFormer backbones (from-scratch or
    HuggingFace-pretrained) expect a 3-channel canvas, matching the existing segformer_multitask
    pipeline's own convention (repo/segformer_multitask/src/datasets.py:load_paired_image_tensor):
    each view grayscale-replicated to 3 channels, concatenated along width. Reproduced here as a
    tensor op instead of at dataset-load time, since nnU-Net's dataloader/augmentation pipeline
    operates on the (B, num_views, H, W) stacked-channel layout, not a pre-built wide canvas."""
    if x.shape[1] != num_views:
        raise ValueError(f"Expected {num_views} input channels (one per view), got {x.shape[1]}.")
    views = [x[:, view_idx:view_idx + 1, :, :].repeat(1, 3, 1, 1) for view_idx in range(num_views)]
    return torch.cat(views, dim=3)


def split_and_restack_view_logits(logits: torch.Tensor, num_views: int) -> torch.Tensor:
    """Inverse of to_paired_view_canvas at the output side: splits a (B, num_classes, H,
    num_views*W) wide-canvas prediction back into num_views per-view slices and restacks them into
    (B, num_views*num_classes, H, W), channel order [view0_class0..N, view1_class0..N, ...]. This
    is the exact layout nnUNetTrainerMultiTask._reshape_paired_multiview_for_loss expects: it
    recovers per-view class maps via `.reshape(B, num_views, num_classes, H, W)`, which only
    groups correctly if the channel axis is packed in this v-major order."""
    batch, num_classes, height, wide_width = logits.shape
    if wide_width % num_views != 0:
        raise ValueError(f"Canvas width {wide_width} is not divisible by num_views={num_views}.")
    view_width = wide_width // num_views
    per_view = torch.stack(
        [logits[:, :, :, v * view_width:(v + 1) * view_width] for v in range(num_views)],
        dim=1,
    )
    return per_view.reshape(batch, num_views * num_classes, height, view_width)
