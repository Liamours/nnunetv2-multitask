from typing import Dict, List, Optional

import torch
from torch import nn

from nnunetv2.architecture.segformer_backbone import HuggingFacePretrainedMiT, MixVisionTransformer, get_mit_spec
from nnunetv2.architecture.segformer_components import (
    SegFormerDecoder,
    SegmentationHead,
    split_and_restack_view_logits,
    to_paired_view_canvas,
)


class _BaseMultiTaskSegFormer(nn.Module):
    """SegFormer counterpart to multitask_unet.py's _BaseMultiTaskUNet: constructed the same way
    (network_class_name + arch_kwargs from the plans file, via get_network_from_plans), sharing the
    same `multitask` kwarg shape and the same Dict[task_name, Tensor] forward contract, so
    nnUNetTrainerMultiTask's loss/target-splitting/metrics machinery drives this family exactly as
    it drives the U-Net one, unmodified. Not a subclass of _BaseMultiTaskUNet: the two architecture
    families share a contract, not an implementation, so a separate hierarchy is cleaner than
    forcing an unrelated base across them.

    Assumes each input channel is one raw single-view image (input_channels == num_views), true for
    every paired-anterior-posterior dataset this project has; num_views is derived from
    input_channels rather than taken as a separate kwarg so the two can never disagree.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        multitask: dict,
        mit_variant: str = "mit_b2",
        pretrained_hf_name: Optional[str] = "nvidia/mit-b2",
        decoder_embedding_dim: int = 768,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if deep_supervision:
            raise NotImplementedError(
                "SegFormer's decode head has no multi-resolution output. Use a trainer with "
                "enable_deep_supervision=False for this architecture family."
            )
        self.num_views = input_channels
        self.num_classes = num_classes
        self.multitask = multitask
        self.task_configs = self._normalize_tasks(multitask)
        self.task_names = [task["name"] for task in self.task_configs]
        self.mit_variant = mit_variant
        self.pretrained_hf_name = pretrained_hf_name
        self.decoder_embedding_dim = decoder_embedding_dim

        spec = get_mit_spec(mit_variant)
        self.backbone = HuggingFacePretrainedMiT(pretrained_hf_name, spec) if pretrained_hf_name \
            else MixVisionTransformer(spec, in_channels=3)

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

    def set_deep_supervision(self, enabled: bool) -> None:
        # Unconditional no-op, both directions. nnUNetTrainerMultiTask.perform_actual_validation
        # calls set_deep_supervision_enabled(False) then unconditionally (True) after every
        # validation pass, regardless of the trainer's own enable_deep_supervision setting -
        # verified directly at nnUNetTrainerMultiTask.py:424,494. This architecture has nothing to
        # toggle; raising here would crash every training run at its first real validation.
        pass


class MultiTaskDualHeadSegFormer(_BaseMultiTaskSegFormer):
    """Hickson et al. 2022, Fig. 1d, Late Fission: one shared SegFormerDecoder, only the final
    per-task SegmentationHead splits. Mirrors MultiTaskDualHeadUNet's role for the U-Net family,
    and DualHeadSegFormer's role in the separate repo/segformer_multitask pipeline."""

    def _build_variant_modules(self) -> None:
        self.decoder = SegFormerDecoder(self.backbone.out_channels, embedding_dim=self.decoder_embedding_dim)
        # Heads predict num_classes channels per canvas pixel, not the task's already-doubled
        # output_channels: the view axis does not exist yet at this point, forward() only produces
        # it afterward via split_and_restack_view_logits, which is what actually multiplies by
        # num_views. Sizing the head to output_channels here would silently double the channel
        # count a second time.
        self.heads = nn.ModuleDict(
            {
                task["name"]: SegmentationHead(self.decoder.output_dim, task["num_classes"])
                for task in self.task_configs
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        canvas = to_paired_view_canvas(x, self.num_views)
        features = self.backbone(canvas)
        decoded = self.decoder(features)
        canvas_size = canvas.shape[2:]
        return {
            task_name: split_and_restack_view_logits(
                self.heads[task_name](decoded, output_size=canvas_size), self.num_views
            )
            for task_name in self.task_names
        }
