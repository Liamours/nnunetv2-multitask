from typing import Dict, List, Optional

import torch

from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform


class ConvertMultiTaskSegmentationToRegionsTransform(SegOnlyTransform):
    """Per-task replacement for the stock ConvertSegmentationToRegionsTransform, which is hardcoded
    to read only channel 0 - correct for a single task, wrong for multi-task multichannel targets.

    For each task, in task_order:
    - multichannel task: its raw channels are already independent per-class {0,1} (or {-1,0,1} before
      RemoveLabelTansform strips -1) - binarize (>0) and pass through, no derivation needed.
    - region-derived task (1 raw channel, tuple-valued labels + regions_class_order): derive regions
      via torch.isin, same logic the stock transform used, just scoped to this task's own channel.
    - plain multi-class task (no regions): pass its single raw channel through unchanged, for
      RobustCrossEntropyLoss to consume as integer class indices.
    """

    def __init__(self, task_order: List[str], task_raw_channel_slices: Dict[str, slice],
                 task_is_multichannel: Dict[str, bool],
                 task_regions: Dict[str, Optional[List]]):
        super().__init__()
        self.task_order = task_order
        self.task_raw_channel_slices = task_raw_channel_slices
        self.task_is_multichannel = task_is_multichannel
        self.task_regions = {
            k: [torch.tensor(r) if isinstance(r, (list, tuple)) else torch.tensor([r]) for r in v]
            for k, v in task_regions.items() if v is not None
        }

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        outputs = []
        for task_name in self.task_order:
            raw = segmentation[self.task_raw_channel_slices[task_name]]

            if self.task_is_multichannel.get(task_name, False):
                outputs.append((raw > 0).to(raw.dtype))
            elif task_name in self.task_regions:
                regions = self.task_regions[task_name]
                region_out = torch.zeros((len(regions), *raw.shape[1:]), dtype=raw.dtype)
                for i, r in enumerate(regions):
                    region_out[i] = torch.isin(raw[0], r)
                outputs.append(region_out)
            else:
                outputs.append(raw)

        return torch.cat(outputs, dim=0)
