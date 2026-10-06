from typing import Dict, List, Union

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, task_losses: Dict[str, nn.Module], task_weights: Dict[str, float], deep_supervision_weights=None):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)
        self.task_weights = task_weights
        self.deep_supervision_weights = deep_supervision_weights
        # Diagnostic side channel only, read by nnUNetTrainerMultiTaskCBAMDiagnostic_5epochs - the
        # unweighted per-task scalar is not otherwise observable once summed into the combined loss
        # returned below. Written every forward call, harmless when nothing reads it.
        self.last_task_losses: Dict[str, float] = {}

    def _compute_task_loss(self, task_name: str, output, target):
        task_loss = self.task_losses[task_name]
        if isinstance(output, list):
            assert isinstance(target, list), "Deep-supervision output requires list targets."
            assert self.deep_supervision_weights is not None
            loss = torch.zeros((), device=output[0].device, dtype=output[0].dtype)
            for weight, out_level, tgt_level in zip(self.deep_supervision_weights, output, target):
                if weight == 0:
                    continue
                loss = loss + (weight * task_loss(out_level, tgt_level))
            return loss
        return task_loss(output, target)

    def forward(self, outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], targets: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        loss = None
        for task_name, output in outputs.items():
            unweighted = self._compute_task_loss(task_name, output, targets[task_name])
            self.last_task_losses[task_name] = float(unweighted.detach())
            task_loss = unweighted * self.task_weights[task_name]
            loss = task_loss if loss is None else loss + task_loss
        return loss
