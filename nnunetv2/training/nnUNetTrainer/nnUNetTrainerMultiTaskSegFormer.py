import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class _DifferentialPolyLRScheduler(PolyLRScheduler):
    """PolyLRScheduler.step() sets every param group to the SAME decayed value, computed from one
    shared `initial_lr` (nnunetv2/training/lr_scheduler/polylr.py:18-20) - correct for nnU-Net's own
    single-LR SGD optimizer, but silently collapses a differential-LR optimizer's backbone/head
    groups to one rate on its very first step. Scales each group relative to its own starting LR
    instead, so the backbone/head ratio this project's pretrained-backbone convention requires
    (repo/segformer_multitask/src/defaults.py: backbone 1e-5, head 1e-4) survives the decay."""

    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, self._base_lrs[0], max_steps, exponent, current_step)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        decay = (1 - current_step / self.max_steps) ** self.exponent
        for param_group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            param_group["lr"] = base_lr * decay
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class nnUNetTrainerMultiTaskSegFormer(nnUNetTrainerMultiTask):
    """SegFormer hosted inside nnU-Net's multi-task trainer. Subclasses nnUNetTrainerMultiTask (not
    the plain base) so augmentation, loss weighting, deep-supervision-level counting, and paired-
    view target splitting are all inherited unchanged - see
    context/experiments/nnunet-hosted-segformer-trainer-plan.md for the full design. Only overrides
    the architecture-appropriate exceptions: deep supervision off (SegFormer's decode head has no
    multi-resolution output) and AdamW with a differential backbone/head learning rate, matching
    this project's own convention for fine-tuning a pretrained backbone
    (repo/segformer_multitask/src/defaults.py) rather than nnU-Net's from-scratch SGD default.
    build_network_architecture is not overridden: the inherited pass-through already dispatches via
    configuration_manager.network_arch_class_name to whichever MultiTaskDualHeadSegFormer/etc. class
    the plans file names.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.enable_deep_supervision = False
        self.backbone_lr = 1e-5
        self.initial_lr = 1e-4
        self.weight_decay = 0.01

    def configure_optimizers(self):
        backbone_params = [p for name, p in self.network.named_parameters() if "backbone" in name]
        other_params = [p for name, p in self.network.named_parameters() if "backbone" not in name]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.backbone_lr},
                {"params": other_params, "lr": self.initial_lr},
            ],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )
        lr_scheduler = _DifferentialPolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, lr_scheduler
