import torch

from nnunetv2.training.nnUNetTrainer.checkpointing import SaveLatestEveryEpochMixin
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_50epochs(SaveLatestEveryEpochMixin, nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 50
