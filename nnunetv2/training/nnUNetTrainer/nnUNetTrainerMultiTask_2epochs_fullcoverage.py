import math

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTask_2epochs_fullcoverage(nnUNetTrainerMultiTask):
    """2-epoch smoke test where every epoch actually iterates over the full train/val split at least
    once, unlike nnUNetTrainerMultiTask_2epochs (which hardcodes 1 iteration/epoch - a pipeline check,
    not a coverage check). Iteration counts are derived from the real split size, so this adapts to
    whatever dataset it's pointed at instead of a fixed constant.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 2
        tr_keys, val_keys = self.do_split()
        batch_size = self.configuration_manager.batch_size
        self.num_iterations_per_epoch = max(1, math.ceil(len(tr_keys) / batch_size))
        self.num_val_iterations_per_epoch = max(1, math.ceil(len(val_keys) / batch_size))
