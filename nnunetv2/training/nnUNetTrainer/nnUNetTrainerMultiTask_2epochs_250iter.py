import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTask_2epochs_250iter(nnUNetTrainerMultiTask):
    """Smoke-test trainer: 2 epochs at nnU-Net's standard 250 train / 50 val iterations per epoch -
    the real iteration count, not full dataset coverage, per project decision to keep smoke tests and
    real training on one consistent iterations/epoch convention across all datasets.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
