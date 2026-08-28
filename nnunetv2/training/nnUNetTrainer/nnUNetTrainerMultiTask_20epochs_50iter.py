import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTask_20epochs_50iter(nnUNetTrainerMultiTask):
    """20 epochs at 50 train / 25 val iterations per epoch - a community-precedented reduction from
    nnU-Net's default 250/50, chosen for IDRiD (64 train cases) and BUSI (266 train cases) to avoid
    excessive same-image repetition at the default convention while keeping a fixed, dataset-size-
    independent iteration count (unlike nnUNetTrainerMultiTask_2epochs_fullcoverage's scaled approach).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 20
        self.num_iterations_per_epoch = 50
        self.num_val_iterations_per_epoch = 25
