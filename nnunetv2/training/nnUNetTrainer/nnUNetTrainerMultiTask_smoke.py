from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTask_smoke(nnUNetTrainerMultiTask):
    """Smoke-test trainer: verifies a run works end to end before committing to the real
    100-epoch config. 2 epochs at nnU-Net's standard 250 train / 50 val iterations per epoch,
    the real per-iteration cost, not full dataset coverage. Saves checkpoint_latest.pth every
    epoch (save_every=1, the base class default of 50 would never trigger within 2 epochs) so
    pause/resume can actually be exercised, not just assumed.
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        import torch

        super().__init__(plans, configuration, fold, dataset_json,
                          device=device if device is not None else torch.device("cuda"))
        self.num_epochs = 2
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.save_every = 1
