from batchgenerators.utilities.file_and_folder_operations import join


class SaveLatestEveryEpochMixin:
    """Force checkpoint_latest.pth to track the last completed epoch."""

    def on_epoch_end(self):
        super().on_epoch_end()
        current_epoch_finished = self.current_epoch - 1
        if current_epoch_finished >= 0:
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
