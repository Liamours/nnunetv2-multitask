from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction_multitask import export_prediction_from_logits_multitask
from nnunetv2.inference.predictor_multitask import nnUNetMultiTaskPredictor
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.loss.multitask_losses import MultiTaskLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.multitask_label_handling import MultiTaskLabelManager


class nnUNetTrainerMultiTask(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        if not isinstance(self.label_manager, MultiTaskLabelManager):
            raise TypeError("nnUNetTrainerMultiTask requires MultiTaskLabelManager.")
        self.task_names = self.label_manager.task_order
        self.multitask_config = self.configuration_manager.multitask_config
        self.task_loss_weights = {
            task.get("name", f"task{i + 1}"): float(task.get("loss_weight", 1.0))
            for i, task in enumerate(self.multitask_config["tasks"])
        }

    def _split_targets(self, target):
        if isinstance(target, list):
            return {
                task_name: [level[:, idx:idx + 1].contiguous() for level in target]
                for idx, task_name in enumerate(self.task_names)
            }
        return {
            task_name: target[:, idx:idx + 1].contiguous()
            for idx, task_name in enumerate(self.task_names)
        }

    def _get_task_output_for_metrics(self, output: Dict[str, torch.Tensor], task_name: str):
        task_output = output[task_name]
        if self.enable_deep_supervision:
            return task_output[0]
        return task_output

    def _get_task_target_for_metrics(self, target: Dict[str, torch.Tensor], task_name: str):
        task_target = target[task_name]
        if self.enable_deep_supervision:
            return task_target[0]
        return task_target

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels: int, num_output_channels: int, enable_deep_supervision: bool = True):
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

    def _build_loss(self):
        task_losses = {}
        for task_name in self.task_names:
            task_label_manager = self.label_manager.get_task_label_manager(task_name)
            if task_label_manager.has_regions:
                task_loss = DC_and_BCE_loss(
                    {},
                    {
                        "batch_dice": self.configuration_manager.batch_dice,
                        "do_bg": True,
                        "smooth": 1e-5,
                        "ddp": self.is_ddp,
                    },
                    use_ignore_label=task_label_manager.ignore_label is not None,
                    dice_class=MemoryEfficientSoftDiceLoss,
                )
            else:
                task_loss = DC_and_CE_loss(
                    {
                        "batch_dice": self.configuration_manager.batch_dice,
                        "smooth": 1e-5,
                        "do_bg": False,
                        "ddp": self.is_ddp,
                    },
                    {},
                    weight_ce=1,
                    weight_dice=1,
                    ignore_label=task_label_manager.ignore_label,
                    dice_class=MemoryEfficientSoftDiceLoss,
                )
            if self._do_i_compile():
                task_loss.dc = torch.compile(task_loss.dc)
            task_losses[task_name] = task_loss

        deep_supervision_weights = None
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float64)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            deep_supervision_weights = (weights / weights.sum()).tolist()

        return MultiTaskLoss(task_losses, self.task_loss_weights, deep_supervision_weights)

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if hasattr(mod, "set_deep_supervision"):
            mod.set_deep_supervision(enabled)
        else:
            super().set_deep_supervision_enabled(enabled)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        target = self._split_targets(target)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        target = self._split_targets(target)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        result = {"loss": loss.detach().cpu().numpy()}
        for task_name in self.task_names:
            task_output = self._get_task_output_for_metrics(output, task_name)
            task_target = self._get_task_target_for_metrics(target, task_name)
            task_label_manager = self.label_manager.get_task_label_manager(task_name)
            axes = [0] + list(range(2, task_output.ndim))

            if task_label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(task_output) > 0.5).long()
            else:
                output_seg = task_output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(task_output.shape, device=task_output.device, dtype=torch.float16)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)

            mask = None
            if task_label_manager.has_ignore_label:
                mask = (task_target != task_label_manager.ignore_label).float()
                task_target = torch.where(mask.bool(), task_target, torch.zeros_like(task_target))

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, task_target, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not task_label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
            result[f"tp_hard__{task_name}"] = tp_hard
            result[f"fp_hard__{task_name}"] = fp_hard
            result[f"fn_hard__{task_name}"] = fn_hard
        return result

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        if self.is_ddp:
            losses_val = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        all_task_dice = []
        for task_name in self.task_names:
            tp = np.sum(outputs_collated[f"tp_hard__{task_name}"], 0)
            fp = np.sum(outputs_collated[f"fp_hard__{task_name}"], 0)
            fn = np.sum(outputs_collated[f"fn_hard__{task_name}"], 0)
            if self.is_ddp:
                world_size = torch.distributed.get_world_size()
                gathered = []
                for arr in (tp, fp, fn):
                    bucket = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(bucket, arr)
                    gathered.append(np.vstack([i[None] for i in bucket]).sum(0))
                tp, fp, fn = gathered
            task_dc = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
            self.logger.log(f"mean_fg_dice__{task_name}", np.nanmean(task_dc), self.current_epoch)
            all_task_dice.extend(task_dc)

        self.logger.log("mean_fg_dice", np.nanmean(all_task_dice), self.current_epoch)
        self.logger.log("dice_per_class_or_region", all_task_dice, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetMultiTaskPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        validation_output_folder = join(self.output_folder, "validation")
        maybe_mkdir_p(validation_output_folder)
        for task_name in self.task_names:
            maybe_mkdir_p(join(validation_output_folder, task_name))

        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )

        for k in dataset_val.identifiers:
            self.print_to_log_file(f"predicting {k}")
            data, _, seg_prev, properties = dataset_val.load_case(k)
            data = data[:]
            if self.is_cascaded:
                raise NotImplementedError("Cascaded multi-task validation is not implemented in v1.")
            with torch.no_grad():
                data = torch.from_numpy(data)
                prediction = predictor.predict_sliding_window_return_logits(data)
            export_prediction_from_logits_multitask(
                prediction,
                properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                join(validation_output_folder, k),
                save_probabilities=save_probabilities,
            )

        for task_name in self.task_names:
            gt_folder = join(self.preprocessed_dataset_folder_base, f"gt_segmentations__{task_name}")
            pred_folder = join(validation_output_folder, task_name)
            if isdir(gt_folder):
                metrics = compute_metrics_on_folder(
                    gt_folder,
                    pred_folder,
                    join(pred_folder, "summary.json"),
                    self.plans_manager.image_reader_writer_class(),
                    self.dataset_json["file_ending"],
                    self.label_manager.get_task_label_manager(task_name).foreground_labels,
                    self.label_manager.get_task_label_manager(task_name).ignore_label,
                    chill=True,
                )
                self.logger.log_summary(f"final_val/{task_name}_foreground_dice", metrics["foreground_mean"]["Dice"])

        self.set_deep_supervision_enabled(True)
