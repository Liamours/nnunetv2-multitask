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
from nnunetv2.training.data_augmentation.custom_transforms.multitask_region_transform import (
    ConvertMultiTaskSegmentationToRegionsTransform,
)
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
        for task_name in self.task_names:
            self.logger.local_logger.my_fantastic_logging.setdefault(f"mean_fg_dice__{task_name}", [])
        self.multitask_config = self.configuration_manager.multitask_config
        self.multitask_dataset_config = self.dataset_json.get("multitask", {})
        self.views = list(self.multitask_dataset_config.get("views", []))
        self.is_paired_multiview = self.multitask_dataset_config.get("case_unit") == "paired_anterior_posterior"
        self._validate_task_config_matches_labels()
        self.task_raw_channel_slices = self._compute_task_raw_channel_slices()
        # when self.label_manager.has_regions, ConvertMultiTaskSegmentationToRegionsTransform runs
        # in the dataloader and re-shapes the target from raw channels to num_segmentation_heads
        # channels per task, in the same task_order - _split_targets must slice by whichever shape
        # actually reached it.
        self.task_output_channel_slices = self._compute_task_output_channel_slices()
        self.task_loss_weights = {
            task.get("name", f"task{i + 1}"): float(task.get("loss_weight", 1.0))
            for i, task in enumerate(self.multitask_config["tasks"])
        }

    def _validate_task_config_matches_labels(self):
        plan_heads = {
            task.get("name", f"task{i + 1}"): int(task["num_classes"])
            for i, task in enumerate(self.multitask_config["tasks"])
        }
        label_heads = self.label_manager.task_num_segmentation_heads()
        if set(plan_heads) != set(label_heads):
            raise ValueError(
                "Multitask plans and dataset_json define different task names. "
                f"Plans: {sorted(plan_heads)}. dataset_json: {sorted(label_heads)}."
            )
        mismatched_heads = {
            task_name: (plan_heads[task_name], label_heads[task_name])
            for task_name in plan_heads
            if plan_heads[task_name] != label_heads[task_name]
        }
        if mismatched_heads:
            raise ValueError(
                "Multitask plans num_classes must match dataset_json task label heads. "
                f"Mismatches: {mismatched_heads}."
            )

    def _compute_task_raw_channel_slices(self) -> Dict[str, slice]:
        """Non-paired raw target tensor is task channels concatenated in task_order; each task
        occupies task_num_raw_channels() consecutive channels (1 for standard tasks, N for
        multichannel tasks whose raw storage already is N independent per-class channels)."""
        raw_channels = self.label_manager.task_num_raw_channels()
        slices = {}
        offset = 0
        for task_name in self.task_names:
            n = raw_channels[task_name]
            slices[task_name] = slice(offset, offset + n)
            offset += n
        return slices

    def _compute_task_output_channel_slices(self) -> Dict[str, slice]:
        """Channel layout after ConvertMultiTaskSegmentationToRegionsTransform has run: a task with
        has_regions (multichannel, or genuinely region-derived) occupies num_segmentation_heads
        channels; a plain CE task is left at 1 channel (its raw integer class map, unchanged - CE
        consumes that directly, never one-hot). Task order is preserved."""
        heads = self.label_manager.task_num_segmentation_heads()
        slices = {}
        offset = 0
        for task_name in self.task_names:
            has_regions = self.label_manager.get_task_label_manager(task_name).has_regions
            n = heads[task_name] if has_regions else 1
            slices[task_name] = slice(offset, offset + n)
            offset += n
        return slices

    def _target_channel_slices(self) -> Dict[str, slice]:
        # has_regions => the custom region transform already ran in the dataloader and reshaped the
        # target to num_segmentation_heads channels per task; otherwise the target is still raw
        # (e.g. BS-80K, where no task is multichannel and has_regions is False - untouched behaviour).
        return self.task_output_channel_slices if self.label_manager.has_regions else self.task_raw_channel_slices

    def _split_targets(self, target):
        if self.is_paired_multiview:
            return self._split_paired_multiview_targets(target)
        slices = self._target_channel_slices()
        if isinstance(target, list):
            return {
                task_name: [level[:, slices[task_name]].contiguous() for level in target]
                for task_name in self.task_names
            }
        return {
            task_name: target[:, slices[task_name]].contiguous()
            for task_name in self.task_names
        }

    def _split_paired_multiview_targets(self, target):
        num_views = len(self.views)
        if num_views < 1:
            raise ValueError("Paired multitask datasets require dataset_json['multitask']['views'].")

        def split_level(level):
            expected_channels = len(self.task_names) * num_views
            if level.shape[1] != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} target channels for {len(self.task_names)} tasks x "
                    f"{num_views} views, got {level.shape[1]}."
                )
            return {
                task_name: level[:, task_idx * num_views:(task_idx + 1) * num_views].contiguous()
                for task_idx, task_name in enumerate(self.task_names)
            }

        if isinstance(target, list):
            per_level = [split_level(level) for level in target]
            return {
                task_name: [level_targets[task_name] for level_targets in per_level]
                for task_name in self.task_names
            }
        return split_level(target)

    def _reshape_paired_multiview_for_loss(self, task_name: str, output, target):
        if not self.is_paired_multiview:
            return output, target

        num_views = len(self.views)
        num_classes = self.label_manager.get_task_label_manager(task_name).num_segmentation_heads

        def reshape_level(out_level, tgt_level):
            expected_channels = num_views * num_classes
            if out_level.shape[1] != expected_channels:
                raise ValueError(
                    f"Task {task_name} expected {expected_channels} output channels "
                    f"({num_views} views x {num_classes} classes), got {out_level.shape[1]}."
                )
            spatial_shape = out_level.shape[2:]
            out_level = out_level.reshape(out_level.shape[0], num_views, num_classes, *spatial_shape)
            out_level = out_level.reshape(out_level.shape[0] * num_views, num_classes, *spatial_shape)
            tgt_level = tgt_level.reshape(tgt_level.shape[0] * num_views, 1, *tgt_level.shape[2:])
            return out_level, tgt_level

        if isinstance(output, list):
            reshaped_outputs = []
            reshaped_targets = []
            for out_level, tgt_level in zip(output, target):
                out_level, tgt_level = reshape_level(out_level, tgt_level)
                reshaped_outputs.append(out_level)
                reshaped_targets.append(tgt_level)
            return reshaped_outputs, reshaped_targets
        return reshape_level(output, target)

    def _reshape_all_outputs_targets_for_loss(self, output, target):
        if not self.is_paired_multiview:
            return output, target
        reshaped_outputs = {}
        reshaped_targets = {}
        for task_name in self.task_names:
            reshaped_outputs[task_name], reshaped_targets[task_name] = self._reshape_paired_multiview_for_loss(
                task_name,
                output[task_name],
                target[task_name],
            )
        return reshaped_outputs, reshaped_targets

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
            # Levels come from the instantiated network, not from the pooling geometry alone: the
            # partial-decoder fission variants (MultiTaskEarlyMidUNet/MultiTaskMidUNet) emit fewer
            # deep-supervision outputs than the full pooling depth, and a weight list normalized
            # against the wrong count silently under-weights the loss instead of erroring. Falls back
            # to the old pooling-derived count when the network isn't built yet (e.g. a trainer used
            # only to exercise loss selection, never assigned a `.network`).
            mod = self.network.module if self.is_ddp else self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            if mod is not None and hasattr(mod, "deep_supervision_num_levels"):
                num_levels = mod.deep_supervision_num_levels()
            else:
                num_levels = len(self._get_deep_supervision_scales())
            weights = np.array([1 / (2 ** i) for i in range(num_levels)], dtype=np.float64)
            # A single level (Mid Fission's tail is one decoder stage) has nothing coarser to
            # de-emphasize - zeroing "the last" weight would zero the only one and divide 0/0 below.
            if num_levels > 1:
                if self.is_ddp and not self._do_i_compile():
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0
            deep_supervision_weights = (weights / weights.sum()).tolist()

        return MultiTaskLoss(task_losses, self.task_loss_weights, deep_supervision_weights)

    def _task_regions_for_transform(self) -> Dict[str, List]:
        """regions_class_order per task that has_regions but is NOT multichannel (i.e. a genuinely
        derived-region task, one raw channel expanded into several via label tuples). Not used by any
        current dataset - kept so the mechanism stays structurally correct if one is ever added."""
        result = {}
        for task_name in self.task_names:
            manager = self.label_manager.get_task_label_manager(task_name)
            if manager.has_regions and not self.label_manager.is_multichannel_task(task_name):
                result[task_name] = manager.all_regions
        return result

    def get_training_transforms(self, *args, **kwargs):
        # never let the stock single-channel-0 region transform run for multi-task data - build our
        # own per-task version instead (see ConvertMultiTaskSegmentationToRegionsTransform docstring).
        kwargs["regions"] = None
        composed = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        if self.label_manager.has_regions:
            composed.transforms.append(ConvertMultiTaskSegmentationToRegionsTransform(
                task_order=self.task_names,
                task_raw_channel_slices=self.task_raw_channel_slices,
                task_is_multichannel={t: self.label_manager.is_multichannel_task(t) for t in self.task_names},
                task_regions=self._task_regions_for_transform(),
            ))
        return composed

    def get_validation_transforms(self, *args, **kwargs):
        kwargs["regions"] = None
        composed = nnUNetTrainer.get_validation_transforms(*args, **kwargs)
        if self.label_manager.has_regions:
            composed.transforms.append(ConvertMultiTaskSegmentationToRegionsTransform(
                task_order=self.task_names,
                task_raw_channel_slices=self.task_raw_channel_slices,
                task_is_multichannel={t: self.label_manager.is_multichannel_task(t) for t in self.task_names},
                task_regions=self._task_regions_for_transform(),
            ))
        return composed

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
            output_for_loss, target_for_loss = self._reshape_all_outputs_targets_for_loss(output, target)
            loss = self.loss(output_for_loss, target_for_loss)

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
            output_for_loss, target_for_loss = self._reshape_all_outputs_targets_for_loss(output, target)
            loss = self.loss(output_for_loss, target_for_loss)

        result = {"loss": loss.detach().cpu().numpy()}
        for task_name in self.task_names:
            task_output = self._get_task_output_for_metrics(output_for_loss, task_name)
            task_target = self._get_task_target_for_metrics(target_for_loss, task_name)
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
