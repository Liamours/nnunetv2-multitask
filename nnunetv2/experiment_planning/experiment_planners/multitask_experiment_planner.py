from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np

from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.utilities.multitask_dataset import is_paired_multiview_multitask_dataset


class MultiTaskExperimentPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        multitask_variant: str = "dual_head",
        multitask_tasks: List[dict] = None,
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "MultiTaskPreprocessor",
        plans_name: str = "nnUNetPlansMultiTask",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
        cbam: dict = None,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb=gpu_memory_target_in_gb,
            preprocessor_name=preprocessor_name,
            plans_name=plans_name,
            overwrite_target_spacing=overwrite_target_spacing,
            suppress_transpose=suppress_transpose,
        )
        # Upstream planners clamp to batch size >= 2. That is too aggressive for heavier multitask variants
        # on small GPUs and can still yield invalid plans after we correct the multitask VRAM estimate.
        self.UNet_min_batch_size = 1
        self.multitask_variant = multitask_variant
        self.cbam = cbam or {"enabled": False}
        if is_paired_multiview_multitask_dataset(self.dataset_json) and self.preprocessor_name == "DefaultPreprocessor":
            self.preprocessor_name = "MultiTaskPreprocessor"
        self.multitask_tasks = multitask_tasks or [
            {"name": "task1", "num_classes": 2, "loss_weight": 1.0},
            {"name": "task2", "num_classes": 2, "loss_weight": 1.0},
        ]
        self.multitask_views = self.dataset_json.get("multitask", {}).get("views", [])

    def _make_multitask_architecture(self, architecture_kwargs: dict) -> dict:
        architecture_kwargs = deepcopy(architecture_kwargs)
        architecture_kwargs["network_class_name"] = (
            "nnunetv2.architecture.multitask_unet.MultiTaskDualHeadUNet"
            if self.multitask_variant == "dual_head"
            else "nnunetv2.architecture.multitask_unet.MultiTaskDualDecoderUNet"
        )
        architecture_kwargs["arch_kwargs"]["multitask"] = {
            "variant": self.multitask_variant,
            "tasks": deepcopy(self._tasks_with_output_channels()),
        }
        architecture_kwargs["arch_kwargs"]["cbam"] = deepcopy(self.cbam)
        return architecture_kwargs

    def _tasks_with_output_channels(self) -> List[dict]:
        tasks = deepcopy(self.multitask_tasks)
        if is_paired_multiview_multitask_dataset(self.dataset_json):
            num_views = len(self.multitask_views)
            for task in tasks:
                task["output_channels"] = int(task["num_classes"]) * num_views
        return tasks

    def _recompute_multitask_memory_plan(
        self,
        plan: dict,
        spacing,
        median_shape,
        approximate_n_voxels_dataset,
        _cache,
    ) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple(
                [min(max_num_features, self.UNet_base_num_features * 2 ** i) for i in range(num_stages)]
            )

        def _keygen(patch_size, strides):
            return str(list(patch_size)) + "_" + str(strides)

        spacing = np.array(spacing)
        median_shape = np.array(median_shape)
        patch_size = np.array(plan["patch_size"]).astype(int)
        num_input_channels = len(
            self.dataset_json["channel_names"].keys()
            if "channel_names" in self.dataset_json
            else self.dataset_json["modality"].keys()
        )
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d

        architecture = self._make_multitask_architecture(plan["architecture"])

        def _update_architecture_for_patch(current_patch_size):
            _, pool_op_kernel_sizes, conv_kernel_sizes, adjusted_patch_size, shape_must_be_divisible_by = (
                get_pool_and_conv_props(
                    spacing,
                    current_patch_size,
                    self.UNet_featuremap_min_edge_length,
                    999999,
                )
            )
            num_stages = len(pool_op_kernel_sizes)
            architecture["arch_kwargs"].update(
                {
                    "n_stages": num_stages,
                    "kernel_sizes": conv_kernel_sizes,
                    "strides": pool_op_kernel_sizes,
                    "features_per_stage": _features_per_stage(num_stages, max_num_features),
                    "n_conv_per_stage": self.UNet_blocks_per_stage_encoder[:num_stages],
                    "n_conv_per_stage_decoder": self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                    "conv_op": convert_dim_to_conv_op(len(spacing)).__module__
                    + "."
                    + convert_dim_to_conv_op(len(spacing)).__name__,
                }
            )
            return np.array(adjusted_patch_size).astype(int), pool_op_kernel_sizes, shape_must_be_divisible_by

        patch_size, pool_op_kernel_sizes, shape_must_be_divisible_by = _update_architecture_for_patch(patch_size)
        estimate = self.static_estimate_VRAM_usage(
            tuple(int(i) for i in patch_size),
            num_input_channels,
            len(self.dataset_json["labels"].keys()),
            architecture["network_class_name"],
            architecture["arch_kwargs"],
            architecture["_kw_requires_import"],
        )
        _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * (
            self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB
        )
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d

        while (estimate / ref_bs * 2) > reference:
            axis_to_be_reduced = np.argsort(
                [i / j for i, j in zip(patch_size, median_shape[: len(spacing)])]
            )[-1]

            tmp = list(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, new_shape_must_be_divisible_by = get_pool_and_conv_props(
                spacing,
                tmp,
                self.UNet_featuremap_min_edge_length,
                999999,
            )
            tmp[axis_to_be_reduced] -= new_shape_must_be_divisible_by[axis_to_be_reduced]

            patch_size, pool_op_kernel_sizes, shape_must_be_divisible_by = _update_architecture_for_patch(tmp)
            cache_key = _keygen(patch_size, pool_op_kernel_sizes)
            if cache_key in _cache:
                estimate = _cache[cache_key]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    tuple(int(i) for i in patch_size),
                    num_input_channels,
                    len(self.dataset_json["labels"].keys()),
                    architecture["network_class_name"],
                    architecture["arch_kwargs"],
                    architecture["_kw_requires_import"],
                )
                _cache[cache_key] = estimate

        batch_size = round((reference / estimate) * ref_bs)
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64)
        )
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        plan["architecture"] = architecture
        plan["patch_size"] = [int(i) for i in patch_size]
        plan["batch_size"] = int(batch_size)
        return plan

    def get_plans_for_configuration(
        self,
        spacing,
        median_shape,
        data_identifier,
        approximate_n_voxels_dataset,
        _cache,
    ) -> dict:
        plan = super().get_plans_for_configuration(
            spacing,
            median_shape,
            data_identifier,
            approximate_n_voxels_dataset,
            _cache,
        )
        return self._recompute_multitask_memory_plan(
            plan,
            spacing,
            median_shape,
            approximate_n_voxels_dataset,
            _cache,
        )

    def plan_experiment(self):
        plans = super().plan_experiment()
        plans["label_manager"] = "MultiTaskLabelManager"
        self.save_plans(plans)
        return plans
