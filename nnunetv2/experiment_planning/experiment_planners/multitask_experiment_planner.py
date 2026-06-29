from copy import deepcopy
from typing import List, Tuple, Union

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class MultiTaskExperimentPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        multitask_variant: str = "dual_head",
        multitask_tasks: List[dict] = None,
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetPlansMultiTask",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb=gpu_memory_target_in_gb,
            preprocessor_name=preprocessor_name,
            plans_name=plans_name,
            overwrite_target_spacing=overwrite_target_spacing,
            suppress_transpose=suppress_transpose,
        )
        self.multitask_variant = multitask_variant
        self.multitask_tasks = multitask_tasks or [
            {"name": "task1", "num_classes": 2, "loss_weight": 1.0},
            {"name": "task2", "num_classes": 2, "loss_weight": 1.0},
        ]

    def _make_multitask_architecture(self, architecture_kwargs: dict) -> dict:
        architecture_kwargs = deepcopy(architecture_kwargs)
        architecture_kwargs["network_class_name"] = (
            "nnunetv2.architecture.multitask_unet.MultiTaskDualHeadUNet"
            if self.multitask_variant == "dual_head"
            else "nnunetv2.architecture.multitask_unet.MultiTaskDualDecoderUNet"
        )
        architecture_kwargs["arch_kwargs"]["multitask"] = {
            "variant": self.multitask_variant,
            "tasks": deepcopy(self.multitask_tasks),
        }
        return architecture_kwargs

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
        plan["architecture"] = self._make_multitask_architecture(plan["architecture"])
        return plan

    def plan_experiment(self):
        plans = super().plan_experiment()
        plans["label_manager"] = "MultiTaskLabelManager"
        self.save_plans(plans)
        return plans
