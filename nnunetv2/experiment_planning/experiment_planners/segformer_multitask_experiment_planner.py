from copy import deepcopy
from typing import List, Optional, Tuple, Union

from nnunetv2.experiment_planning.experiment_planners.multitask_experiment_planner import MultiTaskExperimentPlanner

_SEGFORMER_MULTITASK_VARIANT_TO_CLASS = {
    "dual_head": "nnunetv2.architecture.multitask_segformer.MultiTaskDualHeadSegFormer",
    # dual_decoder / dual_fuse added once those network classes are implemented for real, see
    # context/experiments/nnunet-hosted-segformer-trainer-plan.md.
}


class SegFormerMultiTaskExperimentPlanner(MultiTaskExperimentPlanner):
    """Plans a SegFormer network for nnUNetTrainerMultiTaskSegFormer, reusing
    MultiTaskExperimentPlanner's task/label bookkeeping (_tasks_with_output_channels,
    plan_experiment) unchanged and overriding only the two methods that are UNet-pooling-specific:
    architecture construction and the patch-size/batch-size search. SegFormer's fixed-shape MiT
    backbone has no pooling depth to search over, so this uses a fixed patch size and batch size
    instead of ExperimentPlanner's VRAM-search loop (that loop, and the VRAM estimator it depends
    on, are documented as UNet-specific and require compute_conv_feature_map_size, which this
    architecture family does not implement).
    """

    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        multitask_variant: str = "dual_head",
        multitask_tasks: List[dict] = None,
        mit_variant: str = "mit_b2",
        pretrained_hf_name: Optional[str] = "nvidia/mit-b2",
        decoder_embedding_dim: int = 768,
        fixed_patch_size: Tuple[int, int] = (896, 256),
        fixed_batch_size: int = 4,
        preprocessor_name: str = "MultiTaskPreprocessor",
        plans_name: str = "nnUNetPlansMultiTaskSegFormer",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            multitask_variant=multitask_variant,
            multitask_tasks=multitask_tasks,
            preprocessor_name=preprocessor_name,
            plans_name=plans_name,
            overwrite_target_spacing=overwrite_target_spacing,
            suppress_transpose=suppress_transpose,
        )
        self.mit_variant = mit_variant
        self.pretrained_hf_name = pretrained_hf_name
        self.decoder_embedding_dim = decoder_embedding_dim
        self.fixed_patch_size = tuple(fixed_patch_size)
        self.fixed_batch_size = fixed_batch_size

    def _make_multitask_architecture(self, architecture_kwargs: dict) -> dict:
        architecture_kwargs = deepcopy(architecture_kwargs)
        try:
            architecture_kwargs["network_class_name"] = _SEGFORMER_MULTITASK_VARIANT_TO_CLASS[self.multitask_variant]
        except KeyError:
            raise ValueError(
                f"Unknown multitask_variant {self.multitask_variant!r} for SegFormer. "
                f"Known: {sorted(_SEGFORMER_MULTITASK_VARIANT_TO_CLASS)}."
            ) from None
        architecture_kwargs["arch_kwargs"] = {
            "mit_variant": self.mit_variant,
            "pretrained_hf_name": self.pretrained_hf_name,
            "decoder_embedding_dim": self.decoder_embedding_dim,
            "multitask": {
                "variant": self.multitask_variant,
                "tasks": deepcopy(self._tasks_with_output_channels()),
            },
        }
        architecture_kwargs["_kw_requires_import"] = []
        return architecture_kwargs

    def _recompute_multitask_memory_plan(
        self,
        plan: dict,
        spacing,
        median_shape,
        approximate_n_voxels_dataset,
        _cache,
    ) -> dict:
        plan["architecture"] = self._make_multitask_architecture(plan["architecture"])
        plan["patch_size"] = list(self.fixed_patch_size)
        plan["batch_size"] = self.fixed_batch_size
        return plan
