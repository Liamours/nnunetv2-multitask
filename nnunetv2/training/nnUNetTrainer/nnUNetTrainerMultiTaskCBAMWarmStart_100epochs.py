"""Warm-start the shared encoder/decoder and lesion head from the already-working single-task CBAM
post_norm checkpoint (nnunetcbam_single, Dataset261), leaving the bone head at random init.

Rationale (root-cause investigation 2026-09-04, see the plan's CBAM section): three instrumented
5-epoch diagnostic runs showed the multitask Late Fission lesion collapse is a smooth, fast (complete
within ~140 training steps, well inside epoch 0), textbook slide into predicting background
everywhere - not an unstable or broken run, and the lesion head's gradient never dies, it just shrinks
as the prediction saturates, the normal shape of softmax gradients near an extreme. A bare reroll is
poorly positioned to escape this specific attractor, since the same severe class-imbalance pressure
would push any random initialization the same direction early on. Single-task CBAM, trained with the
identical post_norm fix on the identical lesion data, already escaped this same attractor (by ~epoch
89, twice, reproducibly) - so instead of re-deriving that climb from scratch inside the harder
multitask setting (where a shared decoder also has to serve bone's competing gradient), this starts
the lesion-relevant weights from where single-task already got to.

Confirmed by direct config comparison before writing this: nnUNetPlansA1ControlledBatch4CBAMPostNorm
(single-task, Dataset261) and nnUNetPlansMultiTaskCBAMPostNorm (this run, Dataset260) use the exact
same architecture class (MultiTaskDualHeadUNet - single-task is just a one-task instance of the same
multitask architecture, not a different network), the same patch_size, n_stages, features_per_stage,
strides, and cbam config including post_norm. Every encoder.*, encoder_attention.*, decoder.*, and
heads.lesion.* parameter should therefore match by name and shape; heads.bone.* simply does not exist
in the single-task checkpoint (it never had a bone head) and stays at fresh random init. This is
verified at runtime below (not just assumed): every skipped key is checked to start with "heads.bone."
and the run aborts loudly if anything else is missing or shape-mismatched, rather than silently
warm-starting only part of what was intended.

nnU-Net's own `-pretrained_weights` CLI flag / load_pretrained_weights.py was not used: it asserts
every target-model key (except .seg_layers.) exists in the source checkpoint, which heads.bone.*
never will here - a correct restriction for its own intended use (matching architectures), wrong for
this deliberately-partial transfer, so this file implements its own permissive merge instead of
editing that shared utility.
"""
from __future__ import annotations

from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask_100epochs import nnUNetTrainerMultiTask_100epochs

_SINGLE_TASK_CHECKPOINT = (
    Path(__file__).resolve().parents[5] / "weights" / "nnunet" / "Dataset261_BS80KLesionOnly"
    / "nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1ControlledBatch4CBAMPostNorm__2d"
    / "fold_0" / "checkpoint_final.pth"
)


class nnUNetTrainerMultiTaskCBAMWarmStart_100epochs(nnUNetTrainerMultiTask_100epochs):
    def initialize(self):
        super().initialize()
        self._warm_start_from_single_task()

    def _warm_start_from_single_task(self):
        if not _SINGLE_TASK_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Warm-start source checkpoint not found: {_SINGLE_TASK_CHECKPOINT}")
        saved = torch.load(_SINGLE_TASK_CHECKPOINT, map_location=self.device, weights_only=False)
        pretrained = saved["network_weights"]
        model_dict = self.network.state_dict()

        matched, missing, shape_mismatched = [], [], []
        for key in model_dict:
            if key not in pretrained:
                missing.append(key)
                continue
            if model_dict[key].shape != pretrained[key].shape:
                shape_mismatched.append(key)
                continue
            model_dict[key] = pretrained[key]
            matched.append(key)

        unexpected_missing = [k for k in missing if not k.startswith("heads.bone.")]
        if unexpected_missing or shape_mismatched:
            raise RuntimeError(
                "Warm-start expected only heads.bone.* to be absent from the single-task checkpoint "
                f"and every other key to match by shape. Unexpected missing: {unexpected_missing}. "
                f"Shape-mismatched: {shape_mismatched}."
            )

        self.network.load_state_dict(model_dict)
        self.print_to_log_file(
            f"[warm_start] loaded {len(matched)}/{len(model_dict)} parameters from "
            f"{_SINGLE_TASK_CHECKPOINT}; {len(missing)} left at random init (bone head, expected)."
        )
