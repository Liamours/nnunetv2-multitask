"""Warm-start dual_decoder (Early Fission) CBAM from the working single-task CBAM checkpoint.

Same rationale as nnUNetTrainerMultiTaskCBAMWarmStart_100epochs.py (Late Fission, dual_head): a
from-scratch CBAM multitask run is expected to face the same fast, structural slide into predicting
background everywhere that Late Fission showed, for the same class-imbalance reason (see that
file's docstring and the plan's root-cause investigation section) - so starting from single-task's
already-competent lesion weights instead of random init is the direct fix, not another blind
training attempt.

Key mapping differs from Late Fission's flat name-match, because dual_decoder's architecture shape
differs from single-task's. Verified directly against multitask_unet.py before writing this: both
single-task (MultiTaskDualHeadUNet with one task) and dual_decoder build every per-task decoder via
`self._make_decoder()` with no `stage_range` argument, which defaults to the full
`(0, n_stages_encoder - 1)` range - so single-task's one `decoder` and dual_decoder's `decoders.lesion`
are the *same* full-range FeatureUNetDecoder shape, not two different things that happen to share a
name. `decoders.lesion.X` in the target therefore matches `decoder.X` in the source under a prefix
swap. `heads.lesion.*` matches directly with no remap (both architectures use "lesion" as the same
ModuleDict key). `decoders.bone.*` and `heads.bone.*` have no source counterpart and stay at random
init, matching how the Late Fission warm-start left `heads.bone.*` untouched.
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
_TARGET_DECODER_PREFIX = "decoders.lesion."
_SOURCE_DECODER_PREFIX = "decoder."


class nnUNetTrainerMultiTaskDualDecoderCBAMWarmStart_100epochs(nnUNetTrainerMultiTask_100epochs):
    def initialize(self):
        super().initialize()
        self._warm_start_from_single_task()

    @staticmethod
    def _source_key(target_key: str) -> str:
        if target_key.startswith(_TARGET_DECODER_PREFIX):
            return _SOURCE_DECODER_PREFIX + target_key[len(_TARGET_DECODER_PREFIX):]
        return target_key

    def _warm_start_from_single_task(self):
        if not _SINGLE_TASK_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Warm-start source checkpoint not found: {_SINGLE_TASK_CHECKPOINT}")
        saved = torch.load(_SINGLE_TASK_CHECKPOINT, map_location=self.device, weights_only=False)
        pretrained = saved["network_weights"]
        model_dict = self.network.state_dict()

        matched, missing, shape_mismatched = [], [], []
        for key in model_dict:
            source_key = self._source_key(key)
            if source_key not in pretrained:
                missing.append(key)
                continue
            if model_dict[key].shape != pretrained[source_key].shape:
                shape_mismatched.append(key)
                continue
            model_dict[key] = pretrained[source_key]
            matched.append(key)

        unexpected_missing = [
            k for k in missing if not (k.startswith("decoders.bone.") or k.startswith("heads.bone."))
        ]
        if unexpected_missing or shape_mismatched:
            raise RuntimeError(
                "Warm-start expected only decoders.bone.*/heads.bone.* to be absent from the "
                f"single-task checkpoint and every other key to match by shape. Unexpected missing: "
                f"{unexpected_missing}. Shape-mismatched: {shape_mismatched}."
            )

        self.network.load_state_dict(model_dict)
        self.print_to_log_file(
            f"[warm_start] loaded {len(matched)}/{len(model_dict)} parameters from "
            f"{_SINGLE_TASK_CHECKPOINT}; {len(missing)} left at random init (bone decoder+head, expected)."
        )
