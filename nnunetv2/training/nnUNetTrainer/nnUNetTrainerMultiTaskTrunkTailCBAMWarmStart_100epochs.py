"""Warm-start trunk_tail (Early-Mid / Mid Fission) CBAM from the working single-task CBAM checkpoint.

Same rationale as the Late Fission (dual_head) and Early Fission (dual_decoder) warm-start trainers:
a from-scratch CBAM multitask run is expected to face the same fast, structural slide into
predicting background everywhere that Late Fission showed, for the same class-imbalance reason (see
that trainer's docstring and the plan's root-cause investigation section) - so starting from
single-task's already-competent lesion weights instead of random init is the direct fix.

Key mapping is a third shape, different from both prior warm-starts, because `trunk_tail` splits the
*same* decoder stage sequence single-task uses into two consecutive pieces rather than either sharing
it whole (dual_head) or duplicating it per task (dual_decoder). Verified directly against real model
instances before writing this (not just reasoning from the architecture code): for single-task's 7
decoder stages (indices 0-6 in `decoder.stages`/`decoder.transpconvs`/`decoder.attention_blocks`),
`FeatureUNetDecoder.__init__` builds each sub-module list with `self.stage_range[0] + 1` as the first
loop index, so a stage_range of `(0, k)` and one of `(k, 7)` produce `ModuleList`s that are each
independently indexed from 0 - the trunk's own index space and the tail's own index space are both
contiguous with single-task's, just offset. Confirmed on real `MultiTaskEarlyMidUNet`/`MultiTaskMidUNet`
instances: `trunk.stages` covers indices `[0, trunk_stages)`, matching `decoder.stages` at the same
indices unchanged; `tails.lesion.stages` covers indices `[0, 7 - trunk_stages)`, matching
`decoder.stages[trunk_stages :]` - i.e. `tails.lesion.stages[i]` corresponds to
`decoder.stages[i + trunk_stages]`. The same index correspondence holds for `transpconvs` and
`attention_blocks`, since `FeatureUNetDecoder` appends to all three lists inside the same loop.
`heads.lesion.*` matches directly (same ModuleDict key in both, no remap). `tails.bone.*` and
`heads.bone.*` have no source counterpart and stay at random init.

`_TRUNK_STAGES` is the one thing each fission point's subclass sets; everything else is shared.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask_100epochs import nnUNetTrainerMultiTask_100epochs

_SINGLE_TASK_CHECKPOINT = (
    Path(__file__).resolve().parents[5] / "weights" / "nnunet" / "Dataset261_BS80KLesionOnly"
    / "nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1ControlledBatch4CBAMPostNorm__2d"
    / "fold_0" / "checkpoint_final.pth"
)
_TRUNK_KEY_RE = re.compile(r"^trunk\.(stages|transpconvs|attention_blocks)\.(\d+)(.*)$")
_TAIL_KEY_RE = re.compile(r"^tails\.lesion\.(stages|transpconvs|attention_blocks)\.(\d+)(.*)$")
# FeatureUNetDecoder.__init__ stores `self.encoder = encoder` - the *same* shared encoder object
# passed in, not a copy - so trunk.encoder.*, tails.lesion.encoder.*, and tails.bone.encoder.* are
# each a second (third, fourth) path to the identical tensors already reachable as the top-level
# encoder.*. Not bone-specific despite the tails.bone. prefix: the encoder is shared by every task.
# These must resolve to the same correct value as encoder.* - left unmapped, a stale/random entry
# under one of these nested paths would silently overwrite the correctly-loaded shared tensor when
# load_state_dict processes it after the top-level encoder.* entry (dict order follows module
# registration order, and encoder is registered before trunk/tails).
_NESTED_ENCODER_RE = re.compile(r"^(?:trunk|tails\.\w+)\.encoder\.(.*)$")


class _TrunkTailCBAMWarmStartMixin:
    _TRUNK_STAGES: int  # set by each fission-point subclass

    def initialize(self):
        super().initialize()
        self._warm_start_from_single_task()

    def _source_key(self, target_key: str) -> str:
        match = _NESTED_ENCODER_RE.match(target_key)
        if match:
            return f"encoder.{match.group(1)}"
        match = _TRUNK_KEY_RE.match(target_key)
        if match:
            component, index, rest = match.group(1), int(match.group(2)), match.group(3)
            return f"decoder.{component}.{index}{rest}"
        match = _TAIL_KEY_RE.match(target_key)
        if match:
            component, index, rest = match.group(1), int(match.group(2)), match.group(3)
            return f"decoder.{component}.{index + self._TRUNK_STAGES}{rest}"
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
            k for k in missing if not (k.startswith("tails.bone.") or k.startswith("heads.bone."))
        ]
        if unexpected_missing or shape_mismatched:
            raise RuntimeError(
                "Warm-start expected only tails.bone.*/heads.bone.* to be absent from the "
                f"single-task checkpoint and every other key to match by shape. Unexpected missing: "
                f"{unexpected_missing}. Shape-mismatched: {shape_mismatched}."
            )

        self.network.load_state_dict(model_dict)
        self.print_to_log_file(
            f"[warm_start] loaded {len(matched)}/{len(model_dict)} parameters from "
            f"{_SINGLE_TASK_CHECKPOINT}; {len(missing)} left at random init (bone tail+head, expected)."
        )


class nnUNetTrainerMultiTaskEarlyMidCBAMWarmStart_100epochs(_TrunkTailCBAMWarmStartMixin, nnUNetTrainerMultiTask_100epochs):
    _TRUNK_STAGES = 1


class nnUNetTrainerMultiTaskMidCBAMWarmStart_100epochs(_TrunkTailCBAMWarmStartMixin, nnUNetTrainerMultiTask_100epochs):
    _TRUNK_STAGES = 6
