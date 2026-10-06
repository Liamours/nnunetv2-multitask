"""Diagnostic instrumentation for the CBAM Late Fission multitask collapse (2026-09-04): both lesion
classes stayed at exactly 0.0 pseudo-dice for all 100 epochs of nnUNetPlansMultiTaskCBAMPostNorm
training, while multiple bone-region channels in the same run climbed cleanly to 0.27-0.35, despite
the post_norm fix that cleanly resolved single-task CBAM twice. This trainer runs a short 5-epoch
diagnostic with per-CBAM-block and per-task observability the normal trainer does not have, so a
collapse (if it reproduces this early) can be traced to where the lesion signal actually dies rather
than just confirming that it dies.

Four independent, additive observers - nothing here changes what the base trainer computes or how a
normal (non-diagnostic) run behaves:

1. Gate statistics. A forward hook on every `nn.Sigmoid` found anywhere in the network. `cbam.py`
   uses `nn.Sigmoid` in exactly two places (ChannelAttention.activation, SpatialAttention.activation)
   and nowhere else in this architecture (the has_regions prediction path uses the *function*
   `torch.sigmoid()` on a plain tensor, not a module, so it is invisible to this hook and does not
   get mixed in). The qualified module name already encodes which stage and which of
   encoder_attention / decoder.attention_blocks the gate belongs to.
2. CBAM block input/output magnitude. A forward hook on every `CBAM` instance (post-norm applied or
   not - see the coverage note below), comparing pre- and post-block activation statistics.
3. Per-task raw (unweighted) loss, read from `MultiTaskLoss.last_task_losses` (multitask_losses.py),
   a diagnostic side channel that module now always populates - not otherwise observable once summed
   into the single combined scalar `train_step` receives.
4. Per-task head gradient norm. `heads["lesion"]` and `heads["bone"]` never share a parameter, so
   their post-backward gradient norms are already cleanly isolated per task with no need for a
   second backward pass.

Coverage note worth testing against the resulting logs: `post_norm` as currently wired
(multitask_components.py's FeatureUNetDecoder) only reaches the *decoder*-side CBAM blocks.
`CBAMFeatureAdapter` (multitask_unet.py), used for the *encoder* and *bottleneck* attention, builds
its CBAM instances with no norm_op at all (cbam.py line ~95), so those gates still feed straight into
the next stage unnormalized - the exact pre-fix failure shape, just not yet observed to matter there.
Single-task CBAM succeeded with this same asymmetric coverage, so it is not sufficient on its own to
explain a multitask-specific collapse, but it is a real, testable gap worth checking the encoder/
bottleneck gate logs for before ruling it out.

Log destination: logs/nnunetcbam_multihead_diagnostic/<run timestamp>.jsonl, one JSON object per
line, independent of nnU-Net's own text training log (which still writes normally, unchanged).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import torch
from torch import autocast, nn

from nnunetv2.architecture.cbam import CBAM
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.helpers import dummy_context

_LOG_ROOT = Path(__file__).resolve().parents[5] / "logs" / "nnunetcbam_multihead_diagnostic"


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach()
    stats = {
        "mean": float(t.mean()),
        "std": float(t.std()) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
    }
    if torch.is_floating_point(t) and t.min() >= 0:  # gate tensors only: fraction near the rails
        stats["frac_near_0"] = float((t < 0.05).float().mean())
        stats["frac_near_1"] = float((t > 0.95).float().mean())
    return stats


class nnUNetTrainerMultiTaskCBAMDiagnostic_5epochs(nnUNetTrainerMultiTask):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 5
        self._diag_step = 0
        self._diag_log_every = 5  # batches; dense enough to see early divergence within 5 epochs
        self._diag_records: list[dict] = []
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._diag_log_path = _LOG_ROOT / f"{stamp}.jsonl"

    def initialize(self):
        super().initialize()
        self._register_diagnostic_hooks()
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.print_to_log_file(f"[diagnostic] logging to {self._diag_log_path}")

    def _register_diagnostic_hooks(self):
        network = self.network
        for name, module in network.named_modules():
            if isinstance(module, nn.Sigmoid):
                module.register_forward_hook(self._gate_hook(name))
            elif isinstance(module, CBAM):
                module.register_forward_hook(self._cbam_io_hook(name))
        for task_name, head in network.heads.items():
            head.register_forward_hook(self._head_hook(task_name))

    def _should_log(self) -> bool:
        return self._diag_step % self._diag_log_every == 0

    def _gate_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._should_log():
                return
            self._diag_records.append({
                "kind": "gate", "step": self._diag_step, "epoch": self.current_epoch,
                "module": name, **_tensor_stats(output),
            })
        return hook

    def _cbam_io_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._should_log():
                return
            self._diag_records.append({
                "kind": "cbam_io", "step": self._diag_step, "epoch": self.current_epoch,
                "module": name, "input": _tensor_stats(inputs[0]), "output": _tensor_stats(output),
            })
        return hook

    def _head_hook(self, task_name: str):
        def hook(module, inputs, output):
            if not self._should_log():
                return
            logits = output[0] if isinstance(output, list) else output  # finest deep-supervision level
            with torch.no_grad():
                # Raw channels are (view, class) stacked as view * num_classes + class - this project
                # channel-stacks the anterior/posterior pair, and per _reshape_paired_multiview_for_loss
                # in nnUNetTrainerMultiTask.py (the actual training/loss code path), softmax is taken
                # over the class axis *within* each view separately, never across all raw channels at
                # once. An earlier version of this hook applied one flat softmax over every raw channel
                # together, which silently mixes two views' background channels against each other and
                # produces meaningless numbers - fixed here to match the real decode exactly.
                num_views = len(self.views)
                b, c, h, w = logits.shape
                num_classes = c // num_views
                per_view = logits.float().reshape(b, num_views, num_classes, h, w)
                probabilities = torch.softmax(per_view, dim=2)
                # [num_views, num_classes]: mean over batch + spatial, kept separate per view since a
                # collapse could plausibly differ by view even though the earlier full-run 3-signal
                # check found both views equally dead at 100 epochs.
                per_view_class_mean_probability = probabilities.mean(dim=(0, 3, 4)).tolist()
            self._diag_records.append({
                "kind": "head_output", "step": self._diag_step, "epoch": self.current_epoch,
                "task": task_name, "logits": _tensor_stats(logits),
                "views": self.views, "num_classes": num_classes,
                "per_view_class_mean_probability": per_view_class_mean_probability,
            })
        return hook

    def _flush_diagnostics(self):
        if not self._diag_records:
            return
        with self._diag_log_path.open("a", encoding="utf-8") as handle:
            for record in self._diag_records:
                handle.write(json.dumps(record) + "\n")
        self._diag_records.clear()

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

        if self._should_log():
            record = {
                "kind": "step", "step": self._diag_step, "epoch": self.current_epoch,
                "combined_loss": float(loss.detach()),
                "task_losses": dict(self.loss.last_task_losses),
                "head_grad_norm": {},
            }
            for task_name, head in self.network.heads.items():
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                norm = torch.norm(torch.stack([g.norm() for g in grads])) if grads else torch.tensor(0.0)
                record["head_grad_norm"][task_name] = float(norm)
            self._diag_records.append(record)
        self._diag_step += 1
        self._flush_diagnostics()

        return {"loss": loss.detach().cpu().numpy()}
