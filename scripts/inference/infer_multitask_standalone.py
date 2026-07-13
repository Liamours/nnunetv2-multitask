"""
Standalone direct-forward inference for the A2 dual-head BS-80K model.

This script intentionally does not use nnUNetv2_predict. It loads the custom
dual-head network from plans.json, restores checkpoint_best.pth, runs one
paired anterior/posterior case, and writes task/view masks.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL_DIR = Path(
    r"C:\Users\lulay\Desktop\nnunetv2-multitask\data\nnUNet_results"
    r"\Dataset260_BS80KLesionBoneMT"
    r"\nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTask2GB__2d"
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "fold_0" / "checkpoint_best.pth"
DEFAULT_PLANS = DEFAULT_MODEL_DIR / "plans.json"
DEFAULT_DATASET_JSON = DEFAULT_MODEL_DIR / "dataset.json"


def import_from_string(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def resolve_imports(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_imports(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_imports(v) for v in value]
    if isinstance(value, str) and (value.startswith("torch.") or value.startswith("nnunetv2.")):
        return import_from_string(value)
    return value


def load_grayscale_png(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def zscore(channel: np.ndarray) -> np.ndarray:
    mean = float(channel.mean())
    std = float(channel.std())
    if std < 1e-8:
        return channel - mean
    return (channel - mean) / std


def pad_to_network_multiple(x: torch.Tensor, multiple_hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = x.shape
    mult_h, mult_w = multiple_hw
    pad_h = (mult_h - height % mult_h) % mult_h
    pad_w = (mult_w - width % mult_w) % mult_w
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), (pad_h, pad_w)


def crop_logits(logits: torch.Tensor, original_hw: tuple[int, int]) -> torch.Tensor:
    height, width = original_hw
    return logits[..., :height, :width]


def downsampling_multiple(strides: list[list[int]]) -> tuple[int, int]:
    mult_h, mult_w = 1, 1
    for stride in strides:
        mult_h *= int(stride[0])
        mult_w *= int(stride[1])
    return mult_h, mult_w


def build_network(plans_file: Path, input_channels: int = 2) -> torch.nn.Module:
    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    architecture = plans["configurations"]["2d"]["architecture"]
    network_cls = import_from_string(architecture["network_class_name"])
    kwargs = resolve_imports(architecture["arch_kwargs"])
    network = network_cls(
        input_channels=input_channels,
        num_classes=0,
        deep_supervision=True,
        **kwargs,
    )
    network.set_deep_supervision(False)
    return network


def split_view_logits(task_logits: torch.Tensor, num_classes: int) -> dict[str, torch.Tensor]:
    if task_logits.ndim != 4 or task_logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, C, H, W], got {tuple(task_logits.shape)}")
    expected_channels = 2 * num_classes
    if task_logits.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {task_logits.shape[1]}")
    return {
        "anterior": task_logits[0, 0:num_classes],
        "posterior": task_logits[0, num_classes:expected_channels],
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def run_case(
    anterior: Path,
    posterior: Path,
    output_dir: Path,
    checkpoint: Path,
    plans: Path,
    save_logits: bool,
    device_name: str,
) -> None:
    ant = load_grayscale_png(anterior)
    post = load_grayscale_png(posterior)
    if ant.shape != post.shape:
        raise ValueError(f"Anterior/posterior shape mismatch: {ant.shape} vs {post.shape}")

    original_hw = ant.shape
    data = np.stack([zscore(ant), zscore(post)], axis=0)[None]
    x = torch.from_numpy(data).float()

    plans_dict = json.loads(plans.read_text(encoding="utf-8"))
    strides = plans_dict["configurations"]["2d"]["architecture"]["arch_kwargs"]["strides"]
    x, _ = pad_to_network_multiple(x, downsampling_multiple(strides))

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    network = build_network(plans).to(device)
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    network.load_state_dict(checkpoint_dict["network_weights"], strict=True)
    network.eval()

    with torch.no_grad():
        outputs = network(x.to(device))
        lesion_logits = crop_logits(outputs["lesion"].detach().cpu(), original_hw)
        bone_logits = crop_logits(outputs["bone"].detach().cpu(), original_hw)

    lesion_by_view = split_view_logits(lesion_logits, 3)
    bone_by_view = split_view_logits(bone_logits, 13)

    for view_name, logits in lesion_by_view.items():
        mask = logits.argmax(0).numpy().astype(np.uint8)
        save_mask(mask, output_dir / "lesion" / view_name / f"{anterior.stem}_lesion_{view_name}.png")
        malignant = (mask == 2).astype(np.uint8)
        save_mask(malignant, output_dir / "lesion_malignant" / view_name / f"{anterior.stem}_lesion_malignant_{view_name}.png")

    for view_name, logits in bone_by_view.items():
        mask = logits.argmax(0).numpy().astype(np.uint8)
        save_mask(mask, output_dir / "bone" / view_name / f"{anterior.stem}_bone_{view_name}.png")

    if save_logits:
        np.savez_compressed(
            output_dir / f"{anterior.stem}_logits.npz",
            lesion=lesion_logits.numpy(),
            bone=bone_logits.numpy(),
        )

    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_dict.get("current_epoch", -1)),
        "best_ema": float(checkpoint_dict.get("_best_ema", np.nan)),
        "input_anterior": str(anterior),
        "input_posterior": str(posterior),
        "input_shape_hw": list(original_hw),
        "output_contract": {
            "lesion": {"classes": {"background": 0, "benign": 1, "malignant": 2}},
            "bone": {"classes": {"background": 0, **{f"bone_{i}": i for i in range(1, 13)}}},
        },
    }
    (output_dir / f"{anterior.stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone A2 dual-head direct-forward inference.")
    parser.add_argument("--anterior", required=True, type=Path, help="Anterior grayscale PNG.")
    parser.add_argument("--posterior", required=True, type=Path, help="Posterior grayscale PNG.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output folder.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--plans", default=DEFAULT_PLANS, type=Path)
    parser.add_argument("--dataset-json", default=DEFAULT_DATASET_JSON, type=Path, help="Kept for deployment metadata compatibility.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--save-logits", action="store_true", help="Also save raw lesion/bone logits as npz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.plans.is_file():
        raise FileNotFoundError(args.plans)
    if not args.dataset_json.is_file():
        raise FileNotFoundError(args.dataset_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_case(args.anterior, args.posterior, args.output_dir, args.checkpoint, args.plans, args.save_logits, args.device)


if __name__ == "__main__":
    main()
