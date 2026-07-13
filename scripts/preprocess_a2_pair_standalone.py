"""
Standalone preprocessing for A2 paired WBBS images.

This does not import nnU-Net. It prepares one anterior/posterior pair for the
A2 dual-head model contract:

    input tensor: [1, 2, H, W], float32
    channel 0: anterior z-score normalized grayscale
    channel 1: posterior z-score normalized grayscale

The script writes an .npz with the tensor and JSON metadata. It is intended as
portable app logic that can be reimplemented in another runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


TARGET_SIZE_WH = (256, 1024)  # PIL order: width, height.


def load_grayscale_resized(path: Path, size_wh: tuple[int, int] = TARGET_SIZE_WH) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size_wh:
        image = image.resize(size_wh, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def zscore(channel: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = float(channel.mean())
    std = float(channel.std())
    if std < 1e-8:
        normalized = channel - mean
        std_used = 1.0
    else:
        normalized = (channel - mean) / std
        std_used = std
    return normalized.astype(np.float32), {"mean": mean, "std": std, "std_used": std_used}


def preprocess_pair(anterior_path: Path, posterior_path: Path) -> tuple[np.ndarray, dict]:
    anterior = load_grayscale_resized(anterior_path)
    posterior = load_grayscale_resized(posterior_path)
    if anterior.shape != posterior.shape:
        raise ValueError(f"Shape mismatch after resize: anterior={anterior.shape}, posterior={posterior.shape}")

    anterior_norm, anterior_stats = zscore(anterior)
    posterior_norm, posterior_stats = zscore(posterior)

    tensor = np.stack([anterior_norm, posterior_norm], axis=0)[None].astype(np.float32)
    metadata = {
        "input_anterior": str(anterior_path),
        "input_posterior": str(posterior_path),
        "target_size_wh": list(TARGET_SIZE_WH),
        "tensor_shape": list(tensor.shape),
        "tensor_layout": "NCHW",
        "channels": {"0": "anterior", "1": "posterior"},
        "normalization": {
            "type": "per_image_per_channel_zscore",
            "anterior": anterior_stats,
            "posterior": posterior_stats,
        },
        "model_output_note": {
            "lesion": "class 2 is malignant",
            "bone": "classes 0..12",
        },
    }
    return tensor, metadata


def save_preview_png(normalized_channel: np.ndarray, output_path: Path) -> None:
    lo = float(np.percentile(normalized_channel, 0.5))
    hi = float(np.percentile(normalized_channel, 99.5))
    if hi <= lo:
        preview = np.zeros_like(normalized_channel, dtype=np.uint8)
    else:
        preview = np.clip((normalized_channel - lo) / (hi - lo), 0, 1)
        preview = (preview * 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one anterior/posterior pair for A2 dual-head inference.")
    parser.add_argument("--anterior", required=True, type=Path)
    parser.add_argument("--posterior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--preview-png", action="store_true", help="Write display-only normalized PNG previews.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_id = args.case_id or args.anterior.stem.replace("_0000", "")

    tensor, metadata = preprocess_pair(args.anterior, args.posterior)
    np.savez_compressed(args.output_dir / f"{case_id}_preprocessed_pair.npz", input=tensor)
    (args.output_dir / f"{case_id}_preprocess_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if args.preview_png:
        save_preview_png(tensor[0, 0], args.output_dir / f"{case_id}_anterior_preprocessed_preview.png")
        save_preview_png(tensor[0, 1], args.output_dir / f"{case_id}_posterior_preprocessed_preview.png")


if __name__ == "__main__":
    main()
