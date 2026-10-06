"""Relay SegFormer's per-case predicted masks into nnU-Net's evaluator layout.

SegFormer's batch inference (`services/inference-segmentation/batch_infer.py`) writes one folder per
case: `<source>/<case_id>/<task>_<view>.png`, single-channel uint8 class-index PNGs, same pixel
encoding nnU-Net's own ground truth and predictions use. `nnUNetv2_evaluate_multitask` expects
`<predictions>/<task>/<view>/<case_id>.png`. This script only reorganizes the folder layout via
hardlinks; it does not touch pixel data, so the two backends' test metrics come from the exact same
evaluator rather than two independently-implemented Dice/sensitivity computations.

dataset/segformer/bs80k_lesion (split_seed42.csv, via --split-csv) + inferences/segformer/<run>/<split>
(--source) -> relay hardlinks, no pixel data touched -> <dest>/<task>/<view>/<case_id>.png, ready for
nnUNetv2_evaluate_multitask --predictions <dest>
"""

import argparse
import csv
from pathlib import Path

from tqdm import tqdm


def relay(source: Path, dest: Path, split_csv: Path, split: str, task: str, views: list[str]) -> int:
    with split_csv.open(encoding="utf-8") as handle:
        case_ids = [row["case_id"] for row in csv.DictReader(handle) if row["split"] == split]
    if not case_ids:
        raise ValueError(f"No cases found for split={split!r} in {split_csv}")

    for view in views:
        (dest / task / view).mkdir(parents=True, exist_ok=True)

    linked = 0
    for case_id in tqdm(case_ids, desc=f"relay {split}"):
        for view in views:
            src_file = source / case_id / f"{task}_{view}.png"
            dst_file = dest / task / view / f"{case_id}.png"
            if not src_file.is_file():
                raise FileNotFoundError(src_file)
            if dst_file.exists():
                dst_file.unlink()
            dst_file.hardlink_to(src_file)
            linked += 1
    return linked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="SegFormer per-case inference output dir.")
    parser.add_argument("--dest", type=Path, required=True, help="Relay target dir (nnU-Net prediction layout).")
    parser.add_argument("--split-csv", type=Path, required=True, help="split_seed42.csv from the nnU-Net raw dataset.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--task", type=str, default="lesion")
    parser.add_argument("--views", type=str, default="anterior,posterior")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    linked = relay(args.source, args.dest, args.split_csv, args.split, args.task, views)
    print(f"Relayed {linked} files to {args.dest}")


if __name__ == "__main__":
    main()
