from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json

from nnunetv2.dataset_conversion.Dataset264_SegRap2023GTVOarMT import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GTV_CLASSES,
    GTV_VOLUME_CACHE,
    N_TEST,
    SPLIT_SEED,
    _copy,
    _copy_binarized_label,
    compute_test_split,
    write_split_manifest,
)
from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity

DATASET_NAME = "Dataset265_SegRap2023GTVOnly"


def convert_segrap2023_gtv_only_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    case_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case folders found under {data_root}")

    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    # identical split to Dataset264 - deterministic given (case list, N_TEST, SPLIT_SEED), which
    # controlled-ablation-rules.md requires so the A1 baseline and the multitask variants are
    # compared on exactly the same data
    test_cases = set(compute_test_split(case_dirs, cache_path=GTV_VOLUME_CACHE))

    images_tr = output_folder / "imagesTr"
    gtv_tr = output_folder / "labelsTr" / "gtv"
    images_ts = output_folder / "imagesTs"
    gtv_ts = output_folder / "labelsTs" / "gtv"
    for d in (images_tr, gtv_tr, images_ts, gtv_ts):
        maybe_mkdir_p(d)

    missing_gtv = []
    n_train = 0
    for case_dir in case_dirs:
        case_id = f"segrap2023_{case_dir.name}"
        is_test = case_dir.name in test_cases
        out_images, out_gtv = (images_ts, gtv_ts) if is_test else (images_tr, gtv_tr)
        if not is_test:
            n_train += 1

        _copy(case_dir / "image.nii.gz", out_images / f"{case_id}_0000.nii.gz")
        _copy(case_dir / "image_contrast.nii.gz", out_images / f"{case_id}_0001.nii.gz")

        for channel_idx, name in enumerate(GTV_CLASSES):
            src = case_dir / f"{name}.nii.gz"
            if not src.is_file():
                missing_gtv.append((case_dir.name, name))
                continue
            _copy_binarized_label(src, out_gtv / f"{case_id}_{channel_idx:04d}.nii.gz")

    if missing_gtv:
        raise FileNotFoundError(f"Missing GTV files: {missing_gtv}")

    write_split_manifest(output_folder, case_dirs, test_cases)

    dataset_json = {
        "channel_names": {"0": "ct", "1": "ct_contrast_enhanced"},
        "labels": {"background": 0},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "SegRap2023 GTV-only baseline (single-task A1 ablation), multichannel "
                        f"(GTVp/GTVnd). Identical images and {N_TEST}-case held-out test split as "
                        f"Dataset264 (seed {SPLIT_SEED}, stratified by GTV volume).",
        "converted_by": "nnunetv2-multitask",
        "multitask": {
            "case_unit": "single_image",
            "views": ["image"],
            "tasks": {
                "gtv": {
                    "label_dir": "labelsTr/gtv",
                    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(GTV_CLASSES)}},
                    "regions_class_order": list(range(1, len(GTV_CLASSES) + 1)),
                    "multichannel": True,
                },
            },
        },
    }
    save_json(dataset_json, output_folder / "dataset.json", sort_keys=False)

    verify_paired_multitask_dataset_integrity(str(output_folder))
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert_segrap2023_gtv_only_to_nnunet_raw(args.data_root, args.output_root, args.dataset_name, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
