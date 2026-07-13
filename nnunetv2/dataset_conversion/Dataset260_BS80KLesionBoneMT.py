from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json
from PIL import Image

from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity
from nnunetv2.paths import nnUNet_raw


DATASET_NAME = "Dataset260_BS80KLesionBoneMT"
DEFAULT_DATA_ROOT = Path(r"C:\Users\lulay\Desktop\wbbs-dataset")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\lulay\Desktop\nnunetv2-multitask\data\nnUNet_raw")
IMAGE_SIZE = (256, 1024)  # PIL uses width, height.


def _id_from_name(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot extract numeric id from {path}.")
    return match.group(1).zfill(4)


def _files_by_id(folder: Path, suffixes: Tuple[str, ...]) -> Dict[str, Path]:
    files = {}
    for suffix in suffixes:
        for path in folder.glob(f"*{suffix}"):
            files[_id_from_name(path)] = path
    return files


def _read_invalid_ids(invalid_xlsx: Path) -> Set[str]:
    if not invalid_xlsx.is_file():
        return set()
    invalid_ids = set()
    sheets = pd.read_excel(invalid_xlsx, sheet_name=None, header=None)
    for sheet in sheets.values():
        for value in sheet.iloc[1:, 0].to_numpy().ravel():
            if pd.isna(value):
                continue
            invalid_ids.add(str(int(value)).zfill(4))
    return invalid_ids


def _ensure_png_l(path: Path, allowed_labels: Iterable[int], size: Tuple[int, int]) -> None:
    with Image.open(path) as img:
        if img.mode != "L":
            raise ValueError(f"{path} must be grayscale L, got {img.mode}.")
        if img.size != size:
            raise ValueError(f"{path} must be size {size}, got {img.size}.")
        if img.palette is not None:
            raise ValueError(f"{path} must not have a palette.")
        found = set(int(i) for i in np.unique(np.asarray(img)))
        unexpected = found - set(allowed_labels)
        if unexpected:
            raise ValueError(f"{path} has unexpected labels {sorted(unexpected)}.")


def _copy_image_as_png(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        img = img.convert("L")
        if img.size != IMAGE_SIZE:
            img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img.save(dst)


def _copy_label_png(src: Path, dst: Path, allowed_labels: Iterable[int]) -> None:
    _ensure_png_l(src, allowed_labels, IMAGE_SIZE)
    shutil.copy2(src, dst)


def _write_background_label(dst: Path) -> None:
    Image.fromarray(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8), mode="L").save(dst)


def _split_ids(ids: List[str], seed: int = 42) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(ids))
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    split = {}
    for patient_id in shuffled[:n_train]:
        split[str(patient_id)] = "train"
    for patient_id in shuffled[n_train:n_train + n_val]:
        split[str(patient_id)] = "val"
    for patient_id in shuffled[n_train + n_val:]:
        split[str(patient_id)] = "test"
    return split


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_bs80k_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    raw_ant = data_root / "bs80k-imaging-raw" / "wholeBodyANT"
    raw_post = data_root / "bs80k-imaging-raw" / "wholeBodyPOST"
    bone_root = data_root / "bs80k-bone_region-segmentation" / "pseudo_label-2607"
    lesion_root = data_root / "bs80k-lesion-segmentation" / "otsu_morphology-guarded_smooth"
    invalid_xlsx = data_root / "bs80k-invalid_list.xlsx"

    ant_images = _files_by_id(raw_ant, (".jpg", ".jpeg", ".png"))
    post_images = _files_by_id(raw_post, (".jpg", ".jpeg", ".png"))
    bone_ant = _files_by_id(bone_root / "anterior", (".png",))
    bone_post = _files_by_id(bone_root / "posterior", (".png",))
    lesion_ant = _files_by_id(lesion_root / "anterior", (".png",))
    lesion_post = _files_by_id(lesion_root / "posterior", (".png",))
    invalid_ids = _read_invalid_ids(invalid_xlsx)

    paired_raw_ids = sorted(set(ant_images) & set(post_images))
    eligible_ids = [
        patient_id for patient_id in paired_raw_ids
        if patient_id not in invalid_ids
        and patient_id in bone_ant
        and patient_id in bone_post
    ]
    split = _split_ids(eligible_ids, seed=42)

    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    images_tr = output_folder / "imagesTr"
    lesion_labels = output_folder / "labelsTr" / "lesion"
    bone_labels = output_folder / "labelsTr" / "bone"
    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(lesion_labels)
    maybe_mkdir_p(bone_labels)

    manifest_rows = []
    invalid_rows = []

    for patient_id in paired_raw_ids:
        reason = []
        if patient_id in invalid_ids:
            reason.append("invalid_list")
        if patient_id not in bone_ant:
            reason.append("missing_bone_anterior")
        if patient_id not in bone_post:
            reason.append("missing_bone_posterior")
        if reason:
            invalid_rows.append({"id": patient_id, "reason": "|".join(reason)})

    for patient_id in eligible_ids:
        case_id = f"bs80k_{patient_id}"
        _copy_image_as_png(ant_images[patient_id], images_tr / f"{case_id}_0000.png")
        _copy_image_as_png(post_images[patient_id], images_tr / f"{case_id}_0001.png")

        _copy_label_png(bone_ant[patient_id], bone_labels / f"{case_id}_0000.png", range(13))
        _copy_label_png(bone_post[patient_id], bone_labels / f"{case_id}_0001.png", range(13))

        lesion_sources = {
            "anterior": lesion_ant.get(patient_id),
            "posterior": lesion_post.get(patient_id),
        }
        for view_idx, view_name in enumerate(("anterior", "posterior")):
            dst = lesion_labels / f"{case_id}_{view_idx:04d}.png"
            if lesion_sources[view_name] is None:
                _write_background_label(dst)
                lesion_source = "generated_background"
            else:
                _copy_label_png(lesion_sources[view_name], dst, range(3))
                lesion_source = str(lesion_sources[view_name])
            manifest_rows.append({
                "id": patient_id,
                "case_id": case_id,
                "split": split[patient_id],
                "view": view_name,
                "image": str(ant_images[patient_id] if view_name == "anterior" else post_images[patient_id]),
                "lesion_label": lesion_source,
                "bone_label": str(bone_ant[patient_id] if view_name == "anterior" else bone_post[patient_id]),
            })

    dataset_json = {
        "channel_names": {"0": "anterior", "1": "posterior"},
        "labels": {"background": 0},
        "numTraining": len(eligible_ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": dataset_name,
        "description": "BS-80K paired anterior/posterior multitask lesion and bone segmentation dataset.",
        "multitask": {
            "case_unit": "paired_anterior_posterior",
            "views": ["anterior", "posterior"],
            "tasks": {
                "lesion": {
                    "label_dir": "labelsTr/lesion",
                    "labels": {"background": 0, "benign": 1, "malignant": 2},
                },
                "bone": {
                    "label_dir": "labelsTr/bone",
                    "labels": {"background": 0, **{f"label_{i}": i for i in range(1, 13)}},
                },
            },
        },
    }
    save_json(dataset_json, output_folder / "dataset.json", sort_keys=False)
    _write_csv(output_folder / "conversion_manifest.csv", manifest_rows,
               ["id", "case_id", "split", "view", "image", "lesion_label", "bone_label"])
    _write_csv(output_folder / "split_seed42.csv",
               [{"id": patient_id, "case_id": f"bs80k_{patient_id}", "split": split[patient_id]}
                for patient_id in eligible_ids],
               ["id", "case_id", "split"])
    _write_csv(output_folder / "invalid_excluded.csv", invalid_rows, ["id", "reason"])

    verify_paired_multitask_dataset_integrity(str(output_folder))
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert_bs80k_to_nnunet_raw(args.data_root, args.output_root, args.dataset_name, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
