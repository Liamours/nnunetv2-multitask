from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json
from PIL import Image, ImageDraw

from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity
from nnunetv2.paths import nnUNet_raw

DATASET_NAME = "Dataset262_ChestXDetDiseaseOrganMT"
DEFAULT_DATA_ROOT = Path(r"C:\research\research-medical_imaging-multitask\data-chestxdet\source")
DEFAULT_ANNOT_ROOT = Path(r"C:\research\research-medical_imaging-multitask\external\ChestX-Det-Dataset")
DEFAULT_PSEUDO_ROOT = Path(r"C:\research\research-medical_imaging-multitask\data-chestxdet\pseudo_labels_organ")
DEFAULT_OUTPUT_ROOT = Path(r"C:\research\research-medical_imaging-multitask\data-chestxdet\nnUNet_raw")

# canonical channel order - same as visualize_samples.py's DISEASE_COLORS / class_order.json,
# kept fixed here so training labels and the earlier figures agree on what channel is what
DISEASE_CLASSES = [
    "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
    "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule",
    "Pleural Thickening", "Pneumothorax",
]


def _polygon_mask(polygons: List[list], size) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        pts = [tuple(p) for p in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return (np.array(mask) > 0).astype(np.uint8)


def _copy_image_as_grayscale_png(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        img.convert("L").save(dst)


def _write_disease_labels(entry: dict, size, out_dir: Path, case_id: str) -> None:
    by_class: Dict[str, np.ndarray] = {name: np.zeros(size[::-1], dtype=np.uint8) for name in DISEASE_CLASSES}
    for sym, poly in zip(entry["syms"], entry["polygons"]):
        if sym not in by_class:
            raise ValueError(f"Unexpected disease category {sym!r} in {entry['file_name']}")
        by_class[sym] |= _polygon_mask([poly], size)
    for channel_idx, name in enumerate(DISEASE_CLASSES):
        # store as 0/1, not 0/255 - matches every other label file in this project (BS-80K, etc.)
        Image.fromarray(by_class[name], mode="L").save(out_dir / f"{case_id}_{channel_idx:04d}.png")


def _write_organ_labels(pseudo_npz_path: Path, organ_classes: List[str], out_dir: Path, case_id: str) -> None:
    data = np.load(pseudo_npz_path)
    shape = tuple(data["shape"])
    stack = np.unpackbits(data["packed"])[:np.prod(shape)].reshape(shape).astype(np.uint8)
    for channel_idx in range(len(organ_classes)):
        Image.fromarray(stack[channel_idx], mode="L").save(out_dir / f"{case_id}_{channel_idx:04d}.png")


def convert_chestxdet_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    annot_root: Path = DEFAULT_ANNOT_ROOT,
    pseudo_root: Path = DEFAULT_PSEUDO_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    organ_classes = json.loads((pseudo_root / "class_order.json").read_text())

    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    images_tr = output_folder / "imagesTr"
    disease_tr = output_folder / "labelsTr" / "disease"
    organ_tr = output_folder / "labelsTr" / "organ"
    images_ts = output_folder / "imagesTs"
    disease_ts = output_folder / "labelsTs" / "disease"
    organ_ts = output_folder / "labelsTs" / "organ"
    for d in (images_tr, disease_tr, organ_tr, images_ts, disease_ts, organ_ts):
        maybe_mkdir_p(d)

    n_train = 0
    for split, annot_file, img_dir, out_images, out_disease, out_organ in [
        ("train", "ChestX_Det_train.json", "train", images_tr, disease_tr, organ_tr),
        ("test", "ChestX_Det_test.json", "test", images_ts, disease_ts, organ_ts),
    ]:
        entries = json.loads((annot_root / annot_file).read_text())
        for entry in entries:
            stem = Path(entry["file_name"]).stem
            case_id = f"chestxdet_{stem}"
            src_img = data_root / img_dir / entry["file_name"]
            with Image.open(src_img) as im:
                size = im.size  # (W, H)

            _copy_image_as_grayscale_png(src_img, out_images / f"{case_id}_0000.png")
            _write_disease_labels(entry, size, out_disease, case_id)
            _write_organ_labels(pseudo_root / split / f"{stem}.npz", organ_classes, out_organ, case_id)

            if split == "train":
                n_train += 1

    dataset_json = {
        "channel_names": {"0": "chest_xray"},
        "labels": {"background": 0},
        "numTraining": n_train,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": dataset_name,
        "description": "ChestX-Det disease (primary, multichannel) and organ (secondary, multichannel) "
                        "multitask segmentation dataset. Official train/test split preserved: imagesTr/labelsTr "
                        "is the 3025-image official train split (nnU-Net's own 5-fold CV runs within it); "
                        "imagesTs/labelsTs is the 553-image official test split, held out entirely from "
                        "nnU-Net's plan/preprocess/train commands, for literature-comparable final evaluation "
                        "via this project's custom evaluate_multitask tooling.",
        "multitask": {
            "case_unit": "single_image",
            "views": ["image"],
            "tasks": {
                "disease": {
                    "label_dir": "labelsTr/disease",
                    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(DISEASE_CLASSES)}},
                    "regions_class_order": list(range(1, len(DISEASE_CLASSES) + 1)),
                    "multichannel": True,
                },
                "organ": {
                    "label_dir": "labelsTr/organ",
                    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(organ_classes)}},
                    "regions_class_order": list(range(1, len(organ_classes) + 1)),
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
    parser.add_argument("--annot-root", type=Path, default=DEFAULT_ANNOT_ROOT)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert_chestxdet_to_nnunet_raw(
        args.data_root, args.annot_root, args.pseudo_root, args.output_root, args.dataset_name, args.overwrite
    )
    print(output)


if __name__ == "__main__":
    main()
