from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json
from PIL import Image

from nnunetv2.dataset_conversion.Dataset262_ChestXDetDiseaseOrganMT import (
    DEFAULT_ANNOT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DISEASE_CLASSES,
    _copy_image_as_grayscale_png,
    _write_disease_labels,
)
from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity

DATASET_NAME = "Dataset263_ChestXDetDiseaseOnly"


def convert_chestxdet_disease_only_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    annot_root: Path = DEFAULT_ANNOT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    images_tr = output_folder / "imagesTr"
    disease_tr = output_folder / "labelsTr" / "disease"
    images_ts = output_folder / "imagesTs"
    disease_ts = output_folder / "labelsTs" / "disease"
    for d in (images_tr, disease_tr, images_ts, disease_ts):
        maybe_mkdir_p(d)

    n_train = 0
    for split, annot_file, img_dir, out_images, out_disease in [
        ("train", "ChestX_Det_train.json", "train", images_tr, disease_tr),
        ("test", "ChestX_Det_test.json", "test", images_ts, disease_ts),
    ]:
        entries = json.loads((annot_root / annot_file).read_text())
        for entry in entries:
            stem = Path(entry["file_name"]).stem
            case_id = f"chestxdet_{stem}"
            src_img = data_root / img_dir / entry["file_name"]
            with Image.open(src_img) as im:
                size = im.size

            _copy_image_as_grayscale_png(src_img, out_images / f"{case_id}_0000.png")
            _write_disease_labels(entry, size, out_disease, case_id)

            if split == "train":
                n_train += 1

    dataset_json = {
        "channel_names": {"0": "chest_xray"},
        "labels": {"background": 0},
        "numTraining": n_train,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": dataset_name,
        "description": "ChestX-Det disease-only baseline (single-task A1 ablation), multichannel, "
                        "official train/test split preserved (see Dataset262 for details).",
        "converted_by": "nnunetv2-multitask",
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert_chestxdet_disease_only_to_nnunet_raw(
        args.data_root, args.annot_root, args.output_root, args.dataset_name, args.overwrite
    )
    print(output)


if __name__ == "__main__":
    main()
