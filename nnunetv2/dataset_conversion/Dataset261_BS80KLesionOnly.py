from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json

from nnunetv2.dataset_conversion.Dataset260_BS80KLesionBoneMT import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    IMAGE_SIZE,
    _copy_image_as_png,
    _copy_label_png,
    _nested_view_files,
    _read_invalid_ids,
    _split_ids,
    _write_background_label,
)
from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity


DATASET_NAME = "Dataset261_BS80KLesionOnly"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_bs80k_lesion_only_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    raw_root = data_root / "bs80k" / "data" / "whole_body-raster-raw"
    lesion_root = data_root / "bs80k" / "labels" / "whole_body-lesion-segmentation" / "otsu_morphology-guarded_smooth"
    invalid_xlsx = data_root / "bs80k" / "data" / "archive" / "bs80k-invalid_list.xlsx"

    raw_views = _nested_view_files(raw_root, ".jpg")
    lesion_views = _nested_view_files(lesion_root, ".png")

    ant_images = {pid: v["anterior"] for pid, v in raw_views.items() if "anterior" in v}
    post_images = {pid: v["posterior"] for pid, v in raw_views.items() if "posterior" in v}
    lesion_ant = {pid: v["anterior"] for pid, v in lesion_views.items() if "anterior" in v}
    lesion_post = {pid: v["posterior"] for pid, v in lesion_views.items() if "posterior" in v}
    invalid_ids = _read_invalid_ids(invalid_xlsx)

    paired_raw_ids = sorted(set(ant_images) & set(post_images))
    eligible_ids = [patient_id for patient_id in paired_raw_ids if patient_id not in invalid_ids]
    split = _split_ids(eligible_ids, seed=42)

    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    images_tr = output_folder / "imagesTr"
    labels_tr = output_folder / "labelsTr" / "lesion"
    maybe_mkdir_p(images_tr)
    maybe_mkdir_p(labels_tr)

    manifest_rows = []
    invalid_rows = [{"id": patient_id, "reason": "invalid_list"} for patient_id in sorted(invalid_ids & set(paired_raw_ids))]

    for patient_id in eligible_ids:
        case_id = f"bs80k_{patient_id}"
        _copy_image_as_png(ant_images[patient_id], images_tr / f"{case_id}_0000.png")
        _copy_image_as_png(post_images[patient_id], images_tr / f"{case_id}_0001.png")

        for view_idx, (view_name, image_map, lesion_map) in enumerate((
            ("anterior", ant_images, lesion_ant),
            ("posterior", post_images, lesion_post),
        )):
            label_dst = labels_tr / f"{case_id}_{view_idx:04d}.png"
            lesion_source = lesion_map.get(patient_id)
            if lesion_source is None:
                _write_background_label(label_dst)
                lesion_source_text = "generated_background"
            else:
                _copy_label_png(lesion_source, label_dst, range(3))
                lesion_source_text = str(lesion_source)
            manifest_rows.append({
                "id": patient_id,
                "case_id": case_id,
                "split": split[patient_id],
                "view": view_name,
                "image": str(image_map[patient_id]),
                "lesion_label": lesion_source_text,
            })

    dataset_json = {
        "channel_names": {"0": "anterior", "1": "posterior"},
        "labels": {"background": 0},
        "numTraining": len(eligible_ids),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": dataset_name,
        "description": "BS-80K paired anterior/posterior lesion-only baseline dataset.",
        "converted_by": "nnunetv2-multitask",
        "multitask": {
            "case_unit": "paired_anterior_posterior",
            "views": ["anterior", "posterior"],
            "tasks": {
                "lesion": {
                    "label_dir": "labelsTr/lesion",
                    "labels": {"background": 0, "benign": 1, "malignant": 2},
                },
            },
        },
    }
    save_json(dataset_json, output_folder / "dataset.json", sort_keys=False)
    _write_csv(output_folder / "conversion_manifest.csv", manifest_rows,
               ["id", "case_id", "split", "view", "image", "lesion_label"])
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
    output = convert_bs80k_lesion_only_to_nnunet_raw(args.data_root, args.output_root, args.dataset_name, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
