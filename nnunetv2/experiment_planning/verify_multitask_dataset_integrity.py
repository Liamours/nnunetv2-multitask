from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json

from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.multitask_dataset import (
    get_multitask_label_paths,
    get_multitask_views,
    get_paired_multitask_filenames,
)


def _expected_labels(task_config: dict) -> List[int]:
    labels = sorted(int(v) for v in task_config["labels"].values())
    if labels != list(range(max(labels) + 1)):
        raise ValueError(f"Labels must be consecutive from 0. Got {labels}.")
    return labels


def _verify_label_file(label_file: str, reader, expected: List[int], image_shape) -> None:
    seg, _ = reader.read_seg(label_file)
    if seg.shape[1:] != image_shape:
        raise RuntimeError(f"Shape mismatch for {label_file}: expected {image_shape}, got {seg.shape[1:]}.")
    found = sorted(int(i) for i in np.unique(seg))
    unexpected = [i for i in found if i not in expected]
    if unexpected:
        raise RuntimeError(f"Unexpected labels in {label_file}. Expected {expected}, found {found}.")


def verify_paired_multitask_dataset_integrity(folder: str) -> None:
    dataset_json_file = join(folder, "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"Missing dataset.json in {folder}.")
    dataset_json = load_json(dataset_json_file)
    if dataset_json.get("multitask", {}).get("case_unit") != "paired_anterior_posterior":
        raise ValueError("verify_paired_multitask_dataset_integrity only supports paired_anterior_posterior datasets.")

    for required_folder in ("imagesTr", "labelsTr"):
        if not isdir(join(folder, required_folder)):
            raise FileNotFoundError(f"Missing {required_folder} in {folder}.")

    views = get_multitask_views(dataset_json)
    tasks: Dict[str, dict] = dataset_json["multitask"]["tasks"]
    dataset = get_paired_multitask_filenames(folder, dataset_json)
    if len(dataset) != int(dataset_json["numTraining"]):
        raise RuntimeError(f"numTraining={dataset_json['numTraining']} but found {len(dataset)} cases.")

    first_case = next(iter(dataset.values()))
    reader = determine_reader_writer_from_dataset_json(dataset_json, first_case["images"][0])()

    for task_name in tasks:
        if not isdir(join(folder, tasks[task_name].get("label_dir", join("labelsTr", task_name)))):
            raise FileNotFoundError(f"Missing label folder for task {task_name}.")

    invalid_ids = set()
    invalid_file = join(folder, "invalid_excluded.csv")
    if isfile(invalid_file):
        with open(invalid_file, "r", encoding="utf-8") as f:
            invalid_ids = {line.split(",")[0].strip() for line in f.readlines()[1:] if line.strip()}

    for case_id, case in dataset.items():
        patient_id = case_id.replace("bs80k_", "")
        if patient_id in invalid_ids:
            raise RuntimeError(f"Invalid patient {patient_id} appears in generated raw dataset.")

        if len(case["images"]) != len(views):
            raise RuntimeError(f"{case_id} expected {len(views)} images, got {len(case['images'])}.")
        for image_file in case["images"]:
            if not os.path.isfile(image_file):
                raise FileNotFoundError(image_file)

        images, _ = reader.read_images(case["images"])
        if len(images) != len(views):
            raise RuntimeError(f"{case_id} expected {len(views)} image channels, got {len(images)}.")
        image_shape = images.shape[1:]

        label_paths = get_multitask_label_paths(folder, case_id, dataset_json)
        for task_name, task_config in tasks.items():
            expected = _expected_labels(task_config)
            if len(label_paths[task_name]) != len(views):
                raise RuntimeError(f"{case_id}/{task_name} expected {len(views)} labels.")
            for label_file in label_paths[task_name]:
                if not os.path.isfile(label_file):
                    raise FileNotFoundError(label_file)
                _verify_label_file(label_file, reader, expected, image_shape)

    print("paired multitask dataset integrity OK")
