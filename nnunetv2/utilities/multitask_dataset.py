from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles


def is_paired_multiview_multitask_dataset(dataset_json: dict) -> bool:
    multitask = dataset_json.get("multitask", {})
    return multitask.get("case_unit") == "paired_anterior_posterior"


def get_multitask_views(dataset_json: dict) -> List[str]:
    return list(dataset_json.get("multitask", {}).get("views", ["anterior", "posterior"]))


def get_multitask_task_names(dataset_json: dict) -> List[str]:
    return list(dataset_json.get("multitask", {}).get("tasks", {}).keys())


def get_multitask_label_paths(raw_dataset_folder: str, case_id: str, dataset_json: dict) -> Dict[str, List[str]]:
    views = get_multitask_views(dataset_json)
    tasks = dataset_json.get("multitask", {}).get("tasks", {})
    label_paths = {}
    for task_name, task_config in tasks.items():
        label_dir = task_config.get("label_dir", join("labelsTr", task_name))
        label_paths[task_name] = [
            join(raw_dataset_folder, label_dir, f"{case_id}_{view_idx:04d}{dataset_json['file_ending']}")
            for view_idx, _ in enumerate(views)
        ]
    return label_paths


def get_paired_multitask_filenames(raw_dataset_folder: str, dataset_json: dict) -> dict:
    file_ending = dataset_json["file_ending"]
    image_files = subfiles(join(raw_dataset_folder, "imagesTr"), suffix=file_ending, join=False)
    channel_suffix_len = len(file_ending) + 5
    identifiers = sorted(np.unique([i[:-channel_suffix_len] for i in image_files]))
    dataset = {}
    for case_id in identifiers:
        images = [
            join(raw_dataset_folder, "imagesTr", f"{case_id}_{channel_idx:04d}{file_ending}")
            for channel_idx in range(len(dataset_json["channel_names"]))
        ]
        dataset[case_id] = {
            "images": images,
            "label": None,
            "multitask_labels": get_multitask_label_paths(raw_dataset_folder, case_id, dataset_json),
        }
    return dataset


def load_multitask_label_stack(label_paths: Dict[str, List[str]], reader_writer, task_order: List[str] = None) -> np.ndarray:
    task_order = task_order or list(label_paths.keys())
    channels = []
    expected_shape = None
    for task_name in task_order:
        for label_file in label_paths[task_name]:
            seg, _ = reader_writer.read_seg(label_file)
            if expected_shape is None:
                expected_shape = seg.shape[1:]
            if seg.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch in multitask labels. Expected {expected_shape}, got {seg.shape[1:]} for {label_file}."
                )
            channels.append(seg[0])
    return np.stack(channels, axis=0)


def make_multitask_union_label(label_stack: np.ndarray) -> np.ndarray:
    union = np.max(label_stack > 0, axis=0, keepdims=True).astype(np.uint8)
    return union


def assert_all_multitask_label_files_exist(dataset: dict) -> None:
    missing = []
    for case_id, case in dataset.items():
        for image_file in case["images"]:
            if not os.path.isfile(image_file):
                missing.append(image_file)
        for task_paths in case.get("multitask_labels", {}).values():
            for label_file in task_paths:
                if not os.path.isfile(label_file):
                    missing.append(label_file)
    if missing:
        raise FileNotFoundError("Missing multitask raw files:\n" + "\n".join(missing[:100]))
