from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles


def is_paired_multiview_multitask_dataset(dataset_json: dict) -> bool:
    multitask = dataset_json.get("multitask", {})
    return multitask.get("case_unit") == "paired_anterior_posterior"


def is_multitask_dataset(dataset_json: dict) -> bool:
    """True for any dataset declaring multitask tasks, paired-view or single-image alike.

    Use this (not is_paired_multiview_multitask_dataset) wherever the question is "does this dataset
    need multitask-aware file discovery/preprocessing/integrity checks at all" - the paired-only check
    is reserved for the small set of call sites that actually need to know whether view-stacking/
    reshaping applies (nnUNetTrainerMultiTask, export_prediction_multitask, output-channel multiplication).
    """
    multitask = dataset_json.get("multitask", {})
    return bool(multitask.get("tasks"))


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
        if task_config.get("multichannel", False):
            # one file per class (background excluded), not one per view - matches nnU-Net's own
            # imagesTr _0000/_0001/... channel convention, just keyed by class order instead
            num_channels = len(task_config["labels"]) - 1
            label_paths[task_name] = [
                join(raw_dataset_folder, label_dir, f"{case_id}_{channel_idx:04d}{dataset_json['file_ending']}")
                for channel_idx in range(num_channels)
            ]
        else:
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


def load_multitask_union_label(label_paths: Dict[str, List[str]], reader_writer,
                               task_order: List[str] = None) -> np.ndarray:
    """Foreground union across every task/channel, without ever holding the full stack in memory.

    load_multitask_label_stack + make_multitask_union_label needs ~2x the size of all channels
    combined (the stack, plus the `stack > 0` bool temporary). That is fine for 2D, but SegRap2023's
    47 channels at 127x1024x1024 come to ~6.3 GB per copy, i.e. ~12.5 GB peak per case - which OOMs
    even before multiprocessing multiplies it. Callers that only need the union (the fingerprint
    extractor) should use this instead: peak is one channel plus the accumulator.
    """
    task_order = task_order or list(label_paths.keys())
    union = None
    expected_shape = None
    for task_name in task_order:
        for label_file in label_paths[task_name]:
            seg, _ = reader_writer.read_seg(label_file)
            if expected_shape is None:
                expected_shape = seg.shape[1:]
                union = np.zeros((1, *expected_shape), dtype=np.uint8)
            if seg.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch in multitask labels. Expected {expected_shape}, got {seg.shape[1:]} for {label_file}."
                )
            np.logical_or(union[0], seg[0] > 0, out=union[0].view(bool))
    if union is None:
        raise RuntimeError("No multitask label files given.")
    return union


def get_multitask_raw_channel_slices(dataset_json: dict) -> Dict[str, slice]:
    """Where each task's raw channel(s) sit in the stacked seg array, in task_order. Single source of
    truth shared by nnUNetTrainerMultiTask (same arithmetic) and sample_multitask_foreground_locations."""
    tasks = dataset_json.get("multitask", {}).get("tasks", {})
    num_views = len(get_multitask_views(dataset_json)) or 1
    slices = {}
    offset = 0
    for name, cfg in tasks.items():
        n = (len(cfg["labels"]) - 1) if cfg.get("multichannel", False) else num_views
        slices[name] = slice(offset, offset + n)
        offset += n
    return slices


def sample_multitask_foreground_locations(seg: np.ndarray, dataset_json: dict, seed: int = 1234,
                                          verbose: bool = False, min_num_samples: int = 10000,
                                          min_percent_coverage: float = 0.01) -> dict:
    """Per-task foreground-location sampling for multitask datasets.

    DefaultPreprocessor._sample_foreground_locations searches the WHOLE stacked seg array for label
    VALUES, channel-blind - correct for one exclusive integer map, wrong once any task is multichannel
    (each channel is its own independent {0,1} mask, not a label value to search for across every
    channel). This scopes each task to its own channel slice and, for multichannel tasks, samples each
    channel independently - reusing the existing (tested, vectorized) sampling algorithm per task/
    channel rather than reimplementing it. Keys are (task_name, class_or_region) tuples so tasks never
    collide (e.g. disease region 1 and organ region 1 are unrelated) - downstream code (get_bbox in
    the dataloader) only needs hashable keys, already handles tuples.
    """
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    tasks = dataset_json.get("multitask", {}).get("tasks", {})
    slices = get_multitask_raw_channel_slices(dataset_json)
    combined = {}
    for task_name, task_cfg in tasks.items():
        task_seg = seg[slices[task_name]]
        if task_cfg.get("multichannel", False):
            for c in range(task_seg.shape[0]):
                result = DefaultPreprocessor._sample_foreground_locations(
                    task_seg[c:c + 1], [1], seed=seed, verbose=verbose,
                    min_num_samples=min_num_samples, min_percent_coverage=min_percent_coverage,
                )
                combined[(task_name, c)] = result[1]
        else:
            regions_or_labels = task_cfg.get("regions_class_order") or sorted(
                v for v in task_cfg["labels"].values() if v != 0
            )
            result = DefaultPreprocessor._sample_foreground_locations(
                task_seg, regions_or_labels, seed=seed, verbose=verbose,
                min_num_samples=min_num_samples, min_percent_coverage=min_percent_coverage,
            )
            for k, v in result.items():
                combined[(task_name, k)] = v
    return combined


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
