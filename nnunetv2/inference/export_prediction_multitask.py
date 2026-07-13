import os
from typing import Dict, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle, maybe_mkdir_p

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape


def convert_predicted_logits_to_segmentation_with_correct_shape_multitask(
    predicted_logits_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    plans_manager,
    configuration_manager,
    multitask_label_manager,
    properties_dict: dict,
    return_probabilities: bool = False,
    num_threads_torch: int = default_num_processes,
):
    result = {}
    for task_name, predicted_logits in predicted_logits_dict.items():
        label_manager = multitask_label_manager.get_task_label_manager(task_name)
        if isinstance(predicted_logits, dict):
            result[task_name] = {
                view_name: convert_predicted_logits_to_segmentation_with_correct_shape(
                    view_logits,
                    plans_manager,
                    configuration_manager,
                    label_manager,
                    properties_dict,
                    return_probabilities=return_probabilities,
                    num_threads_torch=num_threads_torch,
                )
                for view_name, view_logits in predicted_logits.items()
            }
        else:
            result[task_name] = convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_logits,
                plans_manager,
                configuration_manager,
                label_manager,
                properties_dict,
                return_probabilities=return_probabilities,
                num_threads_torch=num_threads_torch,
            )
    return result


def export_prediction_from_logits_multitask(
    predicted_array_or_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    properties_dict: dict,
    configuration_manager,
    plans_manager,
    dataset_json_dict_or_file,
    output_file_truncated: str,
    save_probabilities: bool = False,
    num_threads_torch: int = default_num_processes,
):
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    multitask_label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    views = dataset_json_dict_or_file.get("multitask", {}).get("views", [])
    is_paired_multiview = dataset_json_dict_or_file.get("multitask", {}).get("case_unit") == "paired_anterior_posterior"
    if is_paired_multiview:
        predicted_array_or_dict = _split_paired_view_logits(
            predicted_array_or_dict,
            multitask_label_manager,
            views,
        )
    ret = convert_predicted_logits_to_segmentation_with_correct_shape_multitask(
        predicted_array_or_dict,
        plans_manager,
        configuration_manager,
        multitask_label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
        num_threads_torch=num_threads_torch,
    )

    rw = plans_manager.image_reader_writer_class()
    for task_name, task_ret in ret.items():
        if isinstance(task_ret, dict):
            for view_name, view_ret in task_ret.items():
                _write_task_prediction(
                    rw,
                    view_ret,
                    properties_dict,
                    dataset_json_dict_or_file,
                    output_file_truncated,
                    task_name,
                    view_name,
                    save_probabilities,
                )
            continue
        _write_task_prediction(
            rw,
            task_ret,
            properties_dict,
            dataset_json_dict_or_file,
            output_file_truncated,
            task_name,
            None,
            save_probabilities,
        )


def _split_paired_view_logits(predicted_logits_dict, multitask_label_manager, views):
    if not views:
        raise ValueError("Paired multiview export requires dataset_json['multitask']['views'].")
    split = {}
    for task_name, predicted_logits in predicted_logits_dict.items():
        label_manager = multitask_label_manager.get_task_label_manager(task_name)
        num_classes = label_manager.num_segmentation_heads
        expected_channels = len(views) * num_classes
        if predicted_logits.shape[0] != expected_channels:
            raise ValueError(
                f"Task {task_name} expected {expected_channels} channels "
                f"({len(views)} views x {num_classes} classes), got {predicted_logits.shape[0]}."
            )
        split[task_name] = {
            view_name: predicted_logits[view_idx * num_classes:(view_idx + 1) * num_classes].detach().cpu()
            if isinstance(predicted_logits, torch.Tensor)
            else predicted_logits[view_idx * num_classes:(view_idx + 1) * num_classes]
            for view_idx, view_name in enumerate(views)
        }
    return split


def _write_task_prediction(
    rw,
    task_ret,
    properties_dict,
    dataset_json_dict,
    output_file_truncated,
    task_name,
    view_name,
    save_probabilities,
):
    suffix = task_name if view_name is None else os.path.join(task_name, view_name)
    task_output = os.path.join(
        os.path.dirname(output_file_truncated),
        suffix,
        os.path.basename(output_file_truncated),
    )
    maybe_mkdir_p(os.path.dirname(task_output))
    if save_probabilities:
        segmentation_final, probabilities_final = task_ret
        np.savez_compressed(task_output + ".npz", probabilities=probabilities_final)
        save_pickle(properties_dict, task_output + ".pkl")
    else:
        segmentation_final = task_ret
    rw.write_seg(segmentation_final, task_output + dataset_json_dict["file_ending"], properties_dict)
