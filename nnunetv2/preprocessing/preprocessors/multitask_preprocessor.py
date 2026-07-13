from __future__ import annotations

import multiprocessing
import shutil
from time import sleep
from typing import List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json, maybe_mkdir_p
from tqdm import tqdm

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, comp_blosc2_params
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.multitask_dataset import (
    get_multitask_task_names,
    load_multitask_label_stack,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class MultiTaskPreprocessor(DefaultPreprocessor):
    def run_case(self, image_files: List[str], seg_file: Union[str, dict, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()
        data, data_properties = rw.read_images(image_files)

        if isinstance(seg_file, dict):
            task_order = get_multitask_task_names(dataset_json)
            seg = load_multitask_label_stack(seg_file, rw, task_order=task_order)
        elif seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if self.verbose:
            print(seg_file)
        data, seg, data_properties = self.run_case_npy(data, seg, data_properties, plans_manager,
                                                       configuration_manager, dataset_json)
        return data, seg, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: Union[str, dict],
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16)
        block_size_data, chunk_size_data = comp_blosc2_params(
            data.shape,
            tuple(configuration_manager.patch_size),
            data.itemsize)
        block_size_seg, chunk_size_seg = comp_blosc2_params(
            seg.shape,
            tuple(configuration_manager.patch_size),
            seg.itemsize)

        nnUNetDatasetBlosc2.save_case(data, seg, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg)

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)
        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            workers = [j for j in p._pool]
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'],
                                           dataset[k].get('multitask_labels', dataset[k]['label']),
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc="Preprocessing multitask cases", total=len(dataset),
                      disable=not getattr(self, 'show_progress_bar', True)) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('A preprocessing worker stopped unexpectedly.')
                    done = [i for i in remaining if r[i].ready()]
                    for i in done:
                        r[i].get()
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
