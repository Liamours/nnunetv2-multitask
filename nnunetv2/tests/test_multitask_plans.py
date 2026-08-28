import unittest
from copy import deepcopy

import torch

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.multitask_experiment_planner import MultiTaskExperimentPlanner
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import _parse_multitask_tasks
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.multitask_dataset import is_multitask_dataset, is_paired_multiview_multitask_dataset
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def make_test_plans():
    return {
        "dataset_name": "Dataset999_Test",
        "plans_name": "nnUNetPlansMultiTask",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [8, 8, 8],
        "image_reader_writer": "NibabelIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "experiment_planner_used": "MultiTaskExperimentPlanner",
        "label_manager": "MultiTaskLabelManager",
        "foreground_intensity_properties_per_channel": {},
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlansMultiTask_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": 2,
                "patch_size": [8, 8, 8],
                "median_image_size_in_voxels": [8, 8, 8],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
                "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
                "resampling_fn_probabilities": "no_resampling_hack",
                "resampling_fn_probabilities_kwargs": {},
                "batch_dice": False,
                "architecture": {
                    "network_class_name": "nnunetv2.architecture.multitask_unet.MultiTaskDualHeadUNet",
                    "arch_kwargs": {
                        "n_stages": 3,
                        "features_per_stage": [8, 16, 32],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                        "n_conv_per_stage": [2, 2, 2],
                        "n_conv_per_stage_decoder": [2, 2],
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.LeakyReLU",
                        "nonlin_kwargs": {"inplace": True},
                        "multitask": {
                            "variant": "dual_head",
                            "tasks": [
                                {"name": "task1", "num_classes": 2, "loss_weight": 1.0},
                                {"name": "task2", "num_classes": 3, "loss_weight": 0.5},
                            ],
                        },
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
                },
            }
        },
    }


def make_test_dataset_json():
    return {
        "channel_names": {"0": "image"},
        "labels": {"background": 0, "foreground": 1},
        "file_ending": ".nii.gz",
        "multitask": {
            "tasks": {
                "task1": {"labels": {"background": 0, "lesion": 1}},
                "task2": {"labels": {"background": 0, "organ_a": 1, "organ_b": 2}},
            }
        },
    }


def make_trainer_test_plans():
    plans = make_test_plans()
    plans["continue_training"] = False
    return plans


class TestMultiTaskPlans(unittest.TestCase):
    def test_configuration_and_label_manager_are_multitask(self):
        plans_manager = PlansManager(make_test_plans())
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = make_test_dataset_json()

        self.assertTrue(configuration_manager.is_multitask)
        self.assertEqual(configuration_manager.multitask_config["variant"], "dual_head")

        label_manager = plans_manager.get_label_manager(dataset_json)
        self.assertTrue(label_manager.is_multitask)
        self.assertEqual(label_manager.task_order, ["task1", "task2"])
        self.assertEqual(label_manager.task_num_segmentation_heads(), {"task1": 2, "task2": 3})

    def test_multitask_cli_task_parser(self):
        tasks = _parse_multitask_tasks(["lesion:3:0.8", "bone:13:0.2"])

        self.assertEqual(tasks, [
            {"name": "lesion", "num_classes": 3, "loss_weight": 0.8},
            {"name": "bone", "num_classes": 13, "loss_weight": 0.2},
        ])

    def test_multitask_cli_task_parser_requires_two_tasks(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            _parse_multitask_tasks(["lesion:3"])

    def test_planner_kwargs_are_validated_before_dataset_access(self):
        with self.assertRaisesRegex(ValueError, "does not support arguments"):
            plan_experiment_dataset(
                999,
                ExperimentPlanner,
                planner_kwargs={"multitask_variant": "dual_decoder"},
            )

        try:
            plan_experiment_dataset(
                999,
                MultiTaskExperimentPlanner,
                planner_kwargs={"multitask_variant": "dual_decoder"},
            )
        except ValueError:
            raise
        except Exception:
            pass

    def test_trainer_rejects_plan_dataset_json_task_name_mismatch(self):
        dataset_json = deepcopy(make_test_dataset_json())
        dataset_json["multitask"]["tasks"]["other"] = dataset_json["multitask"]["tasks"].pop("task2")

        with self.assertRaisesRegex(ValueError, "different task names"):
            nnUNetTrainerMultiTask(make_trainer_test_plans(), "3d_fullres", 0, dataset_json, device=torch.device("cpu"))

    def test_is_multitask_dataset_covers_paired_and_single_image_alike(self):
        single_image = make_test_dataset_json()  # no case_unit at all, tasks defined
        self.assertTrue(is_multitask_dataset(single_image))
        self.assertFalse(is_paired_multiview_multitask_dataset(single_image))

        single_image_explicit = deepcopy(single_image)
        single_image_explicit["multitask"]["case_unit"] = "single_image"
        single_image_explicit["multitask"]["views"] = ["image"]
        self.assertTrue(is_multitask_dataset(single_image_explicit))
        self.assertFalse(is_paired_multiview_multitask_dataset(single_image_explicit))

        paired = deepcopy(single_image)
        paired["multitask"]["case_unit"] = "paired_anterior_posterior"
        paired["multitask"]["views"] = ["anterior", "posterior"]
        self.assertTrue(is_multitask_dataset(paired))
        self.assertTrue(is_paired_multiview_multitask_dataset(paired))

        not_multitask = {"channel_names": {"0": "image"}, "labels": {"background": 0, "fg": 1}}
        self.assertFalse(is_multitask_dataset(not_multitask))
        self.assertFalse(is_paired_multiview_multitask_dataset(not_multitask))

    def test_trainer_rejects_plan_dataset_json_head_count_mismatch(self):
        dataset_json = deepcopy(make_test_dataset_json())
        dataset_json["multitask"]["tasks"]["task2"]["labels"] = {"background": 0, "organ_a": 1}

        with self.assertRaisesRegex(ValueError, "num_classes must match"):
            nnUNetTrainerMultiTask(make_trainer_test_plans(), "3d_fullres", 0, dataset_json, device=torch.device("cpu"))

    def test_multichannel_task_reports_regions_and_raw_channel_count(self):
        # ChestX-Det/SegRap2023 shape: independently-overlapping per-class channels within one
        # task (not derivable from a single exclusive integer map like BS-80K's bone regions).
        dataset_json = make_test_dataset_json()
        dataset_json["multitask"]["tasks"]["task2"] = {
            "labels": {"background": 0, "organ_a": 1, "organ_b": 2, "organ_c": 3},
            "regions_class_order": [1, 2, 3],
            "multichannel": True,
        }
        plans_manager = PlansManager(make_test_plans())
        label_manager = plans_manager.get_label_manager(dataset_json)

        self.assertFalse(label_manager.is_multichannel_task("task1"))
        self.assertTrue(label_manager.is_multichannel_task("task2"))
        self.assertEqual(label_manager.task_num_raw_channels(), {"task1": 1, "task2": 3})
        self.assertEqual(label_manager.task_num_segmentation_heads(), {"task1": 2, "task2": 3})

        task2_manager = label_manager.get_task_label_manager("task2")
        self.assertTrue(task2_manager.has_regions)
        self.assertEqual(task2_manager.inference_nonlin, torch.sigmoid)


if __name__ == "__main__":
    unittest.main()
