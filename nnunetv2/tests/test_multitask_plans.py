import unittest

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


if __name__ == "__main__":
    unittest.main()
