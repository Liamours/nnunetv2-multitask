import unittest

import torch

from nnunetv2.architecture.segformer_components import split_and_restack_view_logits, to_paired_view_canvas
from nnunetv2.architecture.multitask_segformer import MultiTaskDualHeadSegFormer
from nnunetv2.experiment_planning.experiment_planners.segformer_multitask_experiment_planner import (
    SegFormerMultiTaskExperimentPlanner,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTaskSegFormer import nnUNetTrainerMultiTaskSegFormer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def make_segformer_test_dataset_json():
    return {
        "channel_names": {"0": "anterior", "1": "posterior"},
        "labels": {"background": 0},
        "file_ending": ".png",
        "multitask": {
            "case_unit": "paired_anterior_posterior",
            "views": ["anterior", "posterior"],
            "tasks": {
                "task1": {"labels": {"background": 0, "a": 1, "b": 2}},
                "task2": {"labels": {"background": 0, "a": 1, "b": 2, "c": 3}},
            },
        },
    }


def make_segformer_test_plans():
    return {
        "dataset_name": "Dataset999_Test",
        "plans_name": "nnUNetPlansMultiTaskSegFormer",
        "original_median_spacing_after_transp": [1.0, 1.0],
        "original_median_shape_after_transp": [32, 32],
        "image_reader_writer": "NaturalImage2DIO",
        "transpose_forward": [0, 1],
        "transpose_backward": [0, 1],
        "experiment_planner_used": "SegFormerMultiTaskExperimentPlanner",
        "label_manager": "MultiTaskLabelManager",
        "foreground_intensity_properties_per_channel": {},
        "configurations": {
            "2d": {
                "data_identifier": "nnUNetPlansMultiTaskSegFormer_2d",
                "preprocessor_name": "MultiTaskPreprocessor",
                "batch_size": 1,
                "patch_size": [32, 32],
                "median_image_size_in_voxels": [32, 32],
                "spacing": [1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization", "ZScoreNormalization"],
                "use_mask_for_norm": [False, False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
                "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
                "resampling_fn_probabilities": "no_resampling_hack",
                "resampling_fn_probabilities_kwargs": {},
                "batch_dice": False,
                "architecture": {
                    "network_class_name": "nnunetv2.architecture.multitask_segformer.MultiTaskDualHeadSegFormer",
                    "arch_kwargs": {
                        "mit_variant": "mit_b0",
                        "pretrained_hf_name": None,
                        "decoder_embedding_dim": 32,
                        "multitask": {
                            "variant": "dual_head",
                            "tasks": [
                                {"name": "task1", "num_classes": 3, "output_channels": 6, "loss_weight": 1.0},
                                {"name": "task2", "num_classes": 4, "output_channels": 8, "loss_weight": 0.5},
                            ],
                        },
                    },
                    "_kw_requires_import": [],
                },
            }
        },
    }


def make_segformer_trainer_test_plans():
    plans = make_segformer_test_plans()
    plans["continue_training"] = False
    return plans


class TestMultiTaskSegFormerArchitecture(unittest.TestCase):
    def _build_network(self):
        plans_manager = PlansManager(make_segformer_test_plans())
        configuration_manager = plans_manager.get_configuration("2d")
        dataset_json = make_segformer_test_dataset_json()
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        return nnUNetTrainerMultiTaskSegFormer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

    def test_dual_head_forward_pass_shape_and_keys(self):
        network = self._build_network()
        self.assertIsInstance(network, MultiTaskDualHeadSegFormer)
        output = network(torch.rand(1, 2, 32, 32))

        self.assertEqual(set(output.keys()), {"task1", "task2"})
        self.assertIsInstance(output["task1"], torch.Tensor)
        self.assertIsInstance(output["task2"], torch.Tensor)
        # output_channels = num_classes * num_views: task1 3*2=6, task2 4*2=8
        self.assertEqual(output["task1"].shape, (1, 6, 32, 32))
        self.assertEqual(output["task2"].shape, (1, 8, 32, 32))

    def test_deep_supervision_true_raises_at_construction(self):
        plans_manager = PlansManager(make_segformer_test_plans())
        configuration_manager = plans_manager.get_configuration("2d")
        dataset_json = make_segformer_test_dataset_json()
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

        with self.assertRaises(NotImplementedError):
            nnUNetTrainerMultiTaskSegFormer.build_network_architecture(
                plans_manager,
                configuration_manager,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=True,
            )

    def test_set_deep_supervision_is_a_safe_noop(self):
        # Direct regression guard for nnUNetTrainerMultiTask.perform_actual_validation, which calls
        # set_deep_supervision_enabled(False) then unconditionally (True) after every validation
        # pass regardless of the trainer's own enable_deep_supervision setting (verified directly
        # at nnUNetTrainerMultiTask.py:424,494). Without a true no-op here, every real
        # end-of-training validation pass would crash.
        network = self._build_network()
        network.set_deep_supervision(True)
        network.set_deep_supervision(False)

    def test_view_split_is_channel_and_order_correct(self):
        anterior = torch.full((1, 1, 4, 4), 1.0)
        posterior = torch.full((1, 1, 4, 4), 2.0)
        x = torch.cat([anterior, posterior], dim=1)  # (1, 2, 4, 4), channel 0 = anterior, 1 = posterior

        canvas = to_paired_view_canvas(x, num_views=2)
        self.assertEqual(canvas.shape, (1, 3, 4, 8))
        self.assertTrue(torch.all(canvas[:, :, :, :4] == 1.0))
        self.assertTrue(torch.all(canvas[:, :, :, 4:] == 2.0))

        # simulate a decoder+head prediction at canvas resolution, anterior/posterior halves
        # distinguishable so a silently swapped or merged view axis would fail the assertions below
        logits = torch.zeros(1, 3, 4, 8)
        logits[:, :, :, :4] = 10.0
        logits[:, :, :, 4:] = -10.0

        restacked = split_and_restack_view_logits(logits, num_views=2)
        self.assertEqual(restacked.shape, (1, 6, 4, 4))
        self.assertTrue(torch.all(restacked[:, 0:3] == 10.0))
        self.assertTrue(torch.all(restacked[:, 3:6] == -10.0))

    def test_trainer_loss_and_backward_pass(self):
        trainer = nnUNetTrainerMultiTaskSegFormer(
            make_segformer_trainer_test_plans(), "2d", 0, make_segformer_test_dataset_json(),
            device=torch.device("cpu"),
        )
        self.assertFalse(trainer.enable_deep_supervision)

        target = torch.cat(
            [
                torch.randint(0, 3, (1, 2, 32, 32), dtype=torch.int16),
                torch.randint(0, 4, (1, 2, 32, 32), dtype=torch.int16),
            ],
            dim=1,
        )
        split_target = trainer._split_targets(target)
        output = {
            "task1": torch.randn((1, 6, 32, 32), requires_grad=True),
            "task2": torch.randn((1, 8, 32, 32), requires_grad=True),
        }
        reshaped_output, reshaped_target = trainer._reshape_all_outputs_targets_for_loss(output, split_target)
        self.assertEqual(reshaped_output["task1"].shape, (2, 3, 32, 32))
        self.assertEqual(reshaped_output["task2"].shape, (2, 4, 32, 32))

        loss = trainer._build_loss()
        computed = loss(reshaped_output, reshaped_target)
        computed.backward()
        self.assertTrue(torch.isfinite(computed))

    def test_configure_optimizers_keeps_differential_lr_through_decay(self):
        trainer = nnUNetTrainerMultiTaskSegFormer(
            make_segformer_trainer_test_plans(), "2d", 0, make_segformer_test_dataset_json(),
            device=torch.device("cpu"),
        )
        trainer.network = self._build_network()
        trainer.num_epochs = 10

        optimizer, lr_scheduler = trainer.configure_optimizers()
        self.assertEqual(len(optimizer.param_groups), 2)
        backbone_group, head_group = optimizer.param_groups
        self.assertAlmostEqual(backbone_group["lr"], 1e-5)
        self.assertAlmostEqual(head_group["lr"], 1e-4)

        lr_scheduler.step(5)
        # PolyLRScheduler.step() would otherwise reset every group to the SAME decayed value
        # computed from one shared initial_lr (nnunetv2/training/lr_scheduler/polylr.py:18-20) -
        # the differential ratio must survive the decay, not just the initial construction.
        self.assertLess(optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"], 1e-5 / 1e-4, places=6
        )


class TestSegFormerMultiTaskPlans(unittest.TestCase):
    def _make_planner(self, **overrides):
        planner = SegFormerMultiTaskExperimentPlanner.__new__(SegFormerMultiTaskExperimentPlanner)
        planner.multitask_variant = "dual_head"
        planner.multitask_tasks = [
            {"name": "lesion", "num_classes": 3, "loss_weight": 0.8},
            {"name": "bone", "num_classes": 13, "loss_weight": 0.2},
        ]
        planner.multitask_views = ["anterior", "posterior"]
        planner.dataset_json = {"multitask": {"case_unit": "paired_anterior_posterior"}}
        planner.mit_variant = "mit_b2"
        planner.pretrained_hf_name = "nvidia/mit-b2"
        planner.decoder_embedding_dim = 768
        planner.fixed_patch_size = (896, 256)
        planner.fixed_batch_size = 4
        for key, value in overrides.items():
            setattr(planner, key, value)
        return planner

    def test_architecture_has_no_unet_pooling_fields(self):
        planner = self._make_planner()
        arch = planner._make_multitask_architecture(
            {"network_class_name": "placeholder", "arch_kwargs": {}, "_kw_requires_import": ["conv_op"]}
        )

        self.assertEqual(
            arch["network_class_name"], "nnunetv2.architecture.multitask_segformer.MultiTaskDualHeadSegFormer"
        )
        self.assertEqual(arch["_kw_requires_import"], [])
        for unet_field in ("n_stages", "kernel_sizes", "strides", "features_per_stage", "conv_op"):
            self.assertNotIn(unet_field, arch["arch_kwargs"])
        self.assertEqual(arch["arch_kwargs"]["mit_variant"], "mit_b2")
        self.assertEqual(arch["arch_kwargs"]["multitask"]["tasks"][0]["output_channels"], 6)
        self.assertEqual(arch["arch_kwargs"]["multitask"]["tasks"][1]["output_channels"], 26)

    def test_unknown_variant_raises(self):
        planner = self._make_planner(multitask_variant="mid_fission")
        with self.assertRaisesRegex(ValueError, "Unknown multitask_variant"):
            planner._make_multitask_architecture({"network_class_name": "x", "arch_kwargs": {}, "_kw_requires_import": []})

    def test_recompute_multitask_memory_plan_sets_fixed_patch_and_batch(self):
        planner = self._make_planner()
        plan = {
            "architecture": {"network_class_name": "placeholder", "arch_kwargs": {}, "_kw_requires_import": []},
            "patch_size": [1, 1],
            "batch_size": 99,
        }
        result = planner._recompute_multitask_memory_plan(
            plan, spacing=None, median_shape=None, approximate_n_voxels_dataset=None, _cache={}
        )
        self.assertEqual(result["patch_size"], [896, 256])
        self.assertEqual(result["batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
