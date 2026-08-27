import unittest

import torch

from nnunetv2.tests.test_multitask_plans import make_test_dataset_json, make_test_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

_FISSION_CLASS_BY_VARIANT = {
    "early_mid_fission": "nnunetv2.architecture.multitask_unet.MultiTaskEarlyMidUNet",
    "mid_fission": "nnunetv2.architecture.multitask_unet.MultiTaskMidUNet",
}


def _make_deep_test_plans(variant: str) -> dict:
    """5 encoder stages / 4 decoder stages. make_test_plans()'s stock 2-decoder-stage fixture can't
    tell Early-Mid (share 1 of 4) and Mid (share 3 of 4) apart - both would collapse onto the same
    single split point. 32-voxel input so four stride-2 pools don't collapse the spatial size."""
    plans = make_test_plans()
    arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    arch_kwargs.update({
        "n_stages": 5,
        "features_per_stage": [8, 16, 32, 64, 128],
        "kernel_sizes": [[3, 3, 3]] * 5,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        "n_conv_per_stage": [2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2],
    })
    plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"] = _FISSION_CLASS_BY_VARIANT[variant]
    arch_kwargs["multitask"]["variant"] = variant
    return plans


class TestMultiTaskArchitecture(unittest.TestCase):
    def _build_network(self, variant: str):
        plans = make_test_plans()
        plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"] = (
            "nnunetv2.architecture.multitask_unet.MultiTaskDualHeadUNet"
            if variant == "dual_head"
            else "nnunetv2.architecture.multitask_unet.MultiTaskDualDecoderUNet"
        )
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["variant"] = variant
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = make_test_dataset_json()
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        return nnUNetTrainerMultiTask.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=True,
        )

    def test_dual_head_forward(self):
        network = self._build_network("dual_head")
        output = network(torch.rand(1, 1, 8, 8, 8))
        self.assertEqual(set(output.keys()), {"task1", "task2"})
        self.assertEqual(output["task1"][0].shape[1], 2)
        self.assertEqual(output["task2"][0].shape[1], 3)

    def test_dual_decoder_forward(self):
        network = self._build_network("dual_decoder")
        output = network(torch.rand(1, 1, 8, 8, 8))
        self.assertEqual(set(output.keys()), {"task1", "task2"})
        self.assertEqual(output["task1"][0].shape[1], 2)
        self.assertEqual(output["task2"][0].shape[1], 3)

    def _build_deep_network(self, variant: str):
        plans_manager = PlansManager(_make_deep_test_plans(variant))
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = make_test_dataset_json()
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        return nnUNetTrainerMultiTask.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=True,
        )

    def test_early_mid_forward(self):
        network = self._build_deep_network("early_mid_fission")
        output = network(torch.rand(1, 1, 32, 32, 32))
        self.assertEqual(set(output.keys()), {"task1", "task2"})
        self.assertEqual(output["task1"][0].shape[1], 2)
        self.assertEqual(output["task2"][0].shape[1], 3)
        self.assertEqual(len(output["task1"]), 3)  # tail owns 3 of 4 decoder stages

    def test_mid_forward(self):
        network = self._build_deep_network("mid_fission")
        output = network(torch.rand(1, 1, 32, 32, 32))
        self.assertEqual(set(output.keys()), {"task1", "task2"})
        self.assertEqual(output["task1"][0].shape[1], 2)
        self.assertEqual(output["task2"][0].shape[1], 3)
        self.assertEqual(len(output["task1"]), 1)  # tail owns only the last/finest stage

    def test_early_mid_shares_trunk_not_tails(self):
        network = self._build_deep_network("early_mid_fission")

        # tails are genuinely separate objects; _iter_decoders' first entry really is the trunk
        self.assertIsNot(network.tails["task1"], network.tails["task2"])
        self.assertIs(network._iter_decoders()[0], network.trunk)

        # trunk parameters are registered exactly once - no accidental per-task duplication
        trunk_param_ids = [id(p) for p in network.trunk.parameters()]
        all_param_ids = [id(p) for p in network.parameters()]
        for pid in trunk_param_ids:
            self.assertEqual(all_param_ids.count(pid), 1)

        # FeatureUNetDecoder stores the shared encoder as `self.encoder`, which nn.Module
        # auto-registers as a submodule - decoder.parameters() therefore also yields the shared
        # encoder's weights, pre-existing behavior, not something this fission split changes. Exclude
        # those to isolate what's actually unique per trunk/tail instance.
        def own_params(decoder):
            return [p for name, p in decoder.named_parameters() if not name.startswith("encoder.")]

        x = torch.rand(1, 1, 32, 32, 32)
        # perturbing ONE tail's own weights must change only that task's output
        with torch.no_grad():
            baseline = network(x)
            for p in own_params(network.tails["task1"]):
                p.add_(1.0)
            perturbed = network(x)
        self.assertFalse(torch.allclose(baseline["task1"][0], perturbed["task1"][0]))
        self.assertTrue(torch.allclose(baseline["task2"][0], perturbed["task2"][0]))

        # perturbing the TRUNK's own weights must change BOTH tasks' outputs
        with torch.no_grad():
            baseline2 = network(x)
            for p in own_params(network.trunk):
                p.add_(1.0)
            perturbed2 = network(x)
        self.assertFalse(torch.allclose(baseline2["task1"][0], perturbed2["task1"][0]))
        self.assertFalse(torch.allclose(baseline2["task2"][0], perturbed2["task2"][0]))


if __name__ == "__main__":
    unittest.main()
