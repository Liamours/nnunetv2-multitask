import unittest

import torch

from nnunetv2.tests.test_multitask_plans import make_test_dataset_json, make_test_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


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


if __name__ == "__main__":
    unittest.main()
