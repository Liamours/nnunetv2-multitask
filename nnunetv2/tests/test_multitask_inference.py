import unittest

import torch

from nnunetv2.inference.export_prediction_multitask import (
    convert_predicted_logits_to_segmentation_with_correct_shape_multitask,
)
from nnunetv2.inference.predictor_multitask import nnUNetMultiTaskPredictor
from nnunetv2.tests.test_multitask_plans import make_test_dataset_json, make_test_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class TestMultiTaskInference(unittest.TestCase):
    def test_predictor_and_conversion_return_task_dicts(self):
        plans = make_test_plans()
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = make_test_dataset_json()
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        network = nnUNetTrainerMultiTask.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )
        predictor = nnUNetMultiTaskPredictor(
            tile_step_size=0.5,
            use_gaussian=False,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(network, plans_manager, configuration_manager, [{}], dataset_json, "nnUNetTrainerMultiTask", None)
        predictor.list_of_parameters = [network.state_dict()]

        prediction = predictor.predict_sliding_window_return_logits(torch.rand(1, 8, 8, 8))
        self.assertEqual(set(prediction.keys()), {"task1", "task2"})
        self.assertEqual(tuple(prediction["task1"].shape), (2, 8, 8, 8))
        self.assertEqual(tuple(prediction["task2"].shape), (3, 8, 8, 8))

        properties = {
            "spacing": [1.0, 1.0, 1.0],
            "shape_after_cropping_and_before_resampling": [8, 8, 8],
            "bbox_used_for_cropping": [[0, 8], [0, 8], [0, 8]],
            "shape_before_cropping": [8, 8, 8],
        }
        segmentations = convert_predicted_logits_to_segmentation_with_correct_shape_multitask(
            prediction,
            plans_manager,
            configuration_manager,
            label_manager,
            properties,
        )
        self.assertEqual(set(segmentations.keys()), {"task1", "task2"})
        self.assertEqual(tuple(segmentations["task1"].shape), (8, 8, 8))
        self.assertEqual(tuple(segmentations["task2"].shape), (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
