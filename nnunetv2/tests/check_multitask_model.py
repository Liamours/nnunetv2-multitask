import os
import pickle
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.inference.export_prediction_multitask import (
    convert_predicted_logits_to_segmentation_with_correct_shape_multitask,
    export_prediction_from_logits_multitask,
)
from nnunetv2.inference.predictor_multitask import nnUNetMultiTaskPredictor
from nnunetv2.tests.test_multitask_plans import make_test_dataset_json, make_test_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _write_dummy_preprocessed_case(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(1, 8, 8, 8).astype(np.float32)
    seg = np.zeros((2, 8, 8, 8), dtype=np.int16)
    np.savez_compressed(folder / "case_0000.npz", data=data, seg=seg)
    properties = {
        "class_locations": {-1: np.array([[0, 0, 0, 0]], dtype=np.int64)},
        "spacing": [1.0, 1.0, 1.0],
        "shape_after_cropping_and_before_resampling": [8, 8, 8],
        "bbox_used_for_cropping": [[0, 8], [0, 8], [0, 8]],
        "shape_before_cropping": [8, 8, 8],
    }
    with open(folder / "case_0000.pkl", "wb") as f:
        pickle.dump(properties, f)


@contextmanager
def _temporary_nnunet_env(dataset_name: str, data_identifier: str):
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        preprocessed = root / "preprocessed"
        results = root / "results"
        raw.mkdir()
        preprocessed.mkdir()
        results.mkdir()
        _write_dummy_preprocessed_case(preprocessed / dataset_name / data_identifier)

        old_env = {k: os.environ.get(k) for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_compile")}
        os.environ["nnUNet_raw"] = str(raw)
        os.environ["nnUNet_preprocessed"] = str(preprocessed)
        os.environ["nnUNet_results"] = str(results)
        os.environ["nnUNet_compile"] = "false"
        try:
            yield root
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def run_multitask_model_checker():
    plans = make_test_plans()
    dataset_json = make_test_dataset_json()
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("3d_fullres")
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)

    imported_modules = [
        "nnunetv2.architecture.multitask_unet",
        "nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask",
        "nnunetv2.training.loss.multitask_losses",
        "nnunetv2.utilities.label_handling.multitask_label_handling",
        "nnunetv2.utilities.plans_handling.multitask_plans",
        "nnunetv2.inference.predictor_multitask",
        "nnunetv2.inference.export_prediction_multitask",
    ]

    report = {
        "imports_checked": imported_modules,
        "discovery": {},
        "variants": {},
        "trainer": {},
        "inference": {},
        "exports": {},
        "checkpoint_reload": {},
    }

    trainer_class = recursive_find_trainer_class_by_name("nnUNetTrainerMultiTask")
    report["discovery"]["trainer_lookup"] = trainer_class.__name__
    report["discovery"]["label_manager_class"] = plans_manager.label_manager_class.__name__

    for variant, class_name in (
        ("dual_head", "nnunetv2.architecture.multitask_unet.MultiTaskDualHeadUNet"),
        ("dual_decoder", "nnunetv2.architecture.multitask_unet.MultiTaskDualDecoderUNet"),
    ):
        variant_plans = make_test_plans()
        variant_plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"] = class_name
        variant_plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["variant"] = variant
        variant_manager = PlansManager(variant_plans)
        variant_configuration = variant_manager.get_configuration("3d_fullres")
        network = nnUNetTrainerMultiTask.build_network_architecture(
            variant_manager,
            variant_configuration,
            num_input_channels,
            variant_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=True,
        )
        direct_network = get_network_from_plans(
            variant_configuration.network_arch_class_name,
            variant_configuration.network_arch_init_kwargs,
            variant_configuration.network_arch_init_kwargs_req_import,
            num_input_channels,
            variant_manager.get_label_manager(dataset_json).num_segmentation_heads,
            allow_init=True,
            deep_supervision=True,
        )
        output = network(torch.rand(1, 1, 8, 8, 8))
        direct_output = direct_network(torch.rand(1, 1, 8, 8, 8))
        report["variants"][variant] = {
            task_name: tuple(task_output[0].shape)
            for task_name, task_output in output.items()
        }
        report["variants"][variant]["factory_class"] = direct_network.__class__.__name__
        report["variants"][variant]["factory_tasks"] = sorted(list(direct_output.keys()))

    trainer_plans = make_test_plans()
    trainer_plans["continue_training"] = False
    with _temporary_nnunet_env(
        trainer_plans["dataset_name"],
        trainer_plans["configurations"]["3d_fullres"]["data_identifier"],
    ):
        trainer = nnUNetTrainerMultiTask(
            trainer_plans,
            "3d_fullres",
            fold=0,
            dataset_json=dataset_json,
            device=torch.device("cpu"),
        )
        trainer.initialize()
        def make_deep_supervision_target(segmentation: torch.Tensor, scales):
            targets = []
            for scale in scales:
                target_size = [max(1, int(round(segmentation.shape[d + 2] * scale[d]))) for d in range(len(scale))]
                if target_size == list(segmentation.shape[2:]):
                    targets.append(segmentation.clone())
                else:
                    resized = F.interpolate(segmentation.float(), size=target_size, mode="nearest")
                    targets.append(resized.to(segmentation.dtype))
            return targets

        batch = {
            "data": torch.rand(2, 1, 8, 8, 8),
            "target": make_deep_supervision_target(
                torch.randint(0, 2, (2, 2, 8, 8, 8), dtype=torch.int16),
                trainer._get_deep_supervision_scales(),
            ),
        }
        train_result = trainer.train_step(batch)
        val_result = trainer.validation_step(batch)
        trainer.set_deep_supervision_enabled(False)
        ds_disabled = not trainer.network.deep_supervision if hasattr(trainer.network, "deep_supervision") else True
        trainer.set_deep_supervision_enabled(True)

        trained_model_dir = Path(os.environ["nnUNet_results"]) / trainer.plans_manager.dataset_name / (
            trainer.__class__.__name__ + "__" + trainer.plans_manager.plans_name + "__" + trainer.configuration_name
        )
        trained_model_dir.mkdir(parents=True, exist_ok=True)
        (trained_model_dir / "fold_0").mkdir(parents=True, exist_ok=True)
        with open(trained_model_dir / "dataset.json", "w", encoding="utf-8") as f:
            import json
            json.dump(dataset_json, f)
        with open(trained_model_dir / "plans.json", "w", encoding="utf-8") as f:
            import json
            json.dump(trainer_plans, f)
        checkpoint_path = trained_model_dir / "fold_0" / "checkpoint_final.pth"
        trainer.save_checkpoint(str(checkpoint_path))

        reloaded_plans = deepcopy(trainer_plans)
        reloaded_plans["continue_training"] = False
        reloaded_trainer = nnUNetTrainerMultiTask(
            reloaded_plans,
            "3d_fullres",
            fold=0,
            dataset_json=dataset_json,
            device=torch.device("cpu"),
        )
        reloaded_trainer.load_checkpoint(str(checkpoint_path))
        predictor_from_folder = nnUNetMultiTaskPredictor(
            tile_step_size=0.5,
            use_gaussian=False,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor_from_folder.initialize_from_trained_model_folder(
            str(trained_model_dir),
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )
        folder_prediction = predictor_from_folder.predict_sliding_window_return_logits(torch.rand(1, 8, 8, 8))

        report["trainer"] = {
            "loss_key_present": "loss" in train_result,
            "validation_loss_key_present": "loss" in val_result,
            "task_metric_keys": sorted(k for k in val_result.keys() if k != "loss"),
            "deep_supervision_toggle_ok": ds_disabled,
        }
        report["checkpoint_reload"] = {
            "checkpoint_exists": checkpoint_path.is_file(),
            "epoch_restored": int(reloaded_trainer.current_epoch),
            "trainer_class": reloaded_trainer.__class__.__name__,
            "predictor_folder_init_tasks": sorted(list(folder_prediction.keys())),
        }

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
    network = nnUNetTrainerMultiTask.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    predictor.manual_initialization(network, plans_manager, configuration_manager, [{}], dataset_json, "nnUNetTrainerMultiTask", None)
    predictor.list_of_parameters = [network.state_dict()]
    prediction = predictor.predict_sliding_window_return_logits(torch.rand(1, 8, 8, 8))
    properties = {
        "spacing": [1.0, 1.0, 1.0],
        "shape_after_cropping_and_before_resampling": [8, 8, 8],
        "bbox_used_for_cropping": [[0, 8], [0, 8], [0, 8]],
        "shape_before_cropping": [8, 8, 8],
        "nibabel_stuff": {
            "original_affine": np.eye(4),
        },
    }
    segmentations = convert_predicted_logits_to_segmentation_with_correct_shape_multitask(
        prediction,
        plans_manager,
        configuration_manager,
        label_manager,
        properties,
    )
    report["inference"] = {
        "prediction_tasks": {task_name: tuple(task_prediction.shape) for task_name, task_prediction in prediction.items()},
        "segmentation_tasks": {task_name: tuple(task_seg.shape) for task_name, task_seg in segmentations.items()},
    }

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_prefix = tmp_path / "prediction" / "case_0000"
        export_prediction_from_logits_multitask(
            prediction,
            properties,
            configuration_manager,
            plans_manager,
            dataset_json,
            str(output_prefix),
            save_probabilities=False,
        )
        report["exports"] = {
            "task1_file": (tmp_path / "prediction" / "task1" / "case_0000.nii.gz").is_file(),
            "task2_file": (tmp_path / "prediction" / "task2" / "case_0000.nii.gz").is_file(),
        }

    return report


if __name__ == "__main__":
    checker_report = run_multitask_model_checker()
    print("multitask model checker passed")
    for section, value in checker_report.items():
        print(f"{section}: {value}")
