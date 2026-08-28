import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nnunetv2.dataset_conversion.Dataset260_BS80KLesionBoneMT import convert_bs80k_to_nnunet_raw
from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity
from nnunetv2.tests.test_multitask_plans import make_test_dataset_json, make_trainer_test_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
from nnunetv2.utilities.multitask_dataset import get_multitask_label_paths, get_multitask_task_names, load_multitask_label_stack
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def _write_gray(path: Path, value: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((1024, 256), value, dtype=np.uint8), mode="L").save(path)


def _write_jpg(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((1024, 256), 128, dtype=np.uint8), mode="L").save(path)


class TestMultiTaskRawDataset(unittest.TestCase):
    def test_bs80k_converter_writes_pair_case_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            output_root = Path(tmp) / "nnUNet_raw"
            for patient_num in (1, 2):
                patient_dir = f"patient_{patient_num:05d}"
                _write_jpg(root / "bs80k" / "data" / "whole_body-raster-raw" / patient_dir / "study_00001" / "anterior.jpg")
                _write_jpg(root / "bs80k" / "data" / "whole_body-raster-raw" / patient_dir / "study_00001" / "posterior.jpg")
                _write_gray(root / "bs80k" / "labels" / "bone_region-segmentation" / "pseudo_label-2607" / patient_dir / "study_00001" / "anterior.png", 12)
                _write_gray(root / "bs80k" / "labels" / "bone_region-segmentation" / "pseudo_label-2607" / patient_dir / "study_00001" / "posterior.png", 5)

            _write_gray(root / "bs80k" / "labels" / "whole_body-lesion-segmentation" / "otsu_morphology-guarded_smooth" / "patient_00001" / "study_00001" / "anterior.png", 2)

            output = convert_bs80k_to_nnunet_raw(root, output_root, overwrite=True)
            self.assertTrue((output / "imagesTr" / "bs80k_0001_0000.png").is_file())
            self.assertTrue((output / "imagesTr" / "bs80k_0001_0001.png").is_file())
            self.assertTrue((output / "labelsTr" / "lesion" / "bs80k_0001_0000.png").is_file())
            self.assertTrue((output / "labelsTr" / "bone" / "bs80k_0001_0001.png").is_file())
            self.assertEqual(np.max(np.asarray(Image.open(output / "labelsTr" / "lesion" / "bs80k_0001_0001.png"))), 0)
            verify_paired_multitask_dataset_integrity(str(output))

            dataset = get_filenames_of_train_images_and_targets(str(output))
            self.assertEqual(sorted(dataset), ["bs80k_0001", "bs80k_0002"])
            self.assertEqual(len(dataset["bs80k_0001"]["images"]), 2)
            self.assertEqual(set(dataset["bs80k_0001"]["multitask_labels"]), {"lesion", "bone"})

    def test_trainer_reshapes_paired_view_logits_for_loss(self):
        plans = make_trainer_test_plans()
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["tasks"][0]["output_channels"] = 4
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["tasks"][1]["output_channels"] = 6
        dataset_json = make_test_dataset_json()
        dataset_json["multitask"]["case_unit"] = "paired_anterior_posterior"
        dataset_json["multitask"]["views"] = ["anterior", "posterior"]

        trainer = nnUNetTrainerMultiTask(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))
        target = torch.zeros((1, 4, 8, 8, 8), dtype=torch.int16)
        split_target = trainer._split_targets(target)
        output = {
            "task1": torch.randn((1, 4, 8, 8, 8)),
            "task2": torch.randn((1, 6, 8, 8, 8)),
        }
        reshaped_output, reshaped_target = trainer._reshape_all_outputs_targets_for_loss(output, split_target)

        self.assertEqual(reshaped_output["task1"].shape, (2, 2, 8, 8, 8))
        self.assertEqual(reshaped_output["task2"].shape, (2, 3, 8, 8, 8))
        self.assertEqual(reshaped_target["task1"].shape, (2, 1, 8, 8, 8))
        self.assertEqual(reshaped_target["task2"].shape, (2, 1, 8, 8, 8))

    def test_single_image_multitask_dataset_file_discovery_and_integrity(self):
        # IDRiD/BUSI shape: one image per case, not BS-80K's paired anterior/posterior views.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_json = {
                "channel_names": {"0": "image"},
                "labels": {"background": 0},
                "file_ending": ".png",
                "numTraining": 2,
                "multitask": {
                    "case_unit": "single_image",
                    "views": ["image"],
                    "tasks": {
                        "lesion": {"labels": {"background": 0, "lesion": 1}},
                        "opticdisc": {"labels": {"background": 0, "opticdisc": 1}},
                    },
                },
            }

            for case_id in ("idrid_0001", "idrid_0002"):
                _write_gray(root / "imagesTr" / f"{case_id}_0000.png", 128)
                _write_gray(root / "labelsTr" / "lesion" / f"{case_id}_0000.png", 1)
                _write_gray(root / "labelsTr" / "opticdisc" / f"{case_id}_0000.png", 0)

            from batchgenerators.utilities.file_and_folder_operations import save_json
            save_json(dataset_json, root / "dataset.json")

            dataset = get_filenames_of_train_images_and_targets(str(root), dataset_json)
            self.assertEqual(sorted(dataset), ["idrid_0001", "idrid_0002"])
            self.assertEqual(len(dataset["idrid_0001"]["images"]), 1)
            self.assertEqual(set(dataset["idrid_0001"]["multitask_labels"]), {"lesion", "opticdisc"})
            self.assertEqual(len(dataset["idrid_0001"]["multitask_labels"]["lesion"]), 1)

            # must not raise - this is the exact regression this test guards against
            verify_paired_multitask_dataset_integrity(str(root))

    def test_multichannel_task_label_discovery_and_stacking(self):
        # ChestX-Det/SegRap2023 shape: a "disease"/"organ" task backed by N independent per-class
        # files (not one exclusive integer map per view like every existing task so far).
        dataset_json = {
            "channel_names": {"0": "image"},
            "labels": {"background": 0},
            "file_ending": ".png",
            "numTraining": 1,
            "multitask": {
                "case_unit": "single_image",
                "views": ["image"],
                "tasks": {
                    "lesion": {"labels": {"background": 0, "lesion": 1}},
                    "organ": {
                        "labels": {"background": 0, "organ_a": 1, "organ_b": 2, "organ_c": 3},
                        "regions_class_order": [1, 2, 3],
                        "multichannel": True,
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "case_0001"
            _write_gray(root / "labelsTr" / "lesion" / f"{case_id}_0000.png", 1)
            _write_gray(root / "labelsTr" / "organ" / f"{case_id}_0000.png", 1)  # organ_a
            _write_gray(root / "labelsTr" / "organ" / f"{case_id}_0001.png", 0)  # organ_b, absent here
            _write_gray(root / "labelsTr" / "organ" / f"{case_id}_0002.png", 1)  # organ_c

            label_paths = get_multitask_label_paths(str(root), case_id, dataset_json)
            self.assertEqual(len(label_paths["lesion"]), 1)
            self.assertEqual(len(label_paths["organ"]), 3)

            from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
            rw = NaturalImage2DIO()
            task_order = get_multitask_task_names(dataset_json)
            stack = load_multitask_label_stack(label_paths, rw, task_order=task_order)

            self.assertEqual(stack.shape[0], 4)  # 1 (lesion) + 3 (organ_a/b/c)
            self.assertTrue(np.all(stack[0] == 1))  # lesion
            self.assertTrue(np.all(stack[1] == 1))  # organ_a
            self.assertTrue(np.all(stack[2] == 0))  # organ_b
            self.assertTrue(np.all(stack[3] == 1))  # organ_c

    def test_trainer_splits_mixed_standard_and_multichannel_targets(self):
        plans = make_trainer_test_plans()
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["tasks"][0]["output_channels"] = 2
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["tasks"][1]["name"] = "task2"
        plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["multitask"]["tasks"][1]["output_channels"] = 3
        dataset_json = make_test_dataset_json()
        dataset_json["multitask"]["tasks"]["task2"] = {
            "labels": {"background": 0, "organ_a": 1, "organ_b": 2, "organ_c": 3},
            "regions_class_order": [1, 2, 3],
            "multichannel": True,
        }

        trainer = nnUNetTrainerMultiTask(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))
        # target shape AFTER ConvertMultiTaskSegmentationToRegionsTransform has run in the dataloader:
        # task1 (plain CE) stays at 1 channel, task2 (multichannel) is already isolated at 3 channels
        target = torch.zeros((1, 4, 8, 8, 8), dtype=torch.int16)
        target[:, 1] = 1  # organ_a channel, distinguishable from the rest
        split_target = trainer._split_targets(target)

        self.assertEqual(split_target["task1"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(split_target["task2"].shape, (1, 3, 8, 8, 8))
        self.assertTrue(torch.all(split_target["task2"][:, 0] == 1))
        self.assertTrue(torch.all(split_target["task2"][:, 1] == 0))

        # loss selection: task2 (multichannel) must get the BCE/region loss, task1 the CE loss
        from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
        loss = trainer._build_loss()
        self.assertIsInstance(loss.task_losses["task1"], DC_and_CE_loss)
        self.assertIsInstance(loss.task_losses["task2"], DC_and_BCE_loss)

        # a real forward+backward pass must not raise
        output = {
            "task1": torch.randn((1, 2, 8, 8, 8), requires_grad=True),
            "task2": torch.randn((1, 3, 8, 8, 8), requires_grad=True),
        }
        computed = loss(output, split_target)
        computed.backward()
        self.assertTrue(torch.isfinite(computed))

    # -1 crop-marker stripping and per-task channel isolation are now tested directly against
    # ConvertMultiTaskSegmentationToRegionsTransform (the actual place it happens) in
    # test_multitask_region_transform.py, not here.

if __name__ == "__main__":
    unittest.main()
