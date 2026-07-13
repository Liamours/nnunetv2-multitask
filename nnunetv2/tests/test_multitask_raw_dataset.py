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
            for patient_id in ("0001", "0002"):
                _write_jpg(root / "bs80k-imaging-raw" / "wholeBodyANT" / f"{patient_id}.jpg")
                _write_jpg(root / "bs80k-imaging-raw" / "wholeBodyPOST" / f"{patient_id}.jpg")
                _write_gray(root / "bs80k-bone_region-segmentation" / "pseudo_label-2607" / "anterior" / f"{patient_id}.png", 12)
                _write_gray(root / "bs80k-bone_region-segmentation" / "pseudo_label-2607" / "posterior" / f"{patient_id}.png", 5)

            _write_gray(root / "bs80k-lesion-segmentation" / "otsu_morphology-guarded_smooth" / "anterior" / "0001.png", 2)

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


if __name__ == "__main__":
    unittest.main()
