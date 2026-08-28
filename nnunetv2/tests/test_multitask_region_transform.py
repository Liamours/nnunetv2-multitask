import unittest

import torch

from nnunetv2.training.data_augmentation.custom_transforms.multitask_region_transform import (
    ConvertMultiTaskSegmentationToRegionsTransform,
)


class TestMultiTaskRegionTransform(unittest.TestCase):
    def test_plain_task_passthrough_and_multichannel_isolation(self):
        # bug: the stock ConvertSegmentationToRegionsTransform reads only channel 0 for every task,
        # scrambling everything else - this must isolate each task's own channels and strip -1.
        t = ConvertMultiTaskSegmentationToRegionsTransform(
            task_order=["plain", "mc"],
            task_raw_channel_slices={"plain": slice(0, 1), "mc": slice(1, 4)},
            task_is_multichannel={"plain": False, "mc": True},
            task_regions={},
        )
        seg = torch.zeros((4, 4, 4, 4), dtype=torch.int16)
        seg[0, 0, 0, 0] = 2   # plain task raw class value - must pass through unchanged
        seg[1, 1, 1, 1] = 1   # mc channel 0 foreground
        seg[2, 2, 2, 2] = -1  # mc channel 1 crop marker - must become 0
        seg[3, 3, 3, 3] = 1   # mc channel 2 foreground

        out = t._apply_to_segmentation(seg)

        self.assertEqual(out.shape, (4, 4, 4, 4))
        self.assertEqual(out[0, 0, 0, 0].item(), 2)
        self.assertEqual(out[1].sum().item(), 1)
        self.assertEqual(out[1, 1, 1, 1].item(), 1)
        self.assertEqual(out[2].sum().item(), 0, "crop marker -1 leaked as foreground")
        self.assertEqual(out[3].sum().item(), 1)
        self.assertEqual(out[3, 3, 3, 3].item(), 1)


if __name__ == "__main__":
    unittest.main()
