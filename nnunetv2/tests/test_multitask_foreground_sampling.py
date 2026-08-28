import unittest

import numpy as np

from nnunetv2.utilities.multitask_dataset import sample_multitask_foreground_locations


class TestMultiTaskForegroundSampling(unittest.TestCase):
    def test_per_task_per_channel_isolation(self):
        # bug: DefaultPreprocessor._sample_foreground_locations searches label VALUES across the
        # whole stacked seg, channel-blind - meaningless once a task's channels are independent {0,1}
        # masks rather than an exclusive integer map. Each task/channel must be sampled from its own
        # data only.
        dataset_json = {
            "multitask": {
                "views": ["image"],
                "tasks": {
                    "plain": {"labels": {"background": 0, "a": 1, "b": 2}},
                    "mc": {
                        "labels": {"background": 0, "x": 1, "y": 2, "z": 3},
                        "multichannel": True,
                        "regions_class_order": [1, 2, 3],
                    },
                },
            }
        }
        # seg channels: [plain (1ch), mc_x, mc_y, mc_z] = 4 channels
        seg = np.zeros((4, 5, 5, 5), dtype=np.int16)
        seg[0, 1, 1, 1] = 2  # plain task class "b"
        seg[1, 2, 2, 2] = 1  # mc channel x foreground
        seg[3, 3, 3, 3] = 1  # mc channel z foreground (y stays empty)

        locs = sample_multitask_foreground_locations(seg, dataset_json, min_num_samples=100)

        self.assertGreater(len(locs[("plain", 2)]), 0)
        self.assertEqual(len(locs[("plain", 1)]), 0)
        self.assertGreater(len(locs[("mc", 0)]), 0)   # channel x
        self.assertEqual(len(locs[("mc", 1)]), 0)     # channel y - empty
        self.assertGreater(len(locs[("mc", 2)]), 0)   # channel z


if __name__ == "__main__":
    unittest.main()
