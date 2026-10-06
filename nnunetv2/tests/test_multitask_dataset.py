import numpy as np

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.multitask_dataset import load_multitask_label_stack, make_multitask_union_label


class _FakeReaderWriter:
    def __init__(self, arrays):
        self._arrays = arrays

    def read_seg(self, path):
        return self._arrays[path], {}


def _label_paths_and_reader():
    # nnU-Net treats 2D data as a degenerate 3D volume: shape (C, Z=1, X, Y).
    lesion = np.zeros((1, 1, 4, 4), dtype=np.uint8)
    lesion[0, 0, 1, 1] = 1
    bone = np.zeros((1, 1, 4, 4), dtype=np.uint8)
    bone[0, 0, 2, 2] = 1
    reader = _FakeReaderWriter({"lesion.png": lesion, "bone.png": bone})
    return {"lesion": ["lesion.png"], "bone": ["bone.png"]}, reader


def test_union_label_survives_crop_to_nonzero_ignore_sentinel():
    label_paths, reader = _label_paths_and_reader()
    stack = load_multitask_label_stack(label_paths, reader)
    union = make_multitask_union_label(stack)
    assert union.dtype == np.int8

    image = np.ones((1, 1, 4, 4), dtype=np.float32)
    image[0, 0, 0, 0] = 0  # gives crop_to_nonzero a background corner to mark with -1
    _, seg_cropped, _ = crop_to_nonzero(image, union.copy())
    assert (seg_cropped == -1).any()


if __name__ == "__main__":
    test_union_label_survives_crop_to_nonzero_ignore_sentinel()
    print("multitask_dataset union-label dtype test passed.")
