from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json

from nnunetv2.experiment_planning.verify_multitask_dataset_integrity import verify_paired_multitask_dataset_integrity

DATASET_NAME = "Dataset264_SegRap2023GTVOarMT"
DEFAULT_DATA_ROOT = Path(
    r"C:\research\research-medical_imaging-multitask\data-segrap2023\source\SegRap2023_Training_Set_120cases"
)
DEFAULT_OUTPUT_ROOT = Path(r"C:\research\research-medical_imaging-multitask\data-segrap2023\nnUNet_raw")

# held-out test split, see context/dataset/dataset-overview.md for the rationale (20% of 120, stratified by
# GTV volume; remaining 96 go to imagesTr where nnU-Net runs its own 5-fold CV)
N_TEST = 24
SPLIT_SEED = 42
GTV_VOLUME_CACHE = Path(
    r"C:\research\research-medical_imaging-multitask\data-segrap2023\gtv_volume_per_case.json"
)

GTV_CLASSES = ["GTVp", "GTVnd"]

# alphabetical - same order used for the earlier visualization/statistics work in context/dataset/dataset-overview.md
OAR_CLASSES = [
    "Brain", "BrainStem", "Chiasm", "Cochlea_L", "Cochlea_R", "ETbone_L", "ETbone_R", "Esophagus",
    "Eye_L", "Eye_R", "Hippocampus_L", "Hippocampus_R", "IAC_L", "IAC_R", "Larynx", "Larynx_Glottic",
    "Larynx_Supraglot", "Lens_L", "Lens_R", "Mandible_L", "Mandible_R", "Mastoid_L", "Mastoid_R",
    "MiddleEar_L", "MiddleEar_R", "OpticNerve_L", "OpticNerve_R", "OralCavity", "Parotid_L", "Parotid_R",
    "PharynxConst", "Pituitary", "SpinalCord", "Submandibular_L", "Submandibular_R", "TMjoint_L",
    "TMjoint_R", "TemporalLobe_L", "TemporalLobe_R", "Thyroid", "Trachea", "TympanicCavity_L",
    "TympanicCavity_R", "VestibulSemi_L", "VestibulSemi_R",
]


def _copy(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)


def gtv_volume_per_case(case_dirs: List[Path], cache_path: Path = None) -> dict:
    """Total GTV (GTVp + GTVnd) foreground voxels per case, used to stratify the test split.

    Reading 2 volumes x N cases is slow enough to be worth caching, since both Dataset264 and
    Dataset265 need the identical split and would otherwise each recompute it.
    """
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if set(cached) == {p.name for p in case_dirs}:
            return cached

    volumes = {}
    for i, case_dir in enumerate(case_dirs):
        total = 0
        for name in GTV_CLASSES:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / f"{name}.nii.gz")))
            total += int((arr > 0).sum())
        volumes[case_dir.name] = total
        if (i + 1) % 20 == 0 or i == len(case_dirs) - 1:
            print(f"  GTV volume computed for {i + 1}/{len(case_dirs)} cases", flush=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(volumes, indent=2))
    return volumes


def compute_test_split(case_dirs: List[Path], n_test: int = N_TEST, seed: int = SPLIT_SEED,
                       cache_path: Path = None) -> List[str]:
    """Stratified-by-GTV-volume test split: sort cases by GTV volume, cut into n_test equal bins,
    pick one case per bin at random. Guarantees the test set spans the tumour-size distribution
    instead of accidentally drawing all-large or all-small tumours, which matters at n=24.

    Deterministic given (case list, n_test, seed) so Dataset264 and Dataset265 get the identical
    split - required by controlled-ablation-rules.md.
    """
    volumes = gtv_volume_per_case(case_dirs, cache_path=cache_path)
    ordered = sorted(volumes, key=lambda name: (volumes[name], name))
    if n_test > len(ordered):
        raise ValueError(f"n_test={n_test} exceeds available cases ({len(ordered)}).")

    rng = np.random.default_rng(seed)
    bins = np.array_split(np.array(ordered), n_test)
    return sorted(str(rng.choice(b)) for b in bins)


def write_split_manifest(output_folder: Path, case_dirs: List[Path], test_cases: set) -> None:
    """Record which case went where, plus its GTV volume - so the split is auditable after the fact
    (matches BS-80K's split_seed42.csv convention)."""
    volumes = gtv_volume_per_case(case_dirs, cache_path=GTV_VOLUME_CACHE)
    rows = [
        {
            "case": case_dir.name,
            "case_id": f"segrap2023_{case_dir.name}",
            "split": "test" if case_dir.name in test_cases else "train",
            "gtv_voxels": volumes[case_dir.name],
        }
        for case_dir in case_dirs
    ]
    path = output_folder / f"split_seed{SPLIT_SEED}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "case_id", "split", "gtv_voxels"])
        writer.writeheader()
        writer.writerows(rows)


def _copy_binarized_label(src: Path, dst: Path) -> None:
    # source masks use 0/255, and dtype is inconsistent across files (confirmed: GTVp=uint8,
    # GTVnd=float64, Brain=uint16, most others=uint8) - normalize to a fixed uint8 0/1, matching
    # every other label file in this project (BS-80K, ChestX-Det)
    img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(img)
    binarized = (arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(binarized)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst))


def convert_segrap2023_to_nnunet_raw(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dataset_name: str = DATASET_NAME,
    overwrite: bool = False,
) -> Path:
    case_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case folders found under {data_root}")

    output_folder = output_root / dataset_name
    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(f"{output_folder} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_folder)

    print(f"computing stratified test split ({N_TEST} of {len(case_dirs)} cases, seed {SPLIT_SEED})...", flush=True)
    test_cases = set(compute_test_split(case_dirs, cache_path=GTV_VOLUME_CACHE))
    print(f"test cases: {sorted(test_cases)}", flush=True)

    images_tr = output_folder / "imagesTr"
    gtv_tr = output_folder / "labelsTr" / "gtv"
    oar_tr = output_folder / "labelsTr" / "oar"
    images_ts = output_folder / "imagesTs"
    gtv_ts = output_folder / "labelsTs" / "gtv"
    oar_ts = output_folder / "labelsTs" / "oar"
    for d in (images_tr, gtv_tr, oar_tr, images_ts, gtv_ts, oar_ts):
        maybe_mkdir_p(d)

    missing_gtv = []
    n_train = 0
    for case_dir in case_dirs:
        case_id = f"segrap2023_{case_dir.name}"
        is_test = case_dir.name in test_cases
        out_images, out_gtv, out_oar = (
            (images_ts, gtv_ts, oar_ts) if is_test else (images_tr, gtv_tr, oar_tr)
        )
        if not is_test:
            n_train += 1

        _copy(case_dir / "image.nii.gz", out_images / f"{case_id}_0000.nii.gz")
        _copy(case_dir / "image_contrast.nii.gz", out_images / f"{case_id}_0001.nii.gz")

        for channel_idx, name in enumerate(GTV_CLASSES):
            src = case_dir / f"{name}.nii.gz"
            if not src.is_file():
                missing_gtv.append((case_dir.name, name))
                continue
            _copy_binarized_label(src, out_gtv / f"{case_id}_{channel_idx:04d}.nii.gz")

        for channel_idx, name in enumerate(OAR_CLASSES):
            _copy_binarized_label(case_dir / f"{name}.nii.gz", out_oar / f"{case_id}_{channel_idx:04d}.nii.gz")

    if missing_gtv:
        raise FileNotFoundError(f"Missing GTV files: {missing_gtv}")

    write_split_manifest(output_folder, case_dirs, test_cases)

    dataset_json = {
        "channel_names": {"0": "ct", "1": "ct_contrast_enhanced"},
        "labels": {"background": 0},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "SegRap2023 GTV (primary, multichannel: GTVp/GTVnd) and OAR (secondary, "
                        "multichannel: 45 organs-at-risk) multitask segmentation dataset. Paired "
                        "non-contrast + contrast-enhanced CT per case, single-image (non-paired-view) "
                        f"case_unit. SegRap2023's own 20-case validation set has no labels available "
                        f"(challenge scored server-side), so the 120 labeled training cases are split "
                        f"here: {N_TEST} held-out test cases in imagesTs/labelsTs (stratified by GTV "
                        f"volume, seed {SPLIT_SEED}) and the rest in imagesTr/labelsTr where nnU-Net "
                        f"runs its own 5-fold CV. See context/dataset/dataset-overview.md and split_seed"
                        f"{SPLIT_SEED}.csv.",
        "multitask": {
            "case_unit": "single_image",
            "views": ["image"],
            "tasks": {
                "gtv": {
                    "label_dir": "labelsTr/gtv",
                    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(GTV_CLASSES)}},
                    "regions_class_order": list(range(1, len(GTV_CLASSES) + 1)),
                    "multichannel": True,
                },
                "oar": {
                    "label_dir": "labelsTr/oar",
                    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(OAR_CLASSES)}},
                    "regions_class_order": list(range(1, len(OAR_CLASSES) + 1)),
                    "multichannel": True,
                },
            },
        },
    }
    save_json(dataset_json, output_folder / "dataset.json", sort_keys=False)

    verify_paired_multitask_dataset_integrity(str(output_folder))
    return output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = convert_segrap2023_to_nnunet_raw(args.data_root, args.output_root, args.dataset_name, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
