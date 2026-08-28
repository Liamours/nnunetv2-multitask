import csv
from pathlib import Path

import numpy as np
from PIL import Image
from wbbs_lguq.paths import EVALUATIONS, NNUNET_RAW, VISUALIZATIONS

RAW_DATASET = NNUNET_RAW / "Dataset260_BS80KLesionBoneMT"
EVAL_ROOT = EVALUATIONS / "a2_dual_head_100epoch_best_latest"
OUT_ROOT = VISUALIZATIONS / "a2_dual_head_100epoch_test_palette_overlay"

VIEWS = ("anterior", "posterior")
RUNS = {
    "best": EVAL_ROOT / "A2_dual_head_best" / "test" / "predictions",
    "latest": EVAL_ROOT / "A2_dual_head_latest" / "test" / "predictions",
}

BONE_PALETTE = {
    1: (176, 230, 13),
    2: (0, 151, 219),
    3: (126, 230, 225),
    4: (166, 55, 167),
    5: (230, 157, 180),
    6: (167, 110, 77),
    7: (121, 0, 24),
    8: (56, 65, 184),
    9: (230, 218, 0),
    10: (230, 114, 35),
    11: (12, 187, 62),
    12: (230, 182, 22),
}

LESION_PALETTE = {
    1: (0, 255, 64),    # benign
    2: (255, 0, 0),     # lesion malignant
}

ALPHA = 0.25


def read_l(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def colorize(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for value, color in palette.items():
        out[mask == value] = color
    return out


def overlay(base: np.ndarray, color: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    result = base.astype(np.float32).copy()
    active = mask > 0
    result[active] = result[active] * (1.0 - alpha) + color[active].astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def load_test_cases() -> list[str]:
    split_file = RAW_DATASET / "split_seed42.csv"
    with split_file.open(newline="", encoding="utf-8") as f:
        return [row["case_id"] for row in csv.DictReader(f) if row["split"] == "test"]


def write_readme(total_rows: int) -> None:
    text = f"""# A2 Dual-Head Test Palette Overlays

These images visualize A2 dual-head predicted test-set masks.

Layer order:
1. raw scan image
2. predicted bone region mask at 25% opacity
3. predicted lesion mask at 25% opacity, drawn on top of bone

Bone palette uses the verified BS-80K mapping in `context/bone-region-palette.md`.
Lesion visualization palette: benign=`#00ff40`, lesion malignant=`#ff0000`.

Generated rows: {total_rows}
Output layout:
- `best/anterior`
- `best/posterior`
- `latest/anterior`
- `latest/posterior`

Lesion malignant is lesion class `2`.
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    cases = load_test_cases()

    for run_name, pred_root in RUNS.items():
        for view_idx, view in enumerate(VIEWS):
            out_dir = OUT_ROOT / run_name / view
            out_dir.mkdir(parents=True, exist_ok=True)

            for case_id in cases:
                raw_file = RAW_DATASET / "imagesTr" / f"{case_id}_{view_idx:04d}.png"
                bone_file = pred_root / "bone" / view / f"{case_id}.png"
                lesion_file = pred_root / "lesion" / view / f"{case_id}.png"
                if not raw_file.is_file():
                    raise FileNotFoundError(raw_file)
                if not bone_file.is_file():
                    raise FileNotFoundError(bone_file)
                if not lesion_file.is_file():
                    raise FileNotFoundError(lesion_file)

                raw = read_l(raw_file)
                bone = read_l(bone_file)
                lesion = read_l(lesion_file)
                if raw.shape != bone.shape or raw.shape != lesion.shape:
                    raise ValueError(
                        f"Shape mismatch for {run_name}/{view}/{case_id}: "
                        f"raw={raw.shape}, bone={bone.shape}, lesion={lesion.shape}"
                    )

                base = np.stack([raw, raw, raw], axis=-1)
                bone_color = colorize(bone, BONE_PALETTE)
                lesion_color = colorize(lesion, LESION_PALETTE)
                composed = overlay(base, bone_color, bone, ALPHA)
                composed = overlay(composed, lesion_color, lesion, ALPHA)

                out_file = out_dir / f"{case_id}_{view}_raw_bone25_lesion25.png"
                Image.fromarray(composed).save(out_file)
                manifest_rows.append({
                    "run": run_name,
                    "split": "test",
                    "case_id": case_id,
                    "view": view,
                    "raw_file": str(raw_file),
                    "bone_prediction": str(bone_file),
                    "lesion_prediction": str(lesion_file),
                    "overlay_file": str(out_file),
                })

    manifest_file = OUT_ROOT / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_readme(len(manifest_rows))
    print(f"Wrote {len(manifest_rows)} overlays to {OUT_ROOT}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
