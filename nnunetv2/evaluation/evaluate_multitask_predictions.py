import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def _read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _safe_div(numerator: float, denominator: float):
    return float(numerator / denominator) if denominator else None


def _load_split_cases(raw_dataset: Path, split: str) -> List[str]:
    split_file = raw_dataset / "split_seed42.csv"
    with split_file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cases = [row["case_id"] for row in rows if row["split"] == split]
    if not cases:
        raise ValueError(f"No cases found for split={split!r} in {split_file}.")
    return cases


def _task_labels(dataset_json: dict, task: str) -> Dict[str, int]:
    labels = dataset_json["multitask"]["tasks"][task]["labels"]
    return {name: int(value) for name, value in labels.items()}


def _pixel_metrics(gt: np.ndarray, pred: np.ndarray, label: int) -> dict:
    gt_mask = gt == label
    pred_mask = pred == label
    tp = int(np.logical_and(gt_mask, pred_mask).sum())
    fp = int(np.logical_and(~gt_mask, pred_mask).sum())
    fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
    tn = int(np.logical_and(~gt_mask, ~pred_mask).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
    }


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    return ndimage.label(mask, structure=structure)


def _lesionwise_counts_for_class(
    gt: np.ndarray,
    pred: np.ndarray,
    label: int,
    min_overlap_pixels: int,
) -> Tuple[int, int, int]:
    gt_cc, n_gt = _connected_components(gt == label)
    pred_cc, n_pred = _connected_components(pred == label)
    matched_gt = set()
    matched_pred = set()

    for gt_idx in range(1, n_gt + 1):
        gt_mask = gt_cc == gt_idx
        overlapping_pred_ids = np.unique(pred_cc[gt_mask])
        overlapping_pred_ids = [int(i) for i in overlapping_pred_ids if i != 0]
        best_pred = None
        best_overlap = 0
        for pred_idx in overlapping_pred_ids:
            if pred_idx in matched_pred:
                continue
            overlap = int(np.logical_and(gt_mask, pred_cc == pred_idx).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred = pred_idx
        if best_pred is not None and best_overlap >= min_overlap_pixels:
            matched_gt.add(gt_idx)
            matched_pred.add(best_pred)

    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    return tp, fp, fn


def _lesionwise_metrics(
    case_metrics: Iterable[Tuple[int, int, int]],
    num_cases: int,
) -> dict:
    tp = int(sum(i[0] for i in case_metrics))
    fp = int(sum(i[1] for i in case_metrics))
    fn = int(sum(i[2] for i in case_metrics))
    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "tp_lesions": tp,
        "fp_lesions": fp,
        "fn_lesions": fn,
        "lesionwise_precision": precision,
        "lesionwise_sensitivity": sensitivity,
        "lesionwise_f1": f1,
        "froc": {
            "threshold_type": "hard_segmentation",
            "lesion_localization_fraction": sensitivity,
            "false_positives_per_case": _safe_div(fp, num_cases),
        },
    }


def _mean_metric(rows: List[dict], key: str):
    vals = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def evaluate_multitask_prediction_folder(
    raw_dataset: Path,
    prediction_folder: Path,
    split: str,
    output_file: Path,
    min_lesion_overlap_pixels: int = 1,
) -> dict:
    dataset_json = json.loads((raw_dataset / "dataset.json").read_text(encoding="utf-8"))
    views = dataset_json["multitask"]["views"]
    tasks = dataset_json["multitask"]["tasks"]
    cases = _load_split_cases(raw_dataset, split)

    result = {
        "raw_dataset": str(raw_dataset),
        "prediction_folder": str(prediction_folder),
        "split": split,
        "num_cases": len(cases),
        "tasks": {},
    }

    for task in tasks:
        labels = _task_labels(dataset_json, task)
        foreground = {name: value for name, value in labels.items() if value != 0}
        task_rows = []
        lesionwise_by_class = {name: [] for name in foreground}
        lesionwise_rows = []

        for view_idx, view in enumerate(views):
            for case_id in cases:
                gt_file = raw_dataset / "labelsTr" / task / f"{case_id}_{view_idx:04d}.png"
                pred_file = prediction_folder / task / view / f"{case_id}.png"
                if not gt_file.is_file():
                    raise FileNotFoundError(gt_file)
                if not pred_file.is_file():
                    raise FileNotFoundError(pred_file)
                gt = _read_png(gt_file)
                pred = _read_png(pred_file)
                if gt.shape != pred.shape:
                    raise ValueError(f"Shape mismatch for {case_id} {task}/{view}: gt={gt.shape}, pred={pred.shape}")

                for label_name, label_value in foreground.items():
                    metrics = _pixel_metrics(gt, pred, label_value)
                    metrics.update(
                        {
                            "case_id": case_id,
                            "view": view,
                            "label_name": label_name,
                            "label": label_value,
                        }
                    )
                    task_rows.append(metrics)
                    if task == "lesion":
                        lesion_counts = _lesionwise_counts_for_class(gt, pred, label_value, min_lesion_overlap_pixels)
                        lesionwise_by_class[label_name].append(lesion_counts)
                        tp_lesions, fp_lesions, fn_lesions = lesion_counts
                        gt_lesions = tp_lesions + fn_lesions
                        lesionwise_rows.append({
                            "case_id": case_id,
                            "view": view,
                            "label_name": label_name,
                            "label": label_value,
                            "tp_lesions": tp_lesions,
                            "fp_lesions": fp_lesions,
                            "fn_lesions": fn_lesions,
                            "gt_lesions": gt_lesions,
                            "pred_lesions": tp_lesions + fp_lesions,
                            "binary_any_gt_detected": int(gt_lesions > 0 and tp_lesions > 0),
                            "binary_all_gt_detected": int(gt_lesions > 0 and fn_lesions == 0),
                            "eligible_positive_gt": int(gt_lesions > 0),
                        })

        task_summary = {
            "pixel_mean": {
                "dice": _mean_metric(task_rows, "dice"),
                "sensitivity": _mean_metric(task_rows, "sensitivity"),
                "specificity": _mean_metric(task_rows, "specificity"),
            },
            "by_label": {},
            "per_case_view_label": task_rows,
        }
        for label_name in foreground:
            label_rows = [row for row in task_rows if row["label_name"] == label_name]
            task_summary["by_label"][label_name] = {
                "dice": _mean_metric(label_rows, "dice"),
                "sensitivity": _mean_metric(label_rows, "sensitivity"),
                "specificity": _mean_metric(label_rows, "specificity"),
            }
            if task == "lesion":
                task_summary["by_label"][label_name].update(
                    _lesionwise_metrics(lesionwise_by_class[label_name], len(cases))
                )
        if task == "lesion":
            all_counts = []
            for counts in lesionwise_by_class.values():
                all_counts.extend(counts)
            task_summary["lesionwise_class_matched"] = _lesionwise_metrics(all_counts, len(cases))
            task_summary["lesionwise_per_case_view_label"] = lesionwise_rows
        result["tasks"][task] = task_summary

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_lesion_overlap_pixels", type=int, default=1)
    args = parser.parse_args()

    result = evaluate_multitask_prediction_folder(
        Path(args.raw_dataset),
        Path(args.predictions),
        args.split,
        Path(args.output),
        args.min_lesion_overlap_pixels,
    )
    print(json.dumps({
        "split": result["split"],
        "num_cases": result["num_cases"],
        "summary": {
            task: {
                "pixel_mean": values["pixel_mean"],
                "lesionwise_class_matched": values.get("lesionwise_class_matched"),
            }
            for task, values in result["tasks"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
