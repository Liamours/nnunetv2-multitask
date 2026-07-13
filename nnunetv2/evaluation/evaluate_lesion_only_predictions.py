import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nnunetv2.evaluation.evaluate_multitask_predictions import (
    _lesionwise_counts_for_class,
    _lesionwise_metrics,
    _mean_metric,
    _pixel_metrics,
    _read_png,
)


def _load_split_rows(raw_dataset: Path, split: str):
    split_file = raw_dataset / "split_seed42.csv"
    with split_file.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split={split!r} in {split_file}.")
    return rows


def _canonical_case_id(case_id: str) -> str:
    if case_id.endswith("_ant"):
        return case_id[:-4]
    if case_id.endswith("_post"):
        return case_id[:-5]
    return case_id


def evaluate_lesion_only_prediction_folder(
    raw_dataset: Path,
    prediction_folder: Path,
    split: str,
    output_file: Path,
    min_lesion_overlap_pixels: int = 1,
) -> dict:
    dataset_json = json.loads((raw_dataset / "dataset.json").read_text(encoding="utf-8"))
    labels = {name: int(value) for name, value in dataset_json["labels"].items()}
    foreground = {name: value for name, value in labels.items() if value != 0}
    rows = _load_split_rows(raw_dataset, split)

    task_rows = []
    lesionwise_rows = []
    lesionwise_by_class = {name: [] for name in foreground}

    for row in rows:
        case_id = row["case_id"]
        canonical_case_id = _canonical_case_id(case_id)
        view = row.get("view", "unknown")
        gt_file = raw_dataset / "labelsTr" / f"{case_id}.png"
        pred_file = prediction_folder / f"{case_id}.png"
        if not gt_file.is_file():
            raise FileNotFoundError(gt_file)
        if not pred_file.is_file():
            raise FileNotFoundError(pred_file)
        gt = _read_png(gt_file)
        pred = _read_png(pred_file)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {case_id}: gt={gt.shape}, pred={pred.shape}")

        for label_name, label_value in foreground.items():
            metrics = _pixel_metrics(gt, pred, label_value)
            metrics.update({
                "case_id": canonical_case_id,
                "view": view,
                "label_name": label_name,
                "label": label_value,
                "source_case_id": case_id,
            })
            task_rows.append(metrics)

            lesion_counts = _lesionwise_counts_for_class(gt, pred, label_value, min_lesion_overlap_pixels)
            lesionwise_by_class[label_name].append(lesion_counts)
            tp_lesions, fp_lesions, fn_lesions = lesion_counts
            gt_lesions = tp_lesions + fn_lesions
            lesionwise_rows.append({
                "case_id": canonical_case_id,
                "view": view,
                "label_name": label_name,
                "label": label_value,
                "source_case_id": case_id,
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
        "lesionwise_per_case_view_label": lesionwise_rows,
    }
    for label_name in foreground:
        label_rows = [row for row in task_rows if row["label_name"] == label_name]
        task_summary["by_label"][label_name] = {
            "dice": _mean_metric(label_rows, "dice"),
            "sensitivity": _mean_metric(label_rows, "sensitivity"),
            "specificity": _mean_metric(label_rows, "specificity"),
        }
        task_summary["by_label"][label_name].update(
            _lesionwise_metrics(lesionwise_by_class[label_name], len(rows))
        )

    all_counts = []
    for counts in lesionwise_by_class.values():
        all_counts.extend(counts)
    task_summary["lesionwise_class_matched"] = _lesionwise_metrics(all_counts, len(rows))

    result = {
        "raw_dataset": str(raw_dataset),
        "prediction_folder": str(prediction_folder),
        "split": split,
        "num_cases": len(rows),
        "tasks": {"lesion": task_summary},
    }
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
    result = evaluate_lesion_only_prediction_folder(
        Path(args.raw_dataset),
        Path(args.predictions),
        args.split,
        Path(args.output),
        args.min_lesion_overlap_pixels,
    )
    print(json.dumps({
        "split": result["split"],
        "num_cases": result["num_cases"],
        "summary": result["tasks"]["lesion"]["pixel_mean"],
        "lesionwise_class_matched": result["tasks"]["lesion"]["lesionwise_class_matched"],
    }, indent=2))


if __name__ == "__main__":
    main()
