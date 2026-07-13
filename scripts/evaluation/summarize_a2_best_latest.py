import csv
import json
from pathlib import Path


EVAL_ROOT = Path(r"C:\Users\lulay\Desktop\nnunetv2-multitask\data\evaluation\a2_dual_head_100epoch_best_latest")

BONE_NAMES = {
    "label_1": "Skull",
    "label_2": "Cervical Vert",
    "label_3": "Thoracic Vert",
    "label_4": "Ribs",
    "label_5": "Sternum",
    "label_6": "Clavicle",
    "label_7": "Scapula",
    "label_8": "Humerus",
    "label_9": "Lumbar Vert",
    "label_10": "Sacrum",
    "label_11": "Pelvis",
    "label_12": "Femur",
}


def fmt(value):
    return "" if value is None else f"{float(value):.4f}"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def by_view(metrics, task, label_name):
    rows = [
        r for r in metrics["tasks"][task]["per_case_view_label"]
        if r["label_name"] == label_name
    ]
    result = []
    for view in ["anterior", "posterior"]:
        view_rows = [r for r in rows if r["view"] == view]
        result.append({
            "view": view,
            "dice": mean(view_rows, "dice"),
            "sensitivity": mean(view_rows, "sensitivity"),
            "specificity": mean(view_rows, "specificity"),
        })
    return result


def main():
    runs = [
        ("best", "A2 dual head best", "checkpoint_best.pth"),
        ("latest", "A2 dual head latest", "checkpoint_final.pth"),
    ]
    splits = ["val", "test"]

    overall_rows = []
    lesion_rows = []
    bone_rows = []
    view_rows = []

    for run_dir, model_name, checkpoint in runs:
        for split in splits:
            metrics_path = EVAL_ROOT / model_name.replace(" ", "_").replace("A2_dual_head_", "A2_dual_head_")
            metrics_file = EVAL_ROOT / f"A2_dual_head_{run_dir}" / split / "metrics.json"
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            lesion = metrics["tasks"]["lesion"]
            bone = metrics["tasks"]["bone"]
            lw = lesion["lesionwise_class_matched"]
            overall_rows.append({
                "model": model_name,
                "split": split,
                "checkpoint": checkpoint,
                "cases": metrics["num_cases"],
                "lesion_dice": fmt(lesion["pixel_mean"]["dice"]),
                "lesion_malignant_dice": fmt(lesion["by_label"]["malignant"]["dice"]),
                "lesion_sensitivity": fmt(lesion["pixel_mean"]["sensitivity"]),
                "lesion_specificity": fmt(lesion["pixel_mean"]["specificity"]),
                "lesionwise_f1": fmt(lw["lesionwise_f1"]),
                "froc_llf": fmt(lw["froc"]["lesion_localization_fraction"]),
                "froc_fp_per_case": fmt(lw["froc"]["false_positives_per_case"]),
                "bone_dice": fmt(bone["pixel_mean"]["dice"]),
                "bone_sensitivity": fmt(bone["pixel_mean"]["sensitivity"]),
                "bone_specificity": fmt(bone["pixel_mean"]["specificity"]),
            })

            for label_name, label_metrics in lesion["by_label"].items():
                lesion_rows.append({
                    "model": model_name,
                    "split": split,
                    "checkpoint": checkpoint,
                    "label": "lesion malignant" if label_name == "malignant" else f"lesion {label_name}",
                    "dice": fmt(label_metrics["dice"]),
                    "sensitivity": fmt(label_metrics["sensitivity"]),
                    "specificity": fmt(label_metrics["specificity"]),
                    "lesionwise_f1": fmt(label_metrics["lesionwise_f1"]),
                    "froc_llf": fmt(label_metrics["froc"]["lesion_localization_fraction"]),
                    "froc_fp_per_case": fmt(label_metrics["froc"]["false_positives_per_case"]),
                })
                for view_metric in by_view(metrics, "lesion", label_name):
                    view_rows.append({
                        "model": model_name,
                        "split": split,
                        "task": "lesion",
                        "label": "lesion malignant" if label_name == "malignant" else f"lesion {label_name}",
                        "view": view_metric["view"],
                        "dice": fmt(view_metric["dice"]),
                        "sensitivity": fmt(view_metric["sensitivity"]),
                        "specificity": fmt(view_metric["specificity"]),
                    })

            for label_name, label_metrics in bone["by_label"].items():
                bone_rows.append({
                    "model": model_name,
                    "split": split,
                    "checkpoint": checkpoint,
                    "label_id": label_name.replace("label_", ""),
                    "bone_region": BONE_NAMES.get(label_name, label_name),
                    "dice": fmt(label_metrics["dice"]),
                    "sensitivity": fmt(label_metrics["sensitivity"]),
                    "specificity": fmt(label_metrics["specificity"]),
                })
                for view_metric in by_view(metrics, "bone", label_name):
                    view_rows.append({
                        "model": model_name,
                        "split": split,
                        "task": "bone",
                        "label": BONE_NAMES.get(label_name, label_name),
                        "view": view_metric["view"],
                        "dice": fmt(view_metric["dice"]),
                        "sensitivity": fmt(view_metric["sensitivity"]),
                        "specificity": fmt(view_metric["specificity"]),
                    })

    write_csv(EVAL_ROOT / "overall_summary_clean.csv", overall_rows)
    write_csv(EVAL_ROOT / "lesion_by_label.csv", lesion_rows)
    write_csv(EVAL_ROOT / "bone_by_label.csv", bone_rows)
    write_csv(EVAL_ROOT / "by_view_label.csv", view_rows)

    report = [
        "# A2 Dual-Head 100-Epoch Best/Latest Evaluation",
        "",
        "Latest means `checkpoint_final.pth`, the final epoch-100 checkpoint. Lesion malignant is lesion class `2`.",
        "",
        "## Overall",
        markdown_table(list(overall_rows[0].keys()), overall_rows),
        "",
        "## Lesion By Label",
        markdown_table(list(lesion_rows[0].keys()), lesion_rows),
        "",
        "## Bone By Label",
        markdown_table(list(bone_rows[0].keys()), bone_rows),
        "",
        "## Output Files",
        f"- Overall CSV: `{EVAL_ROOT / 'overall_summary_clean.csv'}`",
        f"- Lesion CSV: `{EVAL_ROOT / 'lesion_by_label.csv'}`",
        f"- Bone CSV: `{EVAL_ROOT / 'bone_by_label.csv'}`",
        f"- View CSV: `{EVAL_ROOT / 'by_view_label.csv'}`",
    ]
    (EVAL_ROOT / "A2_100epoch_best_latest_evaluation.md").write_text("\n".join(report), encoding="utf-8")

    print(EVAL_ROOT / "A2_100epoch_best_latest_evaluation.md")


if __name__ == "__main__":
    main()
