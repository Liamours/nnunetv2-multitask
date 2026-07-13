import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2, wilcoxon


def _load_model_specs(specs: List[str]) -> Dict[str, dict]:
    models = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be NAME=metrics.json. Got: {spec}")
        name, path = spec.split("=", 1)
        models[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    return models


def _row_key(row: dict) -> Tuple[str, str, str]:
    return row["case_id"], row["view"], row["label_name"]


def _continuous_rows(metrics: dict, task: str, label_name: str, metric: str) -> Dict[Tuple[str, str, str], float]:
    rows = metrics["tasks"][task]["per_case_view_label"]
    selected = {}
    for row in rows:
        if row["label_name"] != label_name:
            continue
        value = row.get(metric)
        if value is not None:
            selected[_row_key(row)] = float(value)
    return selected


def _binary_rows(metrics: dict, task: str, label_name: str, binary_field: str) -> Dict[Tuple[str, str, str], int]:
    if task != "lesion":
        rows = metrics["tasks"][task]["per_case_view_label"]
        return {
            _row_key(row): int(row.get(binary_field, 0))
            for row in rows
            if row["label_name"] == label_name
        }
    rows = metrics["tasks"][task].get("lesionwise_per_case_view_label", [])
    selected = {}
    for row in rows:
        if row["label_name"] != label_name:
            continue
        if int(row.get("eligible_positive_gt", 1)) == 0:
            continue
        selected[_row_key(row)] = int(row[binary_field])
    return selected


def _paired_vectors(a: Dict[Tuple[str, str, str], float], b: Dict[Tuple[str, str, str], float]):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("No paired samples found.")
    return keys, np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def wilcoxon_signed_rank(models: Dict[str, dict], model_a: str, model_b: str, task: str, label_name: str, metric: str):
    rows_a = _continuous_rows(models[model_a], task, label_name, metric)
    rows_b = _continuous_rows(models[model_b], task, label_name, metric)
    keys, values_a, values_b = _paired_vectors(rows_a, rows_b)
    result = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
    return {
        "test": "Wilcoxon Signed-Rank Test",
        "model_a": model_a,
        "model_b": model_b,
        "task": task,
        "label_name": label_name,
        "metric": metric,
        "n_pairs": len(keys),
        "model_a_mean": float(np.mean(values_a)),
        "model_b_mean": float(np.mean(values_b)),
        "mean_difference_a_minus_b": float(np.mean(values_a - values_b)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def cochran_q(models: Dict[str, dict], task: str, label_name: str, binary_field: str):
    model_rows = {
        model_name: _binary_rows(metrics, task, label_name, binary_field)
        for model_name, metrics in models.items()
    }
    common_keys = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    if len(models) < 3:
        raise ValueError("Cochran's Q requires at least three models.")
    if not common_keys:
        raise ValueError("No paired binary samples found across all models.")

    matrix = np.array([[model_rows[name][key] for name in models] for key in common_keys], dtype=np.float64)
    n, k = matrix.shape
    col_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = col_sums.sum()
    denominator = k * total - np.sum(row_sums ** 2)
    q_stat = 0.0 if denominator == 0 else (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2) / denominator
    p_value = chi2.sf(q_stat, k - 1)

    pairwise = {}
    for a, b in itertools.combinations(models.keys(), 2):
        _, values_a, values_b = _paired_vectors(model_rows[a], model_rows[b])
        pairwise[f"{a}_vs_{b}"] = {
            "discordant_a_success_b_fail": int(np.logical_and(values_a == 1, values_b == 0).sum()),
            "discordant_a_fail_b_success": int(np.logical_and(values_a == 0, values_b == 1).sum()),
        }

    return {
        "test": "Cochran's Q Test",
        "models": list(models.keys()),
        "task": task,
        "label_name": label_name,
        "binary_field": binary_field,
        "n_pairs": int(n),
        "success_rate_by_model": {
            name: float(matrix[:, idx].mean())
            for idx, name in enumerate(models.keys())
        },
        "statistic": float(q_stat),
        "degrees_of_freedom": int(k - 1),
        "p_value": float(p_value),
        "pairwise_discordance_counts": pairwise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=metrics.json. Repeat for each model.")
    parser.add_argument("--test", choices=["wilcoxon", "cochran_q"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--label_name", required=True)
    parser.add_argument("--metric", default="dice", help="Continuous metric for Wilcoxon. Example: dice.")
    parser.add_argument("--binary_field", default="binary_all_gt_detected",
                        help="Binary field for Cochran's Q. Example: binary_all_gt_detected.")
    parser.add_argument("--model_a")
    parser.add_argument("--model_b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    models = _load_model_specs(args.model)
    if args.test == "wilcoxon":
        if not args.model_a or not args.model_b:
            raise ValueError("--model_a and --model_b are required for Wilcoxon.")
        result = wilcoxon_signed_rank(models, args.model_a, args.model_b, args.task, args.label_name, args.metric)
    else:
        result = cochran_q(models, args.task, args.label_name, args.binary_field)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
