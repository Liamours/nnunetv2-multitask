"""Pin every plan in a controlled ablation to the SAME compute-shape parameters.

The auto-planner sizes each config independently against a VRAM budget, so heavier variants (dual
decoder, +CBAM) silently get a smaller patch size and/or fewer stages than the lighter ones. That
turns the sharing-pattern comparison into a confounded one: an A3-vs-A2 gap would reflect patch size
and network depth as much as the architecture. `context/nnunetv2/controlled-ablation-rules.md`
requires all configs to share these values, dictated by the most-constrained config.

This copies the reference (most-constrained) config's compute shape onto the others, while leaving
each plan's own intentional variables untouched:
  copied   - batch_size, patch_size, and the architecture topology derived from patch size
             (n_stages, features_per_stage, kernel_sizes, strides, n_conv_per_stage,
             n_conv_per_stage_decoder)
  preserved - network_class_name (dual_head vs dual_decoder), multitask.tasks, cbam

It also optionally unifies data_identifier so plans whose preprocessing parameters are identical
share one preprocessed cache instead of duplicating it per config.
"""
import argparse
import json
from pathlib import Path

# derived from patch size by the planner - must match across configs for a fair comparison
TOPOLOGY_KEYS = [
    "n_stages",
    "features_per_stage",
    "kernel_sizes",
    "strides",
    "n_conv_per_stage",
    "n_conv_per_stage_decoder",
]
# must match across configs, but are NOT architecture-topology
SHAPE_KEYS = ["batch_size", "patch_size"]
# preprocessing-relevant: if these differ, plans genuinely cannot share a preprocessed cache
PREPROC_KEYS = [
    "spacing",
    "normalization_schemes",
    "use_mask_for_norm",
    "resampling_fn_data",
    "resampling_fn_seg",
    "resampling_fn_data_kwargs",
    "resampling_fn_seg_kwargs",
    "preprocessor_name",
]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def save(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=4, sort_keys=False))


def summarize(plans_path: Path, config: str) -> dict:
    c = load(plans_path)["configurations"][config]
    a = c["architecture"]
    return {
        "plan": plans_path.stem,
        "batch_size": c["batch_size"],
        "patch_size": c["patch_size"],
        "n_stages": a["arch_kwargs"]["n_stages"],
        "network": a["network_class_name"].rsplit(".", 1)[-1],
        "cbam": a["arch_kwargs"]["cbam"].get("enabled", False),
        "tasks": [t["name"] for t in a["arch_kwargs"]["multitask"]["tasks"]],
        "data_identifier": c["data_identifier"],
    }


def pin(reference: Path, targets: list, config: str, shared_data_identifier: str = None,
        dry_run: bool = False) -> None:
    ref_plans = load(reference)
    ref_cfg = ref_plans["configurations"][config]
    ref_arch = ref_cfg["architecture"]["arch_kwargs"]

    print(f"reference: {reference.stem}")
    print(f"  batch_size={ref_cfg['batch_size']} patch_size={ref_cfg['patch_size']} "
          f"n_stages={ref_arch['n_stages']}")

    # the reference joins the shared cache too, otherwise it preprocesses to its own copy - but only
    # when it lives alongside the targets; a cross-dataset reference has its own cache elsewhere
    same_dir = targets and reference.parent == targets[0].parent
    if same_dir and shared_data_identifier is not None and ref_cfg["data_identifier"] != shared_data_identifier:
        print(f"    data_identifier: {ref_cfg['data_identifier']} -> {shared_data_identifier}")
        ref_cfg["data_identifier"] = shared_data_identifier
        if not dry_run:
            save(reference, ref_plans)
    print()

    for target in targets:
        plans = load(target)
        cfg = plans["configurations"][config]
        arch = cfg["architecture"]["arch_kwargs"]

        # refuse to unify a preprocessed cache across plans that would preprocess differently
        if shared_data_identifier is not None:
            for k in PREPROC_KEYS:
                if json.dumps(cfg.get(k), sort_keys=True) != json.dumps(ref_cfg.get(k), sort_keys=True):
                    raise ValueError(
                        f"{target.stem}: {k} differs from reference ({cfg.get(k)} vs {ref_cfg.get(k)}); "
                        f"cannot share a preprocessed cache."
                    )

        changes = []
        for k in SHAPE_KEYS:
            if cfg[k] != ref_cfg[k]:
                changes.append(f"{k}: {cfg[k]} -> {ref_cfg[k]}")
                cfg[k] = ref_cfg[k]
        for k in TOPOLOGY_KEYS:
            if arch[k] != ref_arch[k]:
                changes.append(f"{k}: {arch[k]} -> {ref_arch[k]}")
                arch[k] = ref_arch[k]
        if shared_data_identifier is not None and cfg["data_identifier"] != shared_data_identifier:
            changes.append(f"data_identifier: {cfg['data_identifier']} -> {shared_data_identifier}")
            cfg["data_identifier"] = shared_data_identifier

        if changes:
            print(f"{target.stem}:")
            for ch in changes:
                print(f"    {ch}")
            if not dry_run:
                save(target, plans)
        else:
            print(f"{target.stem}: already matches reference")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", type=Path, required=True,
                    help="nnUNet_preprocessed/<DatasetXXX_Name> containing the plans json files")
    ap.add_argument("--reference", required=True, help="plans name (no .json) to copy compute shape FROM")
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="directory holding --reference; defaults to --preprocessed-dir. Use this to pin "
                         "against the most-constrained config of a DIFFERENT dataset (e.g. the "
                         "single-task baseline lives in its own dataset but must match the multitask one)")
    ap.add_argument("--targets", nargs="+", required=True, help="plans names (no .json) to pin")
    ap.add_argument("--config", default="2d")
    ap.add_argument("--shared-data-identifier", default=None,
                    help="if set, point every plan at this one preprocessed cache")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref_dir = args.reference_dir if args.reference_dir is not None else args.preprocessed_dir
    ref = ref_dir / f"{args.reference}.json"
    targets = [args.preprocessed_dir / f"{t}.json" for t in args.targets]
    for p in [ref, *targets]:
        if not p.is_file():
            raise FileNotFoundError(p)

    pin(ref, targets, args.config, args.shared_data_identifier, args.dry_run)

    print("=== post-pin audit ===")
    rows = [summarize(p, args.config) for p in [ref, *targets]]
    width = max(len(r["plan"]) for r in rows)
    for r in rows:
        print(f"{r['plan']:<{width}}  bs={r['batch_size']}  patch={r['patch_size']}  "
              f"stages={r['n_stages']}  net={r['network']:<24} cbam={str(r['cbam']):<5} "
              f"tasks={','.join(r['tasks'])}  cache={r['data_identifier']}")

    varying = {k: {json.dumps(r[k]) for r in rows} for k in ["batch_size", "patch_size", "n_stages"]}
    bad = {k: v for k, v in varying.items() if len(v) > 1}
    print()
    if bad:
        print("FAIL - these must be identical across configs but are not:", bad)
    else:
        print("OK - batch_size, patch_size, n_stages identical across all configs")


if __name__ == "__main__":
    main()
