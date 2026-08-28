import json
import math
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from wbbs_lguq.paths import EVALUATIONS, NNUNET_RESULTS

RESULTS = NNUNET_RESULTS
EVAL_OUT = EVALUATIONS

MODELS = {
    "A1_lesion_only": RESULTS / "Dataset261_BS80KLesionOnly" / "nnUNetTrainerMultiTask_100epochs__nnUNetPlansA1Lesion2GB__2d",
    "A2_dual_head": RESULTS / "Dataset260_BS80KLesionBoneMT" / "nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTask2GB__2d",
    "A3_dual_decoder": RESULTS / "Dataset260_BS80KLesionBoneMT" / "nnUNetTrainerMultiTask_100epochs__nnUNetPlansMultiTaskDualDecoder__2d",
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def flatten_output_to_loss(output):
    if isinstance(output, dict):
        return sum(flatten_output_to_loss(v) for v in output.values())
    if isinstance(output, list):
        return sum(flatten_output_to_loss(v) for v in output)
    return output.float().square().mean()


def format_int(n):
    return f"{int(n):,}"


def measure_flops(model, sample):
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            model(sample)
        return int(flop_counter.get_total_flops())


def benchmark_train_step(model, batch_size, patch_size, input_channels, device, warmup=1, runs=2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
    data = torch.randn((batch_size, input_channels, *patch_size), device=device)

    def run_once():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(data)
            loss = flatten_output_to_loss(output)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_once()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "peak_mem_mb": peak_mem_mb,
    }


def load_model_info(model_root: Path):
    plans = json.loads((model_root / "plans.json").read_text(encoding="utf-8"))
    dataset_json = json.loads((model_root / "dataset.json").read_text(encoding="utf-8"))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("2d")
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    network = nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=True,
    )
    return plans, dataset_json, plans_manager, configuration_manager, num_input_channels, network


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    rows = []
    bench_rows = []

    for name, model_root in MODELS.items():
        plans, dataset_json, plans_manager, configuration_manager, input_channels, network = load_model_info(model_root)
        patch_size = tuple(int(i) for i in configuration_manager.patch_size)
        batch_size = int(configuration_manager.batch_size)
        total_params, trainable_params = count_params(network)
        conv_feature_map_estimate = int(network.compute_conv_feature_map_size(patch_size))

        flop_network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            allow_init=True,
            deep_supervision=False,
        ).to(device)
        sample = torch.randn((1, input_channels, *patch_size), device=device)
        flops_forward = measure_flops(flop_network, sample)
        del flop_network, sample
        torch.cuda.empty_cache()

        rows.append({
            "model": name,
            "plans_name": plans["plans_name"],
            "network_class": configuration_manager.network_arch_class_name,
            "batch_size": batch_size,
            "patch_size": list(patch_size),
            "input_channels": input_channels,
            "num_segmentation_heads": int(plans_manager.get_label_manager(dataset_json).num_segmentation_heads),
            "param_total": total_params,
            "param_trainable": trainable_params,
            "forward_flops_batch1_ds_off": flops_forward,
            "conv_feature_map_size_estimate": conv_feature_map_estimate,
        })

        for test_bs in sorted(set([1, min(4, batch_size), batch_size])):
            bench_model = get_network_from_plans(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                allow_init=True,
                deep_supervision=True,
            )
            benchmark = benchmark_train_step(bench_model, test_bs, patch_size, input_channels, device)
            bench_rows.append({
                "model": name,
                "batch_size": test_bs,
                "mean_step_s": benchmark["mean_s"],
                "min_step_s": benchmark["min_s"],
                "max_step_s": benchmark["max_s"],
                "peak_mem_mb": benchmark["peak_mem_mb"],
            })
            del bench_model
            torch.cuda.empty_cache()

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_OUT / "model-efficiency-audit.json"
    md_path = EVAL_OUT / "model-efficiency-audit.md"
    payload = {"models": rows, "benchmarks": bench_rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Model Efficiency Audit",
        "",
        "Forward FLOPs are measured with batch size 1 and deep supervision disabled.",
        "Step timing benchmarks are synthetic GPU train-step surrogates with deep supervision enabled.",
        "",
        "## Model Summary",
        "| Model | Plans | Batch | Params | FLOPs (batch1) | Conv Feature Estimate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['plans_name']} | {row['batch_size']} | "
            f"{format_int(row['param_total'])} | {format_int(row['forward_flops_batch1_ds_off'])} | "
            f"{format_int(row['conv_feature_map_size_estimate'])} |"
        )

    lines.extend([
        "",
        "## GPU Step Benchmarks",
        "| Model | Batch | Mean Step (s) | Min Step (s) | Max Step (s) | Peak GPU Mem (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in bench_rows:
        lines.append(
            f"| {row['model']} | {row['batch_size']} | {row['mean_step_s']:.4f} | "
            f"{row['min_step_s']:.4f} | {row['max_step_s']:.4f} | {row['peak_mem_mb']:.1f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
