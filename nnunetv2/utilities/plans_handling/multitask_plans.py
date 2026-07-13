from typing import List


def get_multitask_config(configuration_manager) -> dict:
    multitask = configuration_manager.network_arch_init_kwargs.get("multitask")
    if multitask is None:
        raise ValueError("Configuration does not define a multitask architecture block.")
    validate_multitask_config(multitask)
    return multitask


def validate_multitask_config(multitask: dict):
    variant = multitask.get("variant")
    if variant not in {"dual_head", "dual_decoder"}:
        raise ValueError(f"Unsupported multitask variant: {variant}")
    tasks = multitask.get("tasks", [])
    if len(tasks) < 1:
        raise ValueError("Multitask plans require at least one task.")
    for idx, task in enumerate(tasks):
        if "num_classes" not in task:
            raise ValueError(f"Task at index {idx} is missing num_classes.")
        if int(task["num_classes"]) < 1:
            raise ValueError(f"Task at index {idx} must have num_classes >= 1.")


def get_multitask_task_names(multitask: dict) -> List[str]:
    validate_multitask_config(multitask)
    return [task.get("name", f"task{i + 1}") for i, task in enumerate(multitask["tasks"])]
