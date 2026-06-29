from __future__ import annotations

from typing import Dict, List, Type, Union

from nnunetv2.utilities.label_handling.label_handling import LabelManager


class MultiTaskLabelManager(object):
    expects_multitask_tasks = True
    is_multitask = True

    def __init__(
        self,
        label_dict: dict,
        regions_class_order: Union[List[int], None],
        multitask_tasks: dict,
        force_use_labels: bool = False,
        inference_nonlin=None,
    ):
        self.label_dict = label_dict
        self.regions_class_order = regions_class_order
        self.task_label_managers: Dict[str, LabelManager] = {}
        self.task_order: List[str] = []

        for idx, (task_name, task_config) in enumerate(multitask_tasks.items()):
            task_regions = task_config.get("regions_class_order")
            task_labels = task_config["labels"]
            self.task_label_managers[task_name] = LabelManager(
                task_labels,
                task_regions,
                force_use_labels=force_use_labels,
                inference_nonlin=inference_nonlin,
            )
            self.task_order.append(task_name)

        if len(self.task_order) != 2:
            raise ValueError("Multi-task v1 requires exactly two tasks in dataset_json['multitask']['tasks'].")

        self._has_regions = any(manager.has_regions for manager in self.task_label_managers.values())
        self._ignore_label = None
        self._all_labels = sorted(
            {
                label
                for manager in self.task_label_managers.values()
                for label in manager.all_labels
            }
        )

    @property
    def has_regions(self) -> bool:
        return self._has_regions

    @property
    def has_ignore_label(self) -> bool:
        return False

    @property
    def all_labels(self):
        return self._all_labels

    @property
    def ignore_label(self):
        return self._ignore_label

    @property
    def foreground_regions(self):
        regions = []
        for manager in self.task_label_managers.values():
            if manager.foreground_regions is not None:
                regions.extend(manager.foreground_regions)
        return regions

    @property
    def foreground_labels(self):
        labels = []
        for manager in self.task_label_managers.values():
            labels.extend(manager.foreground_labels)
        return sorted(set(labels))

    @property
    def num_segmentation_heads(self):
        return sum(manager.num_segmentation_heads for manager in self.task_label_managers.values())

    def get_task_label_manager(self, task_name: str) -> LabelManager:
        return self.task_label_managers[task_name]

    def task_num_segmentation_heads(self) -> Dict[str, int]:
        return {
            task_name: manager.num_segmentation_heads
            for task_name, manager in self.task_label_managers.items()
        }
