# Operational Scripts

Scripts are grouped by purpose. Run PowerShell helpers from the repository root with `uv` available; data, checkpoints, logs, and evaluation outputs are stored outside this repository.

- `analysis/`: parameter and FLOP benchmarking.
- `evaluation/`: best/latest checkpoint evaluation and six-experiment result tables.
- `inference/`: standalone multitask inference without the nnU-Net prediction CLI.
- `planning/`: canonical plan aliases for controlled ablations.
- `preprocessing/`: standalone paired-view preprocessing.
- `training/`: A1 setup, smoke tests, and sequential CBAM training launchers.
- `visualization/`: palette overlays for multitask predictions.

Use `training/setup_a1_lesion_only.ps1` to validate the A1 data setup, `training/run_ablation_smoke_tests.ps1` for short training checks, and the named scripts under `evaluation/` for completed-run evaluation. Before any comparison, follow the controlled-ablation policy maintained with the project context.
