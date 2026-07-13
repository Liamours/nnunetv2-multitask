# nnunetv2-multitask

Research fork of nnU-Net v2 for multi-task segmentation.

## Goal

Modify the nnU-Net v2 pipeline to support multi-task segmentation with:
- multi-head segmentation
- multi-decoder segmentation
- configuration-driven branching
- optional CBAM integration in encoder, decoder, and possibly bottleneck blocks

## Current status

Implemented:
- generic two-task `dual_head` and `dual_decoder` architectures
- configuration-driven multitask plans, trainer, loss, inference, export, and evaluation
- optional CBAM attention for the supported ablations
- paired anterior/posterior lesion and bone-region dataset conversion and preprocessing
- targeted architecture, plans, inference, and raw-dataset tests

Not implemented:
- local-gradient or pixel-wise uncertainty quantification

The fork should stay generic. WBS lesion/bone segmentation is a downstream use case, not a hardcoded project boundary.

## Operational scripts

Repository helpers are organized under [scripts/README.md](scripts/README.md). Data, checkpoints, logs, and generated evaluation artifacts remain outside the repository.
