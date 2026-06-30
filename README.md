# nnunetv2-multitask

Research fork of nnU-Net v2 for multi-task segmentation.

## Goal

Modify the nnU-Net v2 pipeline to support multi-task segmentation with:
- multi-head segmentation
- multi-decoder segmentation
- configuration-driven branching
- optional CBAM integration in encoder, decoder, and possibly bottleneck blocks

## Current status

Implemented baseline multi-task support:
- generic two-task `dual_head`
- generic two-task `dual_decoder`
- multitask plans/config parsing
- multitask trainer and loss wrapper
- per-task inference and export path
- targeted tests and model checker

Not implemented yet:
- CBAM integration
- LGUQ uncertainty scoring
- dataset-specific WBS experiment pipeline

The fork should stay generic. WBS lesion/bone segmentation is a downstream use case, not a hardcoded project boundary.
