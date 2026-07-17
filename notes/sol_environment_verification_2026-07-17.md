# SOL Environment Verification - 2026-07-17

## 1. Purpose

This document records the verified execution environment for the STP 499 signed/directed GNN experiments on the SOL cluster.

This repository currently contains the earlier GCN baseline work. No signed/directed experiment implementation was added in this milestone.

## 2. Repository State

- **Repository**: `/scratch/shirpa/gnn-stability/research/GNN-Stability-Analysis`
- **Branch**: `main`
- **Remote**: `https://github.com/sileshith/GNN-Stability-Analysis.git`
- **Repository status**: Clean before this documentation change
- **Starting commit**: `8356aa3 Verify GCN baseline notebook on Sol`

## 3. Storage Protection

- Scratch research directory was copied to permanent storage:
  `/home/shirpa/gnn-stability-backup-2026-07-17/research/`
- Copy completed successfully
- Copied size reported by rsync: 55,800,898 bytes
- Scratch storage remains temporary and must not be treated as the sole copy
- The Conda environment export is stored at:
  `/home/shirpa/gnn-stability-backup-2026-07-17/gnn_env.yml`

## 4. Verified GNN Environment

- **Environment path**: `/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env`
- **Python**: 3.11.15
- **PyTorch**: 2.12.0+cu130
- **PyTorch CUDA build**: 13.0
- **PyTorch Geometric**: 2.7.0
- **PyTorch Geometric Signed Directed**: 1.1.1
- **NumPy**: 1.26.4
- **SciPy**: 1.17.1
- **NetworkX**: 3.6.1
- **CUDA availability**: False in this verification session because the VSCode job used the CPU-only `lightwork` partition. Do not interpret this as a broken CUDA installation.

## 5. Verified PyGSD Components

Successfully discovered components:

- `MSGNN_link_prediction`
- `MSGNN_node_classification`
- `MagNet_link_prediction`
- `SSSNET_link_prediction`
- magnetic Laplacian utility
- magnetic-signed Laplacian utility

Module discovery confirms the required architecture families are importable, but does not yet verify training correctness, dataset compatibility, checkpoint behavior, or perturbation experiments.

## 6. Aider Environment

- A separate permanent Aider environment was created at:
  `/home/shirpa/.conda/envs/aider_env`
- **Aider version**: 0.86.2
- Aider is isolated from `gnn_env`
- Aider temporary history and cache files are excluded locally through `.git/info/exclude`; no tracked `.gitignore` change was made

## 7. Verification Verdict

- **Environment status**: Verified
- **GPU execution status**: Not tested in this CPU-only session
- **PyGSD import status**: Verified
- **Architecture-module discovery**: Verified
- **End-to-end signed/directed model execution**: Pending
- **Dataset loading and perturbation protocol**: Pending

## 8. Next Controlled Coding Milestone

The next milestone is a minimal CPU smoke test that imports and instantiates the selected PyGSD architecture classes using documented constructor signatures, without training, downloading datasets, or changing the baseline results.
