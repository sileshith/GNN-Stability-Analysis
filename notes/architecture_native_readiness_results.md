# Architecture-Native GPU Readiness Results

**Project:** Stability and Robustness of Signed and Directed Graph Neural Networks Under Structural Perturbations

**Execution date:** 2026-07-25

**Status:** PASS with an SGCN model-quality warning

## 1. Purpose

This readiness gate tested whether the three primary theory-aligned baseline architectures can load an architecture-native graph view, train for one clean seed on a SOL GPU, produce evaluation metrics, save a clean checkpoint, reload that checkpoint into a fresh model, and expose outputs needed for later stability diagnostics.

This gate did not apply structural perturbations and does not support robustness, stability, or architecture-ranking conclusions.

## 2. Locked Scope

- One initialization seed: 0
- One synthetic architecture-native dataset per model
- Clean training only
- NVIDIA A100 MIG 1g.20gb
- PyTorch 2.12.0+cu130
- PyG 2.7.0
- PyGSD 1.1.1
- No perturbation budgets
- No cross-architecture performance comparison

## 3. Results

| Model | Graph view | Native task | Epochs | Best epoch | Final train loss | Best val NLL | Test accuracy | Test macro-F1 | Reload max difference | Exposed object | Operator |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| MSGNN | signed-directed | four_class_signed_digraph | 30 | 30 | 1.211027 | 1.238663 | 0.473984 | 0.329925 | 1.669e-06 | query-edge log-probabilities | MSConv |
| MagNet | directed-unsigned | direction | 30 | 25 | 0.638113 | 0.715399 | 0.545858 | 0.545777 | 4.768e-07 | query-edge log-probabilities | MagNetConv |
| SGCN | signed-undirected | link_sign_prediction | 16 | 6 | 1.098682 | 1.094243 | 0.568075 | 0.362275 | 1.192e-07 | node embeddings and query-edge sign log-probabilities | SGCNConv |

## 4. Readiness Findings

- MSGNN passed GPU training, evaluation, durable checkpoint saving, fresh-model reload, query-edge log-probability exposure, and MSConv detection.
- MagNet passed GPU training, evaluation, durable checkpoint saving, fresh-model reload, direction-logit exposure, and MagNetConv detection.
- SGCN passed GPU training, evaluation, durable checkpoint saving, fresh-model reload, node-embedding exposure, and SGCNConv detection.
- All checkpoint reload differences were below the accepted CUDA tolerance of 1e-5.
- The MSGNN dataset fingerprint matches the previously verified Gate E clean dataset fingerprint.

## 5. SGCN Model-Quality Warning

SGCN passed the software-readiness gate, but its restored best checkpoint predicted only class 1 on the held-out test edges. The resulting test accuracy was 0.568075 and macro-F1 was 0.362275.

This class collapse does not invalidate execution readiness. It does mean that the present SGCN configuration is not yet suitable as a scientifically meaningful clean baseline. The signed-data balance, native objective, feature initialization, classifier behavior, learning rate, training duration, and split protocol require review before any perturbation comparison.

## 6. Evidence Locations

- `results/architecture_readiness/msgnn_seed0_readiness.json`
- `results/architecture_readiness/magnet_seed0_readiness.json`
- `results/architecture_readiness/sgcn_seed0_readiness.json`
- Durable checkpoints: `/home/shirpa/stp499_persistent/checkpoints/architecture_readiness/`
- Runner: `scripts/architecture_native_readiness.py`

The checkpoint files are intentionally excluded from Git because they are binary runtime artifacts. The small JSON evidence records and this note are suitable for version control.

## 7. Decision

The three primary baseline architectures are technically ready on SOL for clean, one-seed execution. MSGNN and MagNet show clear clean-training learning signals. SGCN remains execution-ready with a model-quality warning.

Structural perturbation execution remains pending experimental-design ratification and resolution of the SGCN baseline-quality warning.

## 8. Boundaries

- No robustness conclusion
- No stability conclusion
- No architecture superiority claim
- No statistical inference from one seed
- No claim of generalization beyond the tested synthetic graph views
