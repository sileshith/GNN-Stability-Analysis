---
title: "Gate D — Minimal Training Compatibility Results"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate D execution record"
status: "PASS"
execution_date: "2026-07-18"
evidence_level: "2 (single functional run with validation)"
---

# Gate D — Minimal Training Compatibility Results

## 1. Execution Status and Scope

**Execution Timestamp:** 2026-07-18T17:51:37.792853+00:00

**Script Identity:**
- Commit: `2edd583`
- Script: `scripts/pygsd_gate_d_training_compatibility.py`

**Interpreter Environment:**
- Python: 3.11.15
- PyTorch: 2.12.0+cu130
- PyG: 2.7.0
- PyGSD: 1.1.1
- Device: CPU

**Task Configuration:**
- Task: `four_class_signed_digraph`
- Generation seed: 0
- Split seed: 0
- Validation probability: 0.15
- Test probability: 0.15

**Training Configuration:**
- Optimizer: Adam (both models)
- Learning rate: 0.01
- Weight decay: 0.0005
- Loss function: `nn.NLLLoss()`
- Required optimizer steps: exactly one per model

**Pre-Execution Verification:**

Independent syntax check completed successfully:

```bash
/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/bin/python -m py_compile scripts/pygsd_gate_d_training_compatibility.py
```

The syntax command returned no error output.

**Execution Confirmation:**

The script was actually executed on the SOL computational environment.

**Scope Limitations:**

- One deterministic tiny graph only
- One initialization seed only
- One optimizer step required per model
- CPU execution only
- No perturbations
- No hyperparameter search
- No production metrics
- No model comparison claims
- No convergence testing
- No robustness or stability testing

---

## 2. Gate Question

**Gate D Question:**

Can `MSGNN_link_prediction` and `SSSNET_link_prediction` with `directed=True` each complete one minimal end-to-end training step on the same tiny signed-directed graph using the PyGSD 1.1.1 `four_class_signed_digraph` task?

**Answer:**

The observed execution answered this question affirmatively for both tested models.

---

## 3. Shared Data Contract

**Graph Structure:**

- Clean graph edges: 24
- Positive clean edges: 12
- Negative clean edges: 12
- Training graph edges: 17
- Training queries: 34
- Validation queries: 6
- Test queries: 8

**Feature Representation:**

- Feature shape: `(12, 2)`
- Feature dtype: `torch.float32`

**Class Distributions:**

- Training: `[8, 9, 8, 9]` (classes 0, 1, 2, 3)
- Validation: `[2, 1, 2, 1]` (classes 0, 1, 2, 3)
- Test: `[2, 2, 2, 2]` (classes 0, 1, 2, 3)

**Data Fingerprint:**

```text
2714be446fbaa495608f7701ff290f07b8ccaf61852cd07da3f0be0cfc7cd556
```

**Feature Fingerprint:**

```text
09cf9fb82e32803a7fbdbd26df9cbc34101688f683739530635350e6f633380c
```

**Verification:**

All four target classes were present in the training split as required by the specification.

---

## 4. MSGNN Runtime Result

### Tested Configuration

```python
MSGNN_link_prediction(
    num_features=2,
    hidden=4,
    q=0.25,
    K=2,
    label_dim=4,
    activation=True,
    trainable_q=False,
    layer=2,
    dropout=0.0,
    normalization='sym',
    cached=False,
    conv_bias=True,
    absolute_degree=True,
)
```

### Optimizer Configuration

```python
torch.optim.Adam(
    lr=0.01,
    weight_decay=0.0005,
)
```

### Training Step Result

- **Initial loss:** 3.052259
- **Post-step loss:** 2.460572
- **Trainable parameter tensors:** 6
- **Trainable parameter elements:** 148
- **Parameter tensors with gradients:** 6
- **Parameter tensors with finite gradients:** 6
- **Parameter tensors with nonzero gradients:** 6
- **Changed parameter tensors:** 6

### Evaluation Path Result

- **Test accuracy:** 0.2500
- **Test macro-F1:** 0.1667
- **Test micro-F1:** 0.2500

### Fingerprint Verification

- **Data fingerprint:** Matched shared-data fingerprint
- **Feature fingerprint:** Matched shared-feature fingerprint

### Final Status

**PASS**

**Clarification:**

The lower post-step loss is not interpreted as convergence evidence. Loss reduction was not required by Gate D.

---

## 5. Directed SSSNET Runtime Result

### Tested Configuration

```python
SSSNET_link_prediction(
    nfeat=2,
    hidden=4,
    nclass=4,
    dropout=0.0,
    hop=2,
    fill_value=0.5,
    directed=True,
    bias=True,
)
```

### Optimizer Configuration

```python
torch.optim.Adam(
    lr=0.01,
    weight_decay=0.0005,
)
```

### Training Step Result

- **Initial loss:** 5.559962
- **Post-step loss:** 4.392590
- **Trainable parameter tensors:** 14
- **Trainable parameter elements:** 240
- **Parameter tensors with gradients:** 14
- **Parameter tensors with finite gradients:** 14
- **Parameter tensors with nonzero gradients:** 14
- **Changed parameter tensors:** 14

### Evaluation Path Result

- **Test accuracy:** 0.3750
- **Test macro-F1:** 0.2429
- **Test micro-F1:** 0.3750

### Fingerprint Verification

- **Data fingerprint:** Matched shared-data fingerprint
- **Feature fingerprint:** Matched shared-feature fingerprint

### Final Status

**PASS**

**Clarification:**

The lower post-step loss is not interpreted as convergence evidence. Loss reduction was not required by Gate D.

---

## 6. Shared-Record Verification

**Fingerprint Cross-Validation:**

- MSGNN and SSSNET full data fingerprints matched each other
- Both full data fingerprints matched the original shared-data fingerprint
- MSGNN and SSSNET feature fingerprints matched each other
- Both feature fingerprints matched the original shared-feature fingerprint

**Shared-Record Verdict:** PASS

**Interpretation:**

Both models used identical clean graphs, training graphs, splits, features, and seeds.

---

## 7. Consolidated Result

| Model | Status | Initial Loss | Post-Step Loss | Gradients | Parameter Change | Evaluation | Fingerprints |
|-------|--------|--------------|----------------|-----------|------------------|------------|--------------|
| **MSGNN** | PASS | 3.052259 | 2.460572 | PASS | PASS | PASS | PASS |
| **SSSNET** | PASS | 5.559962 | 4.392590 | PASS | PASS | PASS | PASS |

**Gate D Evaluation Result:** PASS

**Precise Definition:**

Both selected model classes completed their approved minimal one-step training tests on the same shared signed-directed four-class split and satisfied all required structural, numerical, gradient, optimizer-step, and evaluation-path assertions.

---

## 8. Supported Conclusion

The following precise conclusion is supported by this execution:

> Under the fixed deterministic CPU configuration, PyGSD 1.1.1 `MSGNN_link_prediction` and directed `SSSNET_link_prediction` each completed one minimal optimizer step on the same tiny signed-directed four-class split. Both produced finite normalized outputs and finite scalar losses, completed backward propagation with finite nonzero gradients, changed at least one trainable parameter, produced finite post-step outputs and losses, executed the held-out test path, and used matching shared-data and feature fingerprints.

---

## 9. Claims Now Supported

The following runtime claims are now supported by this execution:

- Both tested model APIs support minimal one-step training on the defined task
- Loss computation executes and produces finite scalar values
- Backward propagation completes without exception
- Gradients are finite and at least partially nonzero
- Optimizer steps change trainable parameters
- Post-step parameters, outputs, and losses remain finite
- Held-out evaluation path executes and produces finite metrics
- Both models used identical shared data as verified by cryptographic fingerprints
- Models can proceed to functional-baseline specification

**Scope Limitation:**

These claims apply only to the exact tested configurations, graph, split, seed, and device.

---

## 10. Claims Still Prohibited

The following claims are **not** supported by this execution:

- Convergence
- Meaningful learning
- Predictive quality or accuracy
- Generalization to other data
- Robustness under perturbation
- Structural stability
- Production readiness
- Architecture superiority
- Valid performance comparison between MSGNN and SSSNET
- Behavior under other seeds, datasets, devices, splits, or hyperparameters
- Multi-seed repeatability
- GPU compatibility
- Scalability to larger graphs

**Metric Interpretation:**

Test accuracy and F1 values are recorded as execution evidence only. They do not constitute performance estimates or model comparisons.

---

## 11. Gate Interpretation

**Gate D:** PASS

**Gate C:** PASS (forward-pass smoke test only)

**Gate B:** Remains PARTIAL

**Critical Clarification:**

Gate D passing establishes minimal training compatibility only. It does not automatically resolve Gate B task-semantics questions or authorize perturbation experiments.

**Next Authorized Action:**

Specification of a clean functional baseline (not robustness experiments).

---

## 12. Repository Verification

**Post-Execution Repository State:**

```bash
git status --short
```

Returned no output, confirming that the run created no tracked or untracked repository changes.

---

## 13. Remaining Technical Risks

The following technical risks remain unresolved:

- Multi-seed training repeatability remains unverified
- Convergence behavior remains unverified
- Hyperparameter sensitivity remains unverified
- GPU training compatibility remains unverified
- Larger-graph scalability remains unverified
- Perturbation tolerance remains unverified
- Changed-graph cache recomputation remains dynamically unverified
- Production-scale suitability remains unverified

---

## 14. Next Controlled Work Unit

**Recommendation:**

Specify a clean functional-baseline experiment that:

- Uses multiple independent seeds
- Trains to a defined stopping criterion
- Records convergence trajectories
- Establishes clean-graph baseline metrics
- Does not yet introduce perturbations

**Critical Requirement:**

No perturbation experiment should be designed until clean-graph functional baselines are established.

---

## 15. Final Evidence Conclusion

**Summary:**

- Gate D was successfully executed
- MSGNN and directed SSSNET each passed all required minimal training tests
- Both models used identical shared data as verified by fingerprints
- Gate D is PASS within its narrow defined scope
- Gate C remains PASS (forward-pass smoke test only)
- Gate B remains PARTIAL
- No robustness or stability experiment has occurred
- Negative findings and unresolved questions remain preserved

**Evidence Classification:**

This execution provides Evidence Level 2 (single functional run with validation). It does not yet constitute multi-seed repetition, convergence verification, or perturbation testing.

**Preserved Limitations:**

All limitations documented in the specification remain in effect. This test establishes minimal one-step training compatibility only.

---

*Execution completed: 2026-07-18*  
*Gate D status: PASS*  
*Gate C status: PASS (forward-pass smoke test only)*  
*Gate B status: PARTIAL*  
*Evidence level: 2*
