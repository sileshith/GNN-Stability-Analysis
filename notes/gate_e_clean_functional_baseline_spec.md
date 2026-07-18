---
title: "Gate E — Clean Functional-Baseline Specification"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate E test design specification"
status: "specification only — not executed"
specification_date: "2026-07-18"
---

# Gate E — Clean Functional-Baseline Specification

**Specification Date:** 2026-07-18

---

## 1. Purpose and Gate Question

Gate D passed only minimal one-step CPU compatibility. Gate E evaluates whether MSGNN and directed SSSNET can complete controlled multi-epoch training on clean signed-directed data before any perturbation experiment is authorized.

**Gate E Question:**

> Under one fixed clean signed-directed dataset and split, can PyGSD 1.1.1 `MSGNN_link_prediction` and directed `SSSNET_link_prediction` complete reproducible multi-epoch training across three initialization seeds, maintain finite optimization behavior, demonstrate a minimal validation-loss learning signal, and execute a held-out test evaluation without data-contract violations?

**Clarification:**

This is not yet a robustness, stability, or architecture-superiority test.

---

## 2. Locked Primary Scope

**Models:**

- `MSGNN_link_prediction`
- `SSSNET_link_prediction(directed=True)`

**Task:**

- `four_class_signed_digraph`

**Environment:**

- CPU first
- PyGSD version `1.1.1`

**Data Condition:**

- Clean graph only
- No edge deletion
- No sign flips
- No direction flips
- No feature perturbations
- No architecture perturbations
- No hyperparameter search
- No model ranking or superiority claim

**Dataset:**

- One fixed synthetic signed-directed dataset
- One fixed train/validation/test split shared by both models

---

## 3. Dataset-Construction Requirement

**Critical Requirement:**

Do not invent an unverified PyGSD generator API.

Implementation must first inspect the installed PyGSD 1.1.1 source and verify the exact supported synthetic signed-directed generator or dataset-construction path.

**Required Dataset Properties:**

- Deterministic construction
- At least `200` nodes
- At least `2,000` directed signed edges before splitting
- Both positive and negative edges
- All four task classes represented in train, validation, and test
- Nonempty train, validation, and test query sets
- No held-out validation or test queried orientation in the training message-passing graph
- No zero edge weights
- Finite shared node features
- Practical CPU runtime

**Fixed Seeds:**

- Dataset-generation seed: `0`
- Split seed: `0`
- Validation probability: `0.15`
- Test probability: `0.15`

**Stop Rule:**

If the verified generator cannot reliably satisfy these properties, stop and revise the dataset design rather than silently changing the gate.

---

## 4. Shared Data Contract

**Required Immutable Shared Record:**

- Clean edge index and weights
- Training graph and weights
- Train queries and labels
- Validation queries and labels
- Test queries and labels
- One shared feature tensor
- Dataset-generation seed
- Split seed
- Full data fingerprint (SHA-256)
- Feature fingerprint (SHA-256)
- Class distributions
- Graph statistics

**Feature Construction:**

Compute the shared feature tensor exactly once from the training graph using the verified `in_out_degree` path.

**Feature Sharing:**

Both models must use the same tensor object or an explicitly fingerprint-verified equivalent.

**Fingerprint Verification:**

Full SHA-256 data and feature fingerprints must be recomputed inside each model run and matched against the original shared record.

---

## 5. Training Seeds

**Fixed Initialization Seeds:**

```text
0, 1, 2
```

**Scope:**

- Exactly three independent model initialization and training seeds
- Graph and split remain fixed across all seeds and both models
- Each model receives its own independently initialized optimizer for every seed

---

## 6. Model Configurations

**Provisionally Locked:**

Use these configurations unless implementation reveals an exact API incompatibility.

### MSGNN

```python
{
    "hidden": 16,
    "q": 0.25,
    "K": 2,
    "label_dim": 4,
    "activation": True,
    "trainable_q": False,
    "layer": 2,
    "dropout": 0.5,
    "normalization": "sym",
    "cached": False,
    "conv_bias": True,
    "absolute_degree": True,
}
```

**Additional Requirements:**

- Set `num_features` from the shared feature width
- Preserve signed training weights in every forward call

### Directed SSSNET

```python
{
    "hidden": 16,
    "nclass": 4,
    "dropout": 0.5,
    "hop": 2,
    "fill_value": 0.5,
    "directed": True,
    "bias": True,
}
```

**Additional Requirements:**

- Set `nfeat` from the shared feature width
- Construct positive and negative training-edge tensors only from the shared training graph and shared training weights

---

## 7. Optimization Protocol

**Optimizer Configuration:**

For each model and seed:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)
criterion = torch.nn.NLLLoss()
```

**Training Protocol:**

- Maximum epochs: `300`
- Evaluate validation loss after every epoch
- Early-stopping patience: `30` epochs
- Minimum validation-loss improvement: `1e-4`
- Save the state associated with the lowest finite validation NLL
- Restore the best validation checkpoint before test evaluation
- Never use test metrics for stopping, checkpoint selection, or tuning
- Complete at least `10` epochs unless a numerical, API, or data-contract failure occurs
- Do not require monotonic loss reduction

**Required Records:**

- Epoch `0` pre-training validation loss
- Best validation loss

---

## 8. Per-Epoch Validation

**Required Checks Every Epoch:**

- Output shape `[Q, 4]`
- Floating dtype
- CPU placement
- Finite output
- Exponentiated rows approximately sum to one
- Scalar finite loss
- Finite observed gradients
- Finite parameters after the optimizer step

**Failure Rule:**

A nonfinite value triggers an immediate stop for that run.

---

## 9. Recorded Evidence

**Record for Every Model and Seed:**

- Exact configuration
- Optimizer configuration
- Seed
- Graph and split fingerprints
- Initial training loss
- Initial validation loss
- Per-epoch training loss
- Per-epoch validation loss
- Best epoch
- Best validation loss
- Stopping epoch
- Stopping reason
- Total epochs completed
- Gradient finiteness verdict
- Parameter finiteness verdict
- Restored-checkpoint verdict
- Test accuracy
- Test macro-F1
- Test micro-F1
- Runtime
- Final run verdict

**Metric Interpretation:**

Accuracy and F1 remain execution and clean-baseline evidence. They must not be used to claim architecture superiority.

---

## 10. Minimal Learning-Signal Criterion

**Per-Model Requirement:**

For each model separately, require at least two of the three seeds to satisfy:

```text
best validation NLL <= initial validation NLL - 0.001
```

**Interpretation:**

This criterion represents only a minimal clean-training learning signal.

It does not establish:

- Convergence
- Useful representation learning
- Predictive quality
- Generalization

---

## 11. Gate Verdict

### PASS

Gate E passes only when:

- All six model-seed runs complete without data-contract, API, numerical, gradient, optimizer, or evaluation-path failure
- All fingerprints match the shared records
- All recorded outputs, losses, gradients, parameters, and metrics are finite
- Both models satisfy the minimal learning-signal criterion in at least two of three seeds
- Best checkpoints are restored and evaluated successfully

### PARTIAL

Gate E is partial when:

- All six runs execute safely and preserve the data contract
- But one or both models fail the minimal learning-signal criterion

**Requirement:**

A PARTIAL verdict requires diagnosis before perturbation experiments.

### FAIL

Gate E fails when any run has:

- A data-contract mismatch
- An unresolved model API failure
- Nonfinite output, loss, gradient, parameter, or metric
- No parameter update
- Invalid checkpoint restoration
- Test leakage
- An incomplete required run

---

## 12. Failure Categories

**Preserved Categories:**

- `data-contract failure`
- `preprocessing failure`
- `model-API failure`
- `numerical failure`
- `gradient failure`
- `optimizer-step failure`
- `checkpoint failure`
- `evaluation-path failure`
- `implementation defect`
- `unresolved package behavior`

**Required Failure Record:**

Every failure record must contain:

- Model
- Seed
- Epoch
- Category
- Exact failed stage
- Exception type
- Exception message
- Full traceback
- Relevant configuration
- Data and feature fingerprints

---

## 13. Stop Rules

**Immediate Stop Required When:**

- The synthetic generator API is unverified
- Required graph properties are not satisfied
- Any held-out query leaks into the training graph
- Fingerprints differ between models or seeds
- An output, loss, gradient, parameter, or metric is nonfinite
- A model uses a different feature tensor or split
- Test data influences training decisions
- An undocumented fallback or task change would be required

**Critical Rule:**

Do not silently repair, replace, or broaden the experiment.

---

## 14. Authorized Artifacts

**A Later Implementation May Create:**

- One clean-baseline execution script
- One machine-readable configuration file
- One per-epoch metrics file per run
- One aggregate summary file
- One human-readable Gate E results note

**Prohibition:**

Do not create those artifacts in this specification work unit.

---

## 15. Claims Boundary

### Permitted After a PASS

A Gate E PASS may support only:

- Clean multi-epoch CPU training compatibility
- Controlled three-seed execution
- Finite optimization behavior
- A minimal validation-loss learning signal
- Successful best-checkpoint restoration
- Successful held-out clean-test execution
- Readiness to design a limited perturbation pilot

### Not Permitted

A Gate E PASS does not establish:

- Robustness
- Stability
- Perturbation tolerance
- Causal explanations
- Architecture superiority
- Statistical significance
- Generalization to other datasets
- Production readiness
- GPU compatibility
- Large-scale performance

---

## 16. Controlled Implementation Sequence

**Future Work Units:**

1. **E1 — Synthetic-data API and construction audit**
2. **E2 — Clean-baseline implementation**
3. **E3 — Syntax and single-seed dry run**
4. **E4 — Three-seed clean-baseline execution**
5. **E5 — Results record and gate decision**

**Requirement:**

Each work unit must be separately reviewed, validated, and committed.

---

## 17. Current Status

**Gate D:** PASS

**Gate E specification:** PROVISIONALLY LOCKED

**Gate E execution:** NOT STARTED

**Perturbation experiments:** NOT AUTHORIZED

---

*Specification completed: 2026-07-18*  
*Gate D: PASS*  
*Gate E specification: PROVISIONALLY LOCKED*  
*Gate E execution: NOT STARTED*  
*Perturbation experiments: NOT AUTHORIZED*
