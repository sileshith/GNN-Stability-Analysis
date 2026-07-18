---
title: "Primary Experiment Scope Decision"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Experiment scope definition"
status: "Provisionally locked"
decision_date: "2026-07-18"
---

# Primary Experiment Scope Decision

**Decision Type:** Researcher-adopted, advisor-aligned provisional decision

**Decision Date:** 2026-07-18

---

## Decision

The primary matched empirical comparison will initially use:

- **MSGNN_link_prediction**
- **SSSNET_link_prediction** with `directed=True`

Both models will be evaluated on the official PyGSD 1.1.1 `four_class_signed_digraph` task.

**MagNet_link_prediction** will remain in the project as a direction-only reference architecture, but it will not be included in the primary signed-directed matched comparison until its treatment of signed edge weights and its scientific validity on signed tasks are explicitly established.

---

## Rationale

### 1. Official Common Training Path

PyGSD 1.1.1 provides an official common training path for MSGNN and directed SSSNET using:

- The same `link_class_split`
- The same training graph
- The same query-edge splits
- The same four-class labels
- `nn.NLLLoss()`
- Accuracy, macro-F1, and micro-F1

### 2. Direct Task Alignment

The four classes directly encode:

- Class 0: Positive edge in queried direction
- Class 1: Negative edge in queried direction
- Class 2: Positive edge in reverse direction
- Class 3: Negative edge in reverse direction

### 3. Perturbation Alignment

This task aligns directly with the project's central perturbations:

- Sign flips
- Direction flips
- Edge deletions applied to the message-passing graph while preserving a fixed clean evaluation set

### 4. Four-Class vs. Five-Class

The four-class task is preferred over the five-class task for the first controlled experiment because it isolates sign and direction semantics without initially adding:

- A no-edge class
- Additional class imbalance
- Existence-prediction complexity

### 5. MagNet Comparability Limitation

MagNet's official PyGSD example establishes only two-class direction prediction on directed data. Forcing it into the signed four-class task before verifying signed-weight semantics would create unsupported three-model comparability.

### 6. Scope Narrowing, Not Removal

This decision narrows the first experiment rather than removing MagNet from the overall project.

---

## Role of MagNet

**Current Status:**

- MagNet remains relevant as a directed-graph architecture
- MagNet is not included in the primary signed-directed matched comparison

**Future Consideration:**

A later secondary experiment may compare MSGNN, directed SSSNET, and MagNet on a two-class direction task.

**Conditional Requirements for Secondary Experiment:**

That secondary experiment is conditional on verifying:

- Scientifically valid MSGNN and SSSNET label remapping
- Whether signed weights should be preserved, converted to absolute weights, or excluded
- Whether such preprocessing changes the research question
- Whether MagNet recomputes graph-dependent operators correctly after perturbations

---

## Primary Experiment Boundary

The first matched experiment will use:

- One synthetic signed-directed dataset
- One fixed clean train/validation/test split
- MSGNN and directed SSSNET only
- The four-class signed-directed task
- Identical query-edge labels and evaluation metrics
- No robustness perturbations until both models pass a minimal training-compatibility gate
- CPU-scale execution first
- Deterministic seeds and complete experiment records

**Not Yet Specified:**

- Final dataset parameters
- Hyperparameter sweeps
- Production seed counts
- Perturbation budgets

---

## Next Controlled Gate

### Gate D — Minimal Training Compatibility

**Purpose:**

Verify that MSGNN and directed SSSNET can each complete a small end-to-end training run on the same tiny signed-directed dataset and four-class split.

**Minimum Checks:**

- Loss is finite before and after optimization
- Backward pass completes
- Gradients are finite
- At least one optimizer step changes trainable parameters
- Output shape matches `[number_of_query_edges, 4]`
- Labels are integer class indices in `[0, 3]`
- Evaluation metrics execute
- The same clean split is used for both models
- Graph-dependent preprocessing is reconstructed separately for each model

**Prohibited Claims:**

No claim of convergence, accuracy superiority, robustness, or stability is permitted.

**Gate Status:**

- Gate B remains PARTIAL
- Gate C remains narrowly PASS
- This decision authorizes specification of Gate D but not production experimentation

---

## Decision Status

- **Status:** Provisionally locked
- **Primary models:** MSGNN and directed SSSNET
- **Primary task:** Four-class signed-directed link prediction
- **MagNet role:** Direction-only reference pending further validation
- **Next action:** Write the Gate D training-compatibility specification

---

*Decision recorded: 2026-07-18*  
*Scope: Primary matched experiment*  
*Gate B: PARTIAL*  
*Gate C: PASS (forward-pass smoke test only)*  
*Gate D: Specification pending*
