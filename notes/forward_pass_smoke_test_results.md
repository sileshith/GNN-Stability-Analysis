---
title: "PyGSD CPU Forward-Pass Smoke-Test Results"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate C execution record"
status: "PASS within defined scope"
execution_date: "2026-07-18"
evidence_level: "2 (single functional run with validation)"
---

# PyGSD CPU Forward-Pass Smoke-Test Results

## 1. Execution Status and Scope

**Execution Timestamp:** 2026-07-18T13:28:20.031503+00:00

**Interpreter Environment:**
- Python: 3.11.15
- PyTorch: 2.12.0+cu130
- PyG: 2.7.0
- PyGSD: 1.1.1
- Device: CPU
- Seed: 0

**Script:** `scripts/pygsd_forward_pass_smoke_test.py`

**Execution Confirmation:**

The script was actually executed on the SOL computational environment.

**Scope Limitations:**

- No dataset was loaded
- No training occurred
- No optimizer or backward pass was used
- No GPU was used
- No perturbation was applied
- No robustness, accuracy, convergence, or stability claim was tested

---

## 2. Gate Question

**Narrow Gate C Question:**

Can each installed architecture complete deterministic CPU forward execution on its approved minimal architecture-compatible synthetic graph and return a finite, correctly shaped, normalized matrix of per-query-edge class log-probabilities?

**Answer:**

The observed execution answered this narrow question affirmatively for all three tested configurations.

---

## 3. Common Runtime Controls

**Fixed Parameters:**

- `num_nodes = 4`
- `num_features = 3`
- `hidden = 2`
- `num_classes = 2`
- `num_query_edges = 3`
- `dropout = 0.0`
- `model.eval()`
- `torch.no_grad()`

**Tensor Dtypes:**

- Float inputs and weights: `torch.float32`
- Index tensors: `torch.int64` / `torch.long`

**Tensor Shapes:**

- `query_edges` shape: `[3, 2]`
- Expected output shape: `[3, 2]`

**Repeatability:**

Unchanged-input repeatability was tested with a second forward call on the same model with identical inputs.

**Clarification:**

Unchanged-input repeatability does not verify changed-graph cache recomputation.

**Cache Settings:**

MSGNN and MagNet used `cached=False`.

---

## 4. MSGNN Runtime Result

### Tested Configuration

```python
MSGNN_link_prediction(
    num_features=3,
    hidden=2,
    q=0.25,
    K=2,
    label_dim=2,
    activation=True,
    trainable_q=False,
    layer=2,
    dropout=0.0,
    normalization="sym",
    cached=False,
    conv_bias=True,
    absolute_degree=True
)
```

### Actual Input Metadata

- `real`: shape `[4, 3]`, `torch.float32`, CPU
- `imag`: shape `[4, 3]`, `torch.float32`, CPU
- `edge_index`: shape `[2, 6]`, `torch.int64`, CPU, contiguous
- `edge_weight`: shape `[6]`, `torch.float32`, CPU
- `query_edges`: shape `[3, 2]`, `torch.int64`, CPU, contiguous

### Actual Result

- **Returned object:** Tensor
- **Actual output shape:** `[3, 2]`
- **Actual output dtype:** `torch.float32`
- **Finite:** PASS
- **NaN-free:** PASS
- **Positive-infinity-free:** PASS
- **Negative-infinity-free:** PASS
- **Probability normalization:** PASS
- **Probability row sums:** `[1.0, 1.0, 1.0]`
- **Unchanged-input repeatability:** PASS
- **Final model status:** PASS

---

## 5. MagNet Runtime Result

### Tested Configuration

```python
MagNet_link_prediction(
    num_features=3,
    hidden=2,
    q=0.25,
    K=1,
    label_dim=2,
    activation=True,
    trainable_q=False,
    layer=2,
    dropout=0.0,
    normalization="sym",
    cached=False
)
```

### Actual Input Metadata

- `real`: shape `[4, 3]`, `torch.float32`, CPU
- `imag`: shape `[4, 3]`, `torch.float32`, CPU
- `edge_index`: shape `[2, 6]`, `torch.int64`, CPU, contiguous
- `edge_weight`: shape `[6]`, `torch.float32`, CPU
- `query_edges`: shape `[3, 2]`, `torch.int64`, CPU, contiguous

### Actual Result

- **Returned object:** Tensor
- **Actual output shape:** `[3, 2]`
- **Actual output dtype:** `torch.float32`
- **Finite:** PASS
- **NaN-free:** PASS
- **Positive-infinity-free:** PASS
- **Negative-infinity-free:** PASS
- **Probability normalization:** PASS
- **Probability row sums:** `[1.0, 1.0, 1.0]`
- **Unchanged-input repeatability:** PASS
- **Final model status:** PASS

**Limitation:**

This test did not evaluate signed-edge support.

---

## 6. SSSNET Runtime Result

### Tested Configuration

```python
SSSNET_link_prediction(
    nfeat=3,
    hidden=2,
    nclass=2,
    dropout=0.0,
    hop=2,
    fill_value=1.0,
    directed=False,
    bias=True
)
```

### Actual Input Metadata

- `features`: shape `[4, 3]`, `torch.float32`, CPU
- `edge_index_p`: shape `[2, 4]`, `torch.int64`, CPU, contiguous
- `edge_weight_p`: shape `[4]`, `torch.float32`, CPU
- `edge_index_n`: shape `[2, 4]`, `torch.int64`, CPU, contiguous
- `edge_weight_n`: shape `[4]`, `torch.float32`, CPU
- `query_edges`: shape `[3, 2]`, `torch.int64`, CPU, contiguous

### Actual Result

- **Returned object:** Tensor
- **Actual output shape:** `[3, 2]`
- **Actual output dtype:** `torch.float32`
- **Finite:** PASS
- **NaN-free:** PASS
- **Positive-infinity-free:** PASS
- **Negative-infinity-free:** PASS
- **Probability normalization:** PASS
- **Probability row sums:** `[1.0, 0.9999999403953552, 1.0]`
- **Unchanged-input repeatability:** PASS
- **Final model status:** PASS

**Floating-Point Normalization:**

The value `0.9999999403953552` represents ordinary floating-point rounding and passed the predefined normalization tolerance (`atol=1e-5`, `rtol=1e-5`).

**Limitations:**

- Only `directed=False` was tested
- `directed=True` remains untested
- `SIMPA` cache behavior remains unresolved
- SSSNET is not being treated as SGCN

---

## 7. Consolidated Result

| Model | Primary Forward Pass | Actual Output Shape | Finite | Normalized | Unchanged-Input Repeatable | Final Status |
|-------|---------------------|---------------------|--------|------------|---------------------------|--------------|
| **MSGNN** | PASS | `[3, 2]` | PASS | PASS | PASS | PASS |
| **MagNet** | PASS | `[3, 2]` | PASS | PASS | PASS | PASS |
| **SSSNET** | PASS | `[3, 2]` | PASS | PASS | PASS | PASS |

**Gate C Evaluation Result:** PASS

**Precise Definition:**

All three selected installed model classes completed their approved primary CPU forward-pass tests and satisfied the required structural and numerical assertions.

---

## 8. Claims Now Supported

The following runtime claims are now supported by this execution:

- All three tested classes completed CPU forward execution with the specified synthetic inputs
- All returned one `torch.Tensor`
- All actual outputs had shape `[3, 2]`
- All actual outputs used `torch.float32`
- All outputs were finite and free of NaN and infinity
- Exponentiated log-probability rows normalized to approximately one
- All were repeatable on unchanged inputs in evaluation mode
- `float32` `edge_weight` was accepted by MSGNN and MagNet in these tested calls
- The visible SSSNET implementation returned one tensor despite its conflicting tuple annotation

**Scope Limitation:**

These claims apply only to the exact tested configurations and inputs.

---

## 9. Claims Still Prohibited

The following claims are **not** supported by this execution:

- Training correctness
- Loss compatibility
- Target-label correctness
- Gradient correctness
- Optimizer compatibility
- Accuracy
- Dataset performance
- Robustness
- Structural perturbation correctness
- Changed-graph cache recomputation
- Mathematical stability
- Convergence
- Architecture superiority
- Common scientific task comparability
- MagNet signed-edge validity
- `directed=True` SSSNET compatibility
- GPU compatibility
- Production-scale suitability

---

## 10. Gate Interpretation

**Gate C:** PASS within the defined minimal CPU forward-pass compatibility test

**Gate B:** Remains PARTIAL because task semantics, label meanings, intended loss, and cross-model task comparability remain unresolved

**Critical Clarification:**

Gate C passing does not automatically resolve Gate B.

**Generalization Limitation:**

Do not claim that all future Gate C-like configurations will pass. This result applies only to the tested configurations.

---

## 11. Remaining Technical Risks

The following technical risks remain unresolved:

- MSGNN first-layer `MSConv` cache behavior remains unresolved
- Changed-graph recomputation remains dynamically unverified
- MagNet signed-edge behavior remains unresolved
- SSSNET `SIMPA` internals remain uninspected
- SSSNET `directed=True` remains untested
- Exact model-specific label meanings remain unresolved
- Intended package losses remain unresolved
- Forward success does not establish trainability or scientific validity

---

## 12. Next Controlled Work Unit

**Recommendation:**

Perform a read-only package-example and training-path audit to determine, for each architecture:

- Intended prediction task
- Class-label meanings
- Target encoding
- Package-author loss function
- Expected train/validation/test split representation
- Architecture-specific preprocessing
- Whether a scientifically comparable common task can be defined

**Critical Requirement:**

No training script should be written until this task/loss audit is complete.

---

## 13. Final Evidence Conclusion

**Summary:**

- The Gate C smoke test was successfully executed
- MSGNN, MagNet, and SSSNET each passed their primary required test
- The tested runtime output shapes matched the source-established expectations
- Gate C is PASS within its narrow defined scope
- Gate B remains PARTIAL
- No robustness or stability experiment has occurred
- Negative findings and unresolved questions remain preserved

**Evidence Classification:**

This execution provides Evidence Level 2 (single functional run with validation). It does not yet constitute multi-seed repetition, controlled perturbation, or theory-aligned validation.

**Preserved Limitations:**

All limitations documented in the specification, audit, and methodology remain in effect. This test establishes forward-pass compatibility only.

---

*Execution completed: 2026-07-18*  
*Gate C status: PASS (within defined scope)*  
*Gate B status: PARTIAL*  
*Evidence level: 2*
