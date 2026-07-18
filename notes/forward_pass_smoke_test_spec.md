---
title: "PyGSD CPU Forward-Pass Smoke-Test Specification"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate C test design specification"
status: "specification only — not executed"
specification_date: "2026-07-18"
---

# PyGSD CPU Forward-Pass Smoke-Test Specification

## 1. Status and Scope

**Specification Date:** 2026-07-18

**Current Evidence:** Level 0–1 constructor/API evidence

**Gate Status:**
- **Gate B (Model API/Task Audit):** PARTIAL
- **Gate C (Finite CPU Forward Pass):** PENDING

**Target Environment:**
- Interpreter: `/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/bin/python`
- Device: CPU only
- Installed distribution: `torch-geometric-signed-directed 1.1.1`

**Scope Limitations:**
- No dataset
- No training
- No optimizer
- No backward pass
- No GPU
- No perturbation experiment
- No robustness, stability, accuracy, or convergence claim

**This specification does not itself pass Gate C.** Gate C passage requires implementation and successful execution of this specification.

---

## 2. Scientific Question

**Narrow Question:**

Can each installed model complete one deterministic CPU forward pass on a minimal architecture-compatible synthetic graph and produce a finite, correctly shaped, normalized matrix of per-query-edge class log-probabilities?

**What This Test Addresses:**

This test addresses executable API compatibility only.

**What This Test Does Not Establish:**

- Correct scientific task labels
- Meaningful predictive performance
- Model comparability
- Robustness
- Perturbation correctness
- Mathematical stability

---

## 3. Common Test Controls

**Fixed Parameters:**

- Seed: `0`
- `torch.manual_seed(0)`
- Device: CPU explicitly selected
- Float tensors: `torch.float32`
- Index tensors: `torch.long`
- `num_nodes = 4`
- `num_features = 3`
- `hidden = 2`
- `num_classes = 2`
- `num_query_edges = 3`
- `dropout = 0.0`
- `model.eval()`
- `torch.no_grad()`

**Graph Construction:**

- No random graph generation
- Tensors written explicitly in the future test script
- No mutation of input tensors between model calls
- One independently constructed model per test

**Cache Settings:**

- MSGNN and MagNet use `cached=False`

**Query Edges:**

Expected shape: `[3, 2]`

```python
query_edges = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 3]
], dtype=torch.long)
```

`query_edges` must be `torch.long` and contiguous.

---

## 4. MSGNN Test Design

**Constructor Settings:**

Use constructor settings consistent with the existing constructor smoke test unless the audit requires otherwise.

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

**Input Tensors:**

**`real`:**
- `torch.float32`
- Shape: `[4, 3]`
- Explicit nonconstant values

Example:
```python
real = torch.tensor([
    [1.0, 0.5, 0.2],
    [0.8, 1.0, 0.3],
    [0.6, 0.4, 1.0],
    [0.9, 0.7, 0.5]
], dtype=torch.float32)
```

**`imag`:**
- `torch.float32`
- Shape: `[4, 3]`
- Initialize to zeros for the first smoke test

```python
imag = torch.zeros(4, 3, dtype=torch.float32)
```

**`edge_index`:**
- `torch.long`
- Shape: `[2, 6]`
- Explicit directed edges
- Use a small connected directed cycle with two additional directed edges
- No node index outside 0 through 3

Example:
```python
edge_index = torch.tensor([
    [0, 1, 2, 3, 0, 1],
    [1, 2, 3, 0, 2, 3]
], dtype=torch.long)
```

**`edge_weight`:**
- Primary planned dtype: `torch.float32`
- Shape: `[6]`
- All values `1.0`

```python
edge_weight = torch.ones(6, dtype=torch.float32)
```

**Edge Weight Dtype Conflict:**

- The installed signature annotates `edge_weight` as `Optional[torch.LongTensor]`
- The docstring describes `torch.FloatTensor`
- The primary test uses `float32` because it represents numerical edge weights
- Runtime acceptance remains to be determined
- A dtype failure must be recorded, not silently corrected
- Omission of `edge_weight` may be tested only in a separate diagnostic retry after preserving the primary failure

**Expected Output:**

- A `torch.Tensor`
- Shape: `[3, 2]`
- One row per query edge
- Class log-probabilities

**Limitations:**

MSGNN first-layer cache behavior remains unresolved even with `cached=False` and is not tested in this first forward-pass unit.

---

## 5. MagNet Test Design

**Constructor Settings:**

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

**Input Tensors:**

Use the same:
- `real` tensor
- `imag` tensor
- `edge_index`
- `query_edges`
- Primary `float32` `edge_weight`

as the MSGNN test, while constructing a separate MagNet model instance.

**Expected Output:**

- A `torch.Tensor`
- Shape: `[3, 2]`
- One row per query edge
- Class log-probabilities

**Preserved Limitations:**

- Signed-edge support is not being tested
- Negative edge weights must not be introduced
- Changed-graph cache recomputation is not being tested
- Scientific equivalence with MSGNN is not implied

---

## 6. SSSNET Test Design

**Constructor Settings:**

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

**Input Tensors:**

**`features`:**
- `torch.float32`
- Shape: `[4, 3]`
- Use the same explicit numerical values as MSGNN `real` inputs

```python
features = torch.tensor([
    [1.0, 0.5, 0.2],
    [0.8, 1.0, 0.3],
    [0.6, 0.4, 1.0],
    [0.9, 0.7, 0.5]
], dtype=torch.float32)
```

**Positive Structure:**

**`edge_index_p`:**
- `torch.long`
- Shape: `[2, 4]`
- Each undirected pair appears in both directions
- All node indices are between 0 and 3

This represents the positive undirected pairs: {0, 1} and {2, 3}

```python
edge_index_p = torch.tensor([
    [0, 1, 2, 3],
    [1, 0, 3, 2]
], dtype=torch.long)
```

**`edge_weight_p`:**
- `torch.float32`
- Shape: `[4]`
- All values `1.0`
- One weight corresponds to each directed representation entry

```python
edge_weight_p = torch.ones(4, dtype=torch.float32)
```

**Negative Structure:**

**`edge_index_n`:**
- `torch.long`
- Shape: `[2, 4]`
- Each undirected pair appears in both directions
- No positive source-target pair is duplicated in the negative structure
- All node indices are between 0 and 3

This represents the negative undirected pairs: {1, 2} and {0, 3}

```python
edge_index_n = torch.tensor([
    [1, 2, 0, 3],
    [2, 1, 3, 0]
], dtype=torch.long)
```

**`edge_weight_n`:**
- `torch.float32`
- Shape: `[4]`
- All values are positive magnitudes of `1.0`
- Sign is represented by separation into the negative adjacency input

```python
edge_weight_n = torch.ones(4, dtype=torch.float32)
```

**`query_edges`:**

Use the same `query_edges` tensor with shape `[3, 2]`.

```python
query_edges = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 3]
], dtype=torch.long)
```

**SSSNET Graph Interpretation:**

- Although `edge_index` tensors store ordered source-target entries, the paired reverse entries encode an undirected signed graph for `directed=False`
- The combined positive and negative structure covers all four nodes
- The smoke test does not determine whether this is the package-author's preferred benchmark preprocessing
- It provides a minimal mode-compatible synthetic graph for executable API validation only

**Expected Output:**

- The visible source predicts one `torch.Tensor`, not a four-element tuple
- Expected shape: `[3, 2]`
- One row per query edge
- Class log-probabilities

**Limitations:**

- This first test uses `directed=False` only
- `directed=True` requires a later independent test
- `SIMPA` internals and caching remain unresolved
- SSSNET must not be treated as SGCN

---

## 7. Required Runtime Assertions

The future script must test each model independently and assert:

1. The returned object is a `torch.Tensor`
2. `output.ndim == 2`
3. `output.shape == (3, 2)`
4. `output.dtype` is floating point
5. Every output value is finite
6. No NaN values
7. No positive infinity or negative infinity
8. `torch.exp(output).sum(dim=1)` is approximately ones
9. Normalization tolerance:
   - `atol = 1e-5`
   - `rtol = 1e-5`
10. Repeated evaluation of the same model on the same unchanged inputs produces matching output within:
    - `atol = 1e-6`
    - `rtol = 1e-6`

**Clarification:**

Deterministic repeatability is checked only for an unchanged graph in `model.eval()` mode. It does not verify changed-graph cache recomputation.

---

## 8. Failure-Handling Policy

The future script must:

- Test models independently
- Preserve the full exception type and message
- Identify which model failed
- Report the failed test stage
- Report input names, shapes, and dtypes
- Never fabricate a passing result
- Never automatically alter tensors without recording the original failure
- Never suppress warnings without review
- Continue to the next model after recording a model-specific failure when technically safe
- Exit with nonzero status if any required model fails

**For MSGNN and MagNet Edge Weight Dtype Conflicts:**

- Preserve the primary `float32` failure
- Any `edge_weight=None` diagnostic retry must be labeled diagnostic
- A diagnostic retry cannot erase or overwrite the primary result

---

## 9. Evidence Record Produced by the Future Test

**Minimum Console Record:**

- Timestamp
- Python version
- PyTorch version
- PyG version
- PyGSD distribution version
- Device
- Seed
- Model name
- Constructor configuration
- Input tensor shapes
- Input tensor dtypes
- Output object type
- Actual output shape
- Output dtype
- Finiteness result
- Normalization result
- Deterministic-repeat result
- PASS or FAIL
- Exception type and message when applicable

**Do not print full tensors unless needed for diagnosing a failure.**

---

## 10. Gate C Decision Rule

**Model-Level PASS:**

A model passes only if one primary, non-diagnostic CPU forward pass:

- Completes without exception
- Returns a tensor
- Has shape `[3, 2]`
- Contains only finite values
- Satisfies probability normalization
- Satisfies unchanged-input repeatability

**Overall Gate C Status:**

- Gate C remains PENDING until the future script is implemented and executed
- After execution, report per-model results separately
- Overall Gate C may pass only when all three selected models pass their primary required test
- A diagnostic fallback does not automatically count as a primary pass
- Any failure must be preserved as evidence and investigated before broad experiments

---

## 11. Explicitly Deferred Tests

The following are explicitly deferred to future work units:

- Training
- Backward pass
- Gradient checks
- Optimizer compatibility
- Loss-function execution
- Target-label encoding
- `directed=True` SSSNET
- Signed-edge MagNet behavior
- Changed-graph cache recomputation
- Edge deletion
- Sign flips
- Direction flips
- Datasets
- Multiple seeds
- GPU tests
- Performance benchmarking
- Accuracy
- Robustness
- Stability bounds

---

## 12. Next Implementation Unit

After this specification is reviewed and committed, the next controlled unit will be:

**Create one new script:**

`scripts/pygsd_forward_pass_smoke_test.py`

The script must implement this specification exactly and must not modify:

`scripts/pygsd_model_smoke_test.py`

**Do not create the script now.**

---

## 13. Final Specification Conclusion

**Summary:**

- The test is intentionally minimal
- It tests executable forward compatibility, not scientific validity
- Architecture-specific inputs are preserved
- Expected output shapes are source-established
- Actual output shapes remain dynamically unverified
- Gate B remains PARTIAL
- Gate C remains PENDING
- No experimental gate is passed by writing this specification

**Current Status:**

- This document defines the test design
- Implementation and execution are required before Gate C can be evaluated
- All negative findings must be preserved and investigated
- Success in this test does not establish model validity, only forward-pass compatibility

---

*Specification completed: 2026-07-18*  
*Gate B status: PARTIAL*  
*Gate C status: PENDING*  
*Implementation: not yet created*
