---
title: "PyGSD Model API and Task Audit"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate B source-inspection evidence"
status: "PARTIAL"
evidence_level: "0–1 (constructor/API inspection)"
audit_date: "2026-07-18"
---

# PyGSD Model API and Task Audit

## 1. Audit Status and Scope

**Audit Date:** 2026-07-18

**Repository Context:**  
This audit supports Gate B (Model API/Task Audit) in the signed and directed GNN robustness project. It documents the installed PyGSD 1.1.1 link prediction architectures through source inspection only.

**Interpreter:**  
`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/bin/python`

**Installed Distribution:**  
`torch-geometric-signed-directed 1.1.1` — **VERIFIED FROM INSPECTION**

**Inspection Method:**  
Python `importlib.metadata` and `inspect` module introspection of installed package source code.

**Scope Limitations:**  
- No model was instantiated during this source-inspection unit
- No forward pass was executed
- No training step was performed
- No dataset operation was performed
- No experiment was conducted
- No runtime behavior was observed

**Current Evidence:**  
Level 0–1 constructor/API evidence from source inspection.

**Gate Status:**  
- **Gate B (Model API/Task Audit):** PARTIAL
- **Gate C (Finite CPU Forward Pass):** PENDING

**Evidence Level Clarification:**  
"Evidence Level 0.5" is not an official project level and must not be used. This audit provides Level 0–1 evidence: package version verification, constructor signature verification, and forward-pass signature verification from installed source code.

---

## 2. Critical Corrections to the Preliminary Audit

This section records corrections to any preliminary understanding based on verified source inspection:

1. **MSGNN forward inputs are:**  
   `real`, `imag`, `edge_index`, `query_edges`, optional `edge_weight`

2. **MagNet uses the same forward argument structure:**  
   `real`, `imag`, `edge_index`, `query_edges`, optional `edge_weight`

3. **SSSNET uses separate positive and negative graph tensors:**  
   `edge_index_p`, `edge_weight_p`, `edge_index_n`, `edge_weight_n`, `features`, `query_edges`

4. **Query-edge tensor layout:**  
   The source indexes query edges with `query_edges[:, 0]` and `query_edges[:, 1]`.  
   Therefore the source-established expected layout is: `[num_query_edges, 2]`  
   Not: `[2, num_query_edges]`

5. **All three implementations apply:**  
   `F.log_softmax(..., dim=1)`

6. **Returned values are:**  
   Class log-probabilities for queried edges, not raw logits and not outputs for all nodes.

7. **Source-established expected output shapes:**
   - **MSGNN:** `[num_query_edges, label_dim]`
   - **MagNet:** `[num_query_edges, label_dim]`
   - **SSSNET:** `[num_query_edges, nclass]`

8. **Actual successful runtime output shape, output finiteness, and forward execution remain unverified.**

9. **NLLLoss is the strongest direct loss hypothesis** from the inspected output semantics, but exact intended loss and label encoding remain unresolved.

10. **The three "link_prediction" class names do not by themselves establish a common scientific task.** Link existence prediction, sign prediction, direction prediction, and multi-class relation prediction are distinct task definitions.

---

## 3. MSGNN_link_prediction

**Installed Source Path:** **VERIFIED FROM INSPECTION**  
`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/general/MSGNN.py`

### Constructor Signature

**VERIFIED FROM SIGNATURE:**

```python
MSGNN_link_prediction(
    num_features: int,
    hidden: int = 2,
    q: float = 0.25,
    K: int = 2,
    label_dim: int = 2,
    activation: bool = True,
    trainable_q: bool = False,
    layer: int = 2,
    dropout: float = 0.5,
    normalization: str = "sym",
    cached: bool = False,
    conv_bias: bool = True,
    absolute_degree: bool = True
)
```

### Forward Signature

**VERIFIED FROM SIGNATURE:**

```python
forward(
    real: torch.FloatTensor,
    imag: torch.FloatTensor,
    edge_index: torch.LongTensor,
    query_edges: torch.LongTensor,
    edge_weight: Optional[torch.LongTensor] = None
) -> torch.FloatTensor
```

### Architecture Behavior

**VERIFIED FROM SOURCE:**

- `real` and `imag` are separate node-representation components
- `MSConv` repeatedly transforms the real and imaginary components
- `complex_relu` is applied when `activation=True`
- Query-edge embeddings concatenate:
  - `real[query_edges[:, 0]]` (source real)
  - `real[query_edges[:, 1]]` (target real)
  - `imag[query_edges[:, 0]]` (source imag)
  - `imag[query_edges[:, 1]]` (target imag)
- The pre-classification query-edge representation has expected width `hidden*4`
- `self.z` stores a clone of that pre-classification representation
- The linear layer maps `hidden*4` to `label_dim`
- Dropout is controlled by `self.training`
- Output applies `F.log_softmax(x, dim=1)`

**Source-established expected output shape:** `[num_query_edges, label_dim]`

**VERIFIED FROM SOURCE:**  
Returned rows correspond to query edges, not all nodes.

**VERIFIED FROM DOCSTRING:**  
- `q` is documented in the range `0 <= q <= 0.25`
- The class docstring identifies the architecture as an MSGNN link-prediction model based on a magnetic signed Laplacian

### Documentation Inconsistencies

**VERIFIED FROM INSPECTION:**

1. **Forward docstring incorrectly calls it a "MagNet node classification model"**  
   The docstring states: "Making a forward pass of the MagNet node classification model."  
   This is incorrect. The class is `MSGNN_link_prediction`.

2. **Forward docstring incorrectly says output corresponds to all nodes**  
   The docstring states: "Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes)."  
   The source returns `[num_query_edges, label_dim]`, not all nodes.

3. **Edge weight dtype annotation inconsistency**  
   - Signature annotates `edge_weight` as `Optional[torch.LongTensor]`
   - Docstring describes `edge_weight` as `torch.FloatTensor`
   - Actual runtime dtype acceptance remains **UNRESOLVED**

### MSGNN Cache Behavior

**VERIFIED FROM SIGNATURE:**  
MSGNN exposes `cached` with default `False`.

**VERIFIED FROM SOURCE:**  
The first `MSConv` constructor shown in `MSGNN.__init__` does not explicitly receive `cached`:

```python
chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
    q=q, trainable_q=trainable_q, normalization=normalization, bias=conv_bias))
```

**VERIFIED FROM SOURCE:**  
Subsequent `MSConv` layers created in the loop receive `cached=cached`:

```python
for _ in range(1, layer):
    chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
        q=q, trainable_q=trainable_q, normalization=normalization, \
            bias=conv_bias, cached=cached, absolute_degree=absolute_degree))
```

**UNRESOLVED:**  
- First-layer effective cache behavior depends on the uninspected `MSConv` default and implementation
- Inspection of `MSGNN_link_prediction` alone does not establish that `cached=False` uniformly forces every layer to recompute graph-dependent normalization

### Experimental Cache Policy

Instantiate with `cached=False` and dynamically verify changed-graph recomputation before perturbation experiments.

**MSGNN cache behavior is not fully verified.**

---

## 4. MagNet_link_prediction

**Installed Source Path:** **VERIFIED FROM INSPECTION**  
`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/directed/MagNet_link_prediction.py`

### Constructor Signature

**VERIFIED FROM SIGNATURE:**

```python
MagNet_link_prediction(
    num_features: int,
    hidden: int = 2,
    q: float = 0.25,
    K: int = 1,
    label_dim: int = 2,
    activation: bool = True,
    trainable_q: bool = False,
    layer: int = 2,
    dropout: float = 0.5,
    normalization: str = "sym",
    cached: bool = False
)
```

### Forward Signature

**VERIFIED FROM SIGNATURE:**

```python
forward(
    real: torch.FloatTensor,
    imag: torch.FloatTensor,
    edge_index: torch.LongTensor,
    query_edges: torch.LongTensor,
    edge_weight: Optional[torch.LongTensor] = None
) -> torch.FloatTensor
```

### Architecture Behavior

**VERIFIED FROM SOURCE:**

- `real` and `imag` are separate node-representation components
- `MagNetConv` repeatedly transforms both components
- `complex_relu` is applied when `activation=True`
- Query-edge embeddings concatenate source and target real and imaginary components:
  - `real[query_edges[:, 0]]`
  - `real[query_edges[:, 1]]`
  - `imag[query_edges[:, 0]]`
  - `imag[query_edges[:, 1]]`
- The linear layer maps `hidden*4` to `label_dim`
- Dropout is controlled by `self.training`
- Output applies `F.log_softmax(x, dim=1)`

**Source-established expected output shape:** `[num_query_edges, label_dim]`

**VERIFIED FROM SOURCE:**  
Returned rows correspond to query edges, not all nodes.

**VERIFIED FROM DOCSTRING:**  
The class is documented as a directed-graph link-prediction architecture.

### Documentation Inconsistencies

**VERIFIED FROM INSPECTION:**

1. **Forward docstring incorrectly calls it a node-classification model**  
   The docstring states: "Making a forward pass of the MagNet node classification model."  
   The class is `MagNet_link_prediction`.

2. **Forward docstring incorrectly says output corresponds to all nodes**  
   The docstring states: "Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes)."  
   The source returns `[num_query_edges, label_dim]`, not all nodes.

3. **Edge weight dtype annotation inconsistency**  
   - Signature annotates `edge_weight` as `Optional[torch.LongTensor]`
   - Docstring describes `edge_weight` as `torch.FloatTensor`
   - Actual runtime dtype acceptance remains **UNRESOLVED**

### MagNet Cache Behavior

**VERIFIED FROM SIGNATURE:**  
MagNet exposes `cached` with default `False`.

**VERIFIED FROM SOURCE:**  
`cached` is passed to the first `MagNetConv` layer:

```python
chebs.append(MagNetConv(in_channels=num_features, out_channels=hidden, K=K,
                        q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
```

**VERIFIED FROM SOURCE:**  
`cached` is passed to all subsequent `MagNetConv` layers:

```python
for _ in range(1, layer):
    chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K,
                            q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
```

**VERIFIED FROM DOCSTRING:**  
When enabled, cached normalization is reused after first execution:  
"If set to :obj:`True`, the layer will cache the __norm__ matrix on first execution, and will use the cached version for further executions."

**SOURCE-BASED INFERENCE:**  
Stale cached normalization could invalidate changed-graph perturbation runs.

**DYNAMIC VERIFICATION REQUIRED:**  
Confirm that `cached=False` recomputes graph-dependent state after graph changes.

**Runtime recomputation has not been observed.**

### MagNet Signed-Edge Support

**Do not claim MagNet supports signed edges merely because `edge_weight` exists.**

**UNRESOLVED:**  
Native signed-edge support. The class is documented as a directed-graph architecture. Whether `edge_weight` can represent signed edges or only magnitude weights requires further investigation of `MagNetConv` internals or package examples.

---

## 5. SSSNET_link_prediction

**Installed Source Path:** **VERIFIED FROM INSPECTION**  
`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/signed/SSSNET_link_prediction.py`

### Constructor Signature

**VERIFIED FROM SIGNATURE:**

```python
SSSNET_link_prediction(
    nfeat: int,
    hidden: int,
    nclass: int,
    dropout: float,
    hop: int,
    fill_value: float,
    directed: bool = False,
    bias: bool = True
)
```

### Forward Signature

**VERIFIED FROM SIGNATURE:**

```python
forward(
    edge_index_p: torch.LongTensor,
    edge_weight_p: torch.FloatTensor,
    edge_index_n: torch.LongTensor,
    edge_weight_n: torch.FloatTensor,
    features: torch.FloatTensor,
    query_edges: torch.LongTensor
) -> Tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor
]
```

### Return Annotation Inconsistency

**VERIFIED FROM INSPECTION:**

- The return annotation claims a four-element tuple
- The visible source returns only one tensor: `log_prob`
- This is a source-versus-annotation inconsistency

### Architecture Behavior

**VERIFIED FROM SOURCE:**

- Positive and negative graph structures are supplied separately
- `directed=False` creates positive and negative feature pathways (`_w_p0`, `_w_p1`, `_w_n0`, `_w_n1`)
- `directed=True` creates source-positive, source-negative, target-positive, and target-negative pathways (`_w_sp0`, `_w_sp1`, `_w_sn0`, `_w_sn1`, `_w_tp0`, `_w_tp1`, `_w_tn0`, `_w_tn1`)
- `SIMPA` receives separate positive and negative graph structures
- Query-edge representations concatenate source and target node representations:
  - `z[query_edges[:, 0]]`
  - `z[query_edges[:, 1]]`
- The output matrix maps to `nclass`
- Optional bias is added
- Output applies `F.log_softmax(output, dim=1)`

**VERIFIED FROM DOCSTRING:**  
`features` have shape `[num_nodes, num_features]`.

**VERIFIED FROM SOURCE:**  
The visible source verifies that `features` is used as a node-by-feature matrix in matrix multiplication. Exact runtime shape acceptance remains pending.

**Source-established expected output shape:** `[num_query_edges, nclass]`

**VERIFIED FROM SOURCE:**  
Returned rows correspond to query edges, not all nodes.

**VERIFIED FROM SOURCE:**  
`nn.Dropout` follows normal train/eval behavior controlled by `self.training`.

**VERIFIED FROM INSPECTION:**  
The inspected class interface and visible matrix operations use real-valued tensors. No explicit `real`/`imag` split appears at the inspected interface level.

**UNRESOLVED:**  
- `SIMPA` internals were not inspected
- Deeper helper-level representation behavior remains outside this audit

**VERIFIED FROM SOURCE:**  
`scripts/pygsd_model_smoke_test.py` constructs `SSSNET_link_prediction` with `directed=False`.

**UNRESOLVED:**  
`directed=True` runtime behavior remains unverified.

### fill_value

**VERIFIED FROM DOCSTRING:**  
`fill_value` is used for added self-loops in the positive part of the adjacency matrix:  
"Value for added self-loops for the positive part of the adjacency matrix."

**UNRESOLVED:**  
Exact `SIMPA` implementation details were not inspected.

### SSSNET Cache Behavior

**VERIFIED FROM SIGNATURE:**  
No `cached` argument exists at the inspected class interface.

**UNRESOLVED:**  
- `SIMPA` internals and any internal caching behavior were not inspected
- Whether `SIMPA` internally caches graph-dependent computations is unknown

**DYNAMIC VERIFICATION REQUIRED:**  
Verify changed-graph recomputation before perturbation experiments.

### Documentation Inconsistencies

**VERIFIED FROM INSPECTION:**

1. **Edge index dtype annotation inconsistency**  
   - Docstring calls `edge_index_p` and `edge_index_n` `PyTorch FloatTensor`
   - Signature annotates them as `torch.LongTensor`

2. **Return annotation inconsistency**  
   - Return annotation specifies a four-element tuple
   - Visible source returns one `log_prob` tensor

3. **Forward docstring incorrectly says outputs correspond to all nodes**  
   The docstring states: "Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes)."  
   The source returns `[num_query_edges, nclass]`, not all nodes.

**UNRESOLVED:**  
Actual runtime dtype acceptance remains unresolved.

**Do not declare one conflicting annotation automatically correct without runtime evidence.**

---

## 6. Cross-Model Comparison

| Model | Documented Graph Type | Node Features | Structural Input | Query-Edge Layout | Class Count Arg | Expected Output Shape | Output Semantics | Dropout | Cache Interface | Directed Mode | Major Unresolved Issue |
|-------|----------------------|---------------|------------------|-------------------|-----------------|----------------------|------------------|---------|-----------------|---------------|------------------------|
| **MSGNN** | Signed-directed magnetic | `real`, `imag` | One `edge_index` + optional `edge_weight` | `[Q, 2]` | `label_dim` | `[Q, label_dim]` | Class log-probabilities | Controlled by `self.training` | Exposes `cached`; subsequent-layer propagation verified; first-layer cache behavior unresolved | N/A | First-layer cache propagation unresolved |
| **MagNet** | Directed magnetic | `real`, `imag` | One `edge_index` + optional `edge_weight` | `[Q, 2]` | `label_dim` | `[Q, label_dim]` | Class log-probabilities | Controlled by `self.training` | `cached` passed to all visible `MagNetConv` layers; runtime recomputation still requires verification | N/A | Runtime changed-graph behavior unverified; signed-edge support unresolved |
| **SSSNET** | Signed with optional directed mode | `features` (real at inspected interface) | Separate positive and negative edge structures | `[Q, 2]` | `nclass` | `[Q, nclass]` | Class log-probabilities | Controlled by `self.training` | No `cached` argument at class interface; `SIMPA` internal caching unresolved | `directed` arg controls pathway count | `SIMPA` internals uninspected; `directed=True` runtime unverified |

### Key Observations

**VERIFIED FROM SOURCE:**  
All three expose per-query-edge class log-probabilities.

**VERIFIED FROM SIGNATURE:**
- MSGNN and MagNet forward signatures require `real`, `imag`, `edge_index`, `query_edges`, and optional `edge_weight`
- SSSNET's forward signature requires separate positive and negative graph tensors, `features`, and `query_edges`

**VERIFIED FROM SOURCE:**
- The visible forward implementations process those architecture-specific inputs accordingly
- Architecture-specific preprocessing is therefore required

**SOURCE-BASED INFERENCE:**  
Similar output shape does not establish a scientifically comparable task. Exact class meanings may differ.

### Task-Definition Distinction

Link existence prediction, sign prediction, direction prediction, and multi-class relation prediction are distinct task definitions.

**UNRESOLVED:**  
- The exact class meanings used by each installed model
- Whether the three selected classes can be configured for one scientifically comparable task

### Project Comparison Constraint

SSSNET and SGCN are distinct architectures and must not be treated as interchangeable.

**UNRESOLVED:**  
Which architecture should serve as the final signed comparison or control remains a separate literature, API, and task-compatibility decision.

---

## 7. Loss and Label Semantics

### Output Semantics

**VERIFIED FROM SOURCE:**  
All three methods apply `F.log_softmax(..., dim=1)`.

**VERIFIED FROM SOURCE:**  
Outputs are class log-probabilities for query edges.

### Loss Hypothesis

**SOURCE-BASED INFERENCE:**  
`NLLLoss` is the most direct loss hypothesis because:
- `NLLLoss` expects log-probabilities as input
- `NLLLoss` expects integer class-index targets
- The inspected outputs are log-probabilities

**SOURCE-BASED INFERENCE:**  
Integer class-index targets are likely.

**UNRESOLVED:**
- Exact target encoding
- Exact class meanings
- Package-author intended loss until package examples or training code are inspected

### CrossEntropyLoss Consideration

**CrossEntropyLoss:**
- Expects raw logits and internally applies `log_softmax`
- Passing normalized log-probabilities causes another `log_softmax` operation
- Mathematically, `log_softmax` of normalized log-probabilities is effectively unchanged (since `softmax(log_softmax(x))` ≈ `softmax(x)` up to numerical precision)
- Therefore this is redundant rather than categorically invalid
- **CrossEntropyLoss must still not be selected until package examples and labels are verified**

### BCEWithLogitsLoss Consideration

**BCEWithLogitsLoss:**
- Expects unnormalized logits
- Internally applies sigmoid
- Is not appropriate for the inspected multi-class log-softmax interface without changing the model output and objective

### Probability Normalization

**Mathematical consequence:**  
`exp(output).sum(dim=1) ≈ 1` is a mathematical consequence of `log_softmax`, subject to floating-point tolerance when dynamically tested.

---

## 8. Cache and Perturbation Safety

### MSGNN Cache Summary

**VERIFIED FROM SIGNATURE:**  
`cached` exists with default `False`.

**VERIFIED FROM SOURCE:**  
Subsequent-layer propagation is source-verified.

**UNRESOLVED:**  
First-layer propagation is unresolved.

**DYNAMIC VERIFICATION REQUIRED:**  
Runtime recomputation is unverified.

### MagNet Cache Summary

**VERIFIED FROM SIGNATURE:**  
`cached` exists with default `False`.

**VERIFIED FROM SOURCE:**  
Propagation to all visible `MagNetConv` layers is source-verified.

**VERIFIED FROM DOCSTRING:**  
Cached reuse is documented.

**DYNAMIC VERIFICATION REQUIRED:**  
Runtime changed-graph behavior is unverified.

### SSSNET Cache Summary

**VERIFIED FROM SIGNATURE:**  
No `cached` argument is exposed.

**UNRESOLVED:**  
- `SIMPA` internals were not inspected
- Internal caching and recomputation remain unresolved

### Experimental Cache Policy

**For MSGNN and MagNet:**
- Use `cached=False`
- Do not assume that this alone proves correct recomputation
- Dynamically test graph changes before robustness experiments
- Preserve any cache-related failure as an implementation finding

**For SSSNET:**
- No `cached` argument to control
- Dynamically test graph changes before robustness experiments
- Preserve any cache-related failure as an implementation finding

**No architecture is already proven cache-safe.**

---

## 9. Gate B Status

### Source-Verified Components

**VERIFIED FROM INSPECTION:**
- Installed distribution version: `torch-geometric-signed-directed 1.1.1`
- Installed source-file paths for all three architectures
- Fact that `importlib.metadata` and `inspect` were used

**VERIFIED FROM SIGNATURE:**
- Constructor signatures
- Forward signatures
- Required and optional argument order
- Exposed `cached` parameters in MSGNN and MagNet
- Absence of an SSSNET `cached` argument
- Type annotations, while preserving conflicts

**VERIFIED FROM DOCSTRING:**
- Documented graph/task descriptions
- `q` range `0 <= q <= 0.25`
- Normalization descriptions
- Cached reuse description for MagNet
- SSSNET feature-shape statement `[num_nodes, num_features]`
- SSSNET `fill_value` purpose

**VERIFIED FROM SOURCE:**
- `query_edges[:, 0]` and `query_edges[:, 1]` row indexing
- Expected query-edge layout `[Q, 2]`
- Architecture-specific visible input processing
- Expected output dimensions
- `F.log_softmax` behavior
- Per-query-edge output construction
- Dropout behavior controlled by `self.training`
- MSGNN cache-constructor asymmetry
- MagNet `cached` propagation to all visible layers
- SSSNET visible pathway differences (`directed=False` vs `directed=True`)
- SSSNET visible single-tensor return

### Remaining Requirements

**DYNAMIC VERIFICATION REQUIRED:**
- Successful forward execution
- Accepted runtime dtypes
- Exact runtime edge-index requirements
- Actual runtime output shape
- Output finiteness
- Probability normalization in runtime
- Gradient flow
- One-step loss execution
- Train/eval determinism
- Changed-graph recomputation

**UNRESOLVED:**
- Exact class-label meanings
- Package-author loss and target encoding
- MagNet signed-edge validity
- `directed=True` SSSNET runtime behavior
- `SIMPA` helper behavior
- MSGNN first-layer `MSConv` cache defaults
- Whether the three selected classes can be configured for one scientifically comparable task

**Gate B remains PARTIAL.**  
**Gate C remains PENDING.**  
**No forward-pass success claim is permitted yet.**

---

## 10. Safe Claims and Prohibited Claims

### Safe Claims

The following claims are supported by this audit:

- PyGSD 1.1.1 installed distribution and source were inspected
- Exact constructors and forward signatures were verified
- Query-edge row indexing was verified from source
- Expected output dimensions were established from source
- `F.log_softmax` output construction was verified
- Architecture-specific input differences were verified
- Documentation inconsistencies were preserved
- No model execution occurred in this inspection unit

### Prohibited Claims

The following claims are **not** supported by this audit:

- Successful forward pass
- Finite runtime output
- Actual runtime output shape
- Correct labels
- Verified loss
- Gradient correctness
- Training correctness
- Baseline accuracy
- Robustness
- Mathematical stability
- Convergence
- Architecture superiority
- Common-task comparability
- Manuscript eligibility
- GPU compatibility
- Production-scale suitability

### Critical Distinction

**Source-established expectation:**

"The source establishes the expected output shape `[num_query_edges, label_dim]`."

**Runtime verification (not yet available):**

"A runtime forward pass successfully produced that shape."

**Only the first claim is currently allowed.**

---

## 11. Next Controlled Work Unit

### Recommendation

**Create a forward-pass smoke-test design specification before writing or executing the test script.**

The future specification must define:

- One minimal synthetic graph per architecture
- Exact tensor shapes
- Exact tensor dtypes
- `real` and `imag` initialization
- Positive and negative graph conversion
- Directed-edge representation
- Query-edge construction
- Class count
- Expected output shape
- Finiteness checks
- Log-probability checks
- `exp(output).sum(dim=1)` normalization check
- `model.eval()`
- Dropout disabled
- `cached=False` for MSGNN and MagNet
- No training
- No datasets
- CPU only

**Do not create that specification now.**

---

## 12. Final Audit Conclusion

### Existing Script Status

`scripts/pygsd_model_smoke_test.py` remains valid as a constructor and parameter-finiteness test. It should not be modified in this work unit.

### Evidence Classification

- **Current evidence:** Level 0–1 constructor/API evidence
- **Gate B:** PARTIAL
- **Gate C:** PENDING

### Audit Limitations

- This audit does not pass a new experimental gate by itself
- Source inspection is not runtime validation
- All negative findings and documentation inconsistencies are preserved

### Key Preserved Findings

- MSGNN first-layer cache propagation is unresolved
- MagNet signed-edge support is unresolved
- SSSNET `SIMPA` internals are uninspected
- Forward docstrings incorrectly describe node classification and all-node outputs
- Edge weight dtype annotations conflict with docstrings
- SSSNET return annotation conflicts with visible source
- Whether the three selected classes can be configured for one scientifically comparable task is unresolved
- SSSNET and SGCN are distinct architectures and must not be treated as interchangeable

---

## 13. PyGSD 1.1.1 Training-Path Audit

### Source Provenance

**VERIFIED FROM INSPECTION:**

- Installed distribution: `torch-geometric-signed-directed==1.1.1`
- Package metadata identifies the official repository as `https://github.com/SherylHYX/pytorch_geometric_signed_directed`
- The exact repository tag `1.1.1` resolves to commit `ecdf3fcf94b148ae9350c848e7774915495170d9`
- The installed wheel did not contain bundled training examples
- The exact tagged repository was inspected separately under `/tmp/pygsd-1.1.1` as reference material only
- The temporary clone is not part of this research repository

### MSGNN Official Link-Prediction Path

**VERIFIED FROM SOURCE:**

Official examples: `examples/msgnn_link.py` and `examples/run_link_sign_direction_tasks.py`

**Training protocol:**
- Uses `link_class_split` for train/validation/test splits
- Supports `four_class_signed_digraph` and `five_class_signed_digraph` tasks
- Uses query-edge integer class targets
- Uses `nn.NLLLoss()`
- Reports accuracy, macro-F1, and micro-F1
- Uses the training graph returned by `link_class_split`, excluding held-out query edges
- Uses `in_out_degree` features, with the real feature tensor cloned for the imaginary input

**Conclusion:**  
This establishes an official signed-directed multiclass training path for MSGNN.

### SSSNET Official Paths

**VERIFIED FROM SOURCE:**

- `examples/sssnet.py` is a node-clustering example using `SSSNET_node_clustering`
- This is not evidence for the `SSSNET_link_prediction` class
- The link-prediction class is trained in `examples/run_link_sign_direction_tasks.py`

**Training protocol in shared script:**
- Uses the same graph splits, query edges, labels, and task definitions as MSGNN
- Uses `nn.NLLLoss()`
- Uses the same evaluation metrics (accuracy, macro-F1, micro-F1)
- The training graph is separated into positive and negative edge tensors through `SignedData.separate_positive_negative()`
- The model is initialized with `directed=data.is_directed`

**VERIFIED FROM SOURCE:**  
The official shared experiment runs SSSNET in directed mode (`directed=True` when `data.is_directed` is true).

**Conclusion:**  
This establishes a direct common-task comparison basis between MSGNN and SSSNET for the four- and five-class signed-directed tasks.

### MagNet Official Link-Prediction Path

**VERIFIED FROM SOURCE:**

Official example: `examples/magnet_link.py`

**Training protocol:**
- Uses a directed real-world dataset
- Uses `link_class_split(..., task='direction')`
- Uses two classes (0 = edge in queried direction, 1 = edge in reversed direction)
- Uses integer targets and `nn.NLLLoss()`
- Uses `in_out_degree` features
- Uses identical real/imaginary initial features (real cloned to imaginary)
- Uses train, validation, and test query-edge splits
- Reports accuracy

**Repository-wide usage:**  
Repository-wide search found MagNet only in:
- `examples/magnet_link.py` (standalone direction-prediction example)
- `test/directed_test.py` (unit tests)
- Implementation files

MagNet is **not** used in `examples/run_link_sign_direction_tasks.py` (the shared signed-directed task script).

**Conclusion:**  
The official version-1.1.1 evidence establishes MagNet for two-class direction prediction, but does not establish its validity or scientific comparability on the four- or five-class signed-directed tasks.

### Exact Label Semantics from link_split.py

**VERIFIED FROM SOURCE:**

`torch_geometric_signed_directed/utils/general/link_split.py` defines:

**Direction task (2 classes):**
- 0 = edge exists in the queried direction
- 1 = edge exists in the reversed direction

**Four-class signed-directed task:**
- 0 = positive edge in queried direction
- 1 = negative edge in queried direction
- 2 = positive edge in reversed direction
- 3 = negative edge in reversed direction

**Five-class signed-directed task:**
- Classes 0–3 as above
- 4 = no edge in either direction

**Documentation defect:**  
The prose documentation for class 3 states "the edge of the reversed direction exists" without the word "negative." However, source assertions verify:

```python
assert label_weight[labels==3].max() < 0
```

This confirms class 3 represents negative edges in the reversed direction. The documentation wording is incomplete but the implementation is correct.

### Audit Conclusion and Gate Status

**VERIFIED FROM SOURCE:**

- Package-author use of `nn.NLLLoss()` is verified for the selected official link-prediction examples
- MSGNN and SSSNET have a verified common signed-directed multiclass task path
- MagNet has a verified two-class direction path only

**UNRESOLVED:**

- A scientifically defensible three-model common task has not yet been established
- MagNet's handling of signed edge weights remains unresolved
- No selected training implementation has yet been validated in this repository

**Gate Status:**

- **Gate B:** Remains PARTIAL
  - Task semantics and loss are now verified for MSGNN and SSSNET
  - Common three-model task comparability is not established
  - MagNet signed-edge validity is unresolved
  - No training implementation has been validated in this repository

- **Gate C:** Remains PASS only for the narrow CPU forward-pass smoke test
  - Forward-pass compatibility is verified
  - Training correctness is not verified
  - Robustness validity is not verified
  - Convergence is not verified

**Prohibited Claims:**

Do not claim:
- Training correctness
- Robustness validity
- Convergence
- Full three-model comparability
- MagNet signed-edge support

### Next Decision

The next controlled step is to decide whether:

**(a)** The primary matched comparison should initially be MSGNN versus directed SSSNET on the official four-class signed-directed task, with MagNet treated as a direction-only reference; or

**(b)** A separate two-class direction task should be implemented for all three models, subject to verifying that:
- SSSNET and MSGNN label remapping is scientifically valid
- MagNet signed-weight handling is scientifically valid

This decision requires coordination between the empirical track and the theory track to determine which task configuration best supports the research questions and theoretical framework.

---

*Audit completed: 2026-07-18*  
*Training-path audit added: 2026-07-18*  
*Evidence level: 0–1 (constructor/API inspection) + package-example inspection*  
*Gate B status: PARTIAL*  
*Gate C status: PASS (forward-pass smoke test only)*
