---
title: "Gate E1 — Synthetic Data API Audit"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate E work unit E1"
status: "PASS"
audit_date: "2026-07-18"
---

# Gate E1 — Synthetic Data API Audit

**Audit Date:** 2026-07-18

---

## 1. Purpose and Scope

Gate E1 audits the PyGSD 1.1.1 synthetic signed-directed data construction API to establish a verified clean-baseline dataset configuration for Gate E multi-epoch training validation.

**Scope:**

- Verify available PyGSD synthetic generators
- Audit SDSBM API, semantics, and reproducibility
- Verify post-generation processing utilities
- Verify four-class signed-directed split semantics
- Verify shared feature construction
- Lock a provisionally validated clean-baseline configuration
- Probe the full construction pipeline for determinism and required properties

**Out of Scope:**

- Training implementation
- Model execution
- Perturbation design
- Robustness experiments
- GPU compatibility
- Production-scale validation

---

## 2. Upstream Attribution

**Critical Attribution:**

PyGSD (`torch-geometric-signed-directed`) is advisor-authored upstream research software developed by Dr. Yixuan He and collaborators. The installed package version 1.1.1 and its exact-version source repository are read-only implementation references for this project.

**Prohibited Claims:**

This project does not claim authorship of:

- SDSBM synthetic generator
- MSGNN architecture
- SSSNET architecture
- MagNet architecture
- PyGSD utility functions
- PyGSD dataset classes
- PyGSD training examples

**Permitted Use:**

This project uses the installed PyGSD 1.1.1 package as a verified implementation reference for signed-directed GNN research, following the package's documented APIs and official examples.

---

## 3. Available PyGSD Synthetic Generators

**VERIFIED FROM INSPECTION:**

PyGSD 1.1.1 provides the following synthetic signed/directed graph generators:

- **DSBM** — Directed Stochastic Block Model
- **SDSBM** — Signed Directed Stochastic Block Model
- **SSBM** — Signed Stochastic Block Model
- **polarized_SSBM** — Polarized Signed Stochastic Block Model

**Selection Rationale:**

SDSBM is selected for Gate E because:

1. It generates signed and directed graphs matching the primary experimental scope
2. PyGSD documentation describes SDSBM as originating from the MSGNN paper (He et al., 2022)
3. It is used in the official PyGSD node-classification example (`examples/msgnn_node.py`)
4. It is used in PyGSD test code (`test/general_test.py`)
5. It supports controllable block structure, edge density, and sign distribution

---

## 4. SDSBM API and Semantics

**VERIFIED FROM SOURCE:**

```python
SDSBM(N, K, p, F, size_ratio=1, eta=0.1)
```

**Parameters:**

- **N** (int) — Total number of nodes
- **K** (int) — Number of blocks/communities
- **p** (float) — Base edge probability multiplier
- **F** (numpy.ndarray) — K-by-K meta-graph matrix encoding blockwise edge probabilities and signs
- **size_ratio** (float, default=1) — Controls block size variation
- **eta** (float, default=0.1) — Fraction of edges whose signs are flipped

**Returns:**

- **A** (scipy.sparse matrix) — Signed directed adjacency matrix
- **y** (numpy.ndarray) — Node cluster assignments (length N)

**Edge Generation Semantics:**

1. Block sizes are determined by `size_ratio` and `K`
2. For each ordered node pair `(i, j)` in blocks `(r, s)`:
   - Edge probability is `p * abs(F[r, s])`
   - If an edge is generated, its initial sign is `sign(F[r, s])`
3. Exactly `int(number_of_edges * eta)` edge signs are randomly flipped
4. No self-loops are generated (NetworkX SBM default behavior)

**Meta-Graph Matrix Requirements:**

- `F` must be K-by-K
- All `p * abs(F[i, j])` must be valid probabilities in `[0, 1]`
- `sign(F[i, j])` determines expected edge sign for block pair `(i, j)`
- Positive `F[i, j]` → positive edges (before eta flips)
- Negative `F[i, j]` → negative edges (before eta flips)

---

## 5. Reproducibility Finding

**CRITICAL FINDING:**

SDSBM uses multiple random number generators:

- **NumPy random** — For node permutation and sign flipping
- **Python random (via NetworkX)** — For stochastic block model edge generation

**Experimental Probe:**

```python
# Test 1: NumPy seed only
np.random.seed(0)
A1, y1 = SDSBM(200, 4, 0.05, F, 1.5, 0.1)
np.random.seed(0)
A2, y2 = SDSBM(200, 4, 0.05, F, 1.5, 0.1)
# Result: numpy_only_equal: False
```

```python
# Test 2: Both Python and NumPy seeds
random.seed(0)
np.random.seed(0)
B1, z1 = SDSBM(200, 4, 0.05, F, 1.5, 0.1)
random.seed(0)
np.random.seed(0)
B2, z2 = SDSBM(200, 4, 0.05, F, 1.5, 0.1)
# Result: python_and_numpy_equal: True
```

**Verified Probe Results:**

- `numpy_only_equal: False`
- `python_and_numpy_equal: True`
- Deterministic hash: `5814d8112e79efe375cf53679c84292d5c76802eb8c9749e92263bc364aecfc2`

**Required Seed Protocol:**

For deterministic SDSBM generation, the following seeds must be set before every call:

```python
random.seed(seed)
np.random.seed(seed)
```

For the full PyGSD pipeline including `SignedData` and `link_class_split`, additionally set:

```python
torch.manual_seed(seed)
```

---

## 6. Post-Generation Processing

### extract_network

**VERIFIED FROM PACKAGE USAGE:**

`extract_network(A, y)` performs graph cleaning:

1. Extracts the largest weakly connected component
2. Iteratively removes nodes with total degree (in + out) less than 2
3. Returns cleaned adjacency matrix and corresponding cluster assignments

**Critical Implication:**

All node counts, edge counts, class distributions, and fingerprints must be recorded **after** `extract_network`, not from raw SDSBM output.

### SignedData Construction

**VERIFIED FROM SOURCE:**

```python
SignedData(A=A, y=y)
```

**Behavior:**

- Converts scipy sparse matrix `A` to COO format
- Derives `edge_index` as `LongTensor` from `A.nonzero()`
- Derives `edge_weight` as `FloatTensor` from `A.data`
- Sets `num_nodes` from `A.shape[0]`
- Stores original sparse matrix as `self.A`

**Critical Implication:**

Fingerprints must use the resulting PyTorch tensors (`edge_index`, `edge_weight`) rather than assuming any particular sparse-matrix ordering.

---

## 7. Four-Class Signed-Directed Split Semantics

**VERIFIED FROM SOURCE:**

`link_class_split` with `task='four_class_signed_digraph'` produces:

**Class Labels:**

- **0** — Positive edge in the queried direction
- **1** — Negative edge in the queried direction
- **2** — Positive edge in the reverse direction
- **3** — Negative edge in the reverse direction

**Internal Class 4:**

The helper function `undirected_label2directed_label` internally generates class 4 for non-edges, but `link_class_split` filters it out for the `four_class_signed_digraph` task:

```python
elif task == 'four_class_signed_digraph':
    ids_train = ids_train[labels_train < 4]
    labels_train = labels_train[labels_train < 4]
    # (same for val and test)
```

**Reciprocal Edge Handling:**

**VERIFIED FROM SOURCE:**

Reciprocal candidate pairs (edges that exist in both directions) are excluded from supervised four-class query labels because their direction is ambiguous. However, reciprocal training edges returned as `undirected_train` may be added back to the observed message-passing graph to preserve training-graph information.

**Critical Implication:**

Runtime validation should verify that supervised train/validation/test query examples are unambiguous (no reciprocal pairs in query sets), not that all reciprocal edges are absent globally from the observed graph.

**Validation/Test Selection:**

- Uses sign-stratified sampling to maintain positive/negative edge ratios
- Uses `np.random.RandomState(seed)` for reproducible shuffling
- When `maintain_connect=True`, preserves minimum spanning forest edges in training
- Removes validation and test query edges from the observed message-passing graph

**Split Protocol:**

- `splits=1` — Single train/val/test split
- `prob_val=0.15` — 15% of edges for validation
- `prob_test=0.15` — 15% of edges for testing
- `seed=0` — Reproducible split
- `maintain_connect=True` — Preserve connectivity
- `ratio=1.0` — Use all edges
- `device='cpu'` — CPU tensors

**Returned Structure:**

```python
datasets[0]['graph']           # Observed training edge_index [2, E_train]
datasets[0]['weights']         # Observed training edge weights [E_train]
datasets[0]['train']['edges']  # Train query edges [Q_train, 2]
datasets[0]['train']['label']  # Train query labels [Q_train]
datasets[0]['val']['edges']    # Validation query edges [Q_val, 2]
datasets[0]['val']['label']    # Validation query labels [Q_val]
datasets[0]['test']['edges']   # Test query edges [Q_test, 2]
datasets[0]['test']['label']   # Test query labels [Q_test]
```

---

## 8. Shared Feature Construction

**VERIFIED FROM SOURCE:**

```python
in_out_degree(edge_index, size=num_nodes, signed=False, edge_weight=abs_weights)
```

**Parameters:**

- `edge_index` — The observed training graph (after validation/test removal)
- `size` — Total number of nodes
- `signed=False` — Return two-column in/out-degree feature tensor (not four-column signed features)
- `edge_weight` — Absolute values of observed training edge weights

**Returns:**

- `degree` — Tensor of shape `[num_nodes, 2]` containing in-degree and out-degree features

**Critical Requirements:**

1. Compute features from the **observed training graph**, not the full graph
2. Use **absolute edge weights** to avoid negative degree values
3. Use `signed=False` to get two-column features
4. The **same feature tensor** must be used by both MSGNN and directed SSSNET
5. Feature fingerprint must be verified at runtime

---

## 9. Provisionally Locked Gate E Configuration

**Generation Seeds:**

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

**SDSBM Parameters:**

- `N = 400`
- `K = 4`
- `p = 0.05`
- `size_ratio = 1.5`
- `eta = 0.1`

**Meta-Graph Construction Parameter:**

- `gamma = 0.1`

**Meta-Graph Matrix F:**

```python
F = np.array([
    [ 0.5,  0.1, -0.1,  0.1],
    [ 0.9,  0.5, -0.1, -0.5],
    [-0.9, -0.9,  0.5, -0.9],
    [-0.9, -0.5, -0.1,  0.5]
])
```

**Note on gamma:**

The parameter `gamma = 0.1` is used to construct the exact four-class meta-graph matrix `F` shown above, following the official MSGNN example pattern. While `gamma` is not a direct `SDSBM` function argument, it determines the off-diagonal structure of `F` and is therefore recorded as part of the provisionally locked configuration.

**Rationale:**

This configuration follows the official MSGNN four-class construction pattern from PyGSD examples while using:

- `N=400` to satisfy Gate E's minimum 200-node requirement with margin
- `p=0.05` to efficiently generate at least 2,000 directed signed edges before splitting
- The documented four-block meta-graph structure with mixed positive/negative inter-block edges
- `eta=0.1` for 10% sign noise as used in official examples
- `gamma=0.1` for the meta-graph construction pattern

**Split Configuration:**

- `task = 'four_class_signed_digraph'`
- `splits = 1`
- `prob_val = 0.15`
- `prob_test = 0.15`
- `seed = 0`
- `maintain_connect = True`
- `ratio = 1.0`
- `device = 'cpu'`

**Feature Configuration:**

- `signed = False`
- Use absolute observed training edge weights

---

## 10. Verified Construction Probe Results

**Full Pipeline Probe:**

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
A, y = SDSBM(400, 4, 0.05, F, 1.5, 0.1)
A, y = extract_network(A, y)
data = SignedData(A=A, y=torch.as_tensor(y, dtype=torch.long))
split = link_class_split(
    data,
    splits=1,
    task='four_class_signed_digraph',
    prob_val=0.15,
    prob_test=0.15,
    seed=0,
    device='cpu'
)[0]
features = in_out_degree(
    split['graph'],
    size=data.num_nodes,
    signed=False,
    edge_weight=torch.abs(split['weights'])
)
```

**Verified Properties:**

- **Nodes:** 400
- **Original directed signed edges:** 4,163
- **Positive edges:** 1,616
- **Negative edges:** 2,547
- **Observed training graph edges:** 2,915
- **Feature shape:** `(400, 2)`

**Class Distribution:**

- **Train:** `[1103, 1756, 1103, 1756]` (classes 0, 1, 2, 3)
- **Validation:** `[237, 376, 237, 376]`
- **Test:** `[238, 377, 238, 377]`

**All four classes present in every split:** True

**Reproducibility:**

- Repeated full-pipeline fingerprint equality: True
- Audit probe fingerprint: `0cffc2af36103de532ed27cc573211660a5d9e1bdea55dbbcb23f9afe58581a7`

**Critical Note:**

This fingerprint is audit-probe evidence only, not the final E2 production fingerprint. Gate E2 implementation must:

1. Recompute and store named component fingerprints during the first validated production construction
2. Ensure repeated E2/E3 executions under the same environment and configuration match those canonical stored fingerprints
3. Treat the E1 audit-probe fingerprint as supporting evidence only, not as the canonical production fingerprint

**Required Component Fingerprints for E2:**

- Clean graph edge index and weights
- Training graph edge index and weights
- Train/validation/test query edges and labels
- Shared features
- Full data fingerprint

---

## 11. Risks and Safeguards

**Version Dependencies:**

SDSBM generation depends on:

- Python random number generator
- NumPy random number generator
- NetworkX stochastic block model implementation
- SciPy sparse matrix operations
- PyTorch tensor operations

**Required Version Recording:**

Gate E2 implementation must record:

- Python version
- NumPy version
- Torch version
- PyTorch Geometric version
- PyGSD version
- NetworkX version
- SciPy version

**Runtime Validation Requirements:**

Every Gate E execution must validate:

1. Node count matches expected value
2. Edge count is within expected range
3. Positive and negative edge counts are both nonzero
4. All four classes present in train, validation, and test
5. No class has zero representation in any split
6. Supervised query examples are unambiguous (no reciprocal pairs in query sets)
7. Features are finite
8. Component fingerprints match canonical stored fingerprints (after E2 establishes them)

**Failure Protocol:**

If any validation fails:

- Stop immediately
- Record exact failure condition
- Preserve all intermediate tensors
- Do not silently repair or substitute
- Report as implementation defect or package-version incompatibility

---

## 12. E1 Decision

**Gate E1 Status:** PASS

**Verified Components:**

- SDSBM API and semantics
- Reproducibility requirements (Python + NumPy + Torch seeds)
- Post-generation processing (`extract_network`, `SignedData`)
- Four-class signed-directed split semantics
- Shared feature construction
- Full pipeline probe with verified properties

**Provisionally Locked Configuration:**

The Gate E clean-baseline configuration is provisionally locked pending E2 implementation validation.

**Authorized Next Steps:**

- **E2 — Clean-baseline implementation** is authorized
- E2 must implement the full construction pipeline
- E2 must record all component fingerprints
- E2 must validate all required properties
- E2 must preserve exact version information

**Prohibited Actions:**

- Structural perturbations remain unauthorized
- Edge deletion experiments remain unauthorized
- Sign flip experiments remain unauthorized
- Direction flip experiments remain unauthorized
- Feature perturbations remain unauthorized
- Architecture perturbations remain unauthorized

---

*Audit completed: 2026-07-18*  
*Gate E1 status: PASS*  
*E2 clean-baseline implementation: AUTHORIZED*  
*Perturbation experiments: NOT AUTHORIZED*
