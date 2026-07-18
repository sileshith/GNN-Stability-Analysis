---
title: "Empirical Validation Methodology for Signed and Directed GNN Robustness Experiments"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "subordinate empirical-methodology guide"
status: "researcher-adopted working document"
governing_framework: "Phase 0–12 master research workflow"
updated: "2026-07-17"
---

# Empirical Validation Methodology for Signed and Directed GNN Robustness Experiments

## Formal Methodology Statement

This study follows a theory-driven, reproducible computational research methodology. General graph-operator stability relationships are developed first and then specialized to signed and directed graph-neural-network architectures. Mathematical analysis and empirical experiments proceed in parallel: each computational experiment is derived from an explicit theoretical hypothesis, and each theoretical quantity is connected to measurable structural, operator, spectral, representation, or prediction-level diagnostics. The empirical lifecycle is informed by selected CRISP-DM principles for data understanding, preparation, modeling, evaluation, and iteration, while literature verification, proof development, architecture specialization, and theory–experiment reconciliation are governed by the project's Phase 0–12 research framework.

---

## Methodology Hierarchy

This document operates within the following hierarchy:

1. **Primary methodology:** Theory-driven, reproducible computational research.

2. **Scientific governance:** The Phase 0–12 framework governs literature verification, mathematical development, architecture specialization, perturbation theory, computational validation, theory–experiment reconciliation, claim approval, and manuscript synthesis.

3. **Parallel research tracks:** Mathematical analysis and controlled empirical validation proceed in parallel and inform one another.

4. **Supporting empirical lifecycle:** Selected CRISP-DM principles may inform data understanding, preparation, modeling, evaluation, iteration, and reproducible research delivery.

5. **Explicit boundary:** CRISP-DM does not govern primary-source verification, theorem development, proof validation, graph-operator perturbation derivations, architecture-specific theory, contraction analysis, or theory–experiment reconciliation by itself.

---

## 1. Purpose, Scope, and Authority

### Document Scope

This document governs only the empirical validation lifecycle in the SOL experimental repository. It provides operational guidance for:

- Translating research questions into controlled experiments
- Understanding signed and directed graph data
- Preparing datasets and perturbation protocols
- Validating architecture implementations
- Executing clean-training/perturbed-inference experiments
- Measuring multi-level diagnostics
- Managing uncertainty and statistical interpretation
- Ensuring reproducibility and traceability
- Classifying evidence maturity

### Authority and Subordination

This document is subordinate to the Phase 0–12 master guideline maintained in the documentation repository. It does not replace:

- The literature-verification process
- The theory-development track
- Architecture-specific derivations
- Experiment-record templates
- The claim–evidence approval process

### Key Distinctions

This methodology distinguishes between:

- **Mathematical stability:** Theoretical properties of graph operators and their perturbations, analyzed through formal mathematical frameworks
- **Empirical robustness:** Observed behavior of trained models under controlled structural perturbations, measured through computational experiments
- **Task degradation:** Changes in prediction performance metrics (accuracy, F1, AUC) under perturbation
- **Reproducibility:** The ability to regenerate experimental results from documented configurations, seeds, and protocols

Mathematical stability is a theoretical property. Empirical robustness is an observed phenomenon. Task degradation is one measurable consequence. Reproducibility is a methodological requirement.

---

## 2. Research-to-Experiment Translation

### Experiment Origin Requirements

Every substantive experiment must originate from:

1. **Research question:** A clear, answerable question derived from the Phase 0–12 framework
2. **Explicit hypothesis:** A testable prediction about expected behavior
3. **Theoretical or literature-based motivation:** Connection to existing theory or prior empirical findings
4. **Measurable quantities:** Specific diagnostics that can be computed and compared
5. **Acceptance criteria:** Conditions under which the hypothesis is supported, contradicted, or inconclusive
6. **Evidence maturity target:** The intended evidence level (0–6) for the experiment

### Research-to-Experiment Cycle

```
research question
  → theoretical expectation
    → measurable diagnostic
      → controlled experiment
        → implementation validation
          → evidence interpretation
            → theory–experiment reconciliation
```

### iMac–SOL Execution Cycle

This project operates through synchronized cycles between the iMac documentation repository and the SOL experimental repository:

- **Research questions, hypotheses, literature support, theoretical expectations, acceptance criteria, and experiment specifications** are developed in the iMac documentation repository
- **Implementation validation, controlled computation, logs, manifests, configurations, and machine-readable results** are produced in the SOL experimental repository
- **The resulting evidence packet** is reviewed and interpreted in the iMac documentation repository
- **Theory–experiment reconciliation** then determines whether to stop, revise, repeat, promote, narrow, or expand the work

Experiments and documentation proceed through short synchronized cycles rather than completing all experiments before documentation. Disagreement between theory and experiment must not be forced into agreement and may be classified as support, contradiction, qualification, inconclusive evidence, implementation concern, or assumption failure.

```
iMac: question, literature, theory, and experiment specification
                         ↓
SOL: implementation validation and controlled experiment
                         ↓
iMac: evidence interpretation and theory–experiment reconciliation
                         ↓
Decision: stop, revise, repeat, promote, narrow, or expand
```

### Interpretation Outcomes

An experiment may:

- **Support** the theoretical expectation (observed trends match predictions)
- **Contradict** the theoretical expectation (observed trends oppose predictions)
- **Qualify** the theoretical expectation (observed trends partially match, with identified boundary conditions)
- **Remain inconclusive** (insufficient evidence, implementation concerns, or assumption violations prevent interpretation)

All four outcomes are scientifically valid when properly documented.

---

## 3. Signed and Directed Graph Data Understanding

### Structural Properties

Where applicable to the dataset and research question, document:

**Basic structure:**
- Node count
- Edge count
- Positive edge count and proportion
- Negative edge count and proportion
- Directed vs. undirected representation

**Degree statistics:**
- In-degree distribution (directed graphs)
- Out-degree distribution (directed graphs)
- Degree distribution (undirected or aggregated)
- Isolated nodes

**Connectivity:**
- Connected components (undirected view)
- Strongly connected components (directed graphs)
- Weakly connected components (directed graphs)
- Density

**Signed and directed properties:**
- Reciprocity (proportion of bidirectional edges in directed graphs)
- Reciprocal-edge sign consistency (for signed directed graphs: do (i,j,+) and (j,i,+) both exist?)
- Balance-related diagnostics when relevant to the dataset and research question

**Task properties:**
- Class distribution (for node classification)
- Label availability
- Train/validation/test proportions

### Dataset-Specific Considerations

**Bitcoin-Alpha and Bitcoin-OTC:**  
These are signed trust/rating networks, not automatically fraud-label datasets. Edges represent trust ratings between users. Task labels (if present) may represent user reputation or other derived properties, not direct fraud labels.

**SDSBM:**  
Controlled synthetic graphs with known generative parameters. Document the generative model, parameters, and any ground-truth properties.

**Cora:**  
Legacy debugging work only. Not current manuscript evidence.

### Leakage Risks

Document and check for:

- **Reverse-edge leakage:** Test edges appearing in the message-passing graph in reverse direction
- **Temporal leakage:** Future information available during training (if dataset has temporal structure)
- **Label leakage:** Test labels influencing training through graph structure

### Optional Statistics

Not every graph statistic is required for every dataset. Focus on properties relevant to:

- The research question
- The perturbation protocol
- The architecture's assumptions
- The task definition

---

## 4. Dataset Preparation and Split Validation

### Data Provenance

Require:

- **Immutable or versioned raw-data references:** Fixed dataset version, commit hash, or checksum
- **Deterministic preprocessing:** Reproducible transformations from raw data to model input
- **Model-compatible edge representations:** Correct format for each architecture (e.g., edge index, adjacency matrix, signed adjacency)

### Feature Preparation

Where applicable:

- Feature normalization (zero mean, unit variance, or min-max scaling)
- Feature dimensionality
- Missing feature handling
- Categorical feature encoding

### Split Protocol

Require:

- **Split seeds:** Explicit random seeds for train/validation/test splitting
- **Class stratification:** Preserve class proportions across splits (for classification tasks)
- **Sign and direction balance:** Consider signed and directed structure when splitting
- **Connectivity checks:** Verify that splits do not create isolated components unless intentional

### Message-Passing Graph Definition

**Critical requirement:**  
Validation and test edges must not remain in the message-passing graph unless explicitly justified and documented.

For link prediction or edge classification:

- Training edges form the message-passing graph
- Validation edges are predicted but not used for message passing during validation
- Test edges are predicted but not used for message passing during testing

For node classification:

- All edges may be used for message passing
- Node labels are split into train/validation/test

### Leakage Checks

Verify:

- **Reverse-edge leakage:** If (i,j) is in test, is (j,i) in the message-passing graph?
- **Message-passing leakage:** Can test information propagate through the graph structure?
- **Negative sampling:** For link prediction, ensure negative samples are truly absent edges

### Task and Loss Compatibility

Verify that:

- Split protocol matches the task (node classification, link prediction, edge classification)
- Loss function is compatible with the task and architecture
- Evaluation metrics are appropriate for the task

---

## 5. Structural Perturbation Protocol

### Primary Perturbations

This project defines three primary structural perturbations:

1. **Sign flip:** (i, j, w) → (i, j, −w)  
   Reverses the polarity of a signed edge.

2. **Direction reversal:** (i, j, w) → (j, i, w)  
   Reverses the orientation of a directed edge, preserving sign.

3. **Edge deletion:** (i, j, w) → removed  
   Removes an edge from the graph.

### Perturbation Specification

For each perturbation, require:

1. **Eligible-edge definition:** Which edges can be perturbed (e.g., all edges, only positive edges, only training edges)
2. **Requested budget:** Intended perturbation severity (e.g., "flip 10% of positive edges")
3. **Denominator:** What the percentage is relative to (e.g., total edges, positive edges, eligible edges)
4. **Realized number of changes:** Actual count of edges perturbed
5. **Random seed:** Seed used to select which edges to perturb
6. **Canonical raw-graph perturbation manifest:** Explicit list of perturbed edges before architecture-specific conversion
7. **Verification:** Confirmation that the manifest was applied exactly as specified
8. **Structural before/after diagnostics:** Edge counts, sign distributions, degree statistics before and after perturbation
9. **Deterministic architecture-specific conversion:** Each architecture converts the same perturbed raw graph to its internal representation

### Critical Principle: Equal Budgets ≠ Equal Operator Perturbations

**Explicit statement:**  
Equal edge-level budgets do not imply equal model-specific operator perturbations.

Example: Flipping 10% of edges in a graph produces:
- One perturbed adjacency matrix
- Different operator perturbations for GCN (symmetric normalized Laplacian), MSGNN (magnetic Laplacian), and MagNet (directed Laplacian)
- Different spectral changes
- Potentially different representation and task impacts

### Dual Severity Reporting

Require reporting at two levels:

1. **Structural severity:**  
   - Requested budget (e.g., "10% of positive edges")
   - Realized count (e.g., "523 edges flipped")
   - Structural diagnostics (sign distribution change, degree distribution change)

2. **Model-specific operator severity:**  
   - Operator norm change (Frobenius or spectral norm)
   - Spectral property changes (where computable)
   - Architecture-specific perturbation magnitude

### Optional Perturbations

**Edge rewiring** (adding edges while removing others to preserve edge count) is labeled as optional stretch work only. It is not a required perturbation for the current project scope.

### Perturbation Manifest Requirements

The canonical perturbation manifest must contain:

- Perturbation type (sign flip, direction reversal, edge deletion)
- Random seed
- Eligible edge set definition
- Requested budget and denominator
- Realized edge list (explicit edges perturbed)
- Timestamp or experiment ID
- Storage location for durable backup

---

## 6. Architecture and Task Validation

### Candidate Architectures

This project evaluates:

- **MSGNN:** Signed and directed magnetic architecture
- **MagNet:** Directed magnetic architecture
- **Selected signed control:** SSSNET or SGCN after the architecture-selection decision
- **GCN:** Debugging or unsigned-control role only

**Explicit statement:**  
SSSNET and SGCN are not automatically interchangeable. The final signed-control architecture selection remains open and must be documented when finalized.

### Staged Validation Protocol

Require validation in stages:

#### Stage 1: Import and Constructor
- Package import succeeds
- Constructor accepts expected arguments
- Constructor returns a model instance
- Parameters are initialized
- Parameters are finite (no NaN or Inf)

#### Stage 2: Forward-Pass Signature
- Forward method accepts expected inputs (node features, edge index, edge weights, etc.)
- Forward method returns expected outputs (node embeddings, logits, etc.)
- Output tensor shapes are correct
- Output tensor dtypes are correct
- Output values are finite

#### Stage 3: Forward-Pass Semantics
- Output values are in expected range (e.g., logits are unbounded, probabilities are in [0,1])
- Output values change when inputs change
- Output values are deterministic given fixed seed
- Gradient computation succeeds (if training)

#### Stage 4: Loss and Task Compatibility
- Loss function accepts model outputs and labels
- Loss value is finite
- Loss value is in expected range
- Gradient flows through loss to model parameters

#### Stage 5: Minimal Functional Run
- Model can complete one training step
- Model can complete one evaluation step
- Metrics can be computed from outputs

#### Stage 6: Baseline Reproduction
- Model can train to convergence on clean graph
- Model achieves performance consistent with literature or prior runs
- Performance is reproducible across seeds (within expected variance)

### Critical Statement

**Constructor success does not establish:**
- Forward-pass correctness
- Training correctness
- Robustness
- Stability
- Convergence
- Accuracy
- Final architecture suitability

Each stage must be validated independently.

---

## 7. Clean-Training, Perturbed-Inference Protocol

### Primary Evaluation Protocol

The default protocol for evaluating structural sensitivity is:

```
clean observed graph
  → train model and select checkpoint
    → freeze learned parameters
      → apply saved perturbation manifest
        → recompute graph-dependent representations
          → evaluate on untouched test edges
```

### Protocol Interpretation

This protocol evaluates:

- **Structural sensitivity of a fixed trained model:** How does a model trained on the clean graph perform when the graph structure changes?
- **Inference robustness:** Can the model generalize to perturbed graph structure without retraining?

This protocol does **not** evaluate:

- **Adaptation after retraining:** How would the model perform if retrained on the perturbed graph?
- **Training stability:** How sensitive is the training process to graph perturbations?

### Traceability Requirements

Require documentation of:

1. **Clean checkpoint:** Model parameters trained on clean graph, with training configuration and final validation performance
2. **Perturbation manifest:** Exact edges perturbed, with seed and verification
3. **Graph-dependent preprocessing:** Any normalization, Laplacian computation, or other graph-dependent operations recomputed on perturbed graph
4. **Evaluation split:** Test edges used for evaluation (must be identical across clean and perturbed evaluations)

### Alternative Protocols

Other protocols may be used for specific research questions:

- **Perturbed training:** Train on perturbed graph, evaluate on perturbed graph
- **Cross-perturbation:** Train on one perturbation, evaluate on another
- **Adaptive retraining:** Train on clean, retrain on perturbed, compare

Alternative protocols must be explicitly justified and documented.

---

## 8. Four-Level Diagnostic Framework

### Level 1: Structural Diagnostics

Measure changes to the graph structure itself:

**Perturbation realization:**
- Requested budget and denominator
- Realized count of changes
- Eligible edge count

**Sign and direction:**
- Positive edge count before/after
- Negative edge count before/after
- Sign distribution change
- Reciprocity before/after (directed graphs)

**Degree statistics:**
- In-degree distribution change (directed)
- Out-degree distribution change (directed)
- Degree distribution change (undirected or aggregated)

**Connectivity:**
- Connected components before/after
- Strongly connected components before/after (directed)
- Isolated nodes before/after
- Density before/after

### Level 2: Operator Diagnostics

Measure changes to model-specific graph operators:

**Where mathematically and computationally appropriate, include:**

- Raw adjacency Frobenius norm change: ||A_perturbed - A_clean||_F
- Raw adjacency spectral norm change: ||A_perturbed - A_clean||_2
- Model-specific operator Frobenius norm change: ||L_perturbed - L_clean||_F
- Model-specific operator spectral norm change: ||L_perturbed - L_clean||_2

**Document operator construction:**
- Normalization conventions (symmetric, row, column, none)
- Self-loop conventions (added, not added, conditional)
- Charge or phase parameters (for magnetic Laplacians)
- Complex vs. real representation

**Critical warning:**  
Cross-architecture operator norms may not be directly comparable because model-specific operators can use different normalizations and scales. A Frobenius norm change of 10.0 for GCN's normalized Laplacian may not be equivalent to a Frobenius norm change of 10.0 for MSGNN's magnetic Laplacian.

### Level 3: Spectral or Subspace Diagnostics

**Where assumptions and computational feasibility permit, include:**

- Eigenvalue drift: changes in individual eigenvalues or eigenvalue distributions
- Eigengap change: changes in spectral gaps (e.g., λ₂ - λ₁)
- Principal-angle or subspace drift: angles between eigenspaces before and after perturbation
- Davis–Kahan-style diagnostic quantities: bounds on eigenvector perturbation
- Approximation method and tolerance: if eigenvalues are approximated (e.g., Lanczos, Arnoldi)

**Do not automatically require:**
- Complex-eigenvalue analysis (may not be necessary or feasible for all operators)
- Full eigendecomposition (may be computationally prohibitive for large graphs)

**Clarifications:**

1. Complex-valued representations or operator entries do not imply complex eigenvalues when the analyzed operator is Hermitian.
2. Spectral diagnostics measure properties of the graph operator, not the trained neural network.
3. Spectral diagnostics do not by themselves prove trained-network output stability.

### Level 4: Representation and Task Diagnostics

**Where applicable, include:**

**Representation changes:**
- Normalized embedding drift: ||h_perturbed - h_clean|| / ||h_clean||
- Embedding cosine similarity
- Embedding cluster coherence

**Prediction changes:**
- Logit drift: changes in raw model outputs before softmax
- Prediction disagreement: proportion of nodes/edges with different predicted classes
- Confidence or margin change: changes in prediction confidence or decision margins

**Task performance:**
- Absolute task degradation: performance_clean - performance_perturbed
- Relative task degradation: (performance_clean - performance_perturbed) / performance_clean
- Macro-F1 (for multi-class classification)
- Accuracy (for classification)
- ROC-AUC (only where appropriate: binary classification or one-vs-rest)
- Confusion matrices (for classification)

**Theoretical connections:**
- Empirical Lipschitz-style ratios: ||output_perturbed - output_clean|| / ||input_perturbed - input_clean||
- Theoretical-bound versus observed-error comparisons: if theory predicts a bound, compare to observed error

**Critical statement:**  
Accuracy decline is not itself a mathematical stability result. It is an empirical observation that may be explained by mathematical stability properties, but the two must not be conflated.

---

## 9. Repetition, Uncertainty, and Statistical Interpretation

### Seed Requirements

Specify minimum seed counts based on evidence maturity:

- **One seed:** Debugging evidence only. Sufficient for verifying code runs without errors.
- **Three seeds:** Pilot assessment. Sufficient for initial exploration and hypothesis refinement.
- **Five seeds:** Default for promoted production experiments. Sufficient for reporting mean and standard deviation.
- **More seeds:** May be required for formal statistical testing or high-stakes claims.

### Uncertainty Reporting

Require:

- Mean across seeds
- Standard deviation across seeds
- Raw per-seed results (not just aggregates)

### Randomness Sources

Where feasible, separate randomness sources:

1. **Initialization randomness:** Model parameter initialization seed
2. **Data split randomness:** Train/validation/test split seed
3. **Synthetic graph generation randomness:** Seed for generating synthetic graphs (e.g., SDSBM)
4. **Perturbation generation randomness:** Seed for selecting which edges to perturb
5. **Negative sampling randomness:** Seed for sampling negative edges (link prediction)

Separating sources enables variance decomposition and more precise uncertainty attribution.

### Statistical Interpretation

**Before formal testing:**
- Examine effect sizes (mean differences)
- Examine uncertainty (standard deviations, ranges)
- Examine raw-seed results (trajectories, outliers)
- Visualize distributions

**Formal statistical tests:**
- Do not mandate t-tests, Wilcoxon tests, confidence intervals, or statistical power analysis universally
- Use formal statistical tests only when:
  - Assumptions are met (e.g., normality for t-tests, independence)
  - Sample size is adequate
  - Interpretation is defensible
- Document test choice, assumptions, and limitations

**Generalization:**
- One dataset or one seed cannot support broad generalization claims
- Cross-dataset validation is required for claims about architecture-general properties
- Multi-seed validation is required for claims about robustness or stability

---

## 10. Experiment Management and Reproducibility

### Evidence-Level Documentation Requirements

**Evidence Levels 0–2 — Reduced Traceable Record**

Evidence Levels 0–2 may use a reduced but traceable record containing:

- objective;
- date;
- environment versions;
- code version or full commit hash;
- command or script executed;
- result or observation;
- warning or failure, if applicable;
- evidence limitation stating what the result does and does not establish.

These reduced records are appropriate for environment checks, constructor or forward-pass smoke tests, and single functional runs.

**Evidence Level 3 or Higher — Full Substantive Record**

Evidence Level 3 or higher requires the complete substantive experiment record defined below.

**Promotion Requirement**

A Level 0–2 result must receive the complete substantive experiment record before it can be promoted to Evidence Level 3 or higher. Reduced documentation is not sufficient for multi-seed, controlled-perturbation, theory-aligned, or manuscript-ready evidence.

### Experiment Record Requirements

Every substantive experiment (Evidence Level 3 or higher) must document the following. (See Section 13 for additional requirements specific to manuscript-eligible Level 6 evidence.)

**Identification:**
- Experiment ID (unique identifier)
- Timestamp
- Researcher

**Code and environment:**
- Git branch
- Full commit hash (40-character SHA)
- Python version
- PyTorch version
- PyG version
- PyGSD version (if applicable)
- Other relevant package versions

**Data:**
- Dataset name and version
- Preprocessing version or commit hash
- Split seeds
- Train/validation/test sizes

**Model:**
- Architecture name (MSGNN, MagNet, SSSNET, SGCN, GCN)
- Constructor arguments (hidden dimensions, layers, dropout, etc.)
- Task (node classification, link prediction, edge classification)
- Loss function

**Perturbation:**
- Perturbation type (sign flip, direction reversal, edge deletion, none)
- Perturbation seeds
- Requested budget and denominator
- Realized count
- Canonical manifest location

**Execution:**
- Machine-readable configuration (YAML, JSON, or similar)
- Exact command or job script
- SLURM job ID (if applicable)
- Compute resources (CPU, GPU, memory, time)

**Outputs:**
- Log file location
- Machine-readable metrics (JSON, CSV, or similar)
- Checkpoint location
- Output paths

**Interpretation:**
- Summary of results
- Hypothesis support/contradiction/qualification/inconclusive
- Warnings and failures
- Limitations

**Linkage:**
- Literature reference (if applicable)
- Theory reference (if applicable)
- Manuscript destination (which section or figure)

**Storage:**
- Durable backup location (not SOL scratch alone)

### SOL and SLURM

Where relevant:

- Use SLURM job arrays for multi-seed experiments
- Use SLURM job dependencies for multi-stage workflows
- Log SLURM job IDs for traceability
- Monitor resource usage (CPU, GPU, memory, time)

**Do not require:**
- Containerization (Docker, Singularity) unless technically necessary
- Specific SLURM partition or resource allocation

### Critical Storage Requirement

**SOL scratch is temporary and cannot be the sole storage location for important evidence.**

All experiment records, configurations, manifests, checkpoints, and results must be:

- Committed to Git (for code and configurations)
- Backed up to durable storage (for data, checkpoints, and results)
- Synchronized with GitHub (for code and documentation)

### Deferred Details

Detailed field schemas for experiment records and configuration templates are deferred to future template documents. This document establishes the principles and requirements; templates will provide the implementation.

---

## 11. Evidence Levels and Gates

### Evidence Levels (0–6)

**Level 0: Environment/Import Verification**
- Package imports succeed
- Environment versions documented
- No code execution beyond import

**Level 1: Constructor or Forward-Pass Smoke Verification**
- Constructor succeeds
- Forward pass produces finite outputs
- No training or evaluation

**Level 2: Single Functional Run**
- Model completes one training epoch
- Model completes one evaluation step
- Metrics are computed
- No multi-seed repetition

**Level 3: Multi-Seed Repetition**
- Experiment repeated across multiple seeds (typically 3–5)
- Mean and standard deviation reported
- No perturbation or controlled comparison

**Level 4: Controlled Perturbation Experiment**
- Clean and perturbed conditions compared
- Perturbation manifest documented
- Multi-level diagnostics measured
- Multi-seed repetition

**Level 5: Theory-Aligned Validation**
- Experiment derived from explicit theoretical hypothesis
- Theoretical quantities connected to empirical diagnostics
- Theory–experiment reconciliation documented
- Cross-dataset or cross-architecture validation (where applicable)

**Level 6: Manuscript-Ready Evidence**
- All Level 5 requirements met
- Reproducibility verified (independent reproduction or comprehensive documentation)
- Limitations documented
- Claim–evidence approval obtained
- Durable backup and traceability complete

### Research Gates (A–I)

**Gate A: Environment Verified**
- Python, PyTorch, PyG, PyGSD versions documented
- Package imports succeed
- Environment export available
- Status: ✅ Passed

**Gate B: Model API/Task Audit**
- Architecture constructors audited
- Forward-pass signatures audited
- Task compatibility verified
- Status: 🟡 Partial

**Gate C: Finite CPU Forward Pass**
- All candidate architectures produce finite outputs on CPU
- Forward-pass smoke tests documented
- Status: ⏳ Pending

**Gate D: Baseline Reproduction**
- Clean-graph baselines reproduced for each architecture
- Performance consistent with literature or prior runs
- Multi-seed variance documented
- Status: ⏳ Pending

**Gate E: Perturbation-Engine Validation**
- Perturbation functions implemented
- Perturbation manifests generated and verified
- Structural diagnostics measured
- Status: ⏳ Pending

**Gate F: Multi-Seed Robustness Campaign**
- Clean-training/perturbed-inference experiments completed
- Multi-seed repetition (5+ seeds)
- Four-level diagnostics measured
- Status: ⏳ Pending

**Gate G: Theory Linkage**
- Theoretical quantities connected to empirical diagnostics
- Theory–experiment reconciliation documented
- Hypothesis support/contradiction/qualification assessed
- Status: ⏳ Pending

**Gate H: Cross-Dataset Validation**
- Experiments repeated on synthetic (SDSBM) and real (Bitcoin) datasets
- Cross-dataset consistency assessed
- Dataset-specific limitations documented
- Status: ⏳ Pending

**Gate I: Manuscript Evidence Lock**
- All evidence classified as Level 6
- Reproducibility verified
- Claim–evidence approval obtained
- Manuscript-ready artifacts finalized
- Status: ⏳ Pending

### Gate Transition Protocol

**Code execution alone does not pass a gate.**

Gate passage requires:

1. Technical completion (code runs, outputs produced)
2. Validation (outputs are correct, not just finite)
3. Documentation (experiment records, configurations, manifests)
4. Interpretation (results connected to research questions)
5. Approval (researcher and advisor agreement, where applicable)

Gates may be:

- **Blocking:** Must pass before proceeding to next gate
- **Non-blocking:** May proceed with caution, but gate must be resolved before manuscript submission

Current gate status: Gates A–B are blocking for Gate C. Gates C–E are blocking for Gate F. Gates F–H are blocking for Gate I.

These blocking relationships govern empirical evidence progression in the SOL experimental track. They do not block literature verification, proof development, architecture-specific theoretical analysis, or unrelated iMac documentation work. An incomplete empirical gate blocks promotion of the affected empirical claim but does not automatically block unrelated theoretical or literature work. Gate G (Theory Linkage) requires coordination between the theory and experiment tracks.

---

## 12. Theory–Experiment Linkage

### Linkage Requirements

Each promoted experiment (Level 4+) must document:

**Theoretical side:**
- Theoretical quantity (e.g., operator Lipschitz constant, spectral gap, contraction rate)
- Theoretical prediction (e.g., "smaller spectral gap implies larger representation drift")
- Assumptions (e.g., "operator is Hermitian," "perturbation is small")

**Empirical side:**
- Empirical proxy or direct measurement (e.g., measured spectral gap, measured embedding drift)
- Measurement method (e.g., Lanczos approximation, Frobenius norm)
- Measurement limitations (e.g., approximation tolerance, computational constraints)

**Connection:**
- Expected trend (e.g., "as spectral gap decreases, embedding drift should increase")
- Observed trend (e.g., "embedding drift increased by 15% when spectral gap decreased by 20%")
- Agreement status: support, contradiction, qualification, or inconclusive

**Follow-up:**
- Implementation concerns (e.g., "measurement may be affected by numerical precision")
- Assumption failures (e.g., "operator is not Hermitian due to directed edges")
- Required follow-up (e.g., "verify measurement on synthetic graph with known properties")

### Reconciliation Protocol

When theory and experiment diverge:

1. **Check implementation:** Verify code correctness, numerical stability, edge cases
2. **Check assumptions:** Verify theoretical assumptions hold for the experimental setting
3. **Check measurement:** Verify empirical diagnostics measure the intended theoretical quantity
4. **Refine theory:** If implementation and measurement are correct, theory may need refinement
5. **Refine experiment:** If theory is sound, experiment may need redesign
6. **Document divergence:** If reconciliation is not immediate, document the divergence and plan follow-up

### Document Boundary

This document records the empirical side of the linkage. Mathematical validity of theoretical quantities, proofs, and derivations remains governed by the theory track and Phase 0–12 framework.

---

## 13. Manuscript Eligibility and Traceability

### Manuscript-Eligible Evidence Requirements

Evidence is manuscript-eligible (Level 6) only if it includes:

**Reproducibility:**
- Reproducible code (version-controlled, documented)
- Fixed configuration (no manual parameter tuning without documentation)
- Recorded seeds (all randomness sources)
- Validated split (leakage checks passed)
- Validated perturbation manifest (verification passed)

**Uncertainty:**
- Multi-seed repetition (5+ seeds for production claims)
- Mean and standard deviation reported
- Raw per-seed results available

**Diagnostics:**
- Operator or structural diagnostics appropriate to the claim
- Multi-level diagnostics where applicable (structural, operator, spectral, representation, task)

**Linkage:**
- Theory or literature linkage documented
- Hypothesis support/contradiction/qualification assessed

**Documentation:**
- Limitation statement (what the evidence does and does not prove)
- Commit hash (exact code version)
- Durable backup (not SOL scratch alone)
- Experiment record (complete metadata)
- Evidence classification (Level 6 confirmed)

### Legacy Evidence Exclusion

**Explicit statements:**

1. **Legacy Cora results are not manuscript evidence.**  
   Results from the pre-reorganization repository phase are excluded from the current manuscript unless independently reconstructed and validated under the current protocol.

2. **Legacy patterns may be independently reconstructed only under the current validation protocol.**  
   If a legacy script or notebook contains a useful pattern (e.g., perturbation function, evaluation loop), it may be adapted for current use only if:
   - The pattern is extracted and rewritten in the current codebase
   - The pattern is validated under the current protocol (gates, evidence levels, diagnostics)
   - The pattern is documented with current experiment records
   - The pattern produces new results, not copied legacy results

3. **Old performance numbers must not be copied into current results.**  
   Accuracy values, F1 scores, or other metrics from legacy experiments must not appear in current result tables, figures, or text unless the experiments are independently reproduced.

4. **Exploratory output must be promoted through the evidence levels before citation.**  
   Preliminary results, debugging output, or exploratory findings must progress through evidence levels 0–6 before being cited in the manuscript.

---

## 14. Scientific Safeguards

### Prohibited Interpretations

This methodology explicitly prohibits:

1. **Equating accuracy decline with mathematical stability**  
   Accuracy is a task-level metric. Stability is a mathematical property of operators or representations. Accuracy decline may be caused by instability, but it is not itself proof of instability.

2. **Conflating eigenspace stability with trained-network output stability**  
   Eigenvalue or eigenvector changes measure properties of the graph operator. Trained-network output stability depends on learned parameters, nonlinearities, and task-specific mappings. Eigenspace stability is a necessary but not sufficient condition for output stability.

3. **Treating equal edge budgets as equal operator perturbations**  
   Flipping 10% of edges produces different operator perturbations for different architectures. Operator norm changes must be measured and reported, not assumed equal.

4. **Transferring unsigned conclusions directly to signed or directed architectures**  
   Results from GCN (unsigned, undirected) do not automatically transfer to MSGNN (signed, directed). Each architecture must be evaluated independently.

5. **Generalizing from one seed or one dataset**  
   One seed provides debugging evidence only. One dataset provides architecture-specific evidence only. Multi-seed and cross-dataset validation are required for general claims.

6. **Treating constructor success as model validation**  
   A model that constructs without errors has not been validated. Forward-pass correctness, training correctness, convergence, and task performance must be verified independently.

7. **Using legacy Cora results as manuscript evidence**  
   Legacy results are excluded unless independently reconstructed under the current protocol.

8. **Treating CRISP-DM completion as theorem verification**  
   Following CRISP-DM steps ensures empirical rigor but does not prove mathematical theorems. Theorem verification requires formal proof, not empirical validation.

9. **Claiming causality from correlation alone**  
   Correlation between two measured quantities (e.g., spectral gap and accuracy) does not prove causation. Causal claims require controlled experiments, ablation studies, or theoretical justification.

10. **Hiding negative or contradictory findings**  
    Experiments that contradict hypotheses are scientifically valuable. Negative results must be documented and reported, not discarded.

11. **Interpreting unexpected results before checking code and data**  
    If results are surprising, verify code correctness and data integrity before proposing scientific explanations.

---

## 15. Adapted Empirical Lifecycle

### CRISP-DM-Informed Mapping

This project's empirical lifecycle is informed by selected CRISP-DM principles:

**Research Understanding:**  
Empirical questions are inherited from the Phase 0–12 framework. Research questions, hypotheses, and theoretical motivations are defined before empirical work begins.

**Data Understanding:**  
Signed and directed graph structure, task labels, leakage risks, and dataset limitations are analyzed. Structural properties relevant to the research question are documented.

**Data Preparation:**  
Deterministic preprocessing, train/validation/test splits, and perturbation manifests are created. Reproducibility and traceability are ensured through version control and seed management.

**Modeling:**  
Validated architecture and task implementations are used. Staged validation (constructor, forward pass, training, baseline reproduction) ensures correctness before perturbation experiments.

**Evaluation:**  
Four-level diagnostics (structural, operator, spectral, representation/task) are measured. Multi-seed repetition and uncertainty quantification are required. Matched comparisons (clean vs. perturbed) are performed.

**Iteration:**  
Hypothesis revision, implementation correction, and scope refinement are performed based on evaluation results. Theory–experiment reconciliation guides iteration.

**Research Delivery:**  
Reproducible repository artifacts (code, configurations, manifests, experiment records) and manuscript-ready evidence (Level 6) are produced.

### Methodological Clarification

**This mapping is CRISP-DM-informed but is not the project's primary scientific methodology.**

- CRISP-DM provides a useful framework for organizing empirical work
- CRISP-DM does not govern mathematical theory development
- CRISP-DM does not replace the Phase 0–12 framework
- CRISP-DM does not substitute for theorem verification

The empirical lifecycle is one component of the larger theory-driven research methodology.

---

## Document Boundary

### Governance Summary

- **Phase 0–12 framework** governs the full project: literature verification, mathematical development, architecture specialization, perturbation theory, computational validation, theory–experiment reconciliation, claim approval, and manuscript synthesis.

- **Theory track** governs mathematical development: operator definitions, stability proofs, perturbation bounds, contraction analysis, and theoretical predictions.

- **This document** governs the empirical validation lifecycle: data understanding, preparation, modeling, evaluation, iteration, and reproducible research delivery.

- **Future experiment-record templates** will govern individual experiment documentation: field schemas, metadata requirements, and storage conventions.

### Document Status

This is a researcher-adopted working document. It reflects current best practices and may be refined as the project progresses. Major changes will be documented and version-controlled.

---

*Document created: 2026-07-17*  
*Governing framework: Phase 0–12 master research workflow*  
*Role: Subordinate empirical-methodology guide*
