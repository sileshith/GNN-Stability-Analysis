---
experiment_id: "<EXP-YYYYMMDD-NNN-short-description>"
title: "<Descriptive experiment title>"
date: "<YYYY-MM-DD>"
researcher: "<Researcher name>"
repository: "<repository-name>"
branch: "<branch-name>"
commit: "<full-40-character-commit-hash>"
machine: "<SOL|local|other>"
sol_job_id: "<SLURM-job-ID-if-applicable>"
phase: "<Phase-0-12-identifier>"
gate: "<Gate-A-I-identifier>"
evidence_level: "<0|1|2|3|4|5|6>"
status: "<pending|running|completed|failed|inconclusive>"
experiment_type: "<smoke-test|baseline|perturbation|robustness|theory-validation|other>"
model: "<MSGNN|MagNet|SSSNET|SGCN|GCN|other>"
dataset: "<SDSBM|Bitcoin-Alpha|Bitcoin-OTC|Cora|other>"
task: "<node-classification|link-prediction|edge-classification|other>"
perturbation: "<none|sign-flip|direction-reversal|edge-deletion|other>"
record_version: "1.0"
---

# Experiment Record — <experiment_id>

---

## 1. Research Purpose

### Research Question
<State the clear, answerable research question this experiment addresses>

### Explicit Hypothesis
<State the testable prediction about expected behavior>

### Literature Motivation
<Cite relevant literature that motivates this experiment>

### Theoretical Motivation
<Describe the theoretical framework or prediction being tested>

### Why This Experiment Is Necessary Now
<Explain why this experiment is the appropriate next step in the research workflow>

### Evidence Maturity Target
<Specify the intended evidence level: 0, 1, 2, 3, 4, 5, or 6>

### Relevant Phase 0–12 Phase
<Identify which phase of the master workflow this experiment supports>

### Relevant Gate
<Identify which gate (A–I) this experiment contributes to>

---

## 2. Acceptance and Interpretation Criteria

**Note:** Acceptance criteria should be written before substantive execution whenever feasible.

### Implementation Acceptance Criteria
<What must be true about the code for the implementation to be considered valid?>

### Data and Split Acceptance Criteria
<What must be true about the data and splits for them to be considered valid?>

### Perturbation Acceptance Criteria
<What must be true about the perturbation for it to be considered correctly applied?>

### Empirical Acceptance Criteria
<What empirical conditions must be met for the experiment to be considered successful?>

### Conditions for Support
<Under what conditions does this experiment support the hypothesis?>

### Conditions for Contradiction
<Under what conditions does this experiment contradict the hypothesis?>

### Conditions for Qualification
<Under what conditions does this experiment partially support the hypothesis with identified boundary conditions?>

### Conditions for Inconclusive Status
<Under what conditions is the experiment inconclusive?>

### Stopping or Promotion Criteria
<When should this experiment be stopped, revised, repeated, or promoted?>

---

## 3. Evidence-Level Documentation

### Evidence Level Classification

- [ ] **Level 0:** Environment/Import Verification
- [ ] **Level 1:** Constructor or Forward-Pass Smoke Verification
- [ ] **Level 2:** Single Functional Run
- [ ] **Level 3:** Multi-Seed Repetition
- [ ] **Level 4:** Controlled Perturbation Experiment
- [ ] **Level 5:** Theory-Aligned Validation
- [ ] **Level 6:** Manuscript-Ready Evidence

### Reduced Record for Levels 0–2

**Objective:**  
<What was being tested?>

**Environment:**  
<Python, PyTorch, PyG, PyGSD versions>

**Code Version:**  
<Commit hash>

**Command:**  
<Exact command or script executed>

**Result:**  
<Success, failure, or observation>

**Warning or Failure:**  
<Any warnings or failures encountered>

**Evidence Limitation:**  
<What this result does and does not establish>

**Note:** The remaining sections become mandatory before promotion to Level 3 or higher.

---

## 4. Environment and Code Provenance

### Repository and Version Control
- **Repository:** <repository-name>
- **Git Branch:** <branch-name>
- **Full Commit Hash:** <full-40-character-commit-hash>
- **Working Tree Status Before Execution:** <clean|modified|untracked-files>

### Python Environment
- **Python Version:** <X.Y.Z>
- **PyTorch Version:** <X.Y.Z+cuXXX>
- **PyG Version:** <X.Y.Z>
- **PyGSD Version:** <X.Y.Z>
- **Other Relevant Dependencies:** <list>

### Compute Environment
- **CPU/GPU:** <CPU-model|GPU-model>
- **CUDA Available:** <yes|no>
- **CUDA Version:** <X.Y>
- **SOL Partition or Execution Context:** <partition-name|local|other>

### Execution Details
- **Exact Command or SLURM Script:** <command-or-script-path>
- **SLURM Job ID:** <job-ID-if-applicable>
- **Start Time:** <YYYY-MM-DD HH:MM:SS>
- **End Time:** <YYYY-MM-DD HH:MM:SS>
- **Duration:** <HH:MM:SS>

---

## 5. Dataset and Data Preparation

### Dataset Identity
- **Dataset Name:** <SDSBM|Bitcoin-Alpha|Bitcoin-OTC|Cora|other>
- **Dataset Version or Source:** <version-or-URL>
- **Raw Data Location:** <path-or-reference>
- **Checksum or Immutable Reference:** <checksum-if-available>

### Structural Summary
- **Node Count:** <N>
- **Edge Count:** <E>
- **Positive Edge Count:** <E+>
- **Negative Edge Count:** <E->
- **Positive Edge Proportion:** <proportion>
- **Negative Edge Proportion:** <proportion>

### Degree Statistics (where applicable)
- **In-Degree Distribution:** <summary-or-path-to-figure>
- **Out-Degree Distribution:** <summary-or-path-to-figure>
- **Degree Distribution:** <summary-or-path-to-figure>

### Connectivity (where applicable)
- **Reciprocity:** <value-for-directed-graphs>
- **Connected Components:** <count>
- **Strongly Connected Components:** <count-for-directed-graphs>
- **Weakly Connected Components:** <count-for-directed-graphs>
- **Isolated Nodes:** <count>
- **Density:** <value>

### Features and Labels
- **Features:** <description-or-dimensionality>
- **Labels:** <description-or-class-count>
- **Task:** <node-classification|link-prediction|edge-classification>

### Preprocessing
- **Preprocessing Steps:** <list-of-transformations>
- **Feature Normalization:** <none|zero-mean-unit-variance|min-max|other>
- **Missing Feature Handling:** <method>

### Split Protocol
- **Split Protocol:** <random|stratified|temporal|other>
- **Split Seeds:** <seed-values>
- **Train Fraction:** <fraction>
- **Validation Fraction:** <fraction>
- **Test Fraction:** <fraction>
- **Train Size:** <count>
- **Validation Size:** <count>
- **Test Size:** <count>

### Split Validation
- **Class Stratification:** <yes|no|N/A>
- **Sign and Direction Balance:** <checked|not-applicable>
- **Reverse-Edge Leakage Check:** <passed|failed|not-applicable>
- **Message-Passing Leakage Check:** <passed|failed|not-applicable>
- **Negative Sampling Rule:** <rule-if-applicable>
- **Connectivity Check:** <passed|failed>
- **Temporal Considerations:** <description-if-applicable>

### Known Dataset Limitations
<List any known limitations, biases, or concerns about the dataset>

**Note:** Bitcoin-Alpha and Bitcoin-OTC are signed trust/rating networks, not automatically fraud-label datasets.

---

## 6. Architecture and Task Definition

### Architecture Identity
- **Architecture:** <MSGNN|MagNet|SSSNET|SGCN|GCN|other>
- **Package and Class:** <package.module.ClassName>
- **Constructor Arguments:** <dict-or-list-of-arguments>

### Forward Signature
- **Expected Tensor Inputs:** <x, edge_index, edge_weight, etc.>
- **Expected Output Semantics:** <node-embeddings|logits|probabilities|other>

### Task and Loss
- **Task:** <node-classification|link-prediction|edge-classification>
- **Loss Function:** <CrossEntropyLoss|BCEWithLogitsLoss|other>
- **Metrics:** <accuracy|F1|AUC|other>
- **Checkpoint Selection Rule:** <best-validation-loss|best-validation-accuracy|other>

### Clean-Training Protocol
- **Clean-Training Protocol:** <train-on-clean-graph|train-on-perturbed-graph|other>
- **Task Compatibility Status:** <verified|pending|failed>

### Validation Checklist

- [ ] Import verified
- [ ] Constructor verified
- [ ] Finite parameters verified
- [ ] Forward signature verified
- [ ] Finite forward output verified
- [ ] Loss compatibility verified
- [ ] Minimal functional run verified
- [ ] Baseline reproduced

**Note:** SSSNET and SGCN are not automatically interchangeable. The final signed-control architecture selection remains open.

---

## 7. Perturbation Specification

### Perturbation Type
- **Type:** <none|sign-flip|direction-reversal|edge-deletion|other>
- **Formal Transformation:**
  - Sign flip: (i, j, w) → (i, j, −w)
  - Direction reversal: (i, j, w) → (j, i, w)
  - Edge deletion: (i, j, w) → removed
  - Other: <specify>

### Eligible Edges
- **Eligible-Edge Definition:** <all-edges|positive-edges|negative-edges|training-edges|other>
- **Eligible Edge Count:** <count>

### Perturbation Budget
- **Requested Budget:** <percentage-or-count>
- **Denominator:** <total-edges|positive-edges|negative-edges|eligible-edges|other>
- **Requested Change Count:** <count>
- **Realized Change Count:** <count>

### Perturbation Seed and Manifest
- **Perturbation Seed:** <seed-value>
- **Canonical Manifest Path:** <path-to-manifest-file>
- **Manifest Checksum or Version:** <checksum-or-version>
- **Verification Method:** <how-manifest-was-verified>

### Structural Before/After Summary
- **Positive Edges Before:** <count>
- **Positive Edges After:** <count>
- **Negative Edges Before:** <count>
- **Negative Edges After:** <count>
- **Reciprocity Before:** <value-if-applicable>
- **Reciprocity After:** <value-if-applicable>
- **Degree Distribution Change:** <summary>

### Architecture-Specific Preprocessing
- **Operator Construction Settings:** <normalization|self-loops|charge|phase|other>
- **Normalization Convention:** <symmetric|row|column|none>
- **Self-Loop Convention:** <added|not-added|conditional>
- **Charge or Phase Parameters:** <values-for-magnetic-Laplacians>

### Critical Principle

**Equal edge-level budgets do not imply equal model-specific operator perturbations.**

Different architectures (GCN, MSGNN, MagNet) convert the same perturbed raw graph into different operators with different perturbation magnitudes.

### Optional Stretch Perturbations

- **Edge Rewiring:** <not-used|description-if-used>
- **Other:** <description-if-applicable>

**Note:** Edge rewiring is optional stretch work only and is not a required perturbation for the current project scope.

---

## 8. Training and Evaluation Configuration

### Randomness Sources
- **Initialization Seed:** <seed>
- **Split Seed:** <seed>
- **Synthetic Data Seed:** <seed-if-applicable>
- **Perturbation Seed:** <seed-if-applicable>
- **Negative Sampling Seed:** <seed-if-applicable>

### Training Hyperparameters
- **Number of Epochs:** <count>
- **Optimizer:** <Adam|SGD|other>
- **Learning Rate:** <value>
- **Weight Decay:** <value>
- **Hidden Dimensions:** <list-or-value>
- **Layers:** <count>
- **Dropout:** <value>

### Early Stopping and Checkpointing
- **Early Stopping:** <enabled|disabled>
- **Patience:** <epochs>
- **Checkpoint Rule:** <best-validation-loss|best-validation-accuracy|other>
- **Batch Settings:** <full-batch|mini-batch-size>

### Multi-Seed Repetition
- **Number of Repeated Seeds:** <1|3|5|other>
- **Seed Values:** <list-of-seeds>

### Clean Checkpoint
- **Clean Checkpoint Path:** <path-to-checkpoint>
- **Clean Validation Performance:** <metric-value>

### Evaluation Split
- **Evaluation Split:** <validation|test>
- **Evaluation Edge Count:** <count>

### Protocol Confirmation
- **Clean-Training/Perturbed-Inference:** <yes|no|N/A>
- **Frozen Parameters During Perturbed Inference:** <yes|no|N/A>
- **Recomputed Graph-Dependent Representations:** <yes|no|N/A>

---

## 9. Four-Level Diagnostics

### 9.1 Structural Diagnostics

**Perturbation Realization:**
- **Requested Budget:** <value>
- **Realized Count:** <value>
- **Eligible Edge Count:** <value>

**Sign and Direction:**
- **Positive Edges Before:** <count>
- **Positive Edges After:** <count>
- **Negative Edges Before:** <count>
- **Negative Edges After:** <count>
- **Sign Distribution Change:** <summary>
- **Reciprocity Before:** <value-if-applicable>
- **Reciprocity After:** <value-if-applicable>

**Degree Statistics:**
- **In-Degree Distribution Change:** <summary-if-applicable>
- **Out-Degree Distribution Change:** <summary-if-applicable>
- **Degree Distribution Change:** <summary>

**Connectivity:**
- **Connected Components Before:** <count>
- **Connected Components After:** <count>
- **Strongly Connected Components Before:** <count-if-applicable>
- **Strongly Connected Components After:** <count-if-applicable>
- **Isolated Nodes Before:** <count>
- **Isolated Nodes After:** <count>
- **Density Before:** <value>
- **Density After:** <value>

---

### 9.2 Operator Diagnostics

**Where mathematically and computationally appropriate:**

**Raw Adjacency:**
- **Raw Adjacency Frobenius Norm Change:** ||A_perturbed - A_clean||_F = <value>
- **Raw Adjacency Spectral Norm Change:** ||A_perturbed - A_clean||_2 = <value>

**Model-Specific Operator:**
- **Model-Specific Operator Frobenius Norm Change:** ||L_perturbed - L_clean||_F = <value>
- **Model-Specific Operator Spectral Norm Change:** ||L_perturbed - L_clean||_2 = <value>

**Operator Construction:**
- **Normalization Convention:** <symmetric|row|column|none>
- **Self-Loop Convention:** <added|not-added|conditional>
- **Charge or Phase Parameters:** <values-for-magnetic-Laplacians>
- **Complex vs. Real Representation:** <complex|real>

**Critical Warning:**  
Cross-architecture operator norms may not be directly comparable because model-specific operators can use different normalizations and scales.

---

### 9.3 Spectral or Subspace Diagnostics

**Where assumptions and computational feasibility permit:**

**Eigenvalue Analysis:**
- **Eigenvalue Drift:** <summary-or-values>
- **Eigengap Change:** Δ(λ₂ - λ₁) = <value>
- **Eigenvalue Distribution Change:** <summary-or-figure-path>

**Subspace Analysis:**
- **Principal-Angle or Subspace Drift:** <value-or-summary>
- **Davis–Kahan-Style Diagnostic:** <value-or-summary>

**Approximation:**
- **Approximation Method:** <Lanczos|Arnoldi|exact|other>
- **Tolerance:** <value-if-applicable>

**Clarifications:**

1. Complex-valued representations or operator entries do not imply complex eigenvalues when the analyzed operator is Hermitian.
2. Spectral diagnostics measure properties of the graph operator, not the trained neural network.
3. **Spectral diagnostics do not by themselves prove trained-network output stability.**

---

### 9.4 Representation and Task Diagnostics

**Representation Changes:**
- **Normalized Embedding Drift:** ||h_perturbed - h_clean|| / ||h_clean|| = <value>
- **Embedding Cosine Similarity:** <value>
- **Embedding Cluster Coherence:** <value-or-summary>

**Prediction Changes:**
- **Logit Drift:** <summary-or-value>
- **Prediction Disagreement:** <proportion-of-nodes/edges-with-different-predictions>
- **Confidence or Margin Change:** <summary>

**Task Performance:**
- **Absolute Task Degradation:** performance_clean - performance_perturbed = <value>
- **Relative Task Degradation:** (performance_clean - performance_perturbed) / performance_clean = <value>
- **Macro-F1:** <value-if-applicable>
- **Accuracy:** <value>
- **ROC-AUC:** <value-if-applicable>
- **Confusion Matrix:** <matrix-or-path-to-figure>

**Theoretical Connections:**
- **Empirical Lipschitz-Style Ratio:** ||output_perturbed - output_clean|| / ||input_perturbed - input_clean|| = <value>
- **Theoretical Bound:** <value-if-applicable>
- **Observed Error:** <value>
- **Bound-to-Observed-Error Ratio:** <value-if-applicable>

**Critical Statement:**  
**Accuracy decline is not itself a mathematical stability result.** It is an empirical observation that may be explained by mathematical stability properties, but the two must not be conflated.

---

## 10. Raw Results and Uncertainty

### Per-Seed Results
<Table or list of results for each seed>

| Seed | Clean Accuracy | Perturbed Accuracy | Absolute Degradation | Relative Degradation |
|------|----------------|--------------------|--------------------|---------------------|
| <seed1> | <value> | <value> | <value> | <value> |
| <seed2> | <value> | <value> | <value> | <value> |
| <seed3> | <value> | <value> | <value> | <value> |
| ... | ... | ... | ... | ... |

### Summary Statistics
- **Mean Clean Performance:** <value> ± <std>
- **Mean Perturbed Performance:** <value> ± <std>
- **Mean Absolute Degradation:** <value> ± <std>
- **Mean Relative Degradation:** <value> ± <std>

### Effect Size and Outliers
- **Effect Size:** <value-or-description>
- **Outliers:** <description-of-any-outliers>
- **Failed Seeds:** <count-and-description>
- **Missing Outputs:** <count-and-description>

### Statistical Testing (if applicable)
- **Confidence Intervals:** <values-if-justified>
- **Formal Statistical Test:** <test-name-if-justified>
- **Test Assumptions:** <assumptions-if-test-used>
- **p-value:** <value-if-applicable>
- **Interpretation Limitations:** <limitations-of-statistical-test>

### Seed Classification
- **One seed:** Debugging evidence only
- **Three seeds:** Pilot evidence
- **Five seeds:** Default production evidence

---

## 11. Validation and Failure Checks

### Expected Behavior Checklist

- [ ] Expected tensor shapes
- [ ] Expected tensor dtypes
- [ ] Finite tensors (no NaN or Inf)
- [ ] Deterministic behavior where expected
- [ ] Gradient flow where applicable
- [ ] Exact perturbation budget realized
- [ ] Untouched evaluation edges
- [ ] No reverse-edge leakage
- [ ] No message-passing leakage
- [ ] Logs captured
- [ ] Metrics saved
- [ ] Manifest saved
- [ ] Configuration saved
- [ ] Checkpoint saved (where applicable)
- [ ] Durable backup confirmed

### Warnings
<List any warnings encountered during execution>

### Failures
<List any failures encountered during execution>

### Implementation Concerns
<List any concerns about code correctness, numerical stability, or edge cases>

### Data Concerns
<List any concerns about data quality, preprocessing, or splits>

### Numerical Concerns
<List any concerns about numerical precision, overflow, underflow, or convergence>

### Unresolved Issues
<List any issues that remain unresolved and require follow-up>

---

## 12. Results

### Concise Result Summary
<One-paragraph summary of the main findings>

### Clean Metrics
<Table or list of metrics on clean graph>

### Perturbed Metrics
<Table or list of metrics on perturbed graph>

### Structural Findings
<Summary of structural diagnostic findings>

### Operator Findings
<Summary of operator diagnostic findings>

### Spectral Findings
<Summary of spectral diagnostic findings>

### Representation Findings
<Summary of representation diagnostic findings>

### Task Findings
<Summary of task diagnostic findings>

### Output Paths
- **Raw Output Directory:** <path>
- **Metric File:** <path>
- **Figure/Table Paths:** <list-of-paths>

---

## 13. Theory–Experiment Reconciliation

### Theoretical Quantity
<What theoretical quantity was this experiment designed to measure or test?>

### Empirical Proxy or Direct Measurement
<What empirical diagnostic was used to measure the theoretical quantity?>

### Expected Trend
<What trend did theory predict?>

### Observed Trend
<What trend was actually observed?>

### Agreement Status

Select one:

- [ ] **Support:** Observed trends match theoretical predictions
- [ ] **Contradiction:** Observed trends oppose theoretical predictions
- [ ] **Qualification:** Observed trends partially match with identified boundary conditions
- [ ] **Inconclusive:** Insufficient evidence, implementation concerns, or assumption violations prevent interpretation
- [ ] **Implementation Concern:** Results may be affected by code or numerical issues
- [ ] **Assumption Failure:** Theoretical assumptions do not hold for this experimental setting

### Explanation
<Explain the agreement status and any discrepancies>

### Alternative Explanations
<List any alternative explanations for the observed results>

### Required Follow-Up
<List any follow-up experiments, code checks, or theoretical refinements required>

**Note:** Theory must not be revised merely to force agreement with observed results. Disagreement is scientifically valuable and must be documented honestly.

---

## 14. Claim–Evidence Classification

For each proposed claim, document:

### Literature-Supported Claim
- **Exact Wording:** <claim-text>
- **Supporting Artifact:** <citation-or-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-claim>
- **Prohibited Overinterpretation:** <what-this-claim-does-not-establish>

### Theoretical Claim
- **Exact Wording:** <claim-text>
- **Supporting Artifact:** <theorem-or-derivation-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-claim>
- **Prohibited Overinterpretation:** <what-this-claim-does-not-establish>

### Empirical Claim
- **Exact Wording:** <claim-text>
- **Supporting Artifact:** <experiment-ID-or-result-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-claim>
- **Prohibited Overinterpretation:** <what-this-claim-does-not-establish>

### Methodological Decision
- **Exact Wording:** <decision-text>
- **Supporting Artifact:** <methodology-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-decision>
- **Prohibited Overinterpretation:** <what-this-decision-does-not-establish>

### Interpretation
- **Exact Wording:** <interpretation-text>
- **Supporting Artifact:** <experiment-or-theory-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-interpretation>
- **Prohibited Overinterpretation:** <what-this-interpretation-does-not-establish>

### Limitation
- **Exact Wording:** <limitation-text>
- **Supporting Artifact:** <experiment-or-methodology-reference>
- **Evidence Level:** <0-6>
- **Permitted Scope:** <scope-of-limitation>
- **Prohibited Overinterpretation:** <what-this-limitation-does-not-establish>

---

## 15. Manuscript and Documentation Linkage

### iMac Evidence Packet
- **iMac Evidence-Packet Location:** <path-in-iMac-documentation-repository>

### Literature and Theory References
- **Literature Note:** <reference-to-literature-note>
- **Theory Note:** <reference-to-theory-note>

### Decision Log
- **Decision-Log Entry:** <reference-to-decision-log-entry>

### Claim–Evidence Map
- **Claim–Evidence Map Entry:** <reference-to-claim-evidence-map>

### Manuscript Destination
- **Manuscript Section:** <section-name-or-number>
- **Proposed Figure or Table:** <figure-or-table-identifier>
- **Reproducibility Appendix Destination:** <appendix-section>

### Legacy Evidence
- **Legacy Evidence Used:** <yes|no>
- **If Yes, Independent Reconstruction Completed:** <yes|no|N/A>

---

## 16. Final Research Decision

### Decision

Select one:

- [ ] **Stop:** Experiment complete; no further work needed
- [ ] **Revise Implementation:** Code or numerical issues require correction
- [ ] **Revise Experimental Design:** Experimental protocol requires modification
- [ ] **Revisit Assumptions:** Theoretical or methodological assumptions require reconsideration
- [ ] **Repeat:** Repeat experiment with same configuration to verify reproducibility
- [ ] **Promote to More Seeds:** Increase seed count for stronger statistical confidence
- [ ] **Expand Budgets:** Increase perturbation severity or other experimental parameters
- [ ] **Add Dataset:** Repeat experiment on additional dataset for cross-dataset validation
- [ ] **Add Architecture:** Repeat experiment with additional architecture for cross-architecture validation
- [ ] **Narrow Scope:** Reduce experimental scope to focus on specific aspect
- [ ] **Manuscript Eligible:** Promote to manuscript-ready evidence (Level 6)
- [ ] **Archive as Negative or Inconclusive Result:** Preserve as scientifically valuable negative finding

### Rationale
<Explain the decision and its justification>

### Next Controlled Work Unit
<Describe the next planned work unit>

### Responsible Repository
<iMac|SOL|both>

### Required Gate Before Proceeding
<Gate-A-I-identifier-or-N/A>

---

## 17. Evidence Limitations

### What This Experiment Establishes
<Explicit statement of what this experiment proves or demonstrates>

### What This Experiment Does Not Establish
<Explicit statement of what this experiment does not prove or demonstrate>

### Generalization Limits
<Limits on generalization to other datasets, architectures, or settings>

### Dataset Limits
<Limits specific to the dataset used>

### Architecture Limits
<Limits specific to the architecture used>

### Statistical Limits
<Limits on statistical inference (e.g., sample size, assumptions)>

### Theoretical Limits
<Limits on theoretical interpretation>

### Implementation Limits
<Limits due to implementation choices or constraints>

---

## 18. Artifact Checklist

- [ ] Configuration file (`config_template.yml`)
- [ ] Command or SLURM script
- [ ] Stdout log
- [ ] Stderr log
- [ ] Metrics file (machine-readable)
- [ ] Perturbation manifest (if applicable)
- [ ] Checkpoint (if applicable)
- [ ] Figure or table (if applicable)
- [ ] Experiment record (this file)
- [ ] Commit hash recorded
- [ ] Durable backup confirmed
- [ ] iMac evidence packet transferred
- [ ] Claim–evidence map updated

---

## 19. Sign-Off

### Researcher Review
- **Researcher:** <name>
- **Review Date:** <YYYY-MM-DD>
- **Review Status:** <approved|revisions-required>

### Mentor Decision (Optional)
- **Mentor:** <name-if-applicable>
- **Decision Date:** <YYYY-MM-DD>
- **Decision:** <approved|revisions-required|N/A>

**Note:** Advisor approval is not required for routine experimental records. Include advisor review only where scientifically necessary (e.g., manuscript-eligible evidence, major methodological decisions).

### Evidence Level Confirmation
- **Evidence Level Confirmed:** <0|1|2|3|4|5|6>

### Gate Status
- **Gate Status:** <passed|partial|pending|failed>

### Manuscript Eligibility
- **Manuscript Eligible:** <yes|no>

### Final Review Date
- **Final Review Date:** <YYYY-MM-DD>

---

*Template version: 1.0*  
*Last updated: 2026-07-17*  
*Governing methodology: notes/empirical_validation_methodology.md*
