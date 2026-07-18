# Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations

**Researcher:** Sileshi T. Hirpa  
**Institution:** Arizona State University  
**Course:** STP 499 — Individualized Instruction  
**Faculty Advisor:** Dr. Yixuan He

This repository serves as an active experimental implementation repository for a researcher-directed, advisor-aligned project investigating the structural sensitivity of signed and directed Graph Neural Networks (GNNs). The complete design and results have not yet been formally approved by the advisor.

---

## Research Objective

This project studies how signed and directed GNNs respond to controlled structural perturbations, including:

- **Sign flips** — reversing edge polarity in signed graphs
- **Direction reversals** — reversing edge orientation in directed graphs  
- **Edge deletion** — removing edges from the graph structure

The work integrates:

- General graph-operator stability theory
- Architecture-specific analysis
- Controlled robustness experiments
- Reproducible experiment-to-manuscript traceability

---

## Working Research Pipeline

```
verified literature
  → general operator framework
    → architecture specialization
      → controlled baseline
        → structural perturbation
          → operator and representation diagnostics
            → theory–experiment reconciliation
              → manuscript evidence
```

Theory and experiments develop in parallel throughout this pipeline.

---

## Candidate Architectures

- **MSGNN** — signed and directed magnetic architecture
- **MagNet** — directed magnetic architecture
- **SSSNET** — available sign-focused architecture
- **SGCN** — separate signed-GCN family under literature and implementation review
- **GCN** — debugging or unsigned-control role only

**Note:** SSSNET is not automatically equivalent to SGCN. The final signed-control architecture selection remains open.

---

## Candidate Datasets

- **SDSBM** — controlled synthetic validation
- **Bitcoin-Alpha** — signed trust/rating network
- **Bitcoin-OTC** — signed trust/rating network
- **Cora** — legacy debugging work only; not current manuscript evidence

**Note:** The Bitcoin datasets are signed trust/rating networks and are not direct fraud-label datasets.

---

## Current Verified Status

### Computational Environment
- Python 3.11.15
- PyTorch 2.12.0+cu130
- PyG 2.7.0
- PyGSD 1.1.1

### Verified Capabilities
- MSGNN, MagNet, and SSSNET imports verified
- CPU constructor compatibility verified
- Initialized parameters verified finite
- Environment documentation exists
- Constructor smoke-test script exists

### What This Proves
The current evidence establishes only:
- Environment availability
- Importability
- Constructor compatibility
- Finite initialized parameters
- CPU instantiation

### What This Does Not Yet Prove
The current evidence does **not** yet establish:
- Forward-pass correctness
- Training correctness
- Convergence
- Accuracy
- Robustness
- Stability
- Perturbation correctness
- Theory–experiment agreement
- Final architecture suitability

---

## Repository Governance

- **`main`** — the active researcher-controlled branch
- **`legacy-exploration`** — preserves the audited pre-reorganization repository state
- **`archive/legacy_exploration/`** — contains preserved unsigned Cora and exploratory artifacts retained for historical reference and possible controlled pattern reuse
- Legacy artifacts are excluded from manuscript evidence unless independently reconstructed and validated under the current protocol

---

## Current Active Repository Structure

```
notes/                          # Environment and repository-governance records
scripts/                        # Verified active utility and smoke-test scripts
archive/legacy_exploration/     # Preserved legacy work
README.md                       # Active project overview
GNN_CRISP_DM_Methodology.md    # Methodology document pending review/refactoring
```

**Note:** Experiment-template and configuration directories do not yet exist.

---

## Research Gates

| Gate | Description | Status |
|------|-------------|--------|
| **Gate A** | Environment verified | ✅ Passed |
| **Gate B** | Model API audit | 🟡 Partial |
| **Gate C** | Forward-pass smoke tests | ⏳ Pending |
| **Gate D** | Controlled baselines | ⏳ Pending |
| **Gate E** | Perturbation generators | ⏳ Pending |
| **Gate F** | Multi-seed robustness | ⏳ Pending |
| **Gate G** | Theory connected | ⏳ Pending |
| **Gate H** | Cross-dataset validation | ⏳ Pending |
| **Gate I** | Manuscript evidence lock | ⏳ Pending |

---

## Immediate Ordered Work

1. Review repository governance and methodology documentation
2. Create experiment-record templates
3. Complete architecture/task/API audit
4. Run controlled forward-pass smoke tests
5. Reproduce signed/directed baselines
6. Implement and validate perturbation functions
7. Begin manuscript-eligible experiments only after earlier gates pass

---

## Reproducibility and Storage

- **SOL scratch is temporary** and is never the only copy
- Code and documentation are synchronized through GitHub
- Durable backup and environment export are maintained outside scratch
- Production results will require:
  - Configurations
  - Seeds
  - Logs
  - Metrics
  - Commit hashes
  - Limitations
  - Experiment records

---

## Evidence Disclaimer

**This repository currently contains verified computational infrastructure and preliminary governance documentation. It does not yet contain manuscript-approved robustness results.**

---

*Last updated: July 2026*
