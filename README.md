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

## Methodology

This project follows a **theory-driven, reproducible computational research methodology** governed scientifically by the **Phase 0–12 master workflow**. The research proceeds through parallel mathematical and empirical tracks:

- **Mathematical track:** Operator definitions, stability proofs, perturbation bounds, and theoretical predictions
- **Empirical track:** Controlled experiments, multi-level diagnostics, and theory–experiment reconciliation

Adapted CRISP-DM principles support the empirical experiment lifecycle (data understanding, preparation, modeling, evaluation, iteration, and delivery) but do not govern literature verification, theorem development, architecture-specific theory, or theory–experiment reconciliation by themselves.

### iMac–SOL Synchronized Workflow

```
iMac: literature, theory, hypothesis, and experiment specification
                         ↓
SOL: implementation validation and controlled computation
                         ↓
iMac: interpretation and theory–experiment reconciliation
                         ↓
Decision: stop, revise, repeat, promote, narrow, or expand
```

Research and experiments proceed through short synchronized cycles rather than completing all experiments before documentation. The SOL evidence packet returns to the iMac documentation repository for interpretation, theory–experiment reconciliation, and selection of the next controlled work unit. Disagreement between theory and experiment is preserved honestly rather than forced into agreement and may be classified as support, contradiction, qualification, inconclusive evidence, implementation concern, or assumption failure.

### Key Scientific Safeguards

- Constructor success establishes only importability, constructor compatibility, finite initialized parameters, and CPU instantiation; it does not establish model validity.
- Legacy Cora results are excluded from manuscript evidence unless independently reconstructed and validated under the current protocol.
- Accuracy decline is an empirical task-level observation, not proof of mathematical instability.
- Equal edge-level perturbation budgets do not imply equal model-specific operator perturbations.
- SSSNET and SGCN are not automatically interchangeable; the signed-control decision remains open.
- Publication is a target, not a guaranteed outcome.
- See `notes/empirical_validation_methodology.md` for the complete empirical safeguards and evidence requirements.

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
notes/                                      # Environment and repository-governance records
notes/empirical_validation_methodology.md   # Active SOL-side empirical validation methodology
scripts/                                    # Verified active utility and smoke-test scripts
archive/legacy_exploration/                 # Preserved legacy work
README.md                                   # Active project overview
```

**Note:** Experiment-template and configuration directories do not yet exist.

### Active Empirical Methodology

**`notes/empirical_validation_methodology.md`** is the active SOL-side empirical validation methodology. It is:

- Subordinate to the Phase 0–12 master guideline maintained in the iMac documentation repository
- Responsible for experiment design discipline, signed/directed data preparation, perturbation protocols, diagnostics, uncertainty quantification, evidence levels, and reproducibility
- The operational guide for all substantive experiments (Evidence Level 3+)

### Legacy Methodology Status

**`archive/legacy_exploration/GNN_CRISP_DM_Methodology.md`** is preserved only as historical documentation. It:

- Reflects the earlier unsigned Cora and edge-deletion-centered project stage
- Presents CRISP-DM too broadly for the current project
- Is not active scientific guidance
- Is not manuscript evidence
- Must not be used to justify current signed/directed experiments

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

1. Create experiment-record and configuration templates.
2. Complete the model forward-API and task audit (Gate B).
3. Run finite CPU forward-pass smoke tests (Gate C).
4. Validate baseline task compatibility and reproduce approved signed/directed baselines (Gate D).
5. Begin perturbation-engine development only after Gates B and C pass; validate perturbation functions and canonical manifests before robustness experiments (Gate E).

No forward-pass validation, baseline reproduction, perturbation engine, robustness campaign, or theory-aligned empirical validation is represented as complete at this stage.

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
