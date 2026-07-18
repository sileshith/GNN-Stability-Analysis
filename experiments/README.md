# Experiment Management

This directory governs SOL-side experiment records, configurations, manifests, logs, metrics, and evidence traceability for the signed and directed GNN robustness project.

---

## Authority Hierarchy

- The **Phase 0–12 master guideline** governs the complete scientific project
- **`notes/empirical_validation_methodology.md`** governs the empirical lifecycle
- **This directory** supplies run-level operational templates
- The **iMac documentation repository** remains authoritative for literature verification, theory, research decisions, claim–evidence synthesis, and manuscript interpretation

---

## Synchronized Workflow

```
iMac: question, literature, theory, and experiment specification
                         ↓
SOL: configuration, implementation validation, and controlled run
                         ↓
SOL: evidence packet, logs, metrics, manifests, and commit reference
                         ↓
iMac: interpretation and theory–experiment reconciliation
                         ↓
Decision: stop, revise, repeat, promote, narrow, or expand
```

Disagreement between theory and experiment must be recorded honestly and must not be forced into agreement.

---

## Directory Structure

```
experiments/
├── README.md
├── templates/
│   ├── experiment_record_template.md
│   └── config_template.yml
├── records/
│   └── EXP-YYYYMMDD-NNN-short-description.md
├── configs/
│   └── EXP-YYYYMMDD-NNN-short-description.yml
├── manifests/
├── logs/
└── outputs/
```

**Note:** Only `README.md` and the two templates are created in this work unit. The `records/`, `configs/`, `manifests/`, `logs/`, and `outputs/` directories are planned conventions and must not be created until needed.

---

## Experiment ID Format

```
EXP-YYYYMMDD-NNN-short-description
```

Where:
- `YYYYMMDD` is the date
- `NNN` is a zero-padded sequential number (001, 002, etc.)
- `short-description` is a brief hyphen-separated identifier

Example: `EXP-20260717-001-msgnn-bitcoin-alpha-sign-flip`

---

## Documentation Tiers

### Evidence Levels 0–2

May use a reduced traceable record containing:

- Objective
- Date
- Environment versions
- Code version or full commit hash
- Command or script executed
- Result or observation
- Warning or failure (if applicable)
- Evidence limitation stating what the result does and does not establish

These reduced records are appropriate for environment checks, constructor or forward-pass smoke tests, and single functional runs.

### Evidence Level 3 or Higher

Requires a complete experiment record and machine-readable configuration:

- Must use the experiment-record template (`experiment_record_template.md`)
- Must use the configuration template (`config_template.yml`)
- Must preserve raw per-seed results where applicable
- Must document all fields required by the empirical validation methodology

### Evidence Level 6

Requires the additional manuscript-eligibility conditions specified in `notes/empirical_validation_methodology.md` Section 13, including:

- Reproducibility verification
- Multi-seed repetition (5+ seeds for production claims)
- Theory or literature linkage
- Limitation statement
- Durable backup
- Claim–evidence approval

---

## Experiment Lifecycle

1. **Define the research question and hypothesis** on the iMac documentation repository
2. **Assign the experiment ID** following the format above
3. **Create the configuration** from `templates/config_template.yml`
4. **Create the experiment record** from `templates/experiment_record_template.md`
5. **Validate code, data, splits, and model API** before interpreting results
6. **Execute the controlled run** on SOL
7. **Record logs, manifests, metrics, warnings, failures, and limitations**
8. **Commit code and lightweight records** to Git
9. **Back up important evidence** outside SOL scratch to durable storage
10. **Return the evidence packet** to the iMac for reconciliation and the next decision

---

## Key Scientific Safeguards

- **Constructor success is not forward-pass validation**
- **Forward-pass success is not baseline validation**
- **One seed is debugging evidence only**
- **Accuracy decline is not mathematical stability**
- **Equal edge budgets do not imply equal operator perturbations**
- **Legacy Cora outputs are not manuscript evidence**
- **Negative and contradictory findings must be preserved**
- **No result is manuscript-ready merely because code executed**

See `notes/empirical_validation_methodology.md` Section 14 for the complete list of prohibited interpretations.

---

## Template Usage

### For Evidence Levels 0–2

Use the reduced-record section of `experiment_record_template.md` (Section 3). The remaining sections are optional but recommended for traceability.

### For Evidence Level 3+

Complete all mandatory sections of `experiment_record_template.md` and `config_template.yml`. Ensure that:

- Research purpose and acceptance criteria are defined before execution
- All randomness sources are seeded and documented
- Perturbation manifests are saved and verified
- Raw per-seed results are preserved
- Theory–experiment reconciliation is documented
- Evidence limitations are explicitly stated

### For Evidence Level 6

In addition to Level 3+ requirements, ensure:

- Multi-seed repetition (5+ seeds)
- Cross-dataset validation where applicable
- Theory or literature linkage documented
- Claim–evidence approval obtained
- Durable backup confirmed
- Manuscript destination specified

---

## Storage and Backup

**SOL scratch is temporary and cannot be the sole storage location for important evidence.**

All experiment records, configurations, manifests, checkpoints, and results must be:

- Committed to Git (for code, configurations, and lightweight records)
- Backed up to durable storage (for data, checkpoints, large results)
- Synchronized with GitHub (for code and documentation)
- Transferred to the iMac documentation repository (for evidence packets and interpretation)

---

## Evidence Packet Contents

Each evidence packet returned to the iMac should contain:

- Experiment ID
- Experiment record (completed)
- Configuration file
- Commit hash
- Logs (stdout, stderr)
- Metrics (machine-readable)
- Perturbation manifest (if applicable)
- Checkpoint location (if applicable)
- Warnings and failures
- Limitations
- Proposed interpretation
- Proposed next decision

---

*Last updated: 2026-07-17*  
*Governing methodology: notes/empirical_validation_methodology.md*  
*Master framework: Phase 0–12 research workflow*
