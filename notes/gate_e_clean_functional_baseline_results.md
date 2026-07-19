---
title: "Gate E — Clean Functional-Baseline Results"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate E execution results and decision record"
status: "completed"
execution_date: "2026-07-18"
gate_decision: "PASS"
---

# Gate E — Clean Functional-Baseline Results

**Execution Date:** 2026-07-18  
**Gate Decision:** **PASS**

---

## 1. Purpose and Scope

Gate E evaluated whether PyGSD 1.1.1 `MSGNN_link_prediction` and directed `SSSNET_link_prediction` can complete reproducible multi-epoch training on a fixed clean signed-directed dataset across three initialization seeds, maintain finite optimization behavior, demonstrate a minimal validation-loss learning signal, and execute held-out test evaluation without data-contract violations.

**Scope Boundaries:**

- Clean graph only — no perturbations
- One fixed synthetic signed-directed dataset
- One fixed train/validation/test split
- Three initialization seeds per model
- CPU execution
- No architecture ranking or superiority claims
- No robustness or stability conclusions

---

## 2. Locked Configuration

### Environment

- Python: `3.11.15`
- PyTorch: `2.12.0+cu130`
- PyG: `2.7.0`
- PyGSD: `1.1.1`
- NumPy: `1.26.4`
- NetworkX: `3.6.1`
- SciPy: `1.17.1`
- Device: `cpu`

### Dataset Construction

- Task: `four_class_signed_digraph`
- Generator: PyGSD 1.1.1 `SDSBM` (Signed Directed Stochastic Block Model)
- Nodes: `N = 400`
- Communities: `K = 4`
- Edge probability: `p = 0.05`
- Size ratio: `1.5`
- Eta: `0.1` (direct SDSBM argument)
- Gamma: `0.1` (used to construct the locked meta-graph matrix F, not passed directly to SDSBM)
- Generation seed: `0`
- Split seed: `0`
- Validation probability: `0.15`
- Test probability: `0.15`

**Locked Meta-Graph Matrix F:**

```
F = [
  [ 0.5,  0.1, -0.1,  0.1],
  [ 0.9,  0.5, -0.1, -0.5],
  [-0.9, -0.9,  0.5, -0.9],
  [-0.9, -0.5, -0.1,  0.5]
]
```

### Model Configurations

**MSGNN_link_prediction:**

```python
{
    "hidden": 16,
    "q": 0.25,
    "K": 2,
    "label_dim": 4,
    "activation": True,
    "trainable_q": False,
    "layer": 2,
    "dropout": 0.5,
    "normalization": "sym",
    "cached": False,
    "conv_bias": True,
    "absolute_degree": True,
    "num_features": 2  # from shared feature width
}
```

**SSSNET_link_prediction (directed=True):**

```python
{
    "hidden": 16,
    "nclass": 4,
    "dropout": 0.5,
    "hop": 2,
    "fill_value": 0.5,
    "directed": True,
    "bias": True,
    "nfeat": 2  # from shared feature width
}
```

### Optimization Protocol

- Optimizer: `Adam(lr=0.01, weight_decay=5e-4)`
- Loss: `NLLLoss()`
- Maximum epochs: `300`
- Early-stopping patience: `30` epochs
- Minimum improvement: `1e-4`
- Minimum epochs: `10`
- Initialization seeds: `[0, 1, 2]`

### Shared Features

Features computed once from the training graph using PyGSD's `in_out_degree` method and shared across all models and seeds.

---

## 3. Shared Data Summary

### Graph Statistics

- Total nodes: `400`
- Clean edges (directed): `4,163`
- Positive edges: `1,616`
- Negative edges: `2,547`
- Training edges: `2,915`
- Training queries: `5,718`
- Validation queries: `1,226`
- Test queries: `1,230`

### Feature Shape

- Shape: `[400, 2]`
- Method: `in_out_degree` from training graph

### Class Distributions

| Split      | Class 0 | Class 1 | Class 2 | Class 3 | Total |
|------------|---------|---------|---------|---------|-------|
| Train      | 1,103   | 1,756   | 1,103   | 1,756   | 5,718 |
| Validation | 237     | 376     | 237     | 376     | 1,226 |
| Test       | 238     | 377     | 238     | 377     | 1,230 |

### Data Fingerprints (SHA-256)

All fingerprints verified across all six model-seed runs.

- **Clean graph:** `5eaf72cf0ded398706060c95b774c11ae4dbe2c15cb8a6b1090fd2bd56a73504`
- **Clean weights:** `26bb02d534b0e8492265bf744c377a27aefc63c0149c9b9a78be772a5341e4c8`
- **Train graph:** `0f0517229a4199bcfc1f435e179352a22184f2a75848ac1842d74e09cfbc6ddb`
- **Train weights:** `1e1fa455542af0f0c21861064bc620d9829cfe33a15879235dad4d6fe7f8f762`
- **Train queries:** `a3a4afce4bbc6fd8e74be7a755016bb1c36913d0bc9a474a57c463bf0f4a4b4a`
- **Train labels:** `4c5bdc0f94e1fb2ded318cdabf505623bb8fc66a39322c091cf2b7e981836e28`
- **Validation queries:** `e0610270543843de386ad614dd132bc2b21a33598ad03c7ca4e9e5d90042032e`
- **Validation labels:** `b051a6afeab7b13e82f406e1d53aec066f04367fe95b4fa7913bede663eac638`
- **Test queries:** `93582066e905b30e36c2b1b607972fe32b9e3739390d70b65d4c38bc35bc4828`
- **Test labels:** `4f86c038f971fc461deb977f258b7b3cac50cedc2a177f141920cd8bdfec3691`
- **Features:** `ab260c1b3dee4f544bd0b700efac7cf61b495e78f20eb3ba000e79816fcc0f76`

**Canonical Bundle Fingerprint:** `fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc`

---

## 4. Per-Model Per-Seed Results

### MSGNN_link_prediction

| Seed | Initial Val NLL | Best Val NLL | Best Epoch | Final Epoch | Restored Val NLL | Test NLL | Test Acc | Test Macro-F1 | Test Micro-F1 | Learning Signal |
|------|-----------------|--------------|------------|-------------|------------------|----------|----------|---------------|---------------|-----------------|
| 0    | 11.5222         | 1.1714       | 134        | 164         | 1.1714           | 1.1279   | 0.5187   | 0.3953        | 0.5187        | ✓               |
| 1    | 4.4582          | 1.1824       | 62         | 92          | 1.1824           | 1.1184   | 0.5171   | 0.3906        | 0.5171        | ✓               |
| 2    | 6.3397          | 1.1795       | 102        | 132         | 1.1795           | 1.1198   | 0.5138   | 0.3896        | 0.5138        | ✓               |

**Learning Signal:** 3/3 seeds satisfied (required: ≥2/3)

### SSSNET_link_prediction (directed=True)

| Seed | Initial Val NLL | Best Val NLL | Best Epoch | Final Epoch | Restored Val NLL | Test NLL | Test Acc | Test Macro-F1 | Test Micro-F1 | Learning Signal |
|------|-----------------|--------------|------------|-------------|------------------|----------|----------|---------------|---------------|-----------------|
| 0    | 26.6887         | 1.1594       | 153        | 183         | 1.1594           | 1.0980   | 0.5520   | 0.4466        | 0.5520        | ✓               |
| 1    | 31.8300         | 1.1665       | 136        | 166         | 1.1665           | 1.1136   | 0.5593   | 0.4861        | 0.5593        | ✓               |
| 2    | 19.7928         | 1.1685       | 193        | 223         | 1.1685           | 1.1020   | 0.5593   | 0.4722        | 0.5593        | ✓               |

**Learning Signal:** 3/3 seeds satisfied (required: ≥2/3)

**Notes:**

- All runs early-stopped before the 300-epoch maximum
- All best checkpoints restored successfully
- All validation NLL values at restored checkpoints matched the recorded best validation NLL exactly
- Learning signal criterion: `best_val_nll <= initial_val_nll - 0.001`

---

## 5. Validation Findings

### Execution Status

- **Total runs:** 6 (2 models × 3 seeds)
- **Passed runs:** 6/6
- **Failed runs:** 0/6

### Finite Value Checks

All six runs passed all finite value checks:

- ✓ All outputs finite
- ✓ All losses finite
- ✓ All gradients finite
- ✓ All parameters finite

### Data Contract Verification

- ✓ All bundle fingerprints matched the canonical fingerprint
- ✓ All individual tensor fingerprints matched shared records
- ✓ No data leakage detected
- ✓ No held-out queries in training graphs

### Checkpoint Restoration

- ✓ All 6 runs restored best validation checkpoints successfully
- ✓ All restored checkpoint validation NLL values matched recorded best values exactly

### Learning Signal

- **MSGNN:** 3/3 seeds satisfied (exceeds 2/3 requirement)
- **SSSNET:** 3/3 seeds satisfied (exceeds 2/3 requirement)
- **Overall:** Both models demonstrated minimal clean-training learning signal

---

## 6. E3-to-E4 Seed-0 Reproducibility Evidence

Gate E3 executed a single-seed dry run (seed 0) for both models. Gate E4 re-executed seed 0 as part of the three-seed clean baseline. The following checks confirm reproducibility:

### Data Fingerprints

**E3 (dry run):**
- Bundle: `fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc`

**E4 (three-seed run):**
- Bundle: `fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc`

**Verdict:** ✓ Exact match

### MSGNN Seed-0 Comparison

| Metric                    | E3 Dry Run | E4 Three-Seed | Match        |
|---------------------------|------------|---------------|--------------|
| Best Epoch                | 134        | 134           | ✓            |
| Final Executed Epoch      | 164        | 164           | ✓            |
| Test Accuracy             | 0.5187     | 0.5187        | ✓            |
| Test Macro-F1             | 0.3953     | 0.3953        | ✓            |
| Test Micro-F1             | 0.5187     | 0.5187        | ✓            |
| Initial Train Loss        | 15.1210    | 15.1210       | within 1e-6  |
| Initial Val Loss          | 11.5222    | 11.5222       | within 1e-6  |
| Best Val Loss             | 1.1714     | 1.1714        | within 1e-6  |
| Restored Checkpoint Val   | 1.1714     | 1.1714        | within 1e-6  |
| Test Loss                 | 1.1279     | 1.1279        | within 1e-6  |

### SSSNET Seed-0 Comparison

| Metric                    | E3 Dry Run | E4 Three-Seed | Match        |
|---------------------------|------------|---------------|--------------|
| Best Epoch                | 153        | 153           | ✓            |
| Final Executed Epoch      | 183        | 183           | ✓            |
| Test Accuracy             | 0.5520     | 0.5520        | ✓            |
| Test Macro-F1             | 0.4466     | 0.4466        | ✓            |
| Test Micro-F1             | 0.5520     | 0.5520        | ✓            |
| Initial Train Loss        | 30.2944    | 30.2944       | within 1e-6  |
| Initial Val Loss          | 26.6887    | 26.6887       | within 1e-6  |
| Best Val Loss             | 1.1594     | 1.1594        | within 1e-6  |
| Restored Checkpoint Val   | 1.1594     | 1.1594        | within 1e-6  |
| Test Loss                 | 1.0980     | 1.0980        | within 1e-6  |

### Selected Scalar Loss Reproducibility

The selected scalar losses compared between E3 and E4 for seed 0 differed by at most `1.1920928955078125e-07`, well within the accepted `1e-6` floating-point tolerance for CPU PyTorch operations.

**Verdict:** ✓ Reproducible within numerical precision

---

## 7. Implementation History

### Commit Timeline

1. **c24f7c7** (2026-07-18): Created `scripts/pygsd_gate_e_clean_baseline.py` implementing the E2 clean-baseline script with shared data construction, fingerprinting, three-seed training loop, and structured failure handling.

2. **b473ea7** (2026-07-18): Added the required per-seed result schema to `scripts/pygsd_gate_e_clean_baseline.py`, including restored-checkpoint validation NLL, test NLL, finite value checks, warnings list, and strict JSON enforcement.

3. **4375dfe** (2026-07-18): Changed the two training loops in `scripts/pygsd_gate_e_clean_baseline.py` from zero-based to one-based epoch numbering so that `total_epochs`, `final_executed_epoch`, `stopping_epoch`, and history lengths have consistent actual-training-epoch semantics.

### Artifact References

- **E2 Dry Run (Seed 0):** `results/gate_e2_seed0_dry_run.json`
- **E4 Three-Seed Baseline:** `results/gate_e4_clean_baseline_results.json`
- **Implementation:** `scripts/pygsd_gate_e_clean_baseline.py`

---

## 8. Gate E Decision

**Decision:** **PASS**

### Pass Criteria Met

✓ All six model-seed runs completed without data-contract, API, numerical, gradient, optimizer, or evaluation-path failure  
✓ All fingerprints matched the shared records  
✓ All recorded outputs, losses, gradients, parameters, and metrics were finite  
✓ Both models satisfied the minimal learning-signal criterion in 3/3 seeds (exceeds required 2/3)  
✓ All best checkpoints restored and evaluated successfully  
✓ E3-to-E4 seed-0 reproducibility confirmed within numerical tolerance

### What Gate E PASS Establishes

The locked clean pipeline supports:

- Reproducible multi-epoch training for MSGNN and directed SSSNET
- Controlled three-seed execution on CPU
- Finite optimization behavior across all seeds
- Minimal validation-loss learning signal on this fixed synthetic signed-directed dataset
- Successful best-checkpoint restoration
- Successful held-out clean-test execution

### What Gate E PASS Does Not Establish

Gate E PASS does **not** establish:

- Robustness to perturbations
- Stability under structural changes
- Perturbation tolerance
- Causal explanations
- Architecture superiority or ranking
- Statistical significance of performance differences
- Convergence guarantees in general
- Generalization to other datasets or tasks
- Production readiness
- GPU compatibility
- Large-scale performance

---

## 9. Interpretation and Scope

### Narrow Interpretation

Gate E validates only that the implemented clean-baseline pipeline can execute reproducible multi-epoch training for two specific PyGSD 1.1.1 models on one fixed synthetic signed-directed dataset with controlled seeds and early stopping.

The test accuracy and F1 scores recorded in this gate serve as **execution evidence** and **clean-baseline reference values** for future perturbation comparisons. They do not represent claims about:

- Model quality
- Predictive performance in general
- Architecture comparison
- Optimal hyperparameters
- Generalization capability

### Attribution Boundary

**Upstream Work:**

- **PyGSD 1.1.1** library, including `MSGNN_link_prediction`, `SSSNET_link_prediction`, and `SDSBM` synthetic data generator
- Original MSGNN and SSSNET architectures and implementations
- PyGSD library developed by Dr. Yixuan He and collaborators

**This Research Project's Contributions:**

- Locked clean-baseline training protocol
- Three-seed reproducibility validation procedure
- SHA-256 fingerprint evidence system
- Structured failure categorization
- Gate E validation framework and decision criteria
- This results record and analysis

The upstream model architectures, their original implementations, and the PyGSD library are attributed to their original authors. The clean-baseline protocol, reproducibility checks, fingerprint evidence, and Gate E analysis are contributions of this research project.

---

## 10. Next Steps

### Authorized Progression

Gate E PASS permits progression to:

- **Gate F Specification:** Design of a controlled structural-perturbation pilot experiment
- **Perturbation Scope Definition:** Specification of perturbation types, magnitudes, and evaluation protocols
- **Validation Stage:** Controlled pilot execution before full perturbation experiments

### Not Authorized

Gate E PASS does **not** authorize:

- Unrestricted perturbation execution
- Robustness conclusions
- Stability claims
- Architecture ranking
- Production deployment
- Generalization claims beyond this specific dataset and configuration

---

## 11. Warnings and Limitations

### No Warnings

The E4 execution produced zero warnings. All runs completed cleanly.

### Known Limitations

1. **Single Dataset:** Results apply only to this specific 400-node SDSBM-generated signed-directed graph
2. **CPU Only:** GPU compatibility not validated
3. **Fixed Hyperparameters:** No hyperparameter search performed
4. **Three Seeds:** Limited statistical power for variance estimation
5. **Clean Data Only:** No perturbation tolerance validated
6. **Task-Specific:** Results specific to four-class signed-directed link prediction

---

## 12. Summary

Gate E successfully validated that PyGSD 1.1.1 `MSGNN_link_prediction` and directed `SSSNET_link_prediction` can complete reproducible, finite, multi-epoch training on a fixed clean signed-directed dataset with controlled seeds and early stopping.

All six model-seed runs passed all validation checks. Both models demonstrated minimal learning signals exceeding the required threshold. E3-to-E4 seed-0 reproducibility was confirmed within numerical tolerance.

**Gate E: PASS**

The project may now proceed to the next controlled structural-perturbation specification and validation stage.

---

*Results recorded: 2026-07-18*  
*Gate E: PASS*  
*Next stage: Gate F specification (structural perturbation pilot design)*  
*Perturbation execution: NOT YET AUTHORIZED*
