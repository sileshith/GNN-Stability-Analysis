---
title: "Gate D — Minimal Training Compatibility Specification"
project: "Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations"
role: "Gate D test design specification"
status: "specification only — not executed"
specification_date: "2026-07-18"
---

# Gate D — Minimal Training Compatibility Specification

**Specification Date:** 2026-07-18

---

## 1. Purpose

Gate D is a narrow engineering and API-compatibility test verifying that:

- `MSGNN_link_prediction`
- `SSSNET_link_prediction` with `directed=True`

can each complete a minimal end-to-end training step on the same tiny signed-directed graph using the PyGSD 1.1.1 `four_class_signed_digraph` task.

**Gate D does not test:**

- Convergence
- Model quality
- Robustness
- Stability
- Generalization
- Architecture superiority

---

## 2. Governing Scope

**Execution Environment:**

- CPU execution only
- PyGSD version `1.1.1`

**Data:**

- One deterministic synthetic signed-directed graph
- One fixed clean split shared by both models
- Four-class signed-directed link prediction

**Training:**

- `nn.NLLLoss()`
- One initialization seed
- One optimizer step required
- A very small optional number of additional steps may be used only to confirm continued numerical execution

**Prohibited:**

- No perturbations
- No hyperparameter search
- No production metrics
- No comparison of accuracy between models

---

## 3. Required Shared Data Contract

A single clean data contract must contain:

- Node count
- Signed directed `edge_index`
- Signed `edge_weight`
- Training message-passing graph returned by `link_class_split`
- Train query edges and labels
- Validation query edges and labels when available
- Test query edges and labels
- Labels with dtype `torch.long`
- Labels restricted to `{0, 1, 2, 3}`
- Identical split objects used for MSGNN and SSSNET
- No held-out query edge included in the message-passing graph

**Class Meanings:**

- **0:** Positive edge in queried direction
- **1:** Negative edge in queried direction
- **2:** Positive edge in reverse direction
- **3:** Negative edge in reverse direction

---

## 4. Tiny Dataset Requirements

The synthetic graph must be intentionally small but structurally sufficient:

- Contains both positive and negative directed edges
- Contains examples of all four target classes in the training query set
- Contains at least one evaluable query edge outside training when the split utility permits
- Avoids isolated-node or empty-class failures
- Uses deterministic generation and split seeds
- Records all graph and split sizes

**Not Locked:**

Final DSBM/SDSBM parameters are not locked in this specification. Either a carefully constructed tiny graph or a small deterministic PyGSD-compatible synthetic generator is permitted, provided all data-contract checks pass.

---

## 5. Model-Specific Preprocessing

### MSGNN

**Required:**

- Node features produced with the verified official `in_out_degree` path
- Real features cloned for imaginary features
- Signed edge weights preserved
- Output dimension `label_dim=4`
- `cached=False` for Gate D to avoid stale graph-operator assumptions
- Model constructed independently from SSSNET

### Directed SSSNET

**Required:**

- The same clean training graph as MSGNN
- Conversion through `SignedData`
- `separate_positive_negative()` called
- Separate positive and negative edge indices and weights
- `directed=True`
- Output dimension `nclass=4`
- Features derived from the same agreed feature policy as the shared official task
- Model constructed independently from MSGNN

**Clarification:**

Equivalent task inputs do not require identical internal graph representations because the architectures expose different APIs.

---

## 6. Optimizer and Loss

**Optimizer:**

- `torch.optim.Adam`
- One fixed learning rate and weight decay chosen before execution
- The same optimizer family for both models

**Loss:**

- `nn.NLLLoss()`
- Integer class targets
- No class weighting in Gate D unless an empty-class issue makes execution impossible
- If an empty-class issue occurs, the gate must stop and the dataset must be revised rather than silently changing the loss

**Not Locked:**

Final production hyperparameters are not locked.

---

## 7. Required Assertions Before Training

For each model assert:

- Model is on CPU
- Model is in training mode
- Query edges have shape `[Q, 2]`
- Output has shape `[Q, 4]`
- Output dtype is floating point
- Output values are finite
- No NaN or positive/negative infinity
- Exponentiated log-probability rows sum to approximately 1
- Labels have shape `[Q]`
- Labels have dtype `torch.long`
- Minimum label is at least 0
- Maximum label is at most 3
- Initial loss is scalar and finite

---

## 8. Backward-Pass and Gradient Checks

**Required:**

- `optimizer.zero_grad()`
- Loss backward pass completes without exception
- Every observed gradient tensor is finite
- At least one trainable parameter receives a non-None gradient
- At least one gradient has a nonzero magnitude

**Record:**

- Count of trainable parameters
- Count of parameters with gradients
- Count of parameters with finite gradients
- Count of parameters with nonzero gradients

**Not Required:**

Every parameter does not need to have a nonzero gradient.

---

## 9. Optimizer-Step Checks

**Before the optimizer step:**

- Clone all trainable parameters or a deterministic subset sufficient to detect change

**After one optimizer step:**

- Verify at least one trainable parameter changed
- Verify all model parameters remain finite
- Run a second forward pass
- Verify post-step output and loss remain finite
- Record initial and post-step loss without interpreting whether the loss improved

**Not Required:**

A lower post-step loss is not required for Gate D.

---

## 10. Evaluation-Path Checks

Using a fixed held-out query set when available:

- Switch to evaluation mode
- Use `torch.no_grad()`
- Run forward inference
- Verify output shape and finiteness
- Compute accuracy, macro-F1, and micro-F1
- Verify metrics execute and are finite
- Record metric values as execution evidence only

**Clarification:**

The values are not performance estimates.

---

## 11. Shared-Split Verification

**Required fingerprint proving both models use the same:**

- Clean graph
- Training graph
- Train query edges
- Train labels
- Validation/test query edges and labels
- Generation seed
- Split seed

**Permitted:**

A deterministic tensor hash or serialized canonical representation.

**Clarification:**

Model-specific positive/negative decomposition for SSSNET must be derived from, and traceable to, the same training graph used by MSGNN.

---

## 12. Pass Criteria

**Gate D passes for an individual model only when all of the following hold:**

- Preprocessing completes
- Forward pass completes
- Output contract is satisfied
- Initial loss is finite
- Backward pass completes
- Gradients satisfy the finite/nonzero checks
- Optimizer step changes at least one parameter
- Post-step parameters, outputs, and loss are finite
- Evaluation metrics execute
- Execution record is complete

**Overall Gate D passes only if:**

Both MSGNN and directed SSSNET individually pass on the same clean split.

---

## 13. Failure and Stop Rules

**Gate D must stop and be marked FAIL or BLOCKED when:**

- A model requires changing the task or labels
- One model receives a different clean split
- Labels do not cover the required four-class contract
- Loss, outputs, gradients, or parameters become non-finite
- Backward pass fails
- No parameter changes after the optimizer step
- Held-out edges leak into the message-passing graph
- Preprocessing silently removes sign or direction information
- Passing would require unrecorded architecture-specific exceptions

**Failure Classification:**

- Data-contract failure
- Preprocessing failure
- Model-API failure
- Numerical failure
- Gradient failure
- Optimizer-step failure
- Evaluation-path failure
- Implementation defect
- Unresolved package behavior

**Do not broaden the experiment to fix a failure.**

---

## 14. Required Evidence Record

The future implementation must record:

- Timestamp
- Environment versions
- Device
- Seeds
- Graph and split sizes
- Class counts
- Model constructor arguments
- Optimizer arguments
- Feature shapes
- Edge tensor shapes
- Output shapes
- Initial and post-step losses
- Gradient summary
- Parameter-change result
- Evaluation metrics
- Pass/fail status for every assertion
- Exception traceback when applicable
- Final Gate D verdict

**Note:**

The script and permanent results document will be created in later controlled work units.

---

## 15. Claims Boundary

### Permitted After a PASS

- Both selected model APIs support a minimal training step on the defined tiny common task
- Loss, backward pass, gradients, parameter update, and evaluation path execute under the tested environment
- The two models can proceed to a later functional-baseline specification

### Not Permitted

- Either model converges
- Either model learns meaningful representations
- One model outperforms the other
- The implementation is production-ready
- Robustness or stability is established
- Perturbation experiments are authorized
- Findings generalize to other datasets, seeds, devices, or hyperparameters

---

## 16. Gate Relationships and Next Action

**Current Gate Status:**

- **Gate B:** PARTIAL
- **Gate C:** PASS only for the narrow CPU forward-pass smoke test
- **Gate D:** SPECIFICATION COMPLETE after this document; execution remains pending

**After Passing Gate D:**

Passing Gate D would authorize specification of a clean functional baseline, not robustness experiments.

---

## Summary

- **Gate:** D — Minimal Training Compatibility
- **Primary models:** MSGNN and directed SSSNET
- **Task:** Four-class signed-directed link prediction
- **Execution scale:** Tiny deterministic CPU test
- **Perturbations:** Prohibited
- **Next controlled action:** Implement the Gate D test script in a separate committed work unit

---

*Specification completed: 2026-07-18*  
*Gate B: PARTIAL*  
*Gate C: PASS (forward-pass smoke test only)*  
*Gate D: SPECIFICATION COMPLETE*  
*Implementation: pending*
