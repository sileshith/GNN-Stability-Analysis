# Gate R2 — PyGSD 1.2.0 Isolated Validation Record

## Status

**PASS — pre-training compatibility validation only**

This gate establishes that the project-relevant `MSConv` and
`MSGNN_link_prediction` behaviors corrected in PyGSD 1.2.0 execute as expected
in an isolated Sol environment.

This gate does **not** authorize or report model training, perturbation
experiments, robustness conclusions, comparative model rankings, or manuscript
changes.

## Date and repository state

Validation date: 2026-09-05

Repository:

`/scratch/shirpa/gnn-stability/research/GNN-Stability-Analysis`

Branch:

`work/v7-base-experiment`

Repository commit at the start of Gate R2:

`c3756758a631359ea62ae85ecced78ed38388504`

Gate R2 editing authorization:

`AUTHORIZE GATE R2 PYGSD 1.2.0 VALIDATION EVIDENCE EDIT`

## Purpose

Gate R2 was opened after the PyGSD maintainer released version 1.2.0 in
response to the `MSConv` correspondence issues documented in Gate R1.

The purposes of Gate R2 were to:

1. preserve the historical PyGSD 1.1.1 environment and evidence;
2. create an isolated PyGSD 1.2.0 validation environment;
3. verify the corrected message-flow configuration;
4. verify the corrected complex computation through a q=0 invariant;
5. exercise `MSGNN_link_prediction` at the mentor-confirmed primary synthetic
   charge `q=1/4`;
6. ensure that historical and current tests remain version-specific;
7. determine whether the implementation is ready for a separate training
   preflight.

## Evidence classification

### Mentor-confirmed

Dr. Yixuan He confirmed that:

- the neural-network input convention begins with `X+iX`, where `X` is the
  input node-feature matrix;
- the earlier complex computation contained issues or typographical errors;
- the `target_to_source` configuration was misplaced;
- the implementation would be revised using direct PyTorch complex
  computation;
- the fixes were released in PyGSD 1.2.0.

Dr. He also specified `q=1/4` as the primary synthetic-data configuration.
The earlier E001 `q=1/8` result remains historical preliminary evidence only.

### Upstream release metadata

Release:

`torch-geometric-signed-directed 1.2.0`

Release page:

<https://github.com/SherylHYX/pytorch_geometric_signed_directed/releases/tag/1.2.0>

Reported release commit:

`2968b55b3378a01ade9f391e194f0077e2c6ee3b`

The upstream release describes:

- direct complex computation for MagNet and MSGNN;
- fixes for trainable `q`;
- correction of the misplaced `target_to_source` setting.

### Code-verified and execution-verified

The installed PyGSD 1.2.0 source and runtime behavior were independently
checked on Sol. The validation did not rely solely on the release description.

### Historical evidence

The accepted E001 run and Gate R1 audit used PyGSD 1.1.1. They must not be
silently reinterpreted as PyGSD 1.2.0 results.

## Environment isolation

### Historical environment

Path:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env`

Verified PyGSD version:

`1.1.1`

This environment was preserved without an in-place package upgrade.

### PyGSD 1.2.0 validation environment

Path:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env_pygsd120`

The validation environment was cloned from the historical environment. PyGSD
alone was upgraded using the exact version requirement `1.2.0` and the
`--no-deps` option, preventing an automatic upgrade of unrelated dependencies.

Verified environment:

- Python: `3.11.15`
- PyTorch: `2.12.0+cu130`
- PyTorch Geometric: `2.7.0`
- PyGSD: `1.2.0`
- NetworkX: `3.6.1`
- device used for Gate R2 tests: CPU

A CUDA-capable PyTorch build was present, but Gate R2 did not require or use a
GPU.

## Installed implementation evidence

Installed `MSConv` source:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env_pygsd120/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/general/MSConv.py`

Installed `MSGNN` source:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env_pygsd120/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/general/MSGNN.py`

SHA-256 of the installed PyGSD 1.2.0 `MSConv.py`:

`f52ae1fa5308f00028a8776614eb0799fa8c6d4b62f156bbd7b7b7d694159e61`

Inspection of the installed `MSConv` implementation established that:

1. `kwargs.setdefault("flow", "target_to_source")` is executed before the
   superclass constructor;
2. the effective runtime `flow` property is `target_to_source`;
3. edge weights are constructed with `torch.complex`;
4. node features are constructed with `torch.complex`;
5. the layer returns the real and imaginary components of the complex result.

These observations directly address the two implementation issues isolated in
Gate R1.

## Test artifacts

### Historical PyGSD 1.1.1 audit

File:

`tests/test_pygsd_msconv_correspondence.py`

SHA-256 after adding the version guard:

`b36a1a03f273c93d4ddd1ffab892c82f6ff528049d5d20a2b1adf5a1372cb12b`

The historical test remains unchanged in substantive expectations. A
version-specific class guard was added so that it executes only when the
installed PyGSD version is exactly `1.1.1`.

### PyGSD 1.2.0 validation

File:

`tests/test_pygsd_120_msconv_validation.py`

SHA-256:

`fcb836726ae7a18b87bc7ec046d894cd2ee527143ea6616a70f2fbef58b02897`

The new test executes only when the installed PyGSD version is exactly `1.2.0`.

## PyGSD 1.2.0 validation results

Five tests were executed under the isolated PyGSD 1.2.0 environment.

### Corrected flow

Test:

`test_effective_flow_is_target_to_source`

Result:

`PASS`

Observed effective flow:

`target_to_source`

### Exact installed release

Test:

`test_installed_release_is_120`

Result:

`PASS`

Observed version:

`1.2.0`

### MSGNN layer inventory at q=1/4

Test:

`test_msgnn_q_quarter_layer_inventory`

Result:

`PASS`

Verified properties:

- two `MSConv` layers;
- both layers use `target_to_source`;
- first weight shape: `(2, 2, 16)`;
- second weight shape: `(2, 16, 16)`;
- model charge: `q=0.25`;
- Chebyshev order: `K=1`;
- `trainable_q=False`.

### q=0 complex-computation invariant

Test:

`test_q_zero_real_input_has_zero_imaginary_output`

Result:

`PASS`

For a purely real input, real learned weights, and `q=0`, the corrected layer
produced:

- a finite real output;
- a finite imaginary output;
- a nonzero propagated real signal;
- a zero imaginary output within absolute tolerance `1e-7`.

This is the invariant that was recorded as an expected failure under PyGSD
1.1.1.

### MSGNN q=1/4 X+iX forward pass

Test:

`test_msgnn_q_quarter_x_plus_ix_forward_is_finite`

Result:

`PASS`

The test used the mentor-confirmed feature convention in which the same real
feature matrix initializes both components, producing `X+iX`.

The resulting link-prediction output:

- had shape `(1, 4)`;
- contained only finite values;
- represented normalized class probabilities after exponentiating the returned
  log probabilities.

### Aggregate result

`Ran 5 tests in 2.130s`

`OK`

## Version-separation results

### Historical test under PyGSD 1.1.1

Results:

- two ordinary passes;
- one expected failure for the historical q=0 defect;
- aggregate result: `OK (expected failures=1)`.

This confirms that the Gate R1 behavior remains reproducible.

### Historical test under PyGSD 1.2.0

Results:

- all three historical tests skipped;
- skip reason: `Historical PyGSD 1.1.1 audit`;
- aggregate result: `OK (skipped=3)`.

### PyGSD 1.2.0 test under PyGSD 1.1.1

Results:

- all five current-release tests skipped;
- skip reason identifies installed version `1.1.1`;
- aggregate result: `OK (skipped=5)`.

The two evidence tracks therefore remain explicitly separated rather than
producing false cross-version failures.

## Existing mathematical and structural-edit regression tests

The following existing test groups were executed under PyGSD 1.2.0:

- restricted exact Frobenius identity;
- Hermitian operator construction;
- charge-controlled edit visibility;
- deletion operator change;
- deletion edit semantics;
- direction-reversal sign preservation;
- sign-reversal endpoint preservation;
- zero-edit identity.

Aggregate result:

`Ran 8 tests in 0.691s`

`OK`

No regression was observed in these eight project-level mathematical and
structural-edit tests.

## Interpretation

Gate R2 resolves the specific pre-training blocker identified in Gate R1 for
the tested `MSConv` and `MSGNN_link_prediction` behaviors.

The installed PyGSD 1.2.0 implementation:

- applies the corrected message-flow setting;
- satisfies the q=0 complex-computation invariant;
- supports the intended `X+iX` feature initialization;
- produces a finite and normalized MSGNN link-prediction output at `q=1/4`;
- remains compatible with the eight existing mathematical and edit-semantics
  tests.

The historical E001 `q=1/8`, PyGSD 1.1.1 result remains useful as a
reproducible diagnostic record, but it is not a publication baseline and must
not be pooled with new PyGSD 1.2.0 results.

## Limitations

Gate R2 does not establish:

1. successful multi-epoch training under PyGSD 1.2.0;
2. successful optimizer or backward-gradient behavior;
3. checkpoint persistence and exact reload behavior under PyGSD 1.2.0;
4. performance or robustness at `q=1/4`;
5. robustness under deletion, sign reversal, or direction reversal;
6. correctness of every model changed by the upstream 1.2.0 release;
7. correctness across multiple datasets, tasks, budgets, or seeds;
8. publication-scale reproducibility;
9. comparative superiority of any architecture.

The upstream release also modified code beyond the exact MSGNN path exercised
here. MagNet, SGCN, SSSNET, DIMPA, and other candidate models must receive
model-appropriate compatibility checks before inclusion in comparative
experiments.

## Authorization boundary

Under the Gate R2 authorization:

- the historical test received only a version guard;
- a separate PyGSD 1.2.0 validation test was added;
- this validation record was created;
- the historical environment was preserved;
- no model training was performed;
- no accepted E001 artifact was overwritten;
- no perturbation result was generated;
- no manuscript or Overleaf content was changed;
- no Git commit or push was performed.

## Recommended next gate

The next gate should be a PyGSD 1.2.0 CPU training preflight at the
mentor-confirmed synthetic setting `q=1/4`.

That preflight should verify:

1. deterministic data construction;
2. finite forward output and loss;
3. finite, nonzero gradients;
4. successful optimizer update;
5. finite model parameters;
6. checkpoint save and file hash;
7. fresh-model checkpoint reload;
8. exact or tolerance-controlled prediction agreement after reload;
9. isolated output paths that cannot overwrite historical E001 evidence.

Only after that preflight passes should a corrected three-seed pilot be
authorized. Five- and ten-seed studies should wait until the experimental
design, perturbation budgets, task matrix, and evaluation protocols are
written and reviewed.
## Final full-suite confirmation

The complete repository test collection was executed in both isolated
environments after all Gate R2 test edits were complete.

### PyGSD 1.2.0 environment

- total discovered tests: 16;
- applicable tests passed: 13;
- historical PyGSD 1.1.1 tests skipped: 3;
- failures: 0;
- unexpected successes: 0;
- aggregate result: `OK (skipped=3)`.

### Preserved PyGSD 1.1.1 environment

- total discovered tests: 16;
- applicable ordinary tests passed: 10;
- PyGSD 1.2.0 tests skipped: 5;
- historical expected failures: 1;
- ordinary failures: 0;
- aggregate result: `OK (skipped=5, expected failures=1)`.

This two-environment execution confirms that the historical defect remains
reproducible under PyGSD 1.1.1 while the corrected behavior passes under
PyGSD 1.2.0.
