# STP 499 — Preliminary E001 Evidence Review

## Project

**Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations**

Prepared for preliminary review by Dr. Yixuan He  
Date: September 1, 2026

Repository branch: `work/v7-base-experiment`  
Local evidence commit: `42fa933`  
Manuscript: seventh internal working draft (“v7”), unchanged during E001

## Naming and status note

The labels in this report are internal project-tracking names:

- **v7** means the seventh internal working draft of the STP 499 manuscript. It is not a published or mentor-approved version number.
- **E001** means the first narrowly scoped, v7-aligned pipeline-validation experiment.
- **Gate** means an internal verification checkpoint used to ensure that required mathematical, implementation, or reproducibility checks are completed before broader experiments begin.
- **Attempt02** identifies the corrected and accepted execution artifact. Earlier attempts are preserved for traceability but are not used as the accepted evidence.

These labels describe workflow history only. They do not imply mentor approval, publication status, or completion of the broader research study.

## Purpose of this review

This preliminary package requests guidance on two connected questions:

1. Does the current theoretical and experimental design align with the intended research direction?
2. Does the narrow E001 implementation provide an appropriate starting point for a larger experimental study?

E001 is a pipeline-validation experiment. It is not presented as a completed robustness study, a statistical result, or evidence of model superiority.

The current manuscript remains unchanged so that theoretical and design alignment can be reviewed separately from any later decision to incorporate experimental results.

## Evidence classification

Important statements are classified using the following labels:

- **Mentor-reported:** The student’s summary of Dr. He’s direction.
- **Manuscript-declared:** Explicitly stated in the current internal manuscript.
- **Code-verified:** Confirmed through inspected or executed source code and preserved outputs.
- **Historically executed:** Supported by preserved historical results but not necessarily aligned with the current manuscript.
- **AI-proposed:** Suggested during AI-assisted planning but not independently approved.
- **Mentor-confirmed:** Explicitly approved by Dr. He with traceable evidence.
- **Unresolved:** Requires source verification, experimental definition, or mentor guidance.

No manuscript-declared or AI-proposed statement is treated as mentor-confirmed.

## Research direction

**Mentor-reported:** The project studies how representations, outputs, predictions, and task behavior in signed and directed graph neural networks change when graph structure is perturbed.

The structural edits of interest are:

- edge deletion;
- sign reversal;
- direction reversal.

**Manuscript-declared:** Sign reversal and direction reversal are distinct graph operations and should not be combined into one generic edge-perturbation category.

## Preliminary correspondence work

### Implementation audit

**Code-verified:** The installed `torch-geometric-signed-directed` package is version `1.1.1`.

The audit inspected the magnetic signed Laplacian implementation and the relevant MSGNN components. It examined:

- magnetic-charge handling;
- symmetric normalization;
- absolute-degree behavior;
- caching behavior;
- fixed versus trainable charge;
- sparse operator construction;
- Hermitian structure;
- effective message flow.

**Code-verified:** The effective message flow observed at runtime in the installed MSConv layer is `source_to_target`.

**Unresolved:** Whether this effective message-flow convention matches the intended theoretical convention requires mentor review.

### Structural-edit and operator tests

**Code-verified:** Eight focused tests passed:

1. zero-edit identity;
2. deletion changes only the edge count under the tested edit construction;
3. sign reversal preserves endpoints;
4. direction reversal preserves sign;
5. all constructed magnetic operators are Hermitian;
6. magnetic charge controls edit visibility;
7. deletion changes the operator;
8. the restricted manuscript identity holds numerically for one eligible edge.

The final test run completed all eight tests in approximately `0.536` seconds with status `OK`.

These tests verify the implemented cases only. They do not establish correctness for every possible graph or experimental condition.

### Restricted exact identities

**Manuscript-declared:** Under the eligibility assumptions stated in the current manuscript, the exact magnetic-adjacency Frobenius changes are:

- sign reversal: `sqrt(2m) |cos(2πq)|`;
- direction reversal: `sqrt(2m) |sin(2πq)|`;
- deletion: `sqrt(m/2)`.

These are restricted identities for the magnetic adjacency operator. They are not automatically identities for the normalized Laplacian, learned representations, logits, output probabilities, predictions, or task metrics.

**Code-verified:** The one-edge sign-reversal identity evaluated in E001 agrees with the implemented magnetic operator to floating-point precision.

## E001 clean baseline

### Experimental configuration

The following configuration is **AI-proposed** as an experimental design and **code-verified** as the configuration that was actually executed.

#### Synthetic graph

- Generator: SDSBM
- Number of nodes: `400`
- Number of communities: `4`
- Edge probability: `0.05`
- Community-size ratio: `1.5`
- Eta: `0.1`
- Gamma: `0.1`
- Generation seed: `0`
- Split seed: `0`
- Validation probability: `0.15`
- Test probability: `0.15`
- Task: `four_class_signed_digraph`

#### MSGNN configuration

- Model-initialization seed: `0`
- Hidden dimension: `16`
- Magnetic charge: `q = 0.125`
- Chebyshev order: `K = 1`
- Number of layers: `2`
- Dropout: `0.5`
- Trainable charge: `False`
- Normalization: `sym`
- Caching: `False`
- Absolute degree: `True`

#### Training configuration

- Execution device: CPU
- Maximum epochs: `300`
- Early-stopping patience: `30`
- Minimum improvement: `0.0001`
- Minimum epochs: `10`
- Learning rate: `0.01`
- Weight decay: `0.0005`

The choice `q = 1/8` was used because, under the restricted identities, both sign reversal and direction reversal remain visible at that charge.

This motivation is **AI-proposed** and has not been mentor-confirmed.

### Clean-baseline result

**Code-verified:**

- Initial training loss: `3.525848`
- Initial validation loss: `2.612920`
- Best epoch: `51`
- Best validation loss: `1.1610015630722046`
- Stopping epoch: `81`
- Stopping reason: early stopping
- Test accuracy: `0.5260162601626016`
- Test macro-F1: `0.42847358361243004`
- Test micro-F1: `0.5260162601626016`
- Majority-class baseline accuracy: `0.3065040650406504`
- Predicted-class counts: `[82, 535, 87, 526]`
- Checkpoint-reload maximum absolute difference: `0.0`
- Clean execution status: `PASS`

The model produced predictions in all four classes and exceeded the majority-class baseline in this single execution.

This supports the narrow conclusion that the implemented pipeline produced a non-collapsed learning signal. It does not establish expected performance across seeds, graphs, datasets, or model configurations.

## Fixed-checkpoint sign-reversal diagnostic

### Diagnostic design

The accepted clean checkpoint was frozen and evaluated without retraining.

A deterministic rule selected the lexicographically first eligible edge in the training graph.

The selected edit was:

- Eligible candidates: `14`
- Stored edge position: `2907`
- Source node: `16`
- Target node: `85`
- Original weight: `-1.0`
- Perturbed weight: `1.0`
- Number of edited edges: `1`

**Code-verified:** The edit:

- preserved both endpoints;
- preserved edge direction;
- changed exactly one edge weight;
- used a unit-weight edge;
- used a non-self-loop edge;
- used an unreciprocated edge;
- used a unique ordered pair;
- excluded query and reverse-query leakage;
- satisfied the restricted manuscript eligibility conditions.

### Operator-level result

**Code-verified:**

- Observed magnetic-adjacency Frobenius change: `1.0`
- Expected value from the restricted identity: `1.0000000000000002`
- Absolute numerical error: `2.220446049250313e-16`
- Normalized-Laplacian Frobenius change: `0.10286890715360641`
- Normalized-Laplacian spectral change: `0.07273930311203003`

The magnetic-adjacency result agrees with the restricted exact identity to numerical precision.

The normalized-Laplacian values are measured changes. They are not claimed to be exact theorem identities.

### Downstream result

**Code-verified:**

- Representation shape: `[1230, 64]`
- Relative representation Frobenius drift: `0.06631334871053696`
- Logit Frobenius change: `2.1920011043548584`
- Output log-probability Frobenius change: `2.6511528491973877`
- Maximum absolute output change: `0.6194145679473877`
- Prediction flips: `0`
- Prediction-flip rate: `0.0`
- Accuracy change: `0.0`
- Macro-F1 change: `0.0`

The sign reversal changed the magnetic operator, representation, logits, and output values. However, it did not move any test query across a predicted-class decision boundary.

This is one diagnostic observation. It does not demonstrate that MSGNN is generally robust or insensitive to sign reversal.

### Integrity result

**Code-verified:**

- All diagnostic tensors were finite.
- The model state was unchanged.
- The saved checkpoint was unchanged.
- Frozen data tensors were unchanged.
- Model parameters were frozen during the diagnostic.
- Repeated evaluation was deterministic.

## Reproducibility and preservation

The accepted source code, tests, JSON results, and provenance record are stored in the local Git commit:

`42fa933 — Add E001 baseline and sign-reversal provenance`

Nothing has been pushed as part of this preliminary workflow.

The committed files are:

- `notes/e001_provenance_record.md`
- `results/e001/e001_k1_q0125_seed0_clean_attempt02.json`
- `results/e001/e001_k1_q0125_seed0_sign_reversal_diagnostic_attempt02.json`
- `scripts/e001_v7_clean_baseline.py`
- `scripts/e001_v7_fixed_checkpoint_sign_reversal.py`
- `tests/test_v7_exact_frobenius_identities.py`
- `tests/test_v7_operator_identities.py`

The checkpoint is excluded from Git by the repository’s existing `*.pt` ignore rule.

The checkpoint and the two accepted JSON files were downloaded from Sol and archived on the iMac at:

`/Users/sileshihirpa/Desktop/ASU/SciML/STP499_E001_Archive/2026-09-01`

Accepted SHA-256 hashes:

- Clean JSON:  
  `9efd03a6f5aac6ec3296924dda9aa5342020f9c2c94c4f671a6470e806599419`
- Sign-reversal diagnostic JSON:  
  `38de0599744981fa6f29a5e96f71e2e98e4faec4cd2d9cc5d5265ba62b426ea2`
- MSGNN checkpoint:  
  `6b0d7f7faa2fb74ad7e2eca5ac186ab0c1313a2773c3b67bc82f0a1395d642ef`

The hashes were independently reproduced on the iMac after downloading the files.

Earlier corrective-history artifacts remain preserved but are not used as the accepted evidence.

## What E001 supports

E001 supports the following narrow statements:

1. The accepted clean MSGNN pipeline executed successfully for one fixed synthetic graph, split, and model-initialization seed.
2. The saved checkpoint reloaded exactly for the tested output.
3. The implemented one-edge sign reversal satisfied its stated structural constraints.
4. The restricted magnetic-adjacency identity matched the implementation to numerical precision.
5. The edit produced measurable changes in continuous downstream quantities.
6. The tested edit produced no predicted-class flips for this checkpoint and test set.
7. The model, checkpoint, and frozen-data integrity checks passed.

## What E001 does not support

E001 does not support claims about:

- statistical robustness;
- expected behavior across seeds or graphs;
- architecture superiority;
- hyperparameter optimality;
- production readiness;
- deletion experiments;
- direction-reversal experiments;
- behavior across perturbation budgets;
- retraining under perturbation;
- statistical significance;
- uncertainty estimates;
- alignment with mentor expectations unless Dr. He explicitly confirms it.

## Current limitations

1. Only one graph-generation seed was used.
2. Only one data split was used.
3. Only one model-initialization seed was used.
4. Only one MSGNN checkpoint was evaluated.
5. Only one deterministically selected edge was edited.
6. Only sign reversal was executed.
7. No perturbation-budget curve was produced.
8. No confidence interval or repeated-run distribution was produced.
9. No comparison model was executed in E001.
10. The clean script retains inherited SSSNET-related implementation material, but E001 evidence is restricted to MSGNN.
11. The installed MSConv message-flow convention requires correspondence review.
12. The current manuscript has not been modified to incorporate E001.

## Guidance requested from Dr. He

Before expanding the experiment, guidance is requested on the following questions:

1. Does the current theory-to-implementation correspondence match the intended research direction?
2. Is the four-class signed directed link-prediction task appropriate as the first experimental task?
3. Is MSGNN the appropriate first model?
4. Is fixed `q = 1/8` appropriate, or should magnetic charge be selected differently?
5. Is `K = 1` appropriate for the first controlled experiment?
6. Should the primary protocol use fixed-checkpoint evaluation, retraining under perturbation, or both?
7. Should sign reversal, direction reversal, and deletion use matched eligible-edge sets?
8. Which perturbation budgets should be evaluated?
9. How many graph-generation, split, and model-initialization seeds are expected?
10. Which representation, output, prediction, and task metrics should be primary?
11. Should the message-flow or direction convention be changed, or should the observed convention simply be documented?
12. Which comparison models, if any, should be included in the first expanded experiment?
13. Should the next manuscript draft incorporate E001 only after these experimental-design decisions are confirmed?

## Proposed next step after mentor review

If the design is approved or corrected, the proposed next experiment would expand the single diagnostic into a controlled experimental matrix covering:

- sign reversal;
- direction reversal;
- edge deletion;
- multiple perturbation budgets;
- multiple eligible-edge samples;
- multiple graph, split, and model seeds;
- clearly separated fixed-checkpoint and retraining protocols;
- uncertainty summaries;
- preserved configurations, code, logs, checkpoints, and artifact hashes.

This expansion is proposed only. It is not yet mentor-confirmed, authorized, or complete.
