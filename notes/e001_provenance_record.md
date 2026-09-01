# E001 Preliminary Execution Provenance Record

Date: 2026-09-01  
Repository branch: `work/v7-base-experiment`  
Starting commit: `25297c2`  
Execution system: ASU Sol  
Execution mode: CPU-only interactive VS Code allocation

## Scope

E001 is a preliminary pipeline-validation experiment for MSGNN link prediction on a synthetic signed directed graph.

It includes:

- one clean training run;
- one fixed-checkpoint, single-edge sign-reversal diagnostic;
- operator-identity tests;
- checkpoint restoration and integrity checks.

It does not establish statistical robustness, model superiority, or broad empirical conclusions.

## Evidence classification

- **Mentor-reported:** The project studies stability and robustness of signed and directed GNNs under structural perturbations.
- **Manuscript-declared:** Sign reversal, direction reversal, and deletion are distinct structural edits. The restricted exact Frobenius identities apply to the magnetic adjacency operator under their stated assumptions.
- **Code-verified:** The tests, clean baseline, checkpoint reload, diagnostic edit, tensor integrity checks, and recorded numerical outputs described below.
- **Historically executed:** Earlier Gate E scripts were used as implementation references but are not treated as v7 evidence.
- **AI-proposed:** The E001 configuration choice `q = 1/8` and the use of one deterministic eligible edge as a diagnostic.
- **Mentor-confirmed:** None of the new E001 experimental choices or results have yet been classified as mentor-confirmed.
- **Unresolved:** Whether `q = 1/8`, the selected task, the edge-selection protocol, metrics, and later experimental scale match Dr. He’s intended design.

## Sol allocation

- Slurm job ID: `62462046`
- Job name: `ood-vscode`
- Cluster: `sol`
- Partition: `lightwork`
- QOS: `public`
- Account: `grp_yixuanh2`
- CPUs per task: `2`
- Memory per node: `16384 MB`
- Device used by E001: `cpu`

## Software environment

- Python: `3.11.15`
- PyTorch: `2.12.0+cu130`
- PyTorch Geometric: `2.7.0`
- torch-geometric-signed-directed: `1.1.1`
- NumPy: `1.26.4`
- NetworkX: `3.6.1`
- SciPy: `1.17.1`

## Locked E001 configuration

### Synthetic data

- Generator: SDSBM
- Nodes: `400`
- Communities: `4`
- Edge probability: `0.05`
- Size ratio: `1.5`
- Eta: `0.1`
- Gamma: `0.1`
- Generation seed: `0`
- Split seed: `0`
- Validation probability: `0.15`
- Test probability: `0.15`
- Task: `four_class_signed_digraph`

### MSGNN

- Model seed: `0`
- Hidden dimension: `16`
- Magnetic charge `q`: `0.125`
- Chebyshev order `K`: `1`
- Layers: `2`
- Dropout: `0.5`
- Trainable q: `False`
- Normalization: `sym`
- Cached: `False`
- Absolute degree: `True`
- Message flow observed in installed MSConv: `source_to_target`

### Training

- Maximum epochs: `300`
- Patience: `30`
- Minimum improvement: `0.0001`
- Minimum epochs: `10`
- Learning rate: `0.01`
- Weight decay: `0.0005`

## Gate 3 operator tests

Command pattern: Python unittest discovery for `test_v7_*.py`

Result: `8` tests ran and all passed.

The tests cover:

- zero-edit identity;
- deletion semantics;
- sign-reversal semantics;
- direction-reversal semantics;
- Hermitian construction;
- charge-dependent edit visibility;
- deletion changing the operator;
- the restricted Theorem 5.3 identity for one eligible edge.

This is **Code-verified** evidence for the tested implementations and cases only.

## Gate 4 accepted clean baseline

Accepted result:

`results/e001/e001_k1_q0125_seed0_clean_attempt02.json`

Accepted checkpoint:

`results/e001/msgnn_k1_q0125_seed0_20260901T163605Z.pt`

Data-bundle fingerprint:

`fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc`

Observed results:

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
- Checkpoint reload maximum absolute difference: `0.0`
- Overall clean-run status: `PASS`

These results show that this single clean run executed successfully and produced a non-collapsed learned output. They do not establish general model performance.

## Gate 5 accepted diagnostic

Accepted result:

`results/e001/e001_k1_q0125_seed0_sign_reversal_diagnostic_attempt02.json`

The diagnostic used the accepted clean checkpoint without retraining.

The deterministic selection rule chose the lexicographically first eligible training-graph edge:

- Eligible candidates: `14`
- Edge-array position: `2907`
- Source: `16`
- Target: `85`
- Original weight: `-1.0`
- Perturbed weight: `1.0`
- Number of edits: `1`

Verified edit properties:

- endpoints preserved;
- direction preserved;
- exactly one weight changed;
- unit-weight edge;
- non-self-loop;
- unreciprocated;
- ordered pair unique;
- query and reverse-query leakage excluded;
- eligible for the restricted Theorem 5.3 identity.

Observed operator changes:

- Magnetic-adjacency Frobenius change: `1.0`
- Theorem 5.3 expected value: `1.0000000000000002`
- Absolute numerical error: `2.220446049250313e-16`
- Normalized-Laplacian Frobenius change: `0.10286890715360641`
- Normalized-Laplacian spectral change: `0.07273930311203003`

Observed downstream changes:

- Representation relative Frobenius drift: `0.06631334871053696`
- Logit Frobenius change: `2.1920011043548584`
- Output Frobenius change: `2.6511528491973877`
- Prediction flips: `0`
- Prediction-flip rate: `0.0`
- Accuracy change: `0.0`
- Macro-F1 change: `0.0`

Interpretation: the single sign reversal was visible at the operator, representation, logit, and output-value levels, but it did not cross any predicted-class decision boundary for the test queries. This is one diagnostic observation, not evidence of general robustness.

## Integrity checks

The accepted diagnostic recorded:

- all diagnostic tensors finite;
- model state unchanged;
- checkpoint file unchanged;
- frozen data tensors unchanged;
- model parameters frozen during the diagnostic;
- deterministic repeated outputs;
- checkpoint hash matched the accepted clean record.

## SHA-256 evidence

### Executed source and tests

- `scripts/e001_v7_clean_baseline.py`  
  `5030375a0b9ad359c642600697333004c4ab5b548204da08488ac4726360262d`
- `scripts/e001_v7_fixed_checkpoint_sign_reversal.py`  
  `eb189b7b72b380f7b556d849a1229c7be9c3efe50143cdc7240735c6a49764d8`
- `tests/test_v7_operator_identities.py`  
  `0b99a2a16868416426e0093513f3c98efe3f869c5798324224f441d167ccbeee`
- `tests/test_v7_exact_frobenius_identities.py`  
  `5a28791277d10ce13f7b960d990dcd5e99138b54bc8cf844003280778969c476`

### Accepted artifacts

- Clean JSON  
  `9efd03a6f5aac6ec3296924dda9aa5342020f9c2c94c4f671a6470e806599419`
- Sign-reversal diagnostic JSON  
  `38de0599744981fa6f29a5e96f71e2e98e4faec4cd2d9cc5d5265ba62b426ea2`
- MSGNN checkpoint  
  `6b0d7f7faa2fb74ad7e2eca5ac186ab0c1313a2773c3b67bc82f0a1395d642ef`

The three accepted artifacts were downloaded from Sol and independently hash-verified on the iMac under:

`/Users/sileshihirpa/Desktop/ASU/SciML/STP499_E001_Archive/2026-09-01`

## Preserved corrective history

The following earlier artifacts are preserved but are not the accepted E001 evidence:

- `e001_k1_q0125_seed0_clean.json`
- `msgnn_k1_q0125_seed0_20260901T162318Z.pt`
- `e001_k1_q0125_seed0_sign_reversal_diagnostic.json`

The earlier clean output retained an inherited `E2` label. The earlier diagnostic omitted the Python-version value. Corrected runs were written as separate `attempt02` artifacts instead of silently replacing this history.

## Limitations and unresolved decisions

1. Only one model seed, checkpoint, and edited edge were evaluated.
2. The selected edge was deterministic and diagnostic, not a random or representative sample.
3. No uncertainty estimate or repeated-run distribution was produced.
4. Only sign reversal was executed in this preliminary diagnostic.
5. Edge deletion and direction reversal remain future experimental work.
6. The exact theorem identity applies to the restricted magnetic adjacency operator, not automatically to the normalized Laplacian or downstream network.
7. Downstream changes are measurements, not exact theorem identities.
8. Zero prediction flips must not be interpreted as a general robustness result.
9. The choice `q = 1/8` is experimentally motivated but not mentor-confirmed.
10. The effective installed MSConv message flow is `source_to_target`; its correspondence with the intended convention requires mentor review.
11. The clean script retains inherited SSSNET-related implementation material that was not executed in E001; E001 evidence is restricted to MSGNN.
12. The manuscript remains unchanged.

## Preliminary conclusion

**Code-verified:** The narrow E001 pipeline successfully trained one MSGNN clean baseline, restored its checkpoint exactly, applied one eligible fixed-checkpoint sign reversal, reproduced the restricted magnetic-adjacency identity to numerical precision, detected downstream continuous changes, and preserved model, checkpoint, and frozen-data integrity.

**Unresolved:** Whether this experimental design adequately implements Dr. He’s intended requirements and should be expanded into a larger robustness study requires explicit mentor review.
