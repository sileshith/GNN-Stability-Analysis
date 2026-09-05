# Gate R1 — PyGSD MSConv Mathematical–Implementation Correspondence Audit

## Record status

- Project: STP 499 — Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations
- Audit date: 2026-09-01
- Execution machine: ASU Sol
- Repository branch: `work/v7-base-experiment`
- Audit type: read-only installed-source inspection and deterministic CPU sanity checks
- Current disposition: **UNRESOLVED — mentor clarification required**
- Publication-scale MSGNN runs: **paused pending resolution**

This record documents observed behavior. It does not modify PyGSD, assert an upstream software defect, revise the manuscript, or replace the preserved E001 evidence.

## Governing evidence classifications

The following classifications are used in this record:

- **Code-verified:** directly observed in installed source code or executed checks on Sol.
- **Mathematically expected:** follows from ordinary complex matrix multiplication under the operator interpretation used in the manuscript.
- **Historically executed:** preserved experimental evidence generated before this audit.
- **Unresolved:** requires confirmation of intended implementation semantics or mentor guidance.

## Audit purpose

The purpose of Gate R1 was to verify whether the installed PyGSD `MSConv` implementation uses the same edge orientation and complex-operator action assumed by the manuscript and the E001 experimental interpretation.

The audit was intentionally narrow. It did not:

- install or update software;
- edit the installed PyGSD package;
- modify the manuscript;
- retrain a model;
- submit a Slurm job;
- alter an existing checkpoint;
- delete or overwrite historical results;
- make a general claim about MSGNN performance.

## Environment

The audit reported:

- Python: `3.11.15`
- PyTorch: `2.12.0+cu130`
- PyTorch Geometric: `2.7.0`
- PyGSD: `1.1.1`
- execution device: CPU

Installed `MSConv` source:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/general/MSConv.py`

Installed `MSGNN_link_prediction` source:

`/scratch/shirpa/gnn-stability/miniconda3/envs/gnn_env/lib/python3.11/site-packages/torch_geometric_signed_directed/nn/general/MSGNN.py`

## Finding 1 — Effective message-passing flow

### Installed constructor behavior

The installed `MSConv.__init__` source performs these operations in the following order:

```python
kwargs.setdefault("aggr", "add")
super(MSConv, self).__init__(**kwargs)
...
kwargs.setdefault("flow", "target_to_source")
```

The `flow` default is added to the local `kwargs` dictionary only after the parent `MessagePassing` constructor has already executed.

The installed PyTorch Geometric `MessagePassing` constructor reports the default:

```text
flow='source_to_target'
```

Direct instantiation of `MSConv` reported:

```text
EFFECTIVE_FLOW: source_to_target
```

### Classification

- **Code-verified:** the effective flow of the installed `MSConv` object is `source_to_target`.
- **Code-verified:** the later `kwargs.setdefault("flow", "target_to_source")` statement does not change the already initialized `layer.flow` attribute in the tested environment.
- **Unresolved:** whether `source_to_target` or `target_to_source` is the intended PyGSD behavior for correspondence with the MSGNN mathematical convention.

## Finding 2 — Complex-component propagation source

Write a complex operator and feature vector as

\[
M=A+iB,
\qquad
x=x_R+i x_I.
\]

Ordinary complex matrix multiplication gives

\[
Mx
=
(Ax_R-Bx_I)
+
i(Bx_R+Ax_I).
\]

Therefore, the four required component operations are:

1. \(Ax_R\);
2. \(Bx_I\);
3. \(Bx_R\);
4. \(Ax_I\).

The inspected `MSConv.forward` source contained the following first-order propagation assignments. Line numbers below are relative to the source returned by `inspect.getsource(MSConv.forward)`:

```python
Tx_1_real_real = self.propagate(
    edge_index_real,
    x=x_real,
    norm=norm_real,
    size=None,
)

Tx_1_imag_imag = self.propagate(
    edge_index_imag,
    x=x_imag,
    norm=norm_imag,
    size=None,
)

Tx_1_imag_real = self.propagate(
    edge_index_real,
    x=x_real,
    norm=norm_real,
    size=None,
)

Tx_1_real_imag = self.propagate(
    edge_index_imag,
    x=x_imag,
    norm=norm_imag,
    size=None,
)
```

The final combination is:

```python
out_real = out_real_real - out_imag_imag
out_imag = out_imag_real + out_real_imag
```

Under the ordinary complex-multiplication interpretation:

- `real_real` represents \(Ax_R\);
- `imag_imag` represents \(Bx_I\);
- the expected `imag_real` cross term is \(Bx_R\);
- the expected `real_imag` cross term is \(Ax_I\).

In the installed source:

- `imag_real` repeats the same real-operator/real-input propagation used for \(Ax_R\);
- `real_imag` repeats the same imaginary-operator/imaginary-input propagation used for \(Bx_I\).

The installed message function itself is:

```python
def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j
```

This message function performs ordinary weighted message scaling. The questioned behavior arises from the selection of edge indices, norms, and feature components supplied to the propagation calls, together with the effective flow.

### Classification

- **Code-verified:** the installed first-order `imag_real` and `real_imag` calls use the component pairings shown above.
- **Mathematically expected:** ordinary complex multiplication would instead use the cross terms \(Bx_R\) and \(Ax_I\).
- **Unresolved:** whether PyGSD intentionally uses a different internal representation that would justify the observed pairing.

## Finding 3 — Two-node \(q=1/8\) operator-action check

A deterministic two-node test used:

- one positive directed edge \(0\rightarrow1\);
- \(q=0.125\);
- `K=1`;
- one input and one output channel;
- the first-order Chebyshev weight isolated;
- no stochastic training;
- one nonzero input node at a time.

The constructed sparse complex operator, represented as a row-column matrix, was:

```text
tensor([
    [ 0.000000+0.000000j, -0.707107-0.707107j],
    [-0.707107+0.707107j,  0.000000+0.000000j],
])
```

For a nonzero real input at node 0, the installed layer produced:

```text
MSCONV_OUTPUT:
tensor([0.000000+0.000000j, -0.707107-0.707107j])

DIFF_VS_M:           1.4142134189605713
DIFF_VS_M_TRANSPOSE: 0.0
```

For a nonzero real input at node 1, the installed layer produced:

```text
MSCONV_OUTPUT:
tensor([-0.707107-0.707107j, 0.000000+0.000000j])

DIFF_VS_M:           0.0
DIFF_VS_M_TRANSPOSE: 1.4142134189605713
```

Thus, under this test, the installed output did not consistently equal either \(Mx\) or \(M^{\mathsf T}x\) for both basis inputs.

### Classification

- **Code-verified:** the two basis-input results above were executed on Sol.
- **Unresolved:** the intended row-column convention and complex action of the installed layer.

## Finding 4 — Decisive \(q=0\) sanity check

At \(q=0\), the magnetic phase is zero, so the tested operator is real. For a purely real input,

\[
x_I=0,
\]

ordinary complex multiplication with a real operator requires

\[
\operatorname{Im}(Mx)=0.
\]

The deterministic test used:

- one positive edge \(0\rightarrow1\);
- `q=0.0`;
- `K=1`;
- first-order weight isolated;
- zeroth-order weight set to zero;
- bias disabled;
- input \(x_R=[1,0]^{\mathsf T}\);
- input \(x_I=[0,0]^{\mathsf T}\).

The executed result was:

```text
EFFECTIVE_FLOW: source_to_target
Q: 0.0
INPUT_REAL: tensor([1., 0.])
INPUT_IMAG: tensor([0., 0.])
OUTPUT_REAL: tensor([ 0.000000, -1.000000])
OUTPUT_IMAG: tensor([ 0.000000, -1.000000])
EXPECTED_AT_Q_ZERO: purely real input produces zero imaginary output
IMAGINARY_OUTPUT_IS_ZERO: False
Q_ZERO_SANITY_CHECK: FAIL
```

The imaginary output was not a small numerical residual. It exactly duplicated the nonzero real output in this test.

### Classification

- **Code-verified:** the installed layer failed the stated \(q=0\), purely real-input sanity check.
- **Mathematically expected:** the imaginary output should be zero under ordinary complex multiplication by a real operator.
- **Unresolved:** whether the layer is intended to implement a different representation whose semantics must be documented separately.

## Finding 5 — Scope within the E001 model

The accepted E001 configuration instantiated:

- model: `MSGNN_link_prediction`;
- number of convolution layers: `2`;
- `q=0.125`;
- `K=1`;
- `normalization="sym"`;
- `cached=False`;
- `absolute_degree=True`.

Direct inspection of the instantiated model reported:

```text
CHEB_LAYER_COUNT: 2

LAYER_1:
{
    'class': 'MSConv',
    'module': 'torch_geometric_signed_directed.nn.general.MSConv',
    'flow': 'source_to_target',
    'q': 0.125,
    'weight_shape': (2, 2, 16),
    'normalization': 'sym'
}

LAYER_2:
{
    'class': 'MSConv',
    'module': 'torch_geometric_signed_directed.nn.general.MSConv',
    'flow': 'source_to_target',
    'q': 0.125,
    'weight_shape': (2, 16, 16),
    'normalization': 'sym'
}

ALL_LAYERS_ARE_MSCONV: True
ALL_LAYERS_SOURCE_TO_TARGET: True
```

The installed `MSGNN_link_prediction.forward` method performs:

```python
for cheb in self.Chebs:
    real, imag = cheb(real, imag, edge_index, edge_weight)
    if self.activation:
        real, imag = self.complex_relu(real, imag)
```

It then constructs the link representation using both real and imaginary endpoint features:

```python
x = torch.cat(
    (
        real[query_edges[:, 0]],
        real[query_edges[:, 1]],
        imag[query_edges[:, 0]],
        imag[query_edges[:, 1]],
    ),
    dim=-1,
)
```

Therefore, both audited `MSConv` layers contribute directly to the representation consumed by the final link classifier.

### Classification

- **Code-verified:** both E001 convolution layers are installed PyGSD `MSConv` layers.
- **Code-verified:** both have effective `source_to_target` flow.
- **Code-verified:** both real and imaginary outputs contribute to the final link classifier.
- **Unresolved:** whether the resulting E001 checkpoint corresponds to the magnetic convolution intended by the manuscript and MSGNN mathematical formulation.

## Impact on preserved E001 evidence

### Evidence that remains valid

The following claims remain supported:

- **Historically executed / code-verified:** the accepted E001 clean run executed successfully using the documented Sol environment and installed PyGSD 1.1.1 package.
- **Historically executed / code-verified:** the checkpoint is reproducible by its recorded SHA-256 digest.
- **Historically executed / code-verified:** checkpoint reload produced zero maximum absolute output difference.
- **Historically executed / code-verified:** the accepted clean run produced the recorded validation and test metrics.
- **Code-verified:** the fixed-checkpoint diagnostic preserved the checkpoint, model state, and frozen data tensors.
- **Code-verified:** the separately constructed magnetic-adjacency calculation satisfied the restricted exact Frobenius identity within numerical precision.

### Evidence requiring narrower interpretation

Until implementation correspondence is resolved:

- E001 test accuracy is evidence of the behavior of the installed PyGSD 1.1.1 execution path.
- It is not yet evidence that the checkpoint implements the manuscript’s intended complex magnetic convolution.
- The diagnostic representation, logits, output changes, and prediction behavior describe the installed model.
- Those downstream measurements should not yet be interpreted as confirmed consequences of the manuscript-defined MSGNN operator.
- The exact custom magnetic-adjacency identity remains logically separate from the downstream MSGNN implementation question.

### Preserved accepted artifacts

Accepted E001 clean JSON:

`results/e001/e001_k1_q0125_seed0_clean_attempt02.json`

Recorded SHA-256:

`9efd03a6f5aac6ec3296924dda9aa5342020f9c2c94c4f671a6470e806599419`

Accepted sign-reversal diagnostic JSON:

`results/e001/e001_k1_q0125_seed0_sign_reversal_diagnostic_attempt02.json`

Recorded SHA-256:

`38de0599744981fa6f29a5e96f71e2e98e4faec4cd2d9cc5d5265ba62b426ea2`

Accepted MSGNN checkpoint:

`results/e001/msgnn_k1_q0125_seed0_20260901T163605Z.pt`

Recorded SHA-256:

`6b0d7f7faa2fb74ad7e2eca5ac186ab0c1313a2773c3b67bc82f0a1395d642ef`

The checkpoint remains intentionally excluded from Git by the repository’s `*.pt` rule and is independently archived on the iMac.

## Gate R1 disposition

### Passed portions

- installed package and source location identified;
- effective message flow directly observed;
- relevant propagation source inspected;
- deterministic two-node operator-action check executed;
- decisive \(q=0\) sanity check executed;
- impact on both E001 convolution layers confirmed;
- no package, checkpoint, or manuscript modification performed.

### Unresolved portion

Mathematical–implementation correspondence has not been established.

Gate R1 therefore has the following split result:

- environment and installed-behavior audit: **PASS**;
- reproducibility of the observed behavior: **PASS**;
- mathematical–implementation correspondence: **UNRESOLVED / FAILING SANITY CHECK**;
- readiness for publication-scale MSGNN execution: **PAUSED**.

## Mentor clarification requested

The evidence should be presented as a correspondence question, not as an assertion that PyGSD contains a confirmed defect.

Proposed question:

> In PyGSD 1.1.1, the installed `MSConv` object has effective `source_to_target` flow because the constructor sets `target_to_source` in `kwargs` only after the parent `MessagePassing` constructor has executed. In addition, the inspected first-order branches use the real operator with the real input for both `real_real` and `imag_real`, and the imaginary operator with the imaginary input for both `imag_imag` and `real_imag`. Under the standard calculation
> \[
> (A+iB)(x_R+i x_I)
> =
> (Ax_R-Bx_I)+i(Bx_R+Ax_I),
> \]
> I expected the imaginary-output branches to use \(Bx_R\) and \(Ax_I\). A two-node check at \(q=0\), with a purely real input and the first-order term isolated, produced a nonzero imaginary output equal to the real output. Is this component pairing an intentional representation used by PyGSD/MSGNN, or should the implementation use the standard complex cross terms? Also, which edge-flow convention should I use for manuscript-to-code correspondence?

## Authorized next actions

Before mentor clarification, permitted work should remain limited to:

- preserving this evidence record;
- preserving a minimal reproducible test;
- verifying hashes and source provenance;
- drafting the mentor clarification question;
- auditing other models read-only if separately authorized.

The following remain unauthorized unless explicitly approved:

- editing installed PyGSD source;
- implementing a local corrected `MSConv`;
- retraining E001;
- starting publication-scale runs;
- modifying manuscript equations or claims;
- replacing accepted historical artifacts;
- committing or pushing this record;
- emailing or messaging the mentor.
