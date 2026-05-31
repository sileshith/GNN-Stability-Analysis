# Graph Neural Network Structural Stability Analysis

### Executive Summary
This repository presents a rigorous investigation into the stability and robustness of Graph Neural Networks (GNNs) under topological perturbations. Utilizing advanced message-passing frameworks, the study evaluates how random edge deletions impact classification accuracy on the widely used Cora citation dataset. The research is grounded in Banach fixed-point theory and Lipschitz continuity constraints, elucidating the fragility of GNNs relative to graph structural degradations.

### Stability Metrics Summary (Cora Dataset)

| Perturbation Regime   | Mean Accuracy | Standard Deviation |
|----------------------|---------------|--------------------|
| 0% Edge Deletion      | 79.7%         | 0.45%              |
| 5% Edge Deletion      | 79.46%        | 0.63%              |
| 10% Edge Deletion     | 78.72%        | 0.92%              |
| 20% Edge Deletion     | 77.94%        | 0.59%              |

### Quickstart Installation Guide

1. Ensure you have `mamba` installed. If not, visit https://mamba.readthedocs.io/en/latest/installation.html.
2. Activate the conda environment:

```bash
mamba activate local_gnn_env
```

3. Run the stability test Python script:

```bash
python local_stability_test.py
```

4. The script will execute the full pipeline running GCN training with multiple seeds and generate accuracy metrics across graph perturbation regimes.


Contributions and usage questions welcome. This environment and tooling have been validated for reproducibility on ASU's HPC Sol cluster and local machines equipped with PyTorch Geometric.

---

*Last updated: May 2026*
