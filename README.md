# Graph Neural Network (GNN) Structural Stability Analysis  
### Scientific Machine Learning (SciML) Sandbox | ASU STP 499 Research Track

This repository documents a rigorous investigation into the stability, mathematical properties, and robustness of Graph Neural Networks (GNNs) under spatial message-passing degradation constraints. Grounded in Banach fixed-point theory and Lipschitz continuity bounding, this sandbox measures how structural topological perturbations degrade representation learning over the network.

## 🌐 Network Topology & Information Flow

Before evaluating adversarial noise or random edge erasure, we project the entire Cora dataset—consisting of 2,708 academic papers and 10,556 citation links—into a condensed, macro-topographic map to analyze how information flows across distinct research fields.

<p align="center">
  <img src="notes/macro_topography.png" alt="Cora Inter-Class Citation Dynamics Map" width="520">
  <br>
  <em>Figure 1: Macro-Topographic Map of Inter-Class Citation Dynamics across the Cora Dataset.</em>
</p>

### Asset Description & Structural Mapping
* **Subject Bubble Scaling:** The physical area of each class node (e.g., *Neural Networks*, *Genetic Algorithms*) is programmatically scaled based on its diagonal intra-class homophily value. Larger bubbles indicate tighter thematic clustering.
* **Directional Arrow Widths:** The thickness of the connecting arcs represents the empirical probability of cross-disciplinary citations (the off-diagonal transition elements) between different academic domains.

### Strategic Interpretation & Robustness Connection
This macro-map exposes the structural mechanics behind **Community Boundary Bleeding**. While the prominent self-loops demonstrate that feature aggregation is heavily concentrated inside homophilous class boundaries, the thin web of inter-class arrows represents active channels of cross-disciplinary communication. 

When a network experiences edge drops, dominant intra-class edges are stripped away first due to pure random distribution. As these internal clusters collapse, the relative proportion of cross-class noise flowing along the off-diagonal links increases. During spatial convolutions, node embeddings begin to drift out of alignment and bleed across semantic boundaries—forcing vanilla Graph Convolutional Networks (GCNs) into a non-linear accuracy drop down to **$77.94\%$**.

## Stability Metrics Summary (Cora Dataset)

The experimental decay curve tracks a monotonic performance degradation across 5 independent initialization seeds to isolate structural fragility from random initialization weight behavior:

| Adjacency Perturbation Level | Mean Test Accuracy (%) | Standard Deviation ($\sigma$) | Performance Delta ($\Delta$) | Impact Classification |
| :--- | :--- | :--- | :--- | :--- |
| **0% Edge Deletion (Baseline)** | 79.70% | ±0.45% | *Reference* | Optimal State |
| **5% Edge Deletion** | 79.46% | ±0.63% | $-0.24\%$ | Topological Buffer Zone |
| **10% Edge Deletion** | 78.72% | ±0.92% | $-0.98\%$ | Accelerated Decay Phase |
| **20% Edge Deletion** | 77.94% | ±0.59% | $-1.76\%$ | Structural Fragmentation |


## Key Framework Capacities

* **Multi-Scale Diagnostics:** Pairs global adjacency matrix sparsity tracking ($\rho \approx 0.14\%$) and Power-Law node degree distribution histograms with micro-level 2-Hop Ego Graph receptive field subplots.
* **Reproducible Controls:** Validated for complete reproducibility across both local workstation machines and ASU's High-Performance Computing (HPC) Sol cluster environment.
* **SciML Benchmarking Sandbox:** Establishes the definitive baseline reference curve needed to evaluate advanced structural continuity constraints, Lipschitz-bounded weight tensors, and Graph Attention Networks (GAT).


##  Core Mathematical Formulation

The structural decay evaluated in this sandbox directly manipulates the spatial convolution operator at layer $l+1$:

$$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

By programmatically deleting target edge percentages from the coordinate tensor (`edge_index`), we alter the perturbed adjacency matrix $\tilde{A}$, which shifts the spectrum of the normalized graph Laplacian:

$$L_{\text{sym}} = I - \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$

This restriction prevents the Laplacian operator from functioning as a reliable low-pass filter, resulting in the representation starvation and classification decay documented throughout this pipeline.
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
