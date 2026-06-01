import nbformat as nbf
from pathlib import Path

notebook_path = Path("notebooks/02_train_gcn_cora_on_sol.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md(r"""# 02 Train a Two-Layer GCN on the Cora Citation Network

**Project:** GNN Structural Stability Analysis  
**Track:** ASU STP 499 Scientific Machine Learning Research Preparation  
**Environment:** Local iMac and ASU Sol Supercomputer  
**Notebook role:** Clean baseline training before perturbation testing

## Purpose

This notebook trains a clean two-layer Graph Convolutional Network, or GCN, on the Cora citation network using PyTorch Geometric.

The goal is to establish a clean baseline model before testing how graph structure changes affect model performance.

Notebook 03 will build on this baseline by testing edge perturbations such as 5%, 10%, and 20% edge deletion.
""")

md(r"""## 1. Research Framing

A Graph Neural Network can be viewed as a mapping

$$
F(G, X) \mapsto Y,
$$

where:

- $G = (V, E)$ is the graph structure,
- $X$ is the node feature matrix,
- $Y$ is the predicted node label output.

In this notebook, the graph structure is kept clean and unchanged. This gives us the reference model needed before we study structural stability.
""")

code(r"""import sys
import platform
import random

import numpy as np
import torch

print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
""")

md(r"""## 2. Reproducibility Setup

We set a random seed so that training behavior is easier to compare across runs.

This does not remove all possible variation, but it improves reproducibility.
""")

code(r"""def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
""")

md(r"""## 3. Load the Cora Dataset

Cora is a citation network dataset.

Each node represents a scientific paper.  
Each edge represents a citation link.  
Each node has a feature vector and a class label.

The task is semi-supervised node classification.
""")

code(r"""from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="./data/Cora", name="Cora")
data = dataset[0].to(device)

print(dataset)
print(data)
""")

md(r"""## 4. Dataset Summary

The most important graph quantities are:

- number of nodes,
- number of edges,
- number of node features,
- number of prediction classes,
- train, validation, and test mask sizes.
""")

code(r"""summary = {
    "nodes": data.num_nodes,
    "edges": data.num_edges,
    "node_features": dataset.num_features,
    "classes": dataset.num_classes,
    "training_nodes": int(data.train_mask.sum()),
    "validation_nodes": int(data.val_mask.sum()),
    "test_nodes": int(data.test_mask.sum()),
}

summary
""")

md(r"""## 5. GCN Mathematical Formulation

A common GCN layer can be written as:

$$
H^{(l+1)} =
\sigma \left(
\tilde{D}^{-1/2}
\tilde{A}
\tilde{D}^{-1/2}
H^{(l)}
W^{(l)}
\right),
$$

where:

- $\tilde{A}$ is the adjacency matrix with self-loops,
- $\tilde{D}$ is the degree matrix,
- $H^{(l)}$ is the node representation at layer $l$,
- $W^{(l)}$ is a trainable weight matrix,
- $\sigma$ is a nonlinear activation function.

In this notebook, we use two GCN layers.
""")

md(r"""## 6. Define the Two-Layer GCN Model

The model has this structure:

$$
1433 \rightarrow 16 \rightarrow 7
$$

That means:

- 1,433 input features per node,
- 16 hidden units,
- 7 output classes.
""")

code(r"""import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = BaselineGCN(
    in_channels=dataset.num_features,
    hidden_channels=16,
    out_channels=dataset.num_classes,
).to(device)

model
""")

md(r"""## 7. Accuracy Function

This helper function evaluates the model on one split of the data.

The split is selected using a mask:

- `train_mask`
- `val_mask`
- `test_mask`
""")

code(r"""@torch.no_grad()
def accuracy(mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    predictions = logits.argmax(dim=1)
    correct = (predictions[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    return correct / total
""")

md(r"""## 8. Training Setup

We use:

- cross-entropy loss,
- Adam optimizer,
- learning rate $0.01$,
- weight decay $5 \times 10^{-4}$,
- 200 training epochs.

The loss is computed only on the training nodes.
""")

code(r"""optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

criterion = torch.nn.CrossEntropyLoss()
""")

md(r"""## 9. Train the GCN

During training, the model learns from the labeled training nodes.

We also monitor validation accuracy to see how well the model generalizes during training.
""")

code(r"""history = []

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0:
        train_acc = accuracy(data.train_mask)
        val_acc = accuracy(data.val_mask)

        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
""")

md(r"""## 10. Final Evaluation

After training, we evaluate the model on:

- training nodes,
- validation nodes,
- test nodes.

The test accuracy becomes the clean baseline for the later stability experiment.
""")

code(r"""final_train_acc = accuracy(data.train_mask)
final_val_acc = accuracy(data.val_mask)
final_test_acc = accuracy(data.test_mask)

results = {
    "train_accuracy": final_train_acc,
    "validation_accuracy": final_val_acc,
    "test_accuracy": final_test_acc,
}

results
""")

md(r"""## 11. Interpretation

This notebook establishes the clean GCN baseline on the original Cora graph.

The result should not be interpreted as a final research conclusion. It is a reference point.

In the next notebook, we will modify the graph structure by removing a controlled percentage of edges. Then we will compare perturbed test accuracy against this clean baseline.

A simple stability comparison will use:

$$
\Delta_{\text{accuracy}} =
\text{Accuracy}_{\text{clean}} -
\text{Accuracy}_{\text{perturbed}}
$$
""")

md(r"""## 12. Next Step

The next notebook is `03_edge_perturbation_stability_test.ipynb`.

It will study how the trained GCN behaves when the graph structure is changed through controlled edge deletion.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (local_gnn_env)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3"
    }
}

with notebook_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created {notebook_path}")
