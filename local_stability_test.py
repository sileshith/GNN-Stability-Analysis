import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PortfolioGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def apply_edge_perturbation(edge_index, drop_rate=0.05, seed=42):
    """Numerically cuts down graph routing channels to measure structural stability."""
    if drop_rate == 0.0:
        return edge_index
    
    np.random.seed(seed)
    edges = edge_index.cpu().numpy()
    num_edges = edges.shape[1]
    num_to_drop = int(drop_rate * num_edges)
    
    all_indices = np.arange(num_edges)
    keep_indices = np.random.choice(all_indices, size=num_edges - num_to_drop, replace=False)
    
    return torch.tensor(edges[:, keep_indices], dtype=torch.long)

def run_crisp_dm_experiment():
    print("--- CRISP-DM PHASE 1 & 2: LOADING GENUINE CORA DATASET ---")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Nodes: {data.num_nodes} | Baseline Edges: {data.num_edges}")
    
    # Staging conditions for portfolio display
    seeds = [42, 101, 2023, 7, 88]
    perturbations = [0.0, 0.05, 0.10, 0.20]
    
    experiment_logs = {}
    
    print("\n--- CRISP-DM PHASE 3, 4 & 5: ITERATIVE MODELING & STABILITY EVALUATION ---")
    for p_rate in perturbations:
        p_accs = []
        print(f"\nEvaluating Graph Instability at {p_rate*100}% Edge Deletion:")
        
        for run_idx, seed in enumerate(seeds):
            set_seed(seed)
            
            # Prepare corrupted structure
            perturbed_edge_index = apply_edge_perturbation(data.edge_index, drop_rate=p_rate, seed=seed).to(device)
            current_data = data.to(device)
            
            # Initialize unique weight matrices per seed
            model = PortfolioGCN(dataset.num_features, 16, dataset.num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            
            # Train GCN on the specific topology loop
            model.train()
            for epoch in range(150):
                optimizer.zero_grad()
                out = model(current_data.x, perturbed_edge_index)
                loss = F.nll_loss(out[current_data.train_mask], current_data.y[current_data.train_mask])
                loss.backward()
                optimizer.step()
            
            # Evaluate Performance Accuracy
            model.eval()
            with torch.no_grad():
                logits = model(current_data.x, perturbed_edge_index)
                preds = logits.argmax(dim=1)
                correct = (preds[current_data.test_mask] == current_data.y[current_data.test_mask]).sum().item()
                acc = correct / current_data.test_mask.sum().item()
                p_accs.append(acc)
                
            print(f" └─ Seed Run {run_idx+1}/5 (Seed {seed}) -> Accuracy: {acc*100:.2f}%")
        
        experiment_logs[f"perturbation_{int(p_rate*100)}pct"] = {
            "mean_accuracy": float(np.mean(p_accs)),
            "std_deviation": float(np.std(p_accs)),
            "raw_scores": p_accs
        }
    
    print("\n--- CRISP-DM PHASE 6: DEPLOYMENT ARTIFACT GENERATION ---")
    # Save structured results to a clean deployment log
    with open("local_stability_metrics.json", "w") as f:
        json.dump(experiment_logs, f, indent=4)
    print("🎯 Success! Numerical logging asset deployed to: local_stability_metrics.json")

if __name__ == '__main__':
    run_crisp_dm_experiment()
