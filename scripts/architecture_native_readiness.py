#!/usr/bin/env python3
"""One-seed GPU readiness checks for MSGNN, MagNet, and SGCN.

This runner is intentionally narrow. Each model uses an architecture-native
synthetic graph view and task:

* MSGNN: SDSBM signed-directed four-class link prediction.
* MagNet: DSBM directed two-class direction prediction.
* SGCN: SSBM signed link-sign prediction.

The runner trains one clean seed, saves a checkpoint, creates a fresh model,
reloads the checkpoint, verifies identical evaluation outputs, and records a
JSON summary. It does not apply structural perturbations or compare model
performance across architectures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric_signed_directed.data import SignedData
from torch_geometric_signed_directed.data.directed.DSBM import DSBM
from torch_geometric_signed_directed.data.signed.SSBM import SSBM
from torch_geometric_signed_directed.nn.directed.MagNet_link_prediction import (
    MagNet_link_prediction,
)
from torch_geometric_signed_directed.nn.general.MSGNN import MSGNN_link_prediction
from torch_geometric_signed_directed.nn.signed.SGCN import SGCN
from torch_geometric_signed_directed.utils.general import (
    in_out_degree,
    link_class_split,
)
from torch_geometric_signed_directed.utils.signed import create_spectral_features


@dataclass
class ReadinessCase:
    model_name: str
    task: str
    model_factory: Callable[[], nn.Module]
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer]
    train_step: Callable[[nn.Module], torch.Tensor]
    validation_output: Callable[[nn.Module], tuple[torch.Tensor, torch.Tensor]]
    test_output: Callable[[nn.Module], tuple[torch.Tensor, torch.Tensor]]
    dataset_summary: dict[str, Any]
    output_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one clean, architecture-native GPU readiness test."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("msgnn", "magnet", "sgcn"),
        help="Architecture to test.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path.home()
        / "stp499_persistent"
        / "checkpoints"
        / "architecture_readiness",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains a non-finite value")


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_record(log_prob: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    require_finite("evaluation output", log_prob)
    if log_prob.ndim != 2:
        raise RuntimeError(f"expected rank-2 class output, received {log_prob.shape}")
    if labels.ndim != 1 or labels.shape[0] != log_prob.shape[0]:
        raise RuntimeError("evaluation labels do not match output rows")
    prediction = log_prob.argmax(dim=1)
    truth_np = labels.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()
    return {
        "nll": float(nn.functional.nll_loss(log_prob, labels).item()),
        "accuracy": float(accuracy_score(truth_np, prediction_np)),
        "macro_f1": float(
            f1_score(truth_np, prediction_np, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(truth_np, prediction_np, average="micro", zero_division=0)
        ),
        "rows": int(log_prob.shape[0]),
        "classes": int(log_prob.shape[1]),
        "predicted_classes": sorted(int(x) for x in prediction.unique().tolist()),
    }


def magnetic_case(
    architecture: str,
    seed: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> ReadinessCase:
    if architecture == "msgnn":
        # Import the previously verified Gate E data contract without changing it.
        import pygsd_gate_e_clean_baseline as gate_e

        shared = gate_e.create_shared_data()
        graph = shared.train_graph.to(device)
        weights = shared.train_weights.to(device)
        features = shared.features.to(device)
        train_edges = shared.train_queries.to(device)
        train_labels = shared.train_labels.to(device)
        val_edges = shared.val_queries.to(device)
        val_labels = shared.val_labels.to(device)
        test_edges = shared.test_queries.to(device)
        test_labels = shared.test_labels.to(device)

        config = dict(gate_e.MSGNN_CONFIG)
        config["num_features"] = int(features.shape[1])

        def model_factory() -> nn.Module:
            return MSGNN_link_prediction(**config).to(device)

        task = "four_class_signed_digraph"
        model_name = "MSGNN_link_prediction"
        fingerprint = shared.bundle_fingerprint
    else:
        meta_graph = np.array(
            [
                [0.5, 0.1, 0.9],
                [0.9, 0.5, 0.1],
                [0.1, 0.9, 0.5],
            ],
            dtype=float,
        )
        adjacency, communities = DSBM(
            N=300,
            K=3,
            p=0.05,
            F=meta_graph,
            size_ratio=1.5,
        )
        data = SignedData(
            A=adjacency,
            y=torch.as_tensor(communities, dtype=torch.long),
        )
        split = link_class_split(
            data,
            splits=1,
            task="direction",
            prob_val=0.15,
            prob_test=0.15,
            seed=seed,
            device="cpu",
        )[0]
        graph_cpu = split["graph"]
        weights_cpu = split["weights"]
        features_cpu = in_out_degree(
            graph_cpu,
            size=data.num_nodes,
            signed=False,
            edge_weight=torch.abs(weights_cpu),
        )

        graph = graph_cpu.to(device)
        weights = weights_cpu.to(device)
        features = features_cpu.to(device)
        train_edges = split["train"]["edges"].to(device)
        train_labels = split["train"]["label"].to(device)
        val_edges = split["val"]["edges"].to(device)
        val_labels = split["val"]["label"].to(device)
        test_edges = split["test"]["edges"].to(device)
        test_labels = split["test"]["label"].to(device)

        config = {
            "num_features": int(features.shape[1]),
            "hidden": 16,
            "q": 0.25,
            "K": 2,
            "label_dim": 2,
            "activation": True,
            "trainable_q": False,
            "layer": 2,
            "dropout": 0.5,
            "normalization": "sym",
            "cached": False,
        }

        def model_factory() -> nn.Module:
            return MagNet_link_prediction(**config).to(device)

        task = "direction"
        model_name = "MagNet_link_prediction"
        fingerprint = tensor_fingerprint(
            graph_cpu,
            weights_cpu,
            split["train"]["edges"],
            split["train"]["label"],
            split["val"]["edges"],
            split["val"]["label"],
            split["test"]["edges"],
            split["test"]["label"],
            features_cpu,
        )

    def forward(model: nn.Module, query_edges: torch.Tensor) -> torch.Tensor:
        return model(
            features,
            features.clone(),
            graph,
            query_edges,
            weights,
        )

    criterion = nn.NLLLoss()

    def train_step(model: nn.Module) -> torch.Tensor:
        return criterion(forward(model, train_edges), train_labels)

    def validation_output(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return forward(model, val_edges), val_labels

    def test_output(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return forward(model, test_edges), test_labels

    def optimizer_factory(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return ReadinessCase(
        model_name=model_name,
        task=task,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        train_step=train_step,
        validation_output=validation_output,
        test_output=test_output,
        dataset_summary={
            "generator": "SDSBM" if architecture == "msgnn" else "DSBM",
            "nodes": int(features.shape[0]),
            "training_graph_edges": int(graph.shape[1]),
            "training_queries": int(train_edges.shape[0]),
            "validation_queries": int(val_edges.shape[0]),
            "test_queries": int(test_edges.shape[0]),
            "feature_shape": list(features.shape),
            "fingerprint": fingerprint,
            "graph_view": "signed-directed"
            if architecture == "msgnn"
            else "directed-unsigned",
        },
        output_kind="query-edge log-probabilities",
    )


def sgcn_case(
    seed: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> ReadinessCase:
    adjacency_pair, communities = SSBM(
        n=300,
        k=3,
        pin=0.08,
        etain=0.1,
        pout=0.03,
        size_ratio=1.5,
        etaout=0.1,
        values="ones",
    )
    data = SignedData(
        A=adjacency_pair,
        y=torch.as_tensor(communities, dtype=torch.long),
    )
    split = link_class_split(
        data,
        splits=1,
        task="sign",
        prob_val=0.15,
        prob_test=0.15,
        seed=seed,
        device="cpu",
    )[0]

    graph_cpu = split["graph"]
    weights_cpu = split["weights"].sign()
    edge_index_s_cpu = torch.cat(
        [graph_cpu.t(), weights_cpu.reshape(-1, 1)],
        dim=1,
    ).long()
    positive_cpu = graph_cpu[:, weights_cpu > 0]
    negative_cpu = graph_cpu[:, weights_cpu < 0]
    initial_embeddings = create_spectral_features(
        pos_edge_index=positive_cpu,
        neg_edge_index=negative_cpu,
        node_num=data.num_nodes,
        dim=32,
    )

    edge_index_s = edge_index_s_cpu.to(device)
    initial_embeddings = initial_embeddings.to(device)
    val_edges = split["val"]["edges"].t().contiguous().to(device)
    val_labels = split["val"]["label"].to(device)
    test_edges = split["test"]["edges"].t().contiguous().to(device)
    test_labels = split["test"]["label"].to(device)

    config = {
        "node_num": int(data.num_nodes),
        "in_dim": 32,
        "out_dim": 32,
        "layer_num": 2,
        "init_emb_grad": False,
        "lamb": 5.0,
        "norm_emb": False,
    }

    def model_factory() -> nn.Module:
        return SGCN(
            edge_index_s=edge_index_s,
            init_emb=initial_embeddings.detach().clone(),
            **config,
        ).to(device)

    def train_step(model: nn.Module) -> torch.Tensor:
        return model.loss()

    def validation_output(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = model()
        return model.lsp_loss.discriminate(embeddings, val_edges), val_labels

    def test_output(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = model()
        return model.lsp_loss.discriminate(embeddings, test_edges), test_labels

    def optimizer_factory(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return ReadinessCase(
        model_name="SGCN",
        task="link_sign_prediction",
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        train_step=train_step,
        validation_output=validation_output,
        test_output=test_output,
        dataset_summary={
            "generator": "SSBM",
            "nodes": int(data.num_nodes),
            "training_graph_edges": int(graph_cpu.shape[1]),
            "training_positive_edges": int((weights_cpu > 0).sum().item()),
            "training_negative_edges": int((weights_cpu < 0).sum().item()),
            "validation_queries": int(val_edges.shape[1]),
            "test_queries": int(test_edges.shape[1]),
            "fingerprint": tensor_fingerprint(
                graph_cpu,
                weights_cpu,
                split["train"]["edges"],
                split["train"]["label"],
                split["val"]["edges"],
                split["val"]["label"],
                split["test"]["edges"],
                split["test"]["label"],
                initial_embeddings,
            ),
            "graph_view": "signed-undirected",
        },
        output_kind="node embeddings and query-edge sign log-probabilities",
    )


def execute(
    case: ReadinessCase,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = case.model_factory()
    optimizer = case.optimizer_factory(model)

    operator_modules = sorted(
        {
            module.__class__.__name__
            for module in model.modules()
            if "conv" in module.__class__.__name__.lower()
        }
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    best_val_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    first_train_loss: float | None = None
    final_train_loss: float | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = case.train_step(model)
        require_finite(f"epoch {epoch} training loss", loss)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("no gradients were produced")
        for gradient in gradients:
            require_finite(f"epoch {epoch} gradient", gradient)

        optimizer.step()
        for parameter in model.parameters():
            require_finite(f"epoch {epoch} parameter", parameter)

        final_train_loss = float(loss.item())
        if first_train_loss is None:
            first_train_loss = final_train_loss

        model.eval()
        with torch.no_grad():
            validation_log_prob, validation_labels = case.validation_output(model)
            validation_nll = float(
                nn.functional.nll_loss(
                    validation_log_prob,
                    validation_labels,
                ).item()
            )
        if validation_nll < best_val_nll - 1e-6:
            best_val_nll = validation_nll
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={final_train_loss:.6f} "
            f"val_nll={validation_nll:.6f}"
        )
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_log_prob, validation_labels = case.validation_output(model)
        test_log_prob, test_labels = case.test_output(model)
        if case.model_name == "SGCN":
            exposed_output = model()
        else:
            exposed_output = test_log_prob

    validation_metrics = metric_record(validation_log_prob, validation_labels)
    test_metrics = metric_record(test_log_prob, test_labels)
    require_finite("exposed model output", exposed_output)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.artifact_dir / f"{args.model}_seed{args.seed}_clean_checkpoint.pt"
    )
    checkpoint = {
        "model": args.model,
        "model_name": case.model_name,
        "task": case.task,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_summary": case.dataset_summary,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric_signed_directed": importlib.metadata.version(
                "torch-geometric-signed-directed"
            ),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
    }
    torch.save(checkpoint, checkpoint_path)

    reloaded_payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    reloaded_model = case.model_factory()
    reloaded_model.load_state_dict(reloaded_payload["model_state_dict"])
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_test_output, reloaded_test_labels = case.test_output(reloaded_model)
    if not torch.equal(test_labels, reloaded_test_labels):
        raise RuntimeError("checkpoint reload changed test labels")
    reload_max_abs_difference = float(
        (test_log_prob - reloaded_test_output).abs().max().item()
    )
    if reload_max_abs_difference > 1e-5:
        raise RuntimeError(
            "checkpoint reload output mismatch: "
            f"{reload_max_abs_difference:.9g}"
        )

    result = {
        "status": "PASS",
        "scope": "clean architecture-native readiness only",
        "model": args.model,
        "model_name": case.model_name,
        "task": case.task,
        "seed": args.seed,
        "device": str(device),
        "device_name": checkpoint["environment"]["device_name"],
        "epochs_requested": args.epochs,
        "epochs_executed": epoch,
        "best_epoch": best_epoch,
        "first_train_loss": first_train_loss,
        "final_train_loss": final_train_loss,
        "best_val_nll": best_val_nll,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_reload_max_abs_difference": reload_max_abs_difference,
        "output_kind": case.output_kind,
        "exposed_output_shape": list(exposed_output.shape),
        "operator_modules": operator_modules,
        "trainable_parameters": trainable_parameters,
        "dataset": case.dataset_summary,
        "environment": checkpoint["environment"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "one clean synthetic dataset",
            "one initialization seed",
            "no structural perturbations",
            "no architecture ranking",
            "no robustness or stability conclusion",
        ],
    }
    result_path = args.artifact_dir / f"{args.model}_seed{args.seed}_readiness.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    result["result_file"] = str(result_path)
    return result


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expected exactly one scheduler-visible GPU, received "
            f"{torch.cuda.device_count()}"
        )

    if args.model in {"msgnn", "magnet"}:
        case = magnetic_case(
            architecture=args.model,
            seed=args.seed,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        case = sgcn_case(
            seed=args.seed,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    result = execute(case, args, device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
