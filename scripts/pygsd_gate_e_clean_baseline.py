"""
Gate E2 — Clean Functional-Baseline Implementation

Implements the locked SDSBM clean-baseline configuration for MSGNN_link_prediction
and directed SSSNET_link_prediction on the four_class_signed_digraph task.

Tests multi-epoch training with early stopping, checkpoint restoration, and
validation/test evaluation under controlled initialization seeds.

Does NOT test: perturbations, robustness, stability, hyperparameter search,
model ranking, or architecture superiority.

CRITICAL ATTRIBUTION:
PyGSD (torch-geometric-signed-directed) is advisor-authored upstream research
software developed by Dr. Yixuan He and collaborators. This script uses the
installed PyGSD 1.1.1 package as a verified implementation reference following
documented APIs and official examples. This project does not claim authorship
of SDSBM, MSGNN, SSSNET, or PyGSD utilities.
"""

import argparse
import hashlib
import json
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric_signed_directed.data import SDSBM, SignedData
from torch_geometric_signed_directed.nn.general.MSGNN import MSGNN_link_prediction
from torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction import (
    SSSNET_link_prediction,
)
from torch_geometric_signed_directed.utils import (
    extract_network,
    in_out_degree,
    link_class_split,
)


# Locked configuration constants
GENERATION_SEED = 0
SPLIT_SEED = 0
N = 400
K = 4
P = 0.05
SIZE_RATIO = 1.5
ETA = 0.1
GAMMA = 0.1
PROB_VAL = 0.15
PROB_TEST = 0.15
TASK = "four_class_signed_digraph"
LABEL_DIM = 4
DEVICE = torch.device("cpu")

# Training configuration
MAX_EPOCHS = 300
PATIENCE = 30
MIN_IMPROVEMENT = 1e-4
MIN_EPOCHS = 10
LR = 0.01
WEIGHT_DECAY = 5e-4

# MSGNN configuration
MSGNN_CONFIG = {
    "hidden": 16,
    "q": 0.25,
    "K": 2,
    "label_dim": LABEL_DIM,
    "activation": True,
    "trainable_q": False,
    "layer": 2,
    "dropout": 0.5,
    "normalization": "sym",
    "cached": False,
    "conv_bias": True,
    "absolute_degree": True,
}

# SSSNET configuration
SSSNET_CONFIG = {
    "hidden": 16,
    "nclass": LABEL_DIM,
    "dropout": 0.5,
    "hop": 2,
    "fill_value": 0.5,
    "directed": True,
    "bias": True,
}


class GateEValidationError(Exception):
    """Raised when a Gate E validation requirement fails."""
    pass


class StagePreservingError(Exception):
    """Exception that preserves the exact stage where failure occurred."""

    def __init__(self, stage: str, original_exception: Exception):
        self.stage = stage
        self.original_exception = original_exception
        self.original_type = type(original_exception).__name__
        self.original_message = str(original_exception)
        self.original_traceback = traceback.format_exc()
        super().__init__(f"Failed at stage '{stage}': {self.original_message}")


@dataclass
class SharedData:
    """Shared data contract for both models."""

    clean_graph: torch.Tensor
    clean_weights: torch.Tensor
    train_graph: torch.Tensor
    train_weights: torch.Tensor
    train_queries: torch.Tensor
    train_labels: torch.Tensor
    val_queries: torch.Tensor
    val_labels: torch.Tensor
    test_queries: torch.Tensor
    test_labels: torch.Tensor
    features: torch.Tensor
    num_nodes: int
    generation_seed: int
    split_seed: int
    clean_graph_fingerprint: str
    clean_weights_fingerprint: str
    train_graph_fingerprint: str
    train_weights_fingerprint: str
    train_queries_fingerprint: str
    train_labels_fingerprint: str
    val_queries_fingerprint: str
    val_labels_fingerprint: str
    test_queries_fingerprint: str
    test_labels_fingerprint: str
    features_fingerprint: str
    bundle_fingerprint: str


@dataclass
class SeedResult:
    """Result record for one model-seed run."""

    model_name: str
    seed: int
    stage: str
    passed: bool
    failure_category: str
    exception_type: str
    exception_message: str
    traceback_text: str
    config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    initial_train_loss: float
    initial_val_loss: float
    best_epoch: int
    best_val_loss: float
    stopping_epoch: int
    stopping_reason: str
    total_epochs: int
    final_executed_epoch: int
    early_stopped: bool
    per_epoch_train_loss: List[float]
    per_epoch_val_loss: List[float]
    gradient_finite: bool
    parameters_finite: bool
    checkpoint_restored: bool
    restored_checkpoint_val_loss: float
    test_loss: float
    test_accuracy: float
    test_macro_f1: float
    test_micro_f1: float
    runtime_seconds: float
    learning_signal_satisfied: bool
    finite_value_checks: Dict[str, bool]
    warnings: List[str]
    bundle_fingerprint: str


def require(condition: bool, message: str) -> None:
    """Raise GateEValidationError if condition is False."""
    if not condition:
        raise GateEValidationError(message)


def compute_tensor_hash(name: str, tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash of tensor metadata and contiguous CPU bytes."""
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(str(tensor.dtype).encode("utf-8"))
    h.update(str(tuple(tensor.shape)).encode("utf-8"))
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compute_bundle_fingerprint(
    clean_graph: torch.Tensor,
    clean_weights: torch.Tensor,
    train_graph: torch.Tensor,
    train_weights: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    val_queries: torch.Tensor,
    val_labels: torch.Tensor,
    test_queries: torch.Tensor,
    test_labels: torch.Tensor,
    features: torch.Tensor,
    generation_seed: int,
    split_seed: int,
) -> str:
    """Compute bundle fingerprint from all shared tensors and seeds."""
    h = hashlib.sha256()
    h.update(compute_tensor_hash("clean_graph", clean_graph).encode("utf-8"))
    h.update(compute_tensor_hash("clean_weights", clean_weights).encode("utf-8"))
    h.update(compute_tensor_hash("train_graph", train_graph).encode("utf-8"))
    h.update(compute_tensor_hash("train_weights", train_weights).encode("utf-8"))
    h.update(compute_tensor_hash("train_queries", train_queries).encode("utf-8"))
    h.update(compute_tensor_hash("train_labels", train_labels).encode("utf-8"))
    h.update(compute_tensor_hash("val_queries", val_queries).encode("utf-8"))
    h.update(compute_tensor_hash("val_labels", val_labels).encode("utf-8"))
    h.update(compute_tensor_hash("test_queries", test_queries).encode("utf-8"))
    h.update(compute_tensor_hash("test_labels", test_labels).encode("utf-8"))
    h.update(compute_tensor_hash("features", features).encode("utf-8"))
    h.update(str(generation_seed).encode("utf-8"))
    h.update(str(split_seed).encode("utf-8"))
    return h.hexdigest()


def classify_failure(stage: str, exception: Exception) -> str:
    """Classify failure into a structured category using explicit stage sets."""

    data_contract_stages = {
        "SDSBM generation",
        "network extraction",
        "SignedData creation",
        "split extraction",
        "split validation",
        "feature construction",
        "fingerprint computation",
        "audit copy creation",
        "audit fingerprint verification",
    }

    preprocessing_stages = {
        "positive/negative decomposition",
    }

    model_api_stages = {
        "model construction",
        "initial forward execution",
        "epoch forward execution",
        "validation forward execution",
        "test forward execution",
    }

    numerical_stages = {
        "initial output validation",
        "initial loss validation",
        "epoch output validation",
        "epoch loss validation",
        "validation output validation",
        "validation loss validation",
        "test output validation",
    }

    gradient_stages = {
        "backward",
        "gradient validation",
    }

    optimizer_step_stages = {
        "optimizer step",
        "parameter validation",
    }

    checkpoint_stages = {
        "checkpoint save",
        "checkpoint restore",
    }

    evaluation_stages = {
        "metric computation",
    }

    if stage in data_contract_stages:
        return "data-contract failure"
    elif stage in preprocessing_stages:
        return "preprocessing failure"
    elif stage in model_api_stages:
        return "model-API failure"
    elif stage in numerical_stages:
        return "numerical failure"
    elif stage in gradient_stages:
        return "gradient failure"
    elif stage in optimizer_step_stages:
        return "optimizer-step failure"
    elif stage in checkpoint_stages:
        return "checkpoint failure"
    elif stage in evaluation_stages:
        return "evaluation-path failure"
    elif isinstance(exception, (AssertionError, GateEValidationError)):
        return "implementation defect"
    else:
        return "unresolved package behavior"


def print_structured_failure(
    *,
    category: str,
    stage: str,
    exception_type: str,
    message: str,
    traceback_text: str,
) -> None:
    """Print structured failure information."""
    print(f"  Failure category: {category}")
    print(f"  Failed stage: {stage}")
    print(f"  Exception type: {exception_type}")
    print(f"  Exception message: {message}")
    print(f"  Traceback:\n{traceback_text}")


def validate_output(
    output: torch.Tensor,
    expected_shape: Tuple[int, int],
    stage_prefix: str,
) -> None:
    """Validate forward-pass output tensor."""
    require(isinstance(output, torch.Tensor), f"{stage_prefix}: expected torch.Tensor")
    require(output.ndim == 2, f"{stage_prefix}: expected 2D tensor")
    require(
        tuple(output.shape) == expected_shape, f"{stage_prefix}: shape mismatch"
    )
    require(output.is_floating_point(), f"{stage_prefix}: expected floating dtype")
    require(output.device == DEVICE, f"{stage_prefix}: expected CPU device")
    require(
        torch.isfinite(output).all().item(), f"{stage_prefix}: non-finite values"
    )


def validate_loss(loss: torch.Tensor, stage_prefix: str) -> None:
    """Validate loss tensor."""
    require(isinstance(loss, torch.Tensor), f"{stage_prefix}: expected torch.Tensor")
    require(loss.ndim == 0, f"{stage_prefix}: expected scalar")
    require(loss.device == DEVICE, f"{stage_prefix}: expected CPU device")
    require(torch.isfinite(loss).item(), f"{stage_prefix}: non-finite loss")


def create_shared_data() -> SharedData:
    """Create shared SDSBM graph, split, and features."""
    stage = "SDSBM generation"

    try:
        # Set all seeds
        random.seed(GENERATION_SEED)
        np.random.seed(GENERATION_SEED)
        torch.manual_seed(GENERATION_SEED)

        # Construct meta-graph matrix F
        F = np.array(
            [
                [0.5, 0.1, -0.1, 0.1],
                [0.9, 0.5, -0.1, -0.5],
                [-0.9, -0.9, 0.5, -0.9],
                [-0.9, -0.5, -0.1, 0.5],
            ]
        )

        # Generate SDSBM
        A, y = SDSBM(N, K, P, F, SIZE_RATIO, ETA)

        # Extract network
        stage = "network extraction"
        A, y = extract_network(A, y)

        # Create SignedData
        stage = "SignedData creation"
        data = SignedData(A=A, y=torch.as_tensor(y, dtype=torch.long))

        # Split
        stage = "split extraction"
        split_result = link_class_split(
            data,
            splits=1,
            task=TASK,
            prob_val=PROB_VAL,
            prob_test=PROB_TEST,
            seed=SPLIT_SEED,
            device="cpu",
        )

        split = split_result[0]

        clean_graph = data.edge_index
        clean_weights = data.edge_weight
        train_graph = split["graph"]
        train_weights = split["weights"]
        train_queries = split["train"]["edges"]
        train_labels = split["train"]["label"]
        val_queries = split["val"]["edges"]
        val_labels = split["val"]["label"]
        test_queries = split["test"]["edges"]
        test_labels = split["test"]["label"]

        # Validate split
        stage = "split validation"
        require(clean_graph.shape[0] == 2, "clean graph not 2D edge index")
        require(clean_weights.ndim == 1, "clean weights not 1D")
        require(
            clean_graph.shape[1] == clean_weights.shape[0],
            "clean graph/weight count mismatch",
        )
        require((clean_weights != 0.0).all().item(), "zero weights in clean graph")
        require(
            (clean_weights > 0.0).any().item(), "no positive weights in clean graph"
        )
        require(
            (clean_weights < 0.0).any().item(), "no negative weights in clean graph"
        )

        require(train_graph.shape[0] == 2, "train graph not 2D edge index")
        require(train_weights.ndim == 1, "train weights not 1D")
        require(
            train_graph.shape[1] == train_weights.shape[0],
            "train graph/weight count mismatch",
        )
        require((train_weights != 0.0).all().item(), "zero weights in train graph")
        require(
            (train_weights > 0.0).any().item(), "no positive weights in train graph"
        )
        require(
            (train_weights < 0.0).any().item(), "no negative weights in train graph"
        )

        require(train_queries.ndim == 2, "train queries not 2D")
        require(train_queries.shape[1] == 2, "train queries wrong width")
        require(train_labels.ndim == 1, "train labels not 1D")
        require(
            train_queries.shape[0] == train_labels.shape[0],
            "train query/label count mismatch",
        )
        require(train_labels.dtype == torch.long, "train labels not long")
        require(train_labels.min().item() >= 0, "train label below 0")
        require(train_labels.max().item() <= 3, "train label above 3")
        require(train_queries.shape[0] > 0, "empty train split")

        unique_train_labels = set(train_labels.tolist())
        require(
            unique_train_labels == {0, 1, 2, 3},
            "training labels missing required classes",
        )

        require(val_queries.ndim == 2, "val queries not 2D")
        require(val_queries.shape[1] == 2, "val queries wrong width")
        require(val_labels.ndim == 1, "val labels not 1D")
        require(
            val_queries.shape[0] == val_labels.shape[0],
            "val query/label count mismatch",
        )
        require(val_labels.dtype == torch.long, "val labels not long")
        require(val_labels.min().item() >= 0, "val label below 0")
        require(val_labels.max().item() <= 3, "val label above 3")
        require(val_queries.shape[0] > 0, "empty val split")

        unique_val_labels = set(val_labels.tolist())
        require(
            unique_val_labels == {0, 1, 2, 3},
            "validation labels missing required classes",
        )

        require(test_queries.ndim == 2, "test queries not 2D")
        require(test_queries.shape[1] == 2, "test queries wrong width")
        require(test_labels.ndim == 1, "test labels not 1D")
        require(
            test_queries.shape[0] == test_labels.shape[0],
            "test query/label count mismatch",
        )
        require(test_labels.dtype == torch.long, "test labels not long")
        require(test_labels.min().item() >= 0, "test label below 0")
        require(test_labels.max().item() <= 3, "test label above 3")
        require(test_queries.shape[0] > 0, "empty test split")

        unique_test_labels = set(test_labels.tolist())
        require(
            unique_test_labels == {0, 1, 2, 3},
            "test labels missing required classes",
        )

        # Compute shared features
        stage = "feature construction"
        features = in_out_degree(
            train_graph, size=data.num_nodes, signed=False, edge_weight=torch.abs(train_weights)
        )

        require(features.shape[0] == data.num_nodes, "feature node count mismatch")
        require(features.ndim == 2, "features not 2D")
        require(features.is_floating_point(), "features not floating point")
        require(features.device == DEVICE, "features not on CPU")
        require(torch.isfinite(features).all().item(), "non-finite features")

        # Compute fingerprints
        stage = "fingerprint computation"
        clean_graph_fp = compute_tensor_hash("clean_graph", clean_graph)
        clean_weights_fp = compute_tensor_hash("clean_weights", clean_weights)
        train_graph_fp = compute_tensor_hash("train_graph", train_graph)
        train_weights_fp = compute_tensor_hash("train_weights", train_weights)
        train_queries_fp = compute_tensor_hash("train_queries", train_queries)
        train_labels_fp = compute_tensor_hash("train_labels", train_labels)
        val_queries_fp = compute_tensor_hash("val_queries", val_queries)
        val_labels_fp = compute_tensor_hash("val_labels", val_labels)
        test_queries_fp = compute_tensor_hash("test_queries", test_queries)
        test_labels_fp = compute_tensor_hash("test_labels", test_labels)
        features_fp = compute_tensor_hash("features", features)

        bundle_fp = compute_bundle_fingerprint(
            clean_graph,
            clean_weights,
            train_graph,
            train_weights,
            train_queries,
            train_labels,
            val_queries,
            val_labels,
            test_queries,
            test_labels,
            features,
            GENERATION_SEED,
            SPLIT_SEED,
        )

        return SharedData(
            clean_graph=clean_graph,
            clean_weights=clean_weights,
            train_graph=train_graph,
            train_weights=train_weights,
            train_queries=train_queries,
            train_labels=train_labels,
            val_queries=val_queries,
            val_labels=val_labels,
            test_queries=test_queries,
            test_labels=test_labels,
            features=features,
            num_nodes=data.num_nodes,
            generation_seed=GENERATION_SEED,
            split_seed=SPLIT_SEED,
            clean_graph_fingerprint=clean_graph_fp,
            clean_weights_fingerprint=clean_weights_fp,
            train_graph_fingerprint=train_graph_fp,
            train_weights_fingerprint=train_weights_fp,
            train_queries_fingerprint=train_queries_fp,
            train_labels_fingerprint=train_labels_fp,
            val_queries_fingerprint=val_queries_fp,
            val_labels_fingerprint=val_labels_fp,
            test_queries_fingerprint=test_queries_fp,
            test_labels_fingerprint=test_labels_fp,
            features_fingerprint=features_fp,
            bundle_fingerprint=bundle_fp,
        )

    except Exception as e:
        print(f"\nShared data creation FAILED")
        failure_category = classify_failure(stage, e)
        print_structured_failure(
            category=failure_category,
            stage=stage,
            exception_type=type(e).__name__,
            message=str(e),
            traceback_text=traceback.format_exc(),
        )
        raise


def verify_audit_copy(shared: SharedData) -> None:
    """Create audit copy and verify all fingerprints match."""
    stage = "audit copy creation"

    try:
        # Set all seeds
        random.seed(GENERATION_SEED)
        np.random.seed(GENERATION_SEED)
        torch.manual_seed(GENERATION_SEED)

        # Construct meta-graph matrix F
        F = np.array(
            [
                [0.5, 0.1, -0.1, 0.1],
                [0.9, 0.5, -0.1, -0.5],
                [-0.9, -0.9, 0.5, -0.9],
                [-0.9, -0.5, -0.1, 0.5],
            ]
        )

        # Generate SDSBM
        A, y = SDSBM(N, K, P, F, SIZE_RATIO, ETA)
        A, y = extract_network(A, y)
        data = SignedData(A=A, y=torch.as_tensor(y, dtype=torch.long))

        split_result = link_class_split(
            data,
            splits=1,
            task=TASK,
            prob_val=PROB_VAL,
            prob_test=PROB_TEST,
            seed=SPLIT_SEED,
            device="cpu",
        )

        split = split_result[0]

        audit_clean_graph = data.edge_index
        audit_clean_weights = data.edge_weight
        audit_train_graph = split["graph"]
        audit_train_weights = split["weights"]
        audit_train_queries = split["train"]["edges"]
        audit_train_labels = split["train"]["label"]
        audit_val_queries = split["val"]["edges"]
        audit_val_labels = split["val"]["label"]
        audit_test_queries = split["test"]["edges"]
        audit_test_labels = split["test"]["label"]

        audit_features = in_out_degree(
            audit_train_graph,
            size=data.num_nodes,
            signed=False,
            edge_weight=torch.abs(audit_train_weights),
        )

        # Verify fingerprints
        stage = "audit fingerprint verification"

        audit_clean_graph_fp = compute_tensor_hash("clean_graph", audit_clean_graph)
        require(
            audit_clean_graph_fp == shared.clean_graph_fingerprint,
            "audit clean_graph fingerprint mismatch",
        )

        audit_clean_weights_fp = compute_tensor_hash(
            "clean_weights", audit_clean_weights
        )
        require(
            audit_clean_weights_fp == shared.clean_weights_fingerprint,
            "audit clean_weights fingerprint mismatch",
        )

        audit_train_graph_fp = compute_tensor_hash("train_graph", audit_train_graph)
        require(
            audit_train_graph_fp == shared.train_graph_fingerprint,
            "audit train_graph fingerprint mismatch",
        )

        audit_train_weights_fp = compute_tensor_hash(
            "train_weights", audit_train_weights
        )
        require(
            audit_train_weights_fp == shared.train_weights_fingerprint,
            "audit train_weights fingerprint mismatch",
        )

        audit_train_queries_fp = compute_tensor_hash(
            "train_queries", audit_train_queries
        )
        require(
            audit_train_queries_fp == shared.train_queries_fingerprint,
            "audit train_queries fingerprint mismatch",
        )

        audit_train_labels_fp = compute_tensor_hash("train_labels", audit_train_labels)
        require(
            audit_train_labels_fp == shared.train_labels_fingerprint,
            "audit train_labels fingerprint mismatch",
        )

        audit_val_queries_fp = compute_tensor_hash("val_queries", audit_val_queries)
        require(
            audit_val_queries_fp == shared.val_queries_fingerprint,
            "audit val_queries fingerprint mismatch",
        )

        audit_val_labels_fp = compute_tensor_hash("val_labels", audit_val_labels)
        require(
            audit_val_labels_fp == shared.val_labels_fingerprint,
            "audit val_labels fingerprint mismatch",
        )

        audit_test_queries_fp = compute_tensor_hash("test_queries", audit_test_queries)
        require(
            audit_test_queries_fp == shared.test_queries_fingerprint,
            "audit test_queries fingerprint mismatch",
        )

        audit_test_labels_fp = compute_tensor_hash("test_labels", audit_test_labels)
        require(
            audit_test_labels_fp == shared.test_labels_fingerprint,
            "audit test_labels fingerprint mismatch",
        )

        audit_features_fp = compute_tensor_hash("features", audit_features)
        require(
            audit_features_fp == shared.features_fingerprint,
            "audit features fingerprint mismatch",
        )

        audit_bundle_fp = compute_bundle_fingerprint(
            audit_clean_graph,
            audit_clean_weights,
            audit_train_graph,
            audit_train_weights,
            audit_train_queries,
            audit_train_labels,
            audit_val_queries,
            audit_val_labels,
            audit_test_queries,
            audit_test_labels,
            audit_features,
            GENERATION_SEED,
            SPLIT_SEED,
        )
        require(
            audit_bundle_fp == shared.bundle_fingerprint,
            "audit bundle fingerprint mismatch",
        )

        print("Audit copy verification: PASS")

    except Exception as e:
        print(f"\nAudit copy verification FAILED")
        failure_category = classify_failure(stage, e)
        print_structured_failure(
            category=failure_category,
            stage=stage,
            exception_type=type(e).__name__,
            message=str(e),
            traceback_text=traceback.format_exc(),
        )
        raise


def compute_metrics(
    output: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute accuracy, macro-F1, and micro-F1."""
    preds = output.argmax(dim=1)

    # Accuracy
    correct = (preds == labels).sum().item()
    total = labels.shape[0]
    accuracy = correct / total if total > 0 else 0.0

    # Macro-F1
    class_f1s = []
    for c in range(LABEL_DIM):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        class_f1s.append(f1)

    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    # Micro-F1
    all_tp = (preds == labels).sum().item()
    all_fp = (preds != labels).sum().item()
    all_fn = all_fp

    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return accuracy, macro_f1, micro_f1


def train_msgnn(shared: SharedData, seed: int) -> SeedResult:
    """Train MSGNN_link_prediction for one seed."""
    model_name = "MSGNN_link_prediction"
    stage = "initialization"
    start_time = datetime.now(timezone.utc)

    result = SeedResult(
        model_name=model_name,
        seed=seed,
        stage=stage,
        passed=False,
        failure_category="",
        exception_type="",
        exception_message="",
        traceback_text="",
        config={},
        optimizer_config={},
        initial_train_loss=float("inf"),
        initial_val_loss=float("inf"),
        best_epoch=-1,
        best_val_loss=float("inf"),
        stopping_epoch=-1,
        stopping_reason="",
        total_epochs=0,
        final_executed_epoch=-1,
        early_stopped=False,
        per_epoch_train_loss=[],
        per_epoch_val_loss=[],
        gradient_finite=False,
        parameters_finite=False,
        checkpoint_restored=False,
        restored_checkpoint_val_loss=float("inf"),
        test_loss=float("inf"),
        test_accuracy=0.0,
        test_macro_f1=0.0,
        test_micro_f1=0.0,
        runtime_seconds=0.0,
        learning_signal_satisfied=False,
        finite_value_checks={},
        warnings=[],
        bundle_fingerprint=shared.bundle_fingerprint,
    )

    try:
        print(f"\n{'='*60}")
        print(f"Training {model_name} with seed {seed}")
        print(f"{'='*60}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Construct model
        stage = "model construction"
        config = MSGNN_CONFIG.copy()
        config["num_features"] = shared.features.shape[1]
        result.config = config

        optimizer_config = {
            "name": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }
        result.optimizer_config = optimizer_config

        model = MSGNN_link_prediction(**config)
        model = model.to(DEVICE)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.NLLLoss()

        # Define forward function
        def msgnn_forward(query_edges):
            x_real = shared.features
            x_imag = x_real.clone()
            return model(
                x_real,
                x_imag,
                shared.train_graph,
                query_edges,
                shared.train_weights,
            )

        # Initial validation loss
        stage = "initial forward execution"
        model.eval()
        with torch.no_grad():
            val_output = msgnn_forward(shared.val_queries)

        stage = "initial output validation"
        validate_output(
            val_output, (shared.val_queries.shape[0], LABEL_DIM), "Initial validation"
        )

        stage = "initial loss validation"
        initial_val_loss = criterion(val_output, shared.val_labels)
        validate_loss(initial_val_loss, "Initial validation loss")
        result.initial_val_loss = initial_val_loss.item()

        # Initial training loss
        model.train()
        train_output = msgnn_forward(shared.train_queries)
        validate_output(
            train_output,
            (shared.train_queries.shape[0], LABEL_DIM),
            "Initial training",
        )
        initial_train_loss = criterion(train_output, shared.train_labels)
        validate_loss(initial_train_loss, "Initial training loss")
        result.initial_train_loss = initial_train_loss.item()

        print(f"Initial train loss: {result.initial_train_loss:.6f}")
        print(f"Initial val loss: {result.initial_val_loss:.6f}")

        # Training loop
        best_val_loss = result.initial_val_loss
        best_epoch = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0

        for epoch in range(MAX_EPOCHS):
            stage = f"epoch {epoch} forward execution"
            model.train()

            # Forward pass
            output = msgnn_forward(shared.train_queries)

            stage = f"epoch {epoch} output validation"
            validate_output(
                output, (shared.train_queries.shape[0], LABEL_DIM), f"Epoch {epoch}"
            )

            stage = f"epoch {epoch} loss validation"
            loss = criterion(output, shared.train_labels)
            validate_loss(loss, f"Epoch {epoch} loss")

            # Backward pass
            stage = f"epoch {epoch} backward"
            optimizer.zero_grad()
            loss.backward()

            # Gradient validation
            stage = f"epoch {epoch} gradient validation"
            gradient_finite = True
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    if not torch.isfinite(param.grad).all().item():
                        gradient_finite = False
                        raise GateEValidationError(
                            f"Non-finite gradient at epoch {epoch}"
                        )

            result.gradient_finite = gradient_finite

            # Optimizer step
            stage = f"epoch {epoch} optimizer step"
            optimizer.step()

            # Parameter validation
            stage = f"epoch {epoch} parameter validation"
            parameters_finite = True
            for param in model.parameters():
                if not torch.isfinite(param).all().item():
                    parameters_finite = False
                    raise GateEValidationError(
                        f"Non-finite parameter at epoch {epoch}"
                    )

            result.parameters_finite = parameters_finite

            # Validation
            stage = f"epoch {epoch} validation forward execution"
            model.eval()
            with torch.no_grad():
                val_output = msgnn_forward(shared.val_queries)

            stage = f"epoch {epoch} validation output validation"
            validate_output(
                val_output,
                (shared.val_queries.shape[0], LABEL_DIM),
                f"Epoch {epoch} validation",
            )

            stage = f"epoch {epoch} validation loss validation"
            val_loss = criterion(val_output, shared.val_labels)
            validate_loss(val_loss, f"Epoch {epoch} validation loss")

            result.per_epoch_train_loss.append(loss.item())
            result.per_epoch_val_loss.append(val_loss.item())
            result.total_epochs = epoch + 1

            # Check for improvement
            if val_loss.item() < best_val_loss - MIN_IMPROVEMENT:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                stage = f"epoch {epoch} checkpoint save"
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
                result.stopping_epoch = epoch
                result.stopping_reason = "early stopping"
                break

        if result.stopping_epoch == -1:
            result.stopping_epoch = MAX_EPOCHS - 1
            result.stopping_reason = "max epochs"

        result.best_epoch = best_epoch
        result.best_val_loss = best_val_loss
        result.final_executed_epoch = result.total_epochs - 1
        result.early_stopped = (result.stopping_reason == "early stopping")

        print(f"Best epoch: {best_epoch}")
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Stopping epoch: {result.stopping_epoch}")
        print(f"Stopping reason: {result.stopping_reason}")

        # Restore best checkpoint
        stage = "checkpoint restore"
        model.load_state_dict(best_state)
        result.checkpoint_restored = True

        # Final validation
        stage = "validation forward execution"
        model.eval()
        with torch.no_grad():
            val_output = msgnn_forward(shared.val_queries)

        stage = "validation output validation"
        validate_output(
            val_output,
            (shared.val_queries.shape[0], LABEL_DIM),
            "Final validation",
        )

        stage = "validation loss validation"
        final_val_loss = criterion(val_output, shared.val_labels)
        validate_loss(final_val_loss, "Final validation loss")
        result.restored_checkpoint_val_loss = final_val_loss.item()

        require(
            abs(final_val_loss.item() - best_val_loss) < 1e-6,
            "Restored checkpoint validation loss mismatch",
        )

        # Test evaluation
        stage = "test forward execution"
        with torch.no_grad():
            test_output = msgnn_forward(shared.test_queries)

        stage = "test output validation"
        validate_output(
            test_output, (shared.test_queries.shape[0], LABEL_DIM), "Test"
        )

        stage = "test loss validation"
        test_loss = criterion(test_output, shared.test_labels)
        validate_loss(test_loss, "Test loss")
        result.test_loss = test_loss.item()

        stage = "metric computation"
        accuracy, macro_f1, micro_f1 = compute_metrics(test_output, shared.test_labels)

        result.test_accuracy = accuracy
        result.test_macro_f1 = macro_f1
        result.test_micro_f1 = micro_f1

        # Finite value checks
        result.finite_value_checks = {
            "outputs_finite": True,
            "losses_finite": True,
            "gradients_finite": result.gradient_finite,
            "parameters_finite": result.parameters_finite,
            "all_finite": True,
        }

        # Learning signal
        result.learning_signal_satisfied = (
            best_val_loss <= result.initial_val_loss - 0.001
        )

        result.passed = True
        result.stage = "complete"

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: PASS")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test macro-F1: {macro_f1:.4f}")
        print(f"Test micro-F1: {micro_f1:.4f}")
        print(f"Learning signal satisfied: {result.learning_signal_satisfied}")

    except StagePreservingError as e:
        result.passed = False
        result.stage = e.stage
        result.failure_category = classify_failure(e.stage, e.original_exception)
        result.exception_type = e.original_type
        result.exception_message = e.original_message
        result.traceback_text = e.original_traceback

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )

    except Exception as e:
        result.passed = False
        result.stage = stage
        result.failure_category = classify_failure(stage, e)
        result.exception_type = type(e).__name__
        result.exception_message = str(e)
        result.traceback_text = traceback.format_exc()

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )

    return result


def train_sssnet(shared: SharedData, seed: int) -> SeedResult:
    """Train SSSNET_link_prediction with directed=True for one seed."""
    model_name = "SSSNET_link_prediction"
    stage = "initialization"
    start_time = datetime.now(timezone.utc)

    result = SeedResult(
        model_name=model_name,
        seed=seed,
        stage=stage,
        passed=False,
        failure_category="",
        exception_type="",
        exception_message="",
        traceback_text="",
        config={},
        optimizer_config={},
        initial_train_loss=float("inf"),
        initial_val_loss=float("inf"),
        best_epoch=-1,
        best_val_loss=float("inf"),
        stopping_epoch=-1,
        stopping_reason="",
        total_epochs=0,
        final_executed_epoch=-1,
        early_stopped=False,
        per_epoch_train_loss=[],
        per_epoch_val_loss=[],
        gradient_finite=False,
        parameters_finite=False,
        checkpoint_restored=False,
        restored_checkpoint_val_loss=float("inf"),
        test_loss=float("inf"),
        test_accuracy=0.0,
        test_macro_f1=0.0,
        test_micro_f1=0.0,
        runtime_seconds=0.0,
        learning_signal_satisfied=False,
        finite_value_checks={},
        warnings=[],
        bundle_fingerprint=shared.bundle_fingerprint,
    )

    try:
        print(f"\n{'='*60}")
        print(f"Training {model_name} with seed {seed}")
        print(f"{'='*60}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Positive/negative decomposition
        stage = "positive/negative decomposition"
        signed_training_data = SignedData(
            edge_index=shared.train_graph,
            edge_weight=shared.train_weights,
            num_nodes=shared.num_nodes,
        ).to(DEVICE)

        signed_training_data.separate_positive_negative()

        edge_index_p = signed_training_data.edge_index_p
        edge_weight_p = signed_training_data.edge_weight_p
        edge_index_n = signed_training_data.edge_index_n
        edge_weight_n = signed_training_data.edge_weight_n

        require(edge_index_p.shape[1] > 0, "Empty positive edge set")
        require(edge_index_n.shape[1] > 0, "Empty negative edge set")

        # Construct model
        stage = "model construction"
        config = SSSNET_CONFIG.copy()
        config["nfeat"] = shared.features.shape[1]
        result.config = config

        optimizer_config = {
            "name": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }
        result.optimizer_config = optimizer_config

        model = SSSNET_link_prediction(**config)
        model = model.to(DEVICE)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.NLLLoss()

        # Define forward function
        def sssnet_forward(query_edges):
            return model(
                edge_index_p,
                edge_weight_p,
                edge_index_n,
                edge_weight_n,
                shared.features,
                query_edges,
            )

        # Initial validation loss
        stage = "initial forward execution"
        model.eval()
        with torch.no_grad():
            val_output = sssnet_forward(shared.val_queries)

        stage = "initial output validation"
        validate_output(
            val_output, (shared.val_queries.shape[0], LABEL_DIM), "Initial validation"
        )

        stage = "initial loss validation"
        initial_val_loss = criterion(val_output, shared.val_labels)
        validate_loss(initial_val_loss, "Initial validation loss")
        result.initial_val_loss = initial_val_loss.item()

        # Initial training loss
        model.train()
        train_output = sssnet_forward(shared.train_queries)
        validate_output(
            train_output,
            (shared.train_queries.shape[0], LABEL_DIM),
            "Initial training",
        )
        initial_train_loss = criterion(train_output, shared.train_labels)
        validate_loss(initial_train_loss, "Initial training loss")
        result.initial_train_loss = initial_train_loss.item()

        print(f"Initial train loss: {result.initial_train_loss:.6f}")
        print(f"Initial val loss: {result.initial_val_loss:.6f}")

        # Training loop
        best_val_loss = result.initial_val_loss
        best_epoch = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0

        for epoch in range(MAX_EPOCHS):
            stage = f"epoch {epoch} forward execution"
            model.train()

            # Forward pass
            output = sssnet_forward(shared.train_queries)

            stage = f"epoch {epoch} output validation"
            validate_output(
                output, (shared.train_queries.shape[0], LABEL_DIM), f"Epoch {epoch}"
            )

            stage = f"epoch {epoch} loss validation"
            loss = criterion(output, shared.train_labels)
            validate_loss(loss, f"Epoch {epoch} loss")

            # Backward pass
            stage = f"epoch {epoch} backward"
            optimizer.zero_grad()
            loss.backward()

            # Gradient validation
            stage = f"epoch {epoch} gradient validation"
            gradient_finite = True
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    if not torch.isfinite(param.grad).all().item():
                        gradient_finite = False
                        raise GateEValidationError(
                            f"Non-finite gradient at epoch {epoch}"
                        )

            result.gradient_finite = gradient_finite

            # Optimizer step
            stage = f"epoch {epoch} optimizer step"
            optimizer.step()

            # Parameter validation
            stage = f"epoch {epoch} parameter validation"
            parameters_finite = True
            for param in model.parameters():
                if not torch.isfinite(param).all().item():
                    parameters_finite = False
                    raise GateEValidationError(
                        f"Non-finite parameter at epoch {epoch}"
                    )

            result.parameters_finite = parameters_finite

            # Validation
            stage = f"epoch {epoch} validation forward execution"
            model.eval()
            with torch.no_grad():
                val_output = sssnet_forward(shared.val_queries)

            stage = f"epoch {epoch} validation output validation"
            validate_output(
                val_output,
                (shared.val_queries.shape[0], LABEL_DIM),
                f"Epoch {epoch} validation",
            )

            stage = f"epoch {epoch} validation loss validation"
            val_loss = criterion(val_output, shared.val_labels)
            validate_loss(val_loss, f"Epoch {epoch} validation loss")

            result.per_epoch_train_loss.append(loss.item())
            result.per_epoch_val_loss.append(val_loss.item())
            result.total_epochs = epoch + 1

            # Check for improvement
            if val_loss.item() < best_val_loss - MIN_IMPROVEMENT:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                stage = f"epoch {epoch} checkpoint save"
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
                result.stopping_epoch = epoch
                result.stopping_reason = "early stopping"
                break

        if result.stopping_epoch == -1:
            result.stopping_epoch = MAX_EPOCHS - 1
            result.stopping_reason = "max epochs"

        result.best_epoch = best_epoch
        result.best_val_loss = best_val_loss
        result.final_executed_epoch = result.total_epochs - 1
        result.early_stopped = (result.stopping_reason == "early stopping")

        print(f"Best epoch: {best_epoch}")
        print(f"Best val loss: {best_val_loss:.6f}")
        print(f"Stopping epoch: {result.stopping_epoch}")
        print(f"Stopping reason: {result.stopping_reason}")

        # Restore best checkpoint
        stage = "checkpoint restore"
        model.load_state_dict(best_state)
        result.checkpoint_restored = True

        # Final validation
        stage = "validation forward execution"
        model.eval()
        with torch.no_grad():
            val_output = sssnet_forward(shared.val_queries)

        stage = "validation output validation"
        validate_output(
            val_output,
            (shared.val_queries.shape[0], LABEL_DIM),
            "Final validation",
        )

        stage = "validation loss validation"
        final_val_loss = criterion(val_output, shared.val_labels)
        validate_loss(final_val_loss, "Final validation loss")
        result.restored_checkpoint_val_loss = final_val_loss.item()

        require(
            abs(final_val_loss.item() - best_val_loss) < 1e-6,
            "Restored checkpoint validation loss mismatch",
        )

        # Test evaluation
        stage = "test forward execution"
        with torch.no_grad():
            test_output = sssnet_forward(shared.test_queries)

        stage = "test output validation"
        validate_output(
            test_output, (shared.test_queries.shape[0], LABEL_DIM), "Test"
        )

        stage = "test loss validation"
        test_loss = criterion(test_output, shared.test_labels)
        validate_loss(test_loss, "Test loss")
        result.test_loss = test_loss.item()

        stage = "metric computation"
        accuracy, macro_f1, micro_f1 = compute_metrics(test_output, shared.test_labels)

        result.test_accuracy = accuracy
        result.test_macro_f1 = macro_f1
        result.test_micro_f1 = micro_f1

        # Finite value checks
        result.finite_value_checks = {
            "outputs_finite": True,
            "losses_finite": True,
            "gradients_finite": result.gradient_finite,
            "parameters_finite": result.parameters_finite,
            "all_finite": True,
        }

        # Learning signal
        result.learning_signal_satisfied = (
            best_val_loss <= result.initial_val_loss - 0.001
        )

        result.passed = True
        result.stage = "complete"

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: PASS")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test macro-F1: {macro_f1:.4f}")
        print(f"Test micro-F1: {micro_f1:.4f}")
        print(f"Learning signal satisfied: {result.learning_signal_satisfied}")

    except StagePreservingError as e:
        result.passed = False
        result.stage = e.stage
        result.failure_category = classify_failure(e.stage, e.original_exception)
        result.exception_type = e.original_type
        result.exception_message = e.original_message
        result.traceback_text = e.original_traceback

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )

    except Exception as e:
        result.passed = False
        result.stage = stage
        result.failure_category = classify_failure(stage, e)
        result.exception_type = type(e).__name__
        result.exception_message = str(e)
        result.traceback_text = traceback.format_exc()

        end_time = datetime.now(timezone.utc)
        result.runtime_seconds = (end_time - start_time).total_seconds()

        print(f"\n{model_name} seed {seed}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )

    return result


def main() -> int:
    """Run Gate E2 clean baseline training."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Gate E2 — Clean Functional-Baseline Implementation"
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        required=True,
        choices=[0, 1, 2],
        help="Model initialization seeds (one or more from 0, 1, 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Validate unique seeds
    if len(args.model_seeds) != len(set(args.model_seeds)):
        print("ERROR: --model-seeds must contain unique values")
        return 1

    model_seeds = sorted(args.model_seeds)
    output_path = Path(args.output)

    # Print header
    print("=" * 60)
    print("Gate E2 — Clean Functional-Baseline Implementation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version: {version('torch-geometric')}")
    print(f"PyGSD version: {version('torch-geometric-signed-directed')}")
    print(f"NumPy version: {np.__version__}")
    print(f"NetworkX version: {version('networkx')}")
    print(f"SciPy version: {version('scipy')}")
    print(f"Device: {DEVICE}")
    print(f"Generation seed: {GENERATION_SEED}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Model seeds: {model_seeds}")
    print(f"Task: {TASK}")
    print(f"N: {N}")
    print(f"K: {K}")
    print(f"p: {P}")
    print(f"size_ratio: {SIZE_RATIO}")
    print(f"eta: {ETA}")
    print(f"gamma: {GAMMA}")
    print(f"Validation probability: {PROB_VAL}")
    print(f"Test probability: {PROB_TEST}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Min improvement: {MIN_IMPROVEMENT}")
    print(f"Min epochs: {MIN_EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Output: {output_path}")
    print("\nTest purpose: Clean multi-epoch training baseline only")
    print("Does NOT test: perturbations, robustness, stability, hyperparameter search")

    # Create shared data
    try:
        print(f"\n{'='*60}")
        print("Creating shared data")
        print(f"{'='*60}")
        shared = create_shared_data()
    except Exception:
        print("\nGate E2 execution result: FAIL")
        print("Shared data creation failed.")
        return 1

    # Print shared data info
    print(f"\nShared data created successfully:")
    print(f"  Nodes: {shared.num_nodes}")
    print(f"  Clean graph edges: {shared.clean_graph.shape[1]}")
    print(f"  Positive edges: {(shared.clean_weights > 0).sum().item()}")
    print(f"  Negative edges: {(shared.clean_weights < 0).sum().item()}")
    print(f"  Training graph edges: {shared.train_graph.shape[1]}")
    print(f"  Train queries: {shared.train_queries.shape[0]}")
    print(f"  Validation queries: {shared.val_queries.shape[0]}")
    print(f"  Test queries: {shared.test_queries.shape[0]}")
    print(f"  Feature shape: {tuple(shared.features.shape)}")
    print(f"  Bundle fingerprint: {shared.bundle_fingerprint}")

    # Print class distributions
    print(f"\nClass distributions:")
    for split_name, labels in [
        ("Train", shared.train_labels),
        ("Validation", shared.val_labels),
        ("Test", shared.test_labels),
    ]:
        counts = [(labels == c).sum().item() for c in range(LABEL_DIM)]
        print(f"  {split_name}: {counts}")

    # Verify audit copy
    try:
        print(f"\n{'='*60}")
        print("Verifying audit copy")
        print(f"{'='*60}")
        verify_audit_copy(shared)
    except Exception:
        print("\nGate E2 execution result: FAIL")
        print("Audit copy verification failed.")
        return 1

    # Train models
    all_results = []

    for seed in model_seeds:
        msgnn_result = train_msgnn(shared, seed)
        all_results.append(msgnn_result)

        sssnet_result = train_sssnet(shared, seed)
        all_results.append(sssnet_result)

    # Compute learning signal satisfaction per model
    msgnn_results = [r for r in all_results if r.model_name == "MSGNN_link_prediction"]
    sssnet_results = [
        r for r in all_results if r.model_name == "SSSNET_link_prediction"
    ]

    msgnn_passed_count = sum(1 for r in msgnn_results if r.passed)
    sssnet_passed_count = sum(1 for r in sssnet_results if r.passed)

    msgnn_learning_count = sum(
        1 for r in msgnn_results if r.passed and r.learning_signal_satisfied
    )
    sssnet_learning_count = sum(
        1 for r in sssnet_results if r.passed and r.learning_signal_satisfied
    )

    # Determine if two-of-three requirement is evaluable
    three_seeds_requested = len(model_seeds) == 3
    msgnn_two_of_three = (
        msgnn_learning_count >= 2 if three_seeds_requested else None
    )
    sssnet_two_of_three = (
        sssnet_learning_count >= 2 if three_seeds_requested else None
    )

    # Overall verdict
    all_passed = all(r.passed for r in all_results)
    all_bundles_match = all(
        r.bundle_fingerprint == shared.bundle_fingerprint for r in all_results
    )

    overall_pass = all_passed and all_bundles_match

    if three_seeds_requested:
        overall_pass = overall_pass and msgnn_two_of_three and sssnet_two_of_three

    # Prepare output
    output_data = {
        "gate": "E2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "pyg_version": version("torch-geometric"),
            "pygsd_version": version("torch-geometric-signed-directed"),
            "numpy_version": np.__version__,
            "networkx_version": version("networkx"),
            "scipy_version": version("scipy"),
            "device": str(DEVICE),
        },
        "configuration": {
            "generation_seed": GENERATION_SEED,
            "split_seed": SPLIT_SEED,
            "model_seeds": model_seeds,
            "task": TASK,
            "N": N,
            "K": K,
            "p": P,
            "size_ratio": SIZE_RATIO,
            "eta": ETA,
            "gamma": GAMMA,
            "prob_val": PROB_VAL,
            "prob_test": PROB_TEST,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_improvement": MIN_IMPROVEMENT,
            "min_epochs": MIN_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "msgnn_config": MSGNN_CONFIG,
            "sssnet_config": SSSNET_CONFIG,
        },
        "shared_data": {
            "num_nodes": shared.num_nodes,
            "clean_edges": shared.clean_graph.shape[1],
            "positive_edges": int((shared.clean_weights > 0).sum().item()),
            "negative_edges": int((shared.clean_weights < 0).sum().item()),
            "train_edges": shared.train_graph.shape[1],
            "train_queries": shared.train_queries.shape[0],
            "val_queries": shared.val_queries.shape[0],
            "test_queries": shared.test_queries.shape[0],
            "feature_shape": list(shared.features.shape),
            "train_class_counts": [
                int((shared.train_labels == c).sum().item()) for c in range(LABEL_DIM)
            ],
            "val_class_counts": [
                int((shared.val_labels == c).sum().item()) for c in range(LABEL_DIM)
            ],
            "test_class_counts": [
                int((shared.test_labels == c).sum().item()) for c in range(LABEL_DIM)
            ],
            "clean_graph_fingerprint": shared.clean_graph_fingerprint,
            "clean_weights_fingerprint": shared.clean_weights_fingerprint,
            "train_graph_fingerprint": shared.train_graph_fingerprint,
            "train_weights_fingerprint": shared.train_weights_fingerprint,
            "train_queries_fingerprint": shared.train_queries_fingerprint,
            "train_labels_fingerprint": shared.train_labels_fingerprint,
            "val_queries_fingerprint": shared.val_queries_fingerprint,
            "val_labels_fingerprint": shared.val_labels_fingerprint,
            "test_queries_fingerprint": shared.test_queries_fingerprint,
            "test_labels_fingerprint": shared.test_labels_fingerprint,
            "features_fingerprint": shared.features_fingerprint,
            "bundle_fingerprint": shared.bundle_fingerprint,
        },
        "results": [
            {
                "model_name": r.model_name,
                "seed": r.seed,
                "passed": r.passed,
                "stage": r.stage,
                "failure_category": r.failure_category,
                "exception_type": r.exception_type,
                "exception_message": r.exception_message,
                "config": r.config,
                "optimizer_config": r.optimizer_config,
                "initial_train_loss": None if r.initial_train_loss == float("inf") else r.initial_train_loss,
                "initial_val_loss": None if r.initial_val_loss == float("inf") else r.initial_val_loss,
                "best_epoch": r.best_epoch,
                "best_val_loss": None if r.best_val_loss == float("inf") else r.best_val_loss,
                "stopping_epoch": r.stopping_epoch,
                "stopping_reason": r.stopping_reason,
                "total_epochs": r.total_epochs,
                "final_executed_epoch": r.final_executed_epoch,
                "early_stopped": r.early_stopped,
                "per_epoch_train_loss": r.per_epoch_train_loss,
                "per_epoch_val_loss": r.per_epoch_val_loss,
                "gradient_finite": r.gradient_finite,
                "parameters_finite": r.parameters_finite,
                "checkpoint_restored": r.checkpoint_restored,
                "restored_checkpoint_val_loss": None if r.restored_checkpoint_val_loss == float("inf") else r.restored_checkpoint_val_loss,
                "test_loss": None if r.test_loss == float("inf") else r.test_loss,
                "test_accuracy": r.test_accuracy,
                "test_macro_f1": r.test_macro_f1,
                "test_micro_f1": r.test_micro_f1,
                "runtime_seconds": r.runtime_seconds,
                "learning_signal_satisfied": r.learning_signal_satisfied,
                "finite_value_checks": r.finite_value_checks,
                "warnings": r.warnings,
                "bundle_fingerprint": r.bundle_fingerprint,
            }
            for r in all_results
        ],
        "summary": {
            "msgnn_passed_count": msgnn_passed_count,
            "msgnn_total_count": len(msgnn_results),
            "msgnn_learning_signal_count": msgnn_learning_count,
            "msgnn_two_of_three_satisfied": msgnn_two_of_three,
            "sssnet_passed_count": sssnet_passed_count,
            "sssnet_total_count": len(sssnet_results),
            "sssnet_learning_signal_count": sssnet_learning_count,
            "sssnet_two_of_three_satisfied": sssnet_two_of_three,
            "all_passed": all_passed,
            "all_bundles_match": all_bundles_match,
            "three_seeds_requested": three_seeds_requested,
            "overall_pass": overall_pass,
        },
        "warnings": [],
        "prohibited_claims": [
            "This execution does NOT establish robustness",
            "This execution does NOT establish stability",
            "This execution does NOT establish perturbation tolerance",
            "This execution does NOT establish architecture superiority",
            "This execution does NOT establish convergence",
            "This execution does NOT establish generalization",
            "This execution does NOT authorize perturbation experiments",
        ],
    }

    # Add warnings
    if not three_seeds_requested:
        output_data["warnings"].append(
            "Two-of-three learning signal requirement not evaluable (fewer than 3 seeds requested)"
        )

    if not all_bundles_match:
        output_data["warnings"].append("Bundle fingerprint mismatch detected")

    # Write output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, allow_nan=False)
        print(f"\nResults written to: {output_path}")
    except Exception as e:
        print(f"\nERROR: Failed to write output file: {e}")
        return 1

    # Print summary
    print(f"\n{'='*60}")
    print("Gate E2 Execution Summary")
    print(f"{'='*60}")
    print(f"MSGNN passed: {msgnn_passed_count}/{len(msgnn_results)}")
    print(f"MSGNN learning signal: {msgnn_learning_count}/{len(msgnn_results)}")
    if msgnn_two_of_three is not None:
        print(f"MSGNN two-of-three: {msgnn_two_of_three}")
    print(f"SSSNET passed: {sssnet_passed_count}/{len(sssnet_results)}")
    print(f"SSSNET learning signal: {sssnet_learning_count}/{len(sssnet_results)}")
    if sssnet_two_of_three is not None:
        print(f"SSSNET two-of-three: {sssnet_two_of_three}")
    print(f"All passed: {all_passed}")
    print(f"All bundles match: {all_bundles_match}")
    print(f"Overall pass: {overall_pass}")

    if overall_pass:
        print("\nGate E2 execution result: PASS")
        return 0
    else:
        print("\nGate E2 execution result: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
