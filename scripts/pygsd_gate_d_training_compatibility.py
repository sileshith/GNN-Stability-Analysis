"""
Gate D — Minimal Training Compatibility Test

Tests that MSGNN_link_prediction and SSSNET_link_prediction (directed=True)
can each complete one minimal training step on a shared tiny signed-directed
graph using the PyGSD four_class_signed_digraph task.

Does NOT test: convergence, model quality, robustness, stability, generalization.
"""

import hashlib
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric_signed_directed.data import SignedData
from torch_geometric_signed_directed.nn.general.MSGNN import (
    MSGNN_link_prediction,
)
from torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction import (
    SSSNET_link_prediction,
)
from torch_geometric_signed_directed.utils import (
    in_out_degree,
    link_class_split,
)


# Constants
NUM_NODES = 12
SEED = 0
SPLIT_SEED = 0
PROB_VAL = 0.15
PROB_TEST = 0.15
DEVICE = torch.device("cpu")
TASK = "four_class_signed_digraph"
LABEL_DIM = 4
HIDDEN = 4
LR = 0.01
WEIGHT_DECAY = 5e-4
NORMALIZATION_ATOL = 1e-5
NORMALIZATION_RTOL = 1e-5


class GateDValidationError(Exception):
    """Raised when a Gate D validation requirement fails."""
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
    generation_seed: int
    split_seed: int
    data_fingerprint: str
    feature_fingerprint: str


@dataclass
class ModelResult:
    """Result record for one model."""
    model_name: str
    stage: str
    passed: bool
    failure_category: str
    exception_type: str
    exception_message: str
    traceback_text: str
    config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    initial_loss: float
    post_step_loss: float
    trainable_tensors: int
    trainable_elements: int
    tensors_with_gradients: int
    tensors_with_finite_gradients: int
    tensors_with_nonzero_gradients: int
    changed_parameter_count: int
    test_accuracy: float
    test_macro_f1: float
    test_micro_f1: float
    data_fingerprint: str
    feature_fingerprint: str


def require(condition: bool, message: str) -> None:
    """Raise GateDValidationError if condition is False."""
    if not condition:
        raise GateDValidationError(message)


def compute_tensor_hash(name: str, tensor: torch.Tensor) -> bytes:
    """Compute SHA-256 hash of tensor metadata and contiguous CPU bytes."""
    h = hashlib.sha256()
    h.update(name.encode('utf-8'))
    h.update(str(tensor.dtype).encode('utf-8'))
    h.update(str(tuple(tensor.shape)).encode('utf-8'))
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def compute_data_fingerprint(
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
    """Compute full data fingerprint from all shared tensors and seeds."""
    h = hashlib.sha256()
    h.update(compute_tensor_hash('clean_graph', clean_graph))
    h.update(compute_tensor_hash('clean_weights', clean_weights))
    h.update(compute_tensor_hash('train_graph', train_graph))
    h.update(compute_tensor_hash('train_weights', train_weights))
    h.update(compute_tensor_hash('train_queries', train_queries))
    h.update(compute_tensor_hash('train_labels', train_labels))
    h.update(compute_tensor_hash('val_queries', val_queries))
    h.update(compute_tensor_hash('val_labels', val_labels))
    h.update(compute_tensor_hash('test_queries', test_queries))
    h.update(compute_tensor_hash('test_labels', test_labels))
    h.update(compute_tensor_hash('features', features))
    h.update(str(generation_seed).encode('utf-8'))
    h.update(str(split_seed).encode('utf-8'))
    return h.hexdigest()


def compute_feature_fingerprint(features: torch.Tensor) -> str:
    """Compute feature-only fingerprint."""
    return compute_tensor_hash('features', features).hex()


def classify_failure(stage: str, exception: Exception) -> str:
    """Classify failure into a structured category using explicit stage sets."""
    
    # Define stage sets for each category
    data_contract_stages = {
        'graph creation',
        'graph validation',
        'SignedData creation',
        'split extraction',
        'split validation',
        'fingerprint computation',
        'data fingerprint verification',
        'feature fingerprint verification',
        'final shared-record verification',
    }
    
    preprocessing_stages = {
        'feature construction',
        'positive/negative decomposition',
    }
    
    model_api_stages = {
        'model construction',
        'initial forward execution',
        'post-step forward execution',
    }
    
    numerical_stages = {
        'initial output validation',
        'initial loss validation',
        'post-step output validation',
        'post-step loss validation',
    }
    
    gradient_stages = {
        'backward',
        'gradient validation',
    }
    
    optimizer_step_stages = {
        'optimizer step',
        'parameter-change validation',
        'post-step parameter validation',
    }
    
    evaluation_stages = {
        'evaluation forward execution',
        'evaluation output validation',
        'metric computation',
    }
    
    # Classify by exact stage membership
    if stage in data_contract_stages:
        return 'data-contract failure'
    elif stage in preprocessing_stages:
        return 'preprocessing failure'
    elif stage in model_api_stages:
        return 'model-API failure'
    elif stage in numerical_stages:
        return 'numerical failure'
    elif stage in gradient_stages:
        return 'gradient failure'
    elif stage in optimizer_step_stages:
        return 'optimizer-step failure'
    elif stage in evaluation_stages:
        return 'evaluation-path failure'
    elif isinstance(exception, (AssertionError, GateDValidationError)):
        return 'implementation defect'
    else:
        return 'unresolved package behavior'


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
    require(tuple(output.shape) == expected_shape, f"{stage_prefix}: shape mismatch")
    require(output.is_floating_point(), f"{stage_prefix}: expected floating dtype")
    require(output.device == DEVICE, f"{stage_prefix}: expected CPU device")
    require(torch.isfinite(output).all().item(), f"{stage_prefix}: non-finite values")
    
    # Probability normalization
    prob_sums = torch.exp(output).sum(dim=1)
    expected_sums = torch.ones_like(prob_sums)
    require(
        torch.allclose(prob_sums, expected_sums, atol=NORMALIZATION_ATOL, rtol=NORMALIZATION_RTOL),
        f"{stage_prefix}: probability normalization failed"
    )


def validate_loss(loss: torch.Tensor, stage_prefix: str) -> None:
    """Validate loss tensor."""
    require(isinstance(loss, torch.Tensor), f"{stage_prefix}: expected torch.Tensor")
    require(loss.ndim == 0, f"{stage_prefix}: expected scalar")
    require(loss.device == DEVICE, f"{stage_prefix}: expected CPU device")
    require(torch.isfinite(loss).item(), f"{stage_prefix}: non-finite loss")


def create_shared_data() -> SharedData:
    """Create shared graph, split, and features."""
    stage = "graph creation"
    
    try:
        # Define exact graph
        positive_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 0),
        ]
        negative_edges = [
            (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (9, 11), (10, 0), (11, 1),
        ]
        
        all_edges = positive_edges + negative_edges
        edge_index = torch.tensor(all_edges, dtype=torch.long, device=DEVICE).t().contiguous()
        edge_weight = torch.cat([
            torch.ones(len(positive_edges), dtype=torch.float32, device=DEVICE),
            -torch.ones(len(negative_edges), dtype=torch.float32, device=DEVICE),
        ])
        
        # Validate clean graph
        stage = "graph validation"
        require(edge_index.shape == (2, 24), "clean graph shape mismatch")
        require(edge_weight.shape == (24,), "clean weight shape mismatch")
        require((edge_weight == 1.0).sum().item() == 12, "positive edge count mismatch")
        require((edge_weight == -1.0).sum().item() == 12, "negative edge count mismatch")
        require((edge_weight != 0.0).all().item(), "zero weights found")
        
        # Check no duplicates
        edge_set = set()
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            require((src, dst) not in edge_set, "duplicate directed edge")
            edge_set.add((src, dst))
        
        # Check no self-loops
        require((edge_index[0] != edge_index[1]).all().item(), "self-loops found")
        
        # Check no reciprocal pairs
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            require((dst, src) not in edge_set, "reciprocal pair found")
        
        # Check index range
        require(edge_index.min().item() >= 0, "negative node index")
        require(edge_index.max().item() <= 11, "node index out of range")
        
        # Check all nodes participate
        unique_nodes = torch.unique(edge_index).tolist()
        require(set(unique_nodes) == set(range(12)), "not all nodes participate")
        
        # Create SignedData
        stage = "SignedData creation"
        data = SignedData(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=NUM_NODES,
        )
        
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
        
        train_graph = split["graph"]
        train_weights = split["weights"]
        
        train_queries = split["train"]["edges"]
        train_labels = split["train"]["label"]
        
        val_queries = split["val"]["edges"]
        val_labels = split["val"]["label"]
        
        test_queries = split["test"]["edges"]
        test_labels = split["test"]["label"]
        
        # Validate train split
        stage = "split validation"
        require(train_queries.ndim == 2, "train queries not 2D")
        require(train_queries.shape[1] == 2, "train queries wrong width")
        require(train_labels.ndim == 1, "train labels not 1D")
        require(train_queries.shape[0] == train_labels.shape[0], "train query/label count mismatch")
        require(train_labels.dtype == torch.long, "train labels not long")
        require(train_queries.device == DEVICE, "train queries not on CPU")
        require(train_labels.device == DEVICE, "train labels not on CPU")
        require(train_labels.min().item() >= 0, "train label below 0")
        require(train_labels.max().item() <= 3, "train label above 3")
        require(train_queries.shape[0] > 0, "empty train split")
        
        # Require all four classes in training
        unique_train_labels = set(train_labels.tolist())
        require(unique_train_labels == {0, 1, 2, 3}, "training labels missing required classes")
        
        # Validate val split
        require(val_queries.ndim == 2, "val queries not 2D")
        require(val_queries.shape[1] == 2, "val queries wrong width")
        require(val_labels.ndim == 1, "val labels not 1D")
        require(val_queries.shape[0] == val_labels.shape[0], "val query/label count mismatch")
        require(val_labels.dtype == torch.long, "val labels not long")
        require(val_queries.device == DEVICE, "val queries not on CPU")
        require(val_labels.device == DEVICE, "val labels not on CPU")
        require(val_labels.min().item() >= 0, "val label below 0")
        require(val_labels.max().item() <= 3, "val label above 3")
        require(val_queries.shape[0] > 0, "empty val split")
        
        # Validate test split
        require(test_queries.ndim == 2, "test queries not 2D")
        require(test_queries.shape[1] == 2, "test queries wrong width")
        require(test_labels.ndim == 1, "test labels not 1D")
        require(test_queries.shape[0] == test_labels.shape[0], "test query/label count mismatch")
        require(test_labels.dtype == torch.long, "test labels not long")
        require(test_queries.device == DEVICE, "test queries not on CPU")
        require(test_labels.device == DEVICE, "test labels not on CPU")
        require(test_labels.min().item() >= 0, "test label below 0")
        require(test_labels.max().item() <= 3, "test label above 3")
        require(test_queries.shape[0] > 0, "empty test split")
        
        # Validate training graph preserves signs
        require((train_weights == 1.0).any().item(), "no positive weights in training graph")
        require((train_weights == -1.0).any().item(), "no negative weights in training graph")
        
        # Validate no query leakage
        train_edge_set = set()
        for i in range(train_graph.shape[1]):
            src = train_graph[0, i].item()
            dst = train_graph[1, i].item()
            train_edge_set.add((src, dst))
        
        for queries, split_name in [
            (val_queries, 'validation'),
            (test_queries, 'test'),
        ]:
            for i in range(queries.shape[0]):
                src = queries[i, 0].item()
                dst = queries[i, 1].item()
                require(
                    (src, dst) not in train_edge_set,
                    f"{split_name} query edge found in training graph"
                )
        
        # Compute shared features
        stage = "feature construction"
        features = in_out_degree(train_graph, size=NUM_NODES)
        
        require(features.shape[0] == NUM_NODES, "feature node count mismatch")
        require(features.is_floating_point(), "features not floating point")
        require(features.device == DEVICE, "features not on CPU")
        require(torch.isfinite(features).all().item(), "non-finite features")
        
        # Compute fingerprints
        stage = "fingerprint computation"
        data_fp = compute_data_fingerprint(
            edge_index, edge_weight,
            train_graph, train_weights,
            train_queries, train_labels,
            val_queries, val_labels,
            test_queries, test_labels,
            features,
            SEED, SPLIT_SEED,
        )
        feature_fp = compute_feature_fingerprint(features)
        
        return SharedData(
            clean_graph=edge_index,
            clean_weights=edge_weight,
            train_graph=train_graph,
            train_weights=train_weights,
            train_queries=train_queries,
            train_labels=train_labels,
            val_queries=val_queries,
            val_labels=val_labels,
            test_queries=test_queries,
            test_labels=test_labels,
            features=features,
            generation_seed=SEED,
            split_seed=SPLIT_SEED,
            data_fingerprint=data_fp,
            feature_fingerprint=feature_fp,
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


def run_one_step_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    shared: SharedData,
    query_edges: torch.Tensor,
    labels: torch.Tensor,
    forward_fn,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Run one training step and return initial loss, post-step loss, and gradient stats.
    
    Raises StagePreservingError with exact internal stage on failure.
    """
    stage = "initial forward execution"
    
    try:
        # Initial forward pass
        output = forward_fn(model, shared, query_edges)
        
        # Validate output
        stage = "initial output validation"
        validate_output(output, (query_edges.shape[0], LABEL_DIM), "Initial forward")
        
        # Compute initial loss
        stage = "initial loss validation"
        initial_loss = criterion(output, labels)
        validate_loss(initial_loss, "Initial loss")
        
        # Backward pass
        stage = "backward"
        optimizer.zero_grad()
        initial_loss.backward()
        
        # Gradient validation
        stage = "gradient validation"
        trainable_tensors = 0
        trainable_elements = 0
        tensors_with_gradients = 0
        tensors_with_finite_gradients = 0
        tensors_with_nonzero_gradients = 0
        
        for param in model.parameters():
            if param.requires_grad:
                trainable_tensors += 1
                trainable_elements += param.numel()
                
                if param.grad is not None:
                    tensors_with_gradients += 1
                    
                    if torch.isfinite(param.grad).all().item():
                        tensors_with_finite_gradients += 1
                    else:
                        raise GateDValidationError("Non-finite gradient detected")
                    
                    if (param.grad != 0.0).any().item():
                        tensors_with_nonzero_gradients += 1
        
        require(tensors_with_gradients > 0, "No gradients computed")
        require(tensors_with_finite_gradients == tensors_with_gradients, "Non-finite gradients")
        require(tensors_with_nonzero_gradients > 0, "All gradients are zero")
        
        gradient_stats = {
            'trainable_tensors': trainable_tensors,
            'trainable_elements': trainable_elements,
            'tensors_with_gradients': tensors_with_gradients,
            'tensors_with_finite_gradients': tensors_with_finite_gradients,
            'tensors_with_nonzero_gradients': tensors_with_nonzero_gradients,
        }
        
        # Clone parameters before step
        stage = "optimizer step"
        param_clones = [p.clone().detach() for p in model.parameters() if p.requires_grad]
        
        # Optimizer step
        optimizer.step()
        
        # Validate parameter change
        stage = "parameter-change validation"
        changed_count = 0
        for original, current in zip(param_clones, [p for p in model.parameters() if p.requires_grad]):
            if not torch.equal(original, current):
                changed_count += 1
        
        require(changed_count > 0, "No parameters changed after optimizer step")
        
        # Validate post-step parameters
        stage = "post-step parameter validation"
        for param in model.parameters():
            require(torch.isfinite(param).all().item(), "Non-finite parameter after step")
        
        # Post-step forward pass
        stage = "post-step forward execution"
        post_output = forward_fn(model, shared, query_edges)
        
        stage = "post-step output validation"
        validate_output(post_output, (query_edges.shape[0], LABEL_DIM), "Post-step forward")
        
        stage = "post-step loss validation"
        post_loss = criterion(post_output, labels)
        validate_loss(post_loss, "Post-step loss")
        
        gradient_stats['changed_parameter_count'] = changed_count
        
        return initial_loss, post_loss, gradient_stats
        
    except Exception as e:
        raise StagePreservingError(stage, e)


def run_evaluation(
    model: nn.Module,
    shared: SharedData,
    forward_fn,
) -> Tuple[float, float, float]:
    """
    Run evaluation on test split.
    
    Raises StagePreservingError with exact internal stage on failure.
    """
    stage = "evaluation forward execution"
    
    try:
        model.eval()
        with torch.no_grad():
            output = forward_fn(model, shared, shared.test_queries)
        
        stage = "evaluation output validation"
        validate_output(output, (shared.test_queries.shape[0], LABEL_DIM), "Evaluation forward")
        
        # Compute predictions
        stage = "metric computation"
        preds = output.argmax(dim=1)
        labels = shared.test_labels
        
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
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_f1s.append(f1)
        
        macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0
        
        # Micro-F1
        all_tp = (preds == labels).sum().item()
        all_fp = (preds != labels).sum().item()
        all_fn = all_fp  # For multi-class, FP = FN
        
        micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Validate metrics
        require(0.0 <= accuracy <= 1.0, "Accuracy out of range")
        require(0.0 <= macro_f1 <= 1.0, "Macro-F1 out of range")
        require(0.0 <= micro_f1 <= 1.0, "Micro-F1 out of range")
        require(torch.isfinite(torch.tensor(accuracy)).item(), "Non-finite accuracy")
        require(torch.isfinite(torch.tensor(macro_f1)).item(), "Non-finite macro-F1")
        require(torch.isfinite(torch.tensor(micro_f1)).item(), "Non-finite micro-F1")
        
        return accuracy, macro_f1, micro_f1
        
    except Exception as e:
        raise StagePreservingError(stage, e)


def test_msgnn(shared: SharedData) -> ModelResult:
    """Test MSGNN_link_prediction."""
    model_name = "MSGNN_link_prediction"
    stage = "initialization"
    
    result = ModelResult(
        model_name=model_name,
        stage=stage,
        passed=False,
        failure_category="",
        exception_type="",
        exception_message="",
        traceback_text="",
        config={},
        optimizer_config={},
        initial_loss=float('nan'),
        post_step_loss=float('nan'),
        trainable_tensors=0,
        trainable_elements=0,
        tensors_with_gradients=0,
        tensors_with_finite_gradients=0,
        tensors_with_nonzero_gradients=0,
        changed_parameter_count=0,
        test_accuracy=float('nan'),
        test_macro_f1=float('nan'),
        test_micro_f1=float('nan'),
        data_fingerprint="",
        feature_fingerprint="",
    )
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Verify fingerprints
        stage = "data fingerprint verification"
        data_fp = compute_data_fingerprint(
            shared.clean_graph, shared.clean_weights,
            shared.train_graph, shared.train_weights,
            shared.train_queries, shared.train_labels,
            shared.val_queries, shared.val_labels,
            shared.test_queries, shared.test_labels,
            shared.features,
            shared.generation_seed, shared.split_seed,
        )
        require(data_fp == shared.data_fingerprint, "Data fingerprint mismatch")
        result.data_fingerprint = data_fp
        
        stage = "feature fingerprint verification"
        feature_fp = compute_feature_fingerprint(shared.features)
        require(feature_fp == shared.feature_fingerprint, "Feature fingerprint mismatch")
        result.feature_fingerprint = feature_fp
        
        # Reset seed and construct model
        stage = "model construction"
        torch.manual_seed(SEED)
        
        config = {
            'num_features': shared.features.shape[1],
            'hidden': HIDDEN,
            'q': 0.25,
            'K': 2,
            'label_dim': LABEL_DIM,
            'activation': True,
            'trainable_q': False,
            'layer': 2,
            'dropout': 0.0,
            'normalization': 'sym',
            'cached': False,
            'conv_bias': True,
            'absolute_degree': True,
        }
        result.config = config
        
        optimizer_config = {
            'name': 'Adam',
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
        }
        result.optimizer_config = optimizer_config
        
        model = MSGNN_link_prediction(**config)
        model = model.to(DEVICE)
        
        # Verify all parameters on CPU
        for param in model.parameters():
            require(param.device == DEVICE, "Parameter not on CPU")
        
        model.train()
        require(model.training, "Model not in training mode")
        
        # Create optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.NLLLoss()
        
        # Define forward function
        def msgnn_forward(model, shared, query_edges):
            x_real = shared.features
            x_imag = x_real.clone()
            return model(
                x_real,
                x_imag,
                shared.train_graph,
                query_edges,
                shared.train_weights,
            )
        
        # Run one-step training
        initial_loss, post_loss, grad_stats = run_one_step_training(
            model, optimizer, criterion,
            shared, shared.train_queries, shared.train_labels,
            msgnn_forward,
        )
        
        result.initial_loss = initial_loss.item()
        result.post_step_loss = post_loss.item()
        result.trainable_tensors = grad_stats['trainable_tensors']
        result.trainable_elements = grad_stats['trainable_elements']
        result.tensors_with_gradients = grad_stats['tensors_with_gradients']
        result.tensors_with_finite_gradients = grad_stats['tensors_with_finite_gradients']
        result.tensors_with_nonzero_gradients = grad_stats['tensors_with_nonzero_gradients']
        result.changed_parameter_count = grad_stats['changed_parameter_count']
        
        # Run evaluation
        accuracy, macro_f1, micro_f1 = run_evaluation(model, shared, msgnn_forward)
        result.test_accuracy = accuracy
        result.test_macro_f1 = macro_f1
        result.test_micro_f1 = micro_f1
        
        result.passed = True
        result.stage = "complete"
        print(f"\n{model_name}: PASS")
        
    except StagePreservingError as e:
        result.passed = False
        result.stage = e.stage
        result.failure_category = classify_failure(e.stage, e.original_exception)
        result.exception_type = e.original_type
        result.exception_message = e.original_message
        result.traceback_text = e.original_traceback
        
        print(f"\n{model_name}: FAIL")
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
        
        print(f"\n{model_name}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )
    
    return result


def test_sssnet(shared: SharedData) -> ModelResult:
    """Test SSSNET_link_prediction with directed=True."""
    model_name = "SSSNET_link_prediction"
    stage = "initialization"
    
    result = ModelResult(
        model_name=model_name,
        stage=stage,
        passed=False,
        failure_category="",
        exception_type="",
        exception_message="",
        traceback_text="",
        config={},
        optimizer_config={},
        initial_loss=float('nan'),
        post_step_loss=float('nan'),
        trainable_tensors=0,
        trainable_elements=0,
        tensors_with_gradients=0,
        tensors_with_finite_gradients=0,
        tensors_with_nonzero_gradients=0,
        changed_parameter_count=0,
        test_accuracy=float('nan'),
        test_macro_f1=float('nan'),
        test_micro_f1=float('nan'),
        data_fingerprint="",
        feature_fingerprint="",
    )
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Verify fingerprints
        stage = "data fingerprint verification"
        data_fp = compute_data_fingerprint(
            shared.clean_graph, shared.clean_weights,
            shared.train_graph, shared.train_weights,
            shared.train_queries, shared.train_labels,
            shared.val_queries, shared.val_labels,
            shared.test_queries, shared.test_labels,
            shared.features,
            shared.generation_seed, shared.split_seed,
        )
        require(data_fp == shared.data_fingerprint, "Data fingerprint mismatch")
        result.data_fingerprint = data_fp
        
        stage = "feature fingerprint verification"
        feature_fp = compute_feature_fingerprint(shared.features)
        require(feature_fp == shared.feature_fingerprint, "Feature fingerprint mismatch")
        result.feature_fingerprint = feature_fp
        
        # Positive/negative decomposition
        stage = "positive/negative decomposition"
        signed_training_data = SignedData(
            edge_index=shared.train_graph,
            edge_weight=shared.train_weights,
            num_nodes=NUM_NODES,
        ).to(DEVICE)
        
        signed_training_data.separate_positive_negative()
        
        edge_index_p = signed_training_data.edge_index_p
        edge_weight_p = signed_training_data.edge_weight_p
        edge_index_n = signed_training_data.edge_index_n
        edge_weight_n = signed_training_data.edge_weight_n
        
        require(edge_index_p.shape[1] > 0, "Empty positive edge set")
        require(edge_index_n.shape[1] > 0, "Empty negative edge set")
        
        # Reset seed and construct model
        stage = "model construction"
        torch.manual_seed(SEED)
        
        config = {
            'nfeat': shared.features.shape[1],
            'hidden': HIDDEN,
            'nclass': LABEL_DIM,
            'dropout': 0.0,
            'hop': 2,
            'fill_value': 0.5,
            'directed': True,
            'bias': True,
        }
        result.config = config
        
        optimizer_config = {
            'name': 'Adam',
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
        }
        result.optimizer_config = optimizer_config
        
        model = SSSNET_link_prediction(**config)
        model = model.to(DEVICE)
        
        # Verify all parameters on CPU
        for param in model.parameters():
            require(param.device == DEVICE, "Parameter not on CPU")
        
        model.train()
        require(model.training, "Model not in training mode")
        
        # Create optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.NLLLoss()
        
        # Define forward function
        def sssnet_forward(model, shared, query_edges):
            return model(
                edge_index_p,
                edge_weight_p,
                edge_index_n,
                edge_weight_n,
                shared.features,
                query_edges,
            )
        
        # Run one-step training
        initial_loss, post_loss, grad_stats = run_one_step_training(
            model, optimizer, criterion,
            shared, shared.train_queries, shared.train_labels,
            sssnet_forward,
        )
        
        result.initial_loss = initial_loss.item()
        result.post_step_loss = post_loss.item()
        result.trainable_tensors = grad_stats['trainable_tensors']
        result.trainable_elements = grad_stats['trainable_elements']
        result.tensors_with_gradients = grad_stats['tensors_with_gradients']
        result.tensors_with_finite_gradients = grad_stats['tensors_with_finite_gradients']
        result.tensors_with_nonzero_gradients = grad_stats['tensors_with_nonzero_gradients']
        result.changed_parameter_count = grad_stats['changed_parameter_count']
        
        # Run evaluation
        accuracy, macro_f1, micro_f1 = run_evaluation(model, shared, sssnet_forward)
        result.test_accuracy = accuracy
        result.test_macro_f1 = macro_f1
        result.test_micro_f1 = micro_f1
        
        result.passed = True
        result.stage = "complete"
        print(f"\n{model_name}: PASS")
        
    except StagePreservingError as e:
        result.passed = False
        result.stage = e.stage
        result.failure_category = classify_failure(e.stage, e.original_exception)
        result.exception_type = e.original_type
        result.exception_message = e.original_message
        result.traceback_text = e.original_traceback
        
        print(f"\n{model_name}: FAIL")
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
        
        print(f"\n{model_name}: FAIL")
        print_structured_failure(
            category=result.failure_category,
            stage=result.stage,
            exception_type=result.exception_type,
            message=result.exception_message,
            traceback_text=result.traceback_text,
        )
    
    return result


def verify_shared_records(
    shared: SharedData,
    msgnn_result: ModelResult,
    sssnet_result: ModelResult,
) -> bool:
    """
    Verify that both models used the same shared data.
    
    Returns True if verification passes, False otherwise.
    """
    stage = "final shared-record verification"
    
    try:
        # Check both models passed
        if not msgnn_result.passed:
            print("SKIP: MSGNN did not pass")
            return False
        
        if not sssnet_result.passed:
            print("SKIP: SSSNET did not pass")
            return False
        
        # Verify data fingerprints match
        require(
            msgnn_result.data_fingerprint == sssnet_result.data_fingerprint,
            "Data fingerprints do not match between models"
        )
        require(
            msgnn_result.data_fingerprint == shared.data_fingerprint,
            "Model data fingerprints do not match original"
        )
        
        # Verify feature fingerprints match
        require(
            msgnn_result.feature_fingerprint == sssnet_result.feature_fingerprint,
            "Feature fingerprints do not match between models"
        )
        require(
            msgnn_result.feature_fingerprint == shared.feature_fingerprint,
            "Model feature fingerprints do not match original"
        )
        
        print("PASS: All fingerprints match")
        return True
        
    except Exception as e:
        failure_category = classify_failure(stage, e)
        print("FAIL: Shared record verification failed")
        print_structured_failure(
            category=failure_category,
            stage=stage,
            exception_type=type(e).__name__,
            message=str(e),
            traceback_text=traceback.format_exc(),
        )
        return False


def main() -> int:
    """Run Gate D training compatibility test."""
    
    # Print header
    print("="*60)
    print("Gate D — Minimal Training Compatibility Test")
    print("="*60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version: {version('torch-geometric')}")
    print(f"PyGSD version: {version('torch-geometric-signed-directed')}")
    print(f"Device: {DEVICE}")
    print(f"Generation seed: {SEED}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Task: {TASK}")
    print(f"Validation probability: {PROB_VAL}")
    print(f"Test probability: {PROB_TEST}")
    print("\nTest purpose: Minimal one-step training compatibility only")
    print("Does NOT test: convergence, model quality, robustness, stability, generalization")
    
    # Create shared data
    try:
        shared = create_shared_data()
    except Exception:
        print("\nGate D evaluation result: FAIL")
        print("Shared data creation failed.")
        return 1
    
    # Print shared data info
    print(f"\nShared data created successfully:")
    print(f"  Clean graph edges: {shared.clean_graph.shape[1]}")
    print(f"  Training graph edges: {shared.train_graph.shape[1]}")
    print(f"  Train queries: {shared.train_queries.shape[0]}")
    print(f"  Validation queries: {shared.val_queries.shape[0]}")
    print(f"  Test queries: {shared.test_queries.shape[0]}")
    print(f"  Feature shape: {tuple(shared.features.shape)}")
    print(f"  Feature dtype: {shared.features.dtype}")
    print(f"  Data fingerprint: {shared.data_fingerprint}")
    print(f"  Feature fingerprint: {shared.feature_fingerprint}")
    
    # Print class distributions
    print(f"\nClass distributions:")
    for split_name, labels in [
        ('Train', shared.train_labels),
        ('Validation', shared.val_labels),
        ('Test', shared.test_labels),
    ]:
        counts = [(labels == c).sum().item() for c in range(LABEL_DIM)]
        print(f"  {split_name}: {counts}")
    
    # Test models
    msgnn_result = test_msgnn(shared)
    sssnet_result = test_sssnet(shared)
    
    # Print results
    print(f"\n{'='*60}")
    print("Individual Model Results")
    print(f"{'='*60}")
    
    for result in [msgnn_result, sssnet_result]:
        print(f"\n{result.model_name}:")
        print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
        if result.passed:
            print(f"  Config: {result.config}")
            print(f"  Optimizer: {result.optimizer_config}")
            print(f"  Initial loss: {result.initial_loss:.6f}")
            print(f"  Post-step loss: {result.post_step_loss:.6f}")
            print(f"  Trainable tensors: {result.trainable_tensors}")
            print(f"  Trainable elements: {result.trainable_elements}")
            print(f"  Tensors with gradients: {result.tensors_with_gradients}")
            print(f"  Tensors with finite gradients: {result.tensors_with_finite_gradients}")
            print(f"  Tensors with nonzero gradients: {result.tensors_with_nonzero_gradients}")
            print(f"  Changed parameters: {result.changed_parameter_count}")
            print(f"  Test accuracy: {result.test_accuracy:.4f}")
            print(f"  Test macro-F1: {result.test_macro_f1:.4f}")
            print(f"  Test micro-F1: {result.test_micro_f1:.4f}")
            print(f"  Data fingerprint: {result.data_fingerprint}")
            print(f"  Feature fingerprint: {result.feature_fingerprint}")
        else:
            print(f"  Failed stage: {result.stage}")
            print(f"  Failure category: {result.failure_category}")
    
    # Verify shared record
    print(f"\n{'='*60}")
    print("Shared Record Verification")
    print(f"{'='*60}")
    
    shared_record_valid = verify_shared_records(shared, msgnn_result, sssnet_result)
    
    # Overall verdict
    print(f"\n{'='*60}")
    print("Gate D Evaluation Result")
    print(f"{'='*60}")
    
    overall_pass = (
        msgnn_result.passed and
        sssnet_result.passed and
        shared_record_valid
    )
    
    if overall_pass:
        print("PASS")
        print("\nBoth models completed minimal one-step training on shared data.")
        print("\nPermitted claims:")
        print("  - Both model APIs support minimal training on the defined task")
        print("  - Loss, backward pass, gradients, parameter update, and evaluation execute")
        print("  - Models can proceed to functional-baseline specification")
        print("\nNOT permitted:")
        print("  - Convergence, learning, or meaningful representations")
        print("  - Model performance comparison or superiority")
        print("  - Production readiness")
        print("  - Robustness or stability")
        print("  - Perturbation experiment authorization")
        print("  - Generalization to other datasets, seeds, devices, or hyperparameters")
        return 0
    else:
        print("FAIL")
        print("\nOne or more models failed minimal training compatibility.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
