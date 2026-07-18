"""
PyGSD CPU Forward-Pass Smoke Test

Implements Gate C specification: tests executable forward compatibility for
MSGNN, MagNet, and SSSNET link prediction architectures on minimal synthetic
graphs with CPU-only execution.

Scope:
- One primary forward pass per model
- One second unchanged-input forward pass for repeatability validation
- Finite output validation
- Shape and dtype verification
- Probability normalization checks
- Unchanged-input repeatability

Does NOT test:
- Training, backward pass, or gradients
- Datasets or data loading
- GPU execution
- Perturbation experiments
- Robustness, stability, or accuracy
- Model comparability or scientific validity
"""

import sys
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Dict, Any, Tuple

import torch
from torch_geometric_signed_directed.nn.general.MSGNN import MSGNN_link_prediction
from torch_geometric_signed_directed.nn.directed.MagNet_link_prediction import MagNet_link_prediction
from torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction import SSSNET_link_prediction


# Global constants
SEED = 0
DEVICE = torch.device("cpu")
NUM_NODES = 4
NUM_FEATURES = 3
HIDDEN = 2
NUM_CLASSES = 2
NUM_QUERY_EDGES = 3
NORMALIZATION_ATOL = 1e-5
NORMALIZATION_RTOL = 1e-5
REPEAT_ATOL = 1e-6
REPEAT_RTOL = 1e-6

# Set random seed
torch.manual_seed(SEED)


def get_msgnn_magnet_tensors() -> Dict[str, torch.Tensor]:
    """
    Create fresh common tensors for MSGNN and MagNet tests.
    
    Returns:
        Dictionary containing real, imag, edge_index, edge_weight, query_edges
    """
    real = torch.tensor([
        [1.0, 0.5, 0.2],
        [0.8, 1.0, 0.3],
        [0.6, 0.4, 1.0],
        [0.9, 0.7, 0.5],
    ], dtype=torch.float32, device=DEVICE)
    
    imag = torch.zeros(
        NUM_NODES,
        NUM_FEATURES,
        dtype=torch.float32,
        device=DEVICE,
    )
    
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 1],
        [1, 2, 3, 0, 2, 3],
    ], dtype=torch.long, device=DEVICE).contiguous()
    
    edge_weight = torch.ones(
        6,
        dtype=torch.float32,
        device=DEVICE,
    )
    
    query_edges = torch.tensor([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=torch.long, device=DEVICE).contiguous()
    
    return {
        'real': real,
        'imag': imag,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'query_edges': query_edges,
    }


def get_sssnet_tensors() -> Dict[str, torch.Tensor]:
    """
    Create fresh SSSNET tensors for directed=False mode.
    
    Positive and negative structures use bidirectional representation
    for undirected signed graph encoding.
    
    Returns:
        Dictionary containing features, edge_index_p, edge_weight_p,
        edge_index_n, edge_weight_n, query_edges
    """
    features = torch.tensor([
        [1.0, 0.5, 0.2],
        [0.8, 1.0, 0.3],
        [0.6, 0.4, 1.0],
        [0.9, 0.7, 0.5],
    ], dtype=torch.float32, device=DEVICE)
    
    # Positive structure: {0,1} and {2,3} as bidirectional pairs
    edge_index_p = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ], dtype=torch.long, device=DEVICE).contiguous()
    
    edge_weight_p = torch.ones(
        4,
        dtype=torch.float32,
        device=DEVICE,
    )
    
    # Negative structure: {1,2} and {0,3} as bidirectional pairs
    edge_index_n = torch.tensor([
        [1, 2, 0, 3],
        [2, 1, 3, 0],
    ], dtype=torch.long, device=DEVICE).contiguous()
    
    edge_weight_n = torch.ones(
        4,
        dtype=torch.float32,
        device=DEVICE,
    )
    
    query_edges = torch.tensor([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=torch.long, device=DEVICE).contiguous()
    
    return {
        'features': features,
        'edge_index_p': edge_index_p,
        'edge_weight_p': edge_weight_p,
        'edge_index_n': edge_index_n,
        'edge_weight_n': edge_weight_n,
        'query_edges': query_edges,
    }


def report_tensor_metadata(name: str, tensor: torch.Tensor) -> None:
    """Print tensor metadata without printing full tensor values."""
    contiguous_str = ""
    if tensor.dtype == torch.long:
        contiguous_str = f", contiguous={tensor.is_contiguous()}"
    print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}{contiguous_str}")


def validate_output(
    output: Any,
    expected_shape: Tuple[int, int] = (NUM_QUERY_EDGES, NUM_CLASSES)
) -> Dict[str, Any]:
    """
    Validate forward-pass output tensor.
    
    Args:
        output: Object returned by model forward pass
        expected_shape: Expected output shape
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        AssertionError: If any validation check fails
    """
    # Check object type
    assert isinstance(output, torch.Tensor), f"Expected torch.Tensor, got {type(output)}"
    
    # Check dimensionality
    assert output.ndim == 2, f"Expected 2D tensor, got {output.ndim}D"
    
    # Check shape
    actual_shape = tuple(output.shape)
    assert actual_shape == expected_shape, f"Expected shape {expected_shape}, got {actual_shape}"
    
    # Check dtype is floating point
    assert output.is_floating_point(), f"Expected floating point dtype, got {output.dtype}"
    
    # Check finiteness
    is_finite = torch.isfinite(output).all().item()
    assert is_finite, "Output contains non-finite values"
    
    # Check for NaN
    has_nan = torch.isnan(output).any().item()
    assert not has_nan, "Output contains NaN values"
    
    # Check for positive infinity
    has_posinf = torch.isposinf(output).any().item()
    assert not has_posinf, "Output contains positive infinity"
    
    # Check for negative infinity
    has_neginf = torch.isneginf(output).any().item()
    assert not has_neginf, "Output contains negative infinity"
    
    # Check probability normalization: exp(log_prob).sum(dim=1) ≈ 1
    prob_sums = torch.exp(output).sum(dim=1)
    expected_sums = torch.ones(
        NUM_QUERY_EDGES,
        dtype=prob_sums.dtype,
        device=prob_sums.device,
    )
    normalization_ok = torch.allclose(
        prob_sums,
        expected_sums,
        atol=NORMALIZATION_ATOL,
        rtol=NORMALIZATION_RTOL
    )
    assert normalization_ok, f"Probability normalization failed: sums={prob_sums.tolist()}"
    
    return {
        'object_type': type(output).__name__,
        'shape': actual_shape,
        'dtype': str(output.dtype),
        'finite': is_finite,
        'nan_free': not has_nan,
        'positive_infinity_free': not has_posinf,
        'negative_infinity_free': not has_neginf,
        'infinity_free': not has_posinf and not has_neginf,
        'normalized': normalization_ok,
        'prob_sums': prob_sums.tolist(),
    }


def test_msgnn() -> bool:
    """
    Test MSGNN_link_prediction forward pass.
    
    Returns:
        True if test passes, False otherwise
    """
    model_name = "MSGNN_link_prediction"
    stage = "initialization"
    config: Dict[str, Any] = {}
    tensors: Dict[str, torch.Tensor] = {}
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Tensor construction
        stage = "tensor construction"
        tensors = get_msgnn_magnet_tensors()
        print("\nInput tensors:")
        for name, tensor in tensors.items():
            report_tensor_metadata(name, tensor)
        
        # Model construction
        stage = "model construction"
        print("\nConstructor configuration:")
        config = {
            'num_features': NUM_FEATURES,
            'hidden': HIDDEN,
            'q': 0.25,
            'K': 2,
            'label_dim': NUM_CLASSES,
            'activation': True,
            'trainable_q': False,
            'layer': 2,
            'dropout': 0.0,
            'normalization': 'sym',
            'cached': False,
            'conv_bias': True,
            'absolute_degree': True,
        }
        for key, value in config.items():
            print(f"  {key}={value}")
        
        model = MSGNN_link_prediction(**config)
        model = model.to(DEVICE)
        model.eval()
        
        # First forward pass
        stage = "first forward pass"
        with torch.no_grad():
            output1 = model(
                tensors['real'],
                tensors['imag'],
                tensors['edge_index'],
                tensors['query_edges'],
                tensors['edge_weight'],
            )
        
        # First output validation
        stage = "first output validation"
        result1 = validate_output(output1)
        print(f"\nFirst forward pass:")
        print(f"  Output type: {result1['object_type']}")
        print(f"  Output shape: {result1['shape']}")
        print(f"  Output dtype: {result1['dtype']}")
        print(f"  Finite: PASS")
        print(f"  NaN-free: PASS")
        print(f"  Positive-infinity-free: PASS")
        print(f"  Negative-infinity-free: PASS")
        print(f"  Infinity-free: PASS")
        print(f"  Probability normalization: PASS")
        print(f"  Probability row sums: {result1['prob_sums']}")
        
        # Second forward pass (repeatability check)
        stage = "second forward pass"
        with torch.no_grad():
            output2 = model(
                tensors['real'],
                tensors['imag'],
                tensors['edge_index'],
                tensors['query_edges'],
                tensors['edge_weight'],
            )
        
        # Second output validation
        stage = "second output validation"
        result2 = validate_output(output2)
        
        # Repeatability validation
        # This verifies unchanged-input repeatability only.
        # It does not verify changed-graph cache recomputation.
        stage = "repeatability validation"
        repeatable = torch.allclose(
            output1,
            output2,
            atol=REPEAT_ATOL,
            rtol=REPEAT_RTOL
        )
        assert repeatable, "Unchanged-input repeatability failed"
        print(f"  Unchanged-input repeatability: PASS")
        
        print(f"\n{model_name}: PASS")
        return True
        
    except Exception as e:
        print(f"\n{model_name}: FAIL")
        print(f"  Failed stage: {stage}")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        
        if config:
            print(f"  Constructor configuration:")
            for key, value in config.items():
                print(f"    {key}={value}")
        else:
            print(f"  Constructor configuration: not yet available")
        
        if tensors:
            print(f"  Input tensor metadata:")
            for name, tensor in tensors.items():
                report_tensor_metadata(name, tensor)
        else:
            print(f"  Input tensor metadata: not yet available")
        
        return False


def test_magnet() -> bool:
    """
    Test MagNet_link_prediction forward pass.
    
    Returns:
        True if test passes, False otherwise
    """
    model_name = "MagNet_link_prediction"
    stage = "initialization"
    config: Dict[str, Any] = {}
    tensors: Dict[str, torch.Tensor] = {}
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Tensor construction
        stage = "tensor construction"
        tensors = get_msgnn_magnet_tensors()
        print("\nInput tensors:")
        for name, tensor in tensors.items():
            report_tensor_metadata(name, tensor)
        
        # Model construction
        stage = "model construction"
        print("\nConstructor configuration:")
        config = {
            'num_features': NUM_FEATURES,
            'hidden': HIDDEN,
            'q': 0.25,
            'K': 1,
            'label_dim': NUM_CLASSES,
            'activation': True,
            'trainable_q': False,
            'layer': 2,
            'dropout': 0.0,
            'normalization': 'sym',
            'cached': False,
        }
        for key, value in config.items():
            print(f"  {key}={value}")
        
        model = MagNet_link_prediction(**config)
        model = model.to(DEVICE)
        model.eval()
        
        # First forward pass
        stage = "first forward pass"
        with torch.no_grad():
            output1 = model(
                tensors['real'],
                tensors['imag'],
                tensors['edge_index'],
                tensors['query_edges'],
                tensors['edge_weight'],
            )
        
        # First output validation
        stage = "first output validation"
        result1 = validate_output(output1)
        print(f"\nFirst forward pass:")
        print(f"  Output type: {result1['object_type']}")
        print(f"  Output shape: {result1['shape']}")
        print(f"  Output dtype: {result1['dtype']}")
        print(f"  Finite: PASS")
        print(f"  NaN-free: PASS")
        print(f"  Positive-infinity-free: PASS")
        print(f"  Negative-infinity-free: PASS")
        print(f"  Infinity-free: PASS")
        print(f"  Probability normalization: PASS")
        print(f"  Probability row sums: {result1['prob_sums']}")
        
        # Second forward pass (repeatability check)
        stage = "second forward pass"
        with torch.no_grad():
            output2 = model(
                tensors['real'],
                tensors['imag'],
                tensors['edge_index'],
                tensors['query_edges'],
                tensors['edge_weight'],
            )
        
        # Second output validation
        stage = "second output validation"
        result2 = validate_output(output2)
        
        # Repeatability validation
        # This verifies unchanged-input repeatability only.
        # It does not verify changed-graph cache recomputation.
        stage = "repeatability validation"
        repeatable = torch.allclose(
            output1,
            output2,
            atol=REPEAT_ATOL,
            rtol=REPEAT_RTOL
        )
        assert repeatable, "Unchanged-input repeatability failed"
        print(f"  Unchanged-input repeatability: PASS")
        
        print(f"\n{model_name}: PASS")
        return True
        
    except Exception as e:
        print(f"\n{model_name}: FAIL")
        print(f"  Failed stage: {stage}")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        
        if config:
            print(f"  Constructor configuration:")
            for key, value in config.items():
                print(f"    {key}={value}")
        else:
            print(f"  Constructor configuration: not yet available")
        
        if tensors:
            print(f"  Input tensor metadata:")
            for name, tensor in tensors.items():
                report_tensor_metadata(name, tensor)
        else:
            print(f"  Input tensor metadata: not yet available")
        
        return False


def test_sssnet() -> bool:
    """
    Test SSSNET_link_prediction forward pass with directed=False.
    
    Returns:
        True if test passes, False otherwise
    """
    model_name = "SSSNET_link_prediction"
    stage = "initialization"
    config: Dict[str, Any] = {}
    tensors: Dict[str, torch.Tensor] = {}
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        # Tensor construction
        stage = "tensor construction"
        tensors = get_sssnet_tensors()
        print("\nInput tensors:")
        for name, tensor in tensors.items():
            report_tensor_metadata(name, tensor)
        
        # Model construction
        stage = "model construction"
        print("\nConstructor configuration:")
        config = {
            'nfeat': NUM_FEATURES,
            'hidden': HIDDEN,
            'nclass': NUM_CLASSES,
            'dropout': 0.0,
            'hop': 2,
            'fill_value': 1.0,
            'directed': False,
            'bias': True,
        }
        for key, value in config.items():
            print(f"  {key}={value}")
        
        model = SSSNET_link_prediction(**config)
        model = model.to(DEVICE)
        model.eval()
        
        # First forward pass
        stage = "first forward pass"
        with torch.no_grad():
            output1 = model(
                tensors['edge_index_p'],
                tensors['edge_weight_p'],
                tensors['edge_index_n'],
                tensors['edge_weight_n'],
                tensors['features'],
                tensors['query_edges'],
            )
        
        # First output validation
        stage = "first output validation"
        result1 = validate_output(output1)
        print(f"\nFirst forward pass:")
        print(f"  Output type: {result1['object_type']}")
        print(f"  Output shape: {result1['shape']}")
        print(f"  Output dtype: {result1['dtype']}")
        print(f"  Finite: PASS")
        print(f"  NaN-free: PASS")
        print(f"  Positive-infinity-free: PASS")
        print(f"  Negative-infinity-free: PASS")
        print(f"  Infinity-free: PASS")
        print(f"  Probability normalization: PASS")
        print(f"  Probability row sums: {result1['prob_sums']}")
        
        # Second forward pass (repeatability check)
        stage = "second forward pass"
        with torch.no_grad():
            output2 = model(
                tensors['edge_index_p'],
                tensors['edge_weight_p'],
                tensors['edge_index_n'],
                tensors['edge_weight_n'],
                tensors['features'],
                tensors['query_edges'],
            )
        
        # Second output validation
        stage = "second output validation"
        result2 = validate_output(output2)
        
        # Repeatability validation
        # This verifies unchanged-input repeatability only.
        # It does not verify changed-graph cache recomputation.
        stage = "repeatability validation"
        repeatable = torch.allclose(
            output1,
            output2,
            atol=REPEAT_ATOL,
            rtol=REPEAT_RTOL
        )
        assert repeatable, "Unchanged-input repeatability failed"
        print(f"  Unchanged-input repeatability: PASS")
        
        print(f"\n{model_name}: PASS")
        return True
        
    except Exception as e:
        print(f"\n{model_name}: FAIL")
        print(f"  Failed stage: {stage}")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        
        if config:
            print(f"  Constructor configuration:")
            for key, value in config.items():
                print(f"    {key}={value}")
        else:
            print(f"  Constructor configuration: not yet available")
        
        if tensors:
            print(f"  Input tensor metadata:")
            for name, tensor in tensors.items():
                report_tensor_metadata(name, tensor)
        else:
            print(f"  Input tensor metadata: not yet available")
        
        return False


def main() -> None:
    """Run PyGSD CPU forward-pass smoke tests."""
    
    # Print environment header
    print("="*60)
    print("PyGSD CPU Forward-Pass Smoke Test")
    print("="*60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyG version: {version('torch-geometric')}")
    print(f"PyGSD version: {version('torch-geometric-signed-directed')}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("\nTest purpose: Executable forward compatibility only")
    print("Does NOT test: training, datasets, GPU, perturbations, robustness, stability, accuracy")
    
    # Run model tests
    results = {
        'MSGNN': test_msgnn(),
        'MagNet': test_magnet(),
        'SSSNET': test_sssnet(),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for model_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{model_name}: {status}")
    
    # Determine overall result
    all_passed = all(results.values())
    
    if all_passed:
        print("\nGate C evaluation result: PASS")
        print("All three models completed primary forward-pass tests successfully.")
        exit_code = 0
    else:
        print("\nGate C evaluation result: FAIL")
        print("One or more models failed primary forward-pass tests.")
        exit_code = 1
    
    print("\nNote: This test establishes forward-pass compatibility only.")
    print("It does not establish training correctness, robustness, or model validity.")
    
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
