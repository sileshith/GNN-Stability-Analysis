"""
PyGSD Model Instantiation Smoke Test

Verifies that MSGNN, MagNet, and SSSNET link prediction architectures
can be imported and instantiated on CPU with valid constructor signatures.

Does not train, run forward passes, download datasets, or require GPU.
"""

import torch
import torch.nn
from importlib.metadata import version
from torch_geometric_signed_directed.nn.general.MSGNN import MSGNN_link_prediction
from torch_geometric_signed_directed.nn.directed.MagNet_link_prediction import MagNet_link_prediction
from torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction import SSSNET_link_prediction


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def verify_parameters_finite(model: torch.nn.Module) -> bool:
    """Verify all parameter tensors contain only finite values."""
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def main() -> None:
    """Run PyGSD model instantiation smoke test."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Print environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyGSD version: {version('torch-geometric-signed-directed')}")
    print(f"Device: CPU")
    print()
    
    # Test MSGNN
    msgnn = MSGNN_link_prediction(
        num_features=4,
        hidden=2,
        q=0.25,
        K=2,
        label_dim=2,
        activation=True,
        trainable_q=False,
        layer=2,
        dropout=0.0,
        normalization="sym",
        cached=False,
        conv_bias=True,
        absolute_degree=True
    )
    assert isinstance(msgnn, torch.nn.Module)
    msgnn = msgnn.to('cpu')
    msgnn_total, msgnn_trainable = count_parameters(msgnn)
    assert verify_parameters_finite(msgnn)
    print(f"MSGNN_link_prediction: total_params={msgnn_total}, trainable_params={msgnn_trainable}, status=PASS")
    
    # Test MagNet
    magnet = MagNet_link_prediction(
        num_features=4,
        hidden=2,
        q=0.25,
        K=1,
        label_dim=2,
        activation=True,
        trainable_q=False,
        layer=2,
        dropout=0.0,
        normalization="sym",
        cached=False
    )
    assert isinstance(magnet, torch.nn.Module)
    magnet = magnet.to('cpu')
    magnet_total, magnet_trainable = count_parameters(magnet)
    assert verify_parameters_finite(magnet)
    print(f"MagNet_link_prediction: total_params={magnet_total}, trainable_params={magnet_trainable}, status=PASS")
    
    # Test SSSNET
    sssnet = SSSNET_link_prediction(
        nfeat=4,
        hidden=2,
        nclass=2,
        dropout=0.0,
        hop=2,
        fill_value=0.5,
        directed=False,
        bias=True
    )
    assert isinstance(sssnet, torch.nn.Module)
    sssnet = sssnet.to('cpu')
    sssnet_total, sssnet_trainable = count_parameters(sssnet)
    assert verify_parameters_finite(sssnet)
    print(f"SSSNET_link_prediction: total_params={sssnet_total}, trainable_params={sssnet_trainable}, status=PASS")
    
    print()
    print("PYGSD MODEL INSTANTIATION SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
