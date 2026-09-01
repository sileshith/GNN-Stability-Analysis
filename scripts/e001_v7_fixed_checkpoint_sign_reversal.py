"""E001 fixed-checkpoint diagnostic: one eligible sign reversal.

This script does not train a model. It loads the accepted clean checkpoint,
freezes it, changes the sign of exactly one eligible message-passing edge,
and compares clean versus perturbed operators and outputs.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch_geometric_signed_directed.utils.general import (
    get_magnetic_signed_Laplacian,
)


DEVICE = torch.device("cpu")
ATOL = 1e-10


def load_clean_module():
    path = Path("scripts/e001_v7_clean_baseline.py")
    spec = importlib.util.spec_from_file_location("e001_clean", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_hash(value).encode("utf-8"))
    return digest.hexdigest()


def query_forbidden_pairs(shared):
    forbidden = set()
    for queries in (
        shared.train_queries,
        shared.val_queries,
        shared.test_queries,
    ):
        for source, target in queries.detach().cpu().tolist():
            source = int(source)
            target = int(target)
            forbidden.add((source, target))
            forbidden.add((target, source))
    return forbidden


def choose_eligible_edge(shared):
    edge_index = shared.train_graph.detach().cpu()
    edge_weight = shared.train_weights.detach().cpu()

    pairs = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    counts = Counter(pairs)
    pair_set = set(pairs)
    forbidden = query_forbidden_pairs(shared)

    candidates = []
    for index, (source, target) in enumerate(pairs):
        weight = float(edge_weight[index])

        if source == target:
            continue
        if abs(abs(weight) - 1.0) > ATOL:
            continue
        if counts[(source, target)] != 1:
            continue
        if (target, source) in pair_set:
            continue
        if (source, target) in forbidden:
            continue

        candidates.append((source, target, index, weight))

    if not candidates:
        raise RuntimeError("No eligible sign-reversal edge was found")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0], len(candidates)


def apply_sign_reversal(edge_index, edge_weight, index):
    edited_index = edge_index.clone()
    edited_weight = edge_weight.clone()
    edited_weight[index] *= -1

    if not torch.equal(edited_index, edge_index):
        raise RuntimeError("Sign reversal changed edge endpoints")

    changed = torch.nonzero(
        edited_weight != edge_weight,
        as_tuple=False,
    ).flatten()

    if changed.tolist() != [index]:
        raise RuntimeError("Sign reversal changed more than one weight")

    if edited_weight[index].item() != -edge_weight[index].item():
        raise RuntimeError("Selected weight was not exactly sign-reversed")

    return edited_index, edited_weight


def magnetic_adjacency(edge_index, edge_weight, q, num_nodes):
    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float64,
    )
    adjacency.index_put_(
        (edge_index[0].cpu(), edge_index[1].cpu()),
        edge_weight.detach().cpu().to(torch.float64),
        accumulate=True,
    )

    amplitude = (adjacency + adjacency.T) / 2
    theta = 2 * math.pi * q * (adjacency - adjacency.T)

    return amplitude.to(torch.complex128) * torch.exp(
        1j * theta.to(torch.complex128)
    )


def normalized_laplacian(edge_index, edge_weight, q, num_nodes):
    lap_index, lap_real, lap_imag = get_magnetic_signed_Laplacian(
        edge_index=edge_index,
        edge_weight=edge_weight,
        normalization="sym",
        dtype=torch.float64,
        num_nodes=num_nodes,
        q=q,
        return_lambda_max=False,
        absolute_degree=True,
    )

    values = torch.complex(lap_real, lap_imag)
    dense = torch.zeros(
        (num_nodes, num_nodes),
        dtype=values.dtype,
    )
    dense.index_put_(
        (lap_index[0], lap_index[1]),
        values,
        accumulate=True,
    )
    return dense


def classification_metrics(log_probabilities, labels):
    predictions = log_probabilities.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())

    f1_values = []
    for class_index in range(log_probabilities.shape[1]):
        predicted_class = predictions == class_index
        actual_class = labels == class_index

        true_positive = int((predicted_class & actual_class).sum())
        false_positive = int((predicted_class & ~actual_class).sum())
        false_negative = int((~predicted_class & actual_class).sum())

        denominator = (
            2 * true_positive + false_positive + false_negative
        )
        f1 = (
            0.0
            if denominator == 0
            else (2 * true_positive) / denominator
        )
        f1_values.append(f1)

    counts = [
        int((predictions == class_index).sum())
        for class_index in range(log_probabilities.shape[1])
    ]

    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "micro_f1": accuracy,
        "predicted_class_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="E001 fixed-checkpoint sign-reversal diagnostic"
    )
    parser.add_argument(
        "--clean-result",
        required=True,
        help="Accepted E001 clean-result JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Diagnostic JSON output path",
    )
    args = parser.parse_args()

    clean_result_path = Path(args.clean_result)
    output_path = Path(args.output)

    clean_record = json.loads(clean_result_path.read_text())
    if clean_record["gate"] != "E001":
        raise RuntimeError("Clean result is not labeled E001")
    if not clean_record["summary"]["overall_pass"]:
        raise RuntimeError("Clean result did not pass")

    run = clean_record["results"][0]
    config = dict(run["config"])

    required_config = {
        "K": 1,
        "q": 0.125,
        "trainable_q": False,
        "normalization": "sym",
        "cached": False,
        "absolute_degree": True,
        "layer": 2,
        "activation": True,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Unexpected clean configuration: {key}={config.get(key)}"
            )

    checkpoint_path = Path(run["checkpoint_path"])
    checkpoint_hash_before = sha256_file(checkpoint_path)
    if checkpoint_hash_before != run["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash does not match clean record")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    clean_module = load_clean_module()
    shared = clean_module.create_shared_data()

    if shared.bundle_fingerprint != run["bundle_fingerprint"]:
        raise RuntimeError("Clean data bundle fingerprint mismatch")

    frozen_tensors = {
        "features": shared.features.clone(),
        "train_graph": shared.train_graph.clone(),
        "train_weights": shared.train_weights.clone(),
        "train_queries": shared.train_queries.clone(),
        "train_labels": shared.train_labels.clone(),
        "val_queries": shared.val_queries.clone(),
        "val_labels": shared.val_labels.clone(),
        "test_queries": shared.test_queries.clone(),
        "test_labels": shared.test_labels.clone(),
    }
    frozen_hashes_before = {
        name: tensor_hash(value)
        for name, value in frozen_tensors.items()
    }

    payload = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model = clean_module.MSGNN_link_prediction(**config).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    model_hash_before = state_hash(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A model parameter remains trainable")

    (source, target, edge_index_position, original_weight), candidate_count = (
        choose_eligible_edge(shared)
    )

    perturbed_graph, perturbed_weights = apply_sign_reversal(
        shared.train_graph,
        shared.train_weights,
        edge_index_position,
    )
    perturbed_weight = float(
        perturbed_weights[edge_index_position].item()
    )

    q = float(config["q"])
    num_nodes = int(shared.num_nodes)

    clean_h = magnetic_adjacency(
        shared.train_graph,
        shared.train_weights,
        q,
        num_nodes,
    )
    perturbed_h = magnetic_adjacency(
        perturbed_graph,
        perturbed_weights,
        q,
        num_nodes,
    )
    delta_h = perturbed_h - clean_h
    delta_h_fro = float(
        torch.linalg.matrix_norm(delta_h, ord="fro")
    )
    expected_delta_h_fro = math.sqrt(2) * abs(
        math.cos(2 * math.pi * q)
    )

    if abs(delta_h_fro - expected_delta_h_fro) > 1e-9:
        raise RuntimeError("Theorem 5.3 sign identity mismatch")

    clean_l = normalized_laplacian(
        shared.train_graph,
        shared.train_weights,
        q,
        num_nodes,
    )
    perturbed_l = normalized_laplacian(
        perturbed_graph,
        perturbed_weights,
        q,
        num_nodes,
    )
    delta_l = perturbed_l - clean_l
    delta_l_fro = float(
        torch.linalg.matrix_norm(delta_l, ord="fro")
    )
    delta_l_spectral = float(
        torch.linalg.matrix_norm(delta_l, ord=2)
    )

    clean_linear_outputs = []
    perturbed_linear_outputs = []

    linear_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    clean_hooks = [
        module.register_forward_hook(
            lambda module, inputs, output: clean_linear_outputs.append(
                output.detach().clone()
            )
        )
        for module in linear_modules
    ]

    with torch.no_grad():
        clean_output = model(
            shared.features,
            shared.features.clone(),
            shared.train_graph,
            shared.test_queries,
            shared.train_weights,
        )
        clean_representation = model.z.detach().clone()

    for hook in clean_hooks:
        hook.remove()

    perturbed_hooks = [
        module.register_forward_hook(
            lambda module, inputs, output: perturbed_linear_outputs.append(
                output.detach().clone()
            )
        )
        for module in linear_modules
    ]

    with torch.no_grad():
        perturbed_output = model(
            shared.features,
            shared.features.clone(),
            perturbed_graph,
            shared.test_queries,
            perturbed_weights,
        )
        perturbed_representation = model.z.detach().clone()

    for hook in perturbed_hooks:
        hook.remove()

    if not clean_linear_outputs or not perturbed_linear_outputs:
        raise RuntimeError("Linear logits were not captured")

    clean_logits = clean_linear_outputs[-1]
    perturbed_logits = perturbed_linear_outputs[-1]

    tensors_to_check = {
        "clean_representation": clean_representation,
        "perturbed_representation": perturbed_representation,
        "clean_logits": clean_logits,
        "perturbed_logits": perturbed_logits,
        "clean_log_probabilities": clean_output,
        "perturbed_log_probabilities": perturbed_output,
    }
    finite_checks = {
        name: bool(torch.isfinite(value).all())
        for name, value in tensors_to_check.items()
    }
    if not all(finite_checks.values()):
        raise RuntimeError("A diagnostic tensor is non-finite")

    clean_predictions = clean_output.argmax(dim=1)
    perturbed_predictions = perturbed_output.argmax(dim=1)

    representation_delta = (
        perturbed_representation - clean_representation
    )
    logits_delta = perturbed_logits - clean_logits
    output_delta = perturbed_output - clean_output

    clean_metrics = classification_metrics(
        clean_output,
        shared.test_labels,
    )
    perturbed_metrics = classification_metrics(
        perturbed_output,
        shared.test_labels,
    )

    flip_count = int(
        (clean_predictions != perturbed_predictions).sum()
    )
    flip_rate = flip_count / clean_predictions.numel()

    model_hash_after = state_hash(model)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    frozen_tensors = {
        "features": shared.features,
        "train_graph": shared.train_graph,
        "train_weights": shared.train_weights,
        "train_queries": shared.train_queries,
        "train_labels": shared.train_labels,
        "val_queries": shared.val_queries,
        "val_labels": shared.val_labels,
        "test_queries": shared.test_queries,
        "test_labels": shared.test_labels,
    }
    frozen_hashes_after = {
        name: tensor_hash(value)
        for name, value in frozen_tensors.items()
    }

    if model_hash_after != model_hash_before:
        raise RuntimeError("Model parameters or buffers changed")
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Checkpoint file changed")
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("A frozen tensor changed")

    result = {
        "gate": "E001-fixed-checkpoint-diagnostic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "evidence_scope": (
            "One deterministic sign reversal using one frozen "
            "K=1 MSGNN checkpoint; not a robustness estimate"
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "pyg": version("torch-geometric"),
            "pygsd": version(
                "torch-geometric-signed-directed"
            ),
            "device": str(DEVICE),
        },
        "clean_artifact": {
            "result_path": str(clean_result_path),
            "result_sha256": sha256_file(clean_result_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash_before,
            "bundle_fingerprint": shared.bundle_fingerprint,
            "config": config,
        },
        "edit_manifest": {
            "edit_type": "sign_reversal",
            "selection_rule": (
                "Lexicographically first eligible training-graph edge"
            ),
            "candidate_count": candidate_count,
            "edge_index_position": edge_index_position,
            "source": source,
            "target": target,
            "original_weight": original_weight,
            "perturbed_weight": perturbed_weight,
            "unit_weight": abs(abs(original_weight) - 1.0) <= ATOL,
            "non_self_loop": source != target,
            "unreciprocated": True,
            "ordered_pair_unique": True,
            "query_and_reverse_query_leakage_excluded": True,
            "direction_preserved": True,
            "endpoints_preserved": True,
            "only_one_weight_changed": True,
            "theorem_5_3_eligible": True,
            "m": 1,
        },
        "operator_diagnostics": {
            "q": q,
            "delta_h_fro": delta_h_fro,
            "theorem_5_3_expected_delta_h_fro": (
                expected_delta_h_fro
            ),
            "theorem_5_3_abs_error": abs(
                delta_h_fro - expected_delta_h_fro
            ),
            "delta_l_fro": delta_l_fro,
            "delta_l_spectral": delta_l_spectral,
            "spectral_le_fro": (
                delta_l_spectral <= delta_l_fro + 1e-10
            ),
        },
        "representation_diagnostics": {
            "shape": list(clean_representation.shape),
            "delta_fro": float(
                torch.linalg.matrix_norm(
                    representation_delta,
                    ord="fro",
                )
            ),
            "clean_fro": float(
                torch.linalg.matrix_norm(
                    clean_representation,
                    ord="fro",
                )
            ),
            "relative_delta_fro": float(
                torch.linalg.matrix_norm(
                    representation_delta,
                    ord="fro",
                )
                / torch.linalg.matrix_norm(
                    clean_representation,
                    ord="fro",
                )
            ),
        },
        "logit_diagnostics": {
            "shape": list(clean_logits.shape),
            "delta_fro": float(
                torch.linalg.matrix_norm(
                    logits_delta,
                    ord="fro",
                )
            ),
            "max_abs_delta": float(
                torch.max(torch.abs(logits_delta))
            ),
        },
        "output_diagnostics": {
            "shape": list(clean_output.shape),
            "delta_fro": float(
                torch.linalg.matrix_norm(
                    output_delta,
                    ord="fro",
                )
            ),
            "max_abs_delta": float(
                torch.max(torch.abs(output_delta))
            ),
            "prediction_flip_count": flip_count,
            "prediction_flip_rate": flip_rate,
        },
        "task_metrics": {
            "clean": clean_metrics,
            "perturbed": perturbed_metrics,
            "accuracy_change": (
                perturbed_metrics["accuracy"]
                - clean_metrics["accuracy"]
            ),
            "macro_f1_change": (
                perturbed_metrics["macro_f1"]
                - clean_metrics["macro_f1"]
            ),
        },
        "integrity": {
            "all_diagnostic_tensors_finite": all(
                finite_checks.values()
            ),
            "finite_checks": finite_checks,
            "model_state_hash_before": model_hash_before,
            "model_state_hash_after": model_hash_after,
            "model_state_unchanged": (
                model_hash_after == model_hash_before
            ),
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_hash_after": checkpoint_hash_after,
            "checkpoint_unchanged": (
                checkpoint_hash_after == checkpoint_hash_before
            ),
            "frozen_tensor_hashes_before": frozen_hashes_before,
            "frozen_tensor_hashes_after": frozen_hashes_after,
            "frozen_tensors_unchanged": (
                frozen_hashes_after == frozen_hashes_before
            ),
            "parameters_require_grad": any(
                parameter.requires_grad
                for parameter in model.parameters()
            ),
        },
        "limitations": [
            "One checkpoint and one edited edge only",
            "The selected edge is diagnostic, not a random sample",
            "No robustness or statistical claim is supported",
            "Theorem 5.3 identity applies to magnetic adjacency H_q only",
            "The normalized-Laplacian and downstream changes are measured, not exact theorem identities",
            "q=1/8 is experimentally motivated but not mentor-confirmed",
            "Effective PyGSD MSConv message flow is source_to_target",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    print("E001 FIXED-CHECKPOINT DIAGNOSTIC: PASS")
    print("EDIT:", result["edit_manifest"])
    print("OPERATOR:", result["operator_diagnostics"])
    print("OUTPUT:", result["output_diagnostics"])
    print("TASK:", result["task_metrics"])
    print("INTEGRITY:", result["integrity"])
    print("RESULT:", output_path)


if __name__ == "__main__":
    main()
