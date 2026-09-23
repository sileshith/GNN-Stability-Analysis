"""E012 cross-graph paired clean/perturbed MSGNN retraining."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


BASE_SOURCE = Path("scripts/e005_charge_comparison_replication.py")
MANIFEST_SOURCE = Path("scripts/e008_expanded_perturbation_manifest.py")
SELECTION_SOURCE = Path("scripts/e009_scaled_fixed_checkpoint_pilot.py")
TRAINING_SOURCE = Path("scripts/e011_multiseed_matched_retraining.py")
EDIT_SOURCE = Path("scripts/e006_fixed_checkpoint_stratified_pilot.py")

SOURCE_HASHES = {
    str(BASE_SOURCE): (
        "5c252bbb56c543e56a5ac43c0f7163df"
        "96393ee69604a7e867e03e0e93721f82"
    ),
    str(MANIFEST_SOURCE): (
        "1000c4f17b91deda217b4ca466754d933"
        "bff6cdb0d8983235b8ef16ce4ea14d6"
    ),
    str(SELECTION_SOURCE): (
        "274ae8048377e52805608ffe724d6fb17"
        "28570fb9de064d9a26faa6cb88e2f7e"
    ),
    str(TRAINING_SOURCE): (
        "4bbd01230e8516b27eaddbbc06097c95"
        "26848aaf75b0516a159198cd67b4a285"
    ),
    str(EDIT_SOURCE): (
        "300fb65268fe61de546bbee6d656638fe"
        "4ccc455d4ad0b7879d8b1afc946198f"
    ),
}

GRAPH_SEEDS = [1, 2, 3, 4]
MODEL_SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 0
FIXED_Q = 0.125
TRAINABLE_Q = False
BUDGET_FRACTION = 0.02
ZERO_METRICS = {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "micro_f1": 0.0,
    "sign_accuracy": 0.0,
    "direction_accuracy": 0.0,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    require(path.is_file(), f"Missing source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"Cannot import source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources():
    for path_string, expected_hash in SOURCE_HASHES.items():
        source = Path(path_string)
        require(source.is_file(), f"Missing source: {source}")
        require(
            sha256_file(source) == expected_hash,
            f"Source hash mismatch: {source}",
        )


def build_manifest(base, manifest_module, selection_module, shared):
    communities = manifest_module.regenerate_communities(base, shared)

    edge_index = shared.train_graph.detach().cpu()
    edge_weight = shared.train_weights.detach().cpu()
    pairs = [
        (
            int(edge_index[0, index].item()),
            int(edge_index[1, index].item()),
        )
        for index in range(edge_index.shape[1])
    ]

    ordered_pair_counts = Counter(pairs)
    pair_set = set(pairs)
    forbidden = manifest_module.evaluation_forbidden_pairs(shared)
    degree = manifest_module.absolute_degree(
        edge_index,
        edge_weight,
        shared.num_nodes,
    )

    common_records = []
    deletion_records = []
    exclusions = Counter()

    for index, (source, target) in enumerate(pairs):
        weight = float(edge_weight[index].item())

        if source == target:
            exclusions["self_loop"] += 1
            continue
        if abs(abs(weight) - 1.0) > manifest_module.ATOL:
            exclusions["non_unit_weight"] += 1
            continue
        if ordered_pair_counts[(source, target)] != 1:
            exclusions["ordered_pair_not_unique"] += 1
            continue
        if (target, source) in pair_set:
            exclusions["reciprocated"] += 1
            continue
        if (source, target) in forbidden:
            exclusions[
                "validation_test_query_or_reverse_query"
            ] += 1
            continue

        record = manifest_module.classify_edge(
            source,
            target,
            index,
            weight,
            communities,
        )
        common_records.append(record)

        removed_degree = abs(weight) / 2.0
        deletion_eligible = (
            degree[source].item() - removed_degree > 0
            and degree[target].item() - removed_degree > 0
        )
        if deletion_eligible:
            deletion_records.append(record)
        else:
            exclusions["deletion_nonpositive_degree"] += 1

    common_records.sort(
        key=lambda record: (
            record["source"],
            record["target"],
            record["edge_index_position"],
        )
    )
    deletion_records.sort(
        key=lambda record: (
            record["source"],
            record["target"],
            record["edge_index_position"],
        )
    )

    disrupting_records = [
        record
        for record in common_records
        if record["sign_reversal_stratum"] == "signal_disrupting"
    ]
    disrupting_records.sort(
        key=lambda record: selection_module.selection_key(
            "sign_reversal",
            "signal_disrupting",
            record,
        )
    )

    training_edge_count = int(edge_index.shape[1])
    budget = max(
        1,
        int(math.floor(BUDGET_FRACTION * training_edge_count + 0.5)),
    )
    require(
        budget <= len(disrupting_records),
        "Insufficient signal-disrupting edges",
    )

    manifest = {
        "artifact": "E012 in-memory graph-specific semantic manifest",
        "status": "PASS",
        "source": {
            "bundle_fingerprint": shared.bundle_fingerprint,
            "generation_seed": base.GENERATION_SEED,
            "split_seed": base.SPLIT_SEED,
        },
        "eligibility": {
            "training_graph_edge_count": training_edge_count,
            "evaluation_forbidden_ordered_pair_count": len(forbidden),
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "sign_reversal": {
            "eligible_count": len(common_records),
            "stratum_counts": manifest_module.count_by(
                common_records,
                "sign_reversal_stratum",
            ),
            "records": common_records,
        },
        "direction_reversal": {
            "eligible_count": len(common_records),
            "stratum_counts": manifest_module.count_by(
                common_records,
                "direction_reversal_stratum",
            ),
            "records": common_records,
        },
        "deletion": {
            "eligible_count": len(deletion_records),
            "stratum_counts": manifest_module.count_by(
                deletion_records,
                "deletion_stratum",
            ),
            "records": deletion_records,
        },
        "selection": {
            "budget_rule": "round_half_up(0.02 * training_graph_edges)",
            "budget_fraction": BUDGET_FRACTION,
            "budget": budget,
            "signal_disrupting_eligible_count": len(disrupting_records),
            "selected_edge_positions": [
                record["edge_index_position"]
                for record in disrupting_records[:budget]
            ],
        },
    }

    require(
        len(manifest["selection"]["selected_edge_positions"]) == budget,
        "Selected-edge count mismatch",
    )
    require(
        len(set(manifest["selection"]["selected_edge_positions"]))
        == budget,
        "Duplicate selected edge",
    )
    return manifest, budget


def metric_references(labels, label_dim):
    labels_cpu = labels.detach().cpu()
    counts = torch.bincount(
        labels_cpu,
        minlength=label_dim,
    ).to(torch.float64)
    total = float(counts.sum().item())
    prevalence = counts / total

    majority_accuracy = float(prevalence.max().item())
    uniform_accuracy = 1.0 / label_dim

    majority_class = int(torch.argmax(counts).item())
    majority_f1_values = []
    uniform_f1_values = []

    for class_index in range(label_dim):
        p_true = float(prevalence[class_index].item())

        if class_index == majority_class:
            majority_f1 = (
                2.0 * p_true / (1.0 + p_true)
                if p_true > 0.0
                else 0.0
            )
        else:
            majority_f1 = 0.0
        majority_f1_values.append(majority_f1)

        p_predicted = 1.0 / label_dim
        denominator = p_true + p_predicted
        uniform_f1_values.append(
            2.0 * p_true * p_predicted / denominator
            if denominator > 0.0
            else 0.0
        )

    return {
        "class_counts": [int(value) for value in counts.tolist()],
        "majority_accuracy": majority_accuracy,
        "uniform_accuracy": uniform_accuracy,
        "majority_macro_f1": (
            sum(majority_f1_values) / label_dim
        ),
        "uniform_expected_macro_f1": (
            sum(uniform_f1_values) / label_dim
        ),
        "accuracy_floor": max(
            majority_accuracy,
            uniform_accuracy,
        ),
        "macro_f1_floor": max(
            sum(majority_f1_values) / label_dim,
            sum(uniform_f1_values) / label_dim,
        ),
    }


def checkpoint_predictions(
    base,
    edit_module,
    shared,
    config,
    checkpoint_path,
):
    payload = torch.load(
        checkpoint_path,
        map_location=base.DEVICE,
        weights_only=True,
    )
    model = base.MSGNN_link_prediction(**config).to(base.DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(
            shared.features,
            shared.features.clone(),
            shared.train_graph,
            shared.test_queries,
            shared.train_weights,
        )

    base.validate_output(
        output,
        (shared.test_queries.shape[0], base.LABEL_DIM),
        "E012 clean checkpoint evaluation",
    )
    predictions = output.argmax(dim=1)
    labels = shared.test_labels

    confusion = torch.zeros(
        (base.LABEL_DIM, base.LABEL_DIM),
        dtype=torch.int64,
    )
    for true_label, predicted_label in zip(
        labels.detach().cpu(),
        predictions.detach().cpu(),
    ):
        confusion[int(true_label), int(predicted_label)] += 1

    prediction_counts = torch.bincount(
        predictions.detach().cpu(),
        minlength=base.LABEL_DIM,
    )
    row_sums = confusion.sum(dim=1)
    per_class_recall = [
        (
            float(confusion[index, index].item())
            / float(row_sums[index].item())
            if row_sums[index].item() > 0
            else 0.0
        )
        for index in range(base.LABEL_DIM)
    ]

    metrics = edit_module.task_metrics(
        base,
        output,
        labels,
    )
    return {
        "metrics": metrics,
        "confusion_rows_true_columns_predicted": confusion.tolist(),
        "prediction_counts": prediction_counts.tolist(),
        "per_class_recall": per_class_recall,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "E012 q=0.125 cross-graph paired matched-retraining"
        )
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        choices=GRAPH_SEEDS,
        required=True,
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        choices=MODEL_SEEDS,
        required=True,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate graph, manifest, and interfaces without training",
    )
    args = parser.parse_args()

    verify_sources()

    base = load_module("e012_base", BASE_SOURCE)
    manifest_module = load_module(
        "e012_manifest",
        MANIFEST_SOURCE,
    )
    selection_module = load_module(
        "e012_selection",
        SELECTION_SOURCE,
    )
    training_module = load_module(
        "e012_training",
        TRAINING_SOURCE,
    )
    edit_module = load_module("e012_edit", EDIT_SOURCE)

    base.GENERATION_SEED = args.generation_seed
    base.SPLIT_SEED = SPLIT_SEED

    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)

    shared = base.create_shared_data()
    frozen_before = training_module.frozen_hashes(base, shared)
    manifest, budget = build_manifest(
        base,
        manifest_module,
        selection_module,
        shared,
    )

    config = dict(base.MSGNN_CONFIG)
    config["num_features"] = int(shared.features.shape[1])
    config["q"] = FIXED_Q
    config["trainable_q"] = TRAINABLE_Q

    require(
        config["num_features"] == int(shared.features.shape[1]),
        "Feature-dimension configuration mismatch",
    )
    require(config["q"] == FIXED_Q, "Fixed-q configuration mismatch")
    require(
        config["trainable_q"] is False,
        "q must remain non-trainable",
    )

    clean_scenario = {
        "condition": "q0125",
        "name": (
            f"g{args.generation_seed}_q0125_clean_m0"
        ),
        "edit_type": "sign_reversal",
        "stratum": "signal_disrupting",
        "budget": 0,
        "budget_role": "clean_comparator",
    }
    perturbed_scenario = {
        "condition": "q0125",
        "name": (
            f"g{args.generation_seed}_q0125_"
            f"sign_reversal_signal_disrupting_m{budget}"
        ),
        "edit_type": "sign_reversal",
        "stratum": "signal_disrupting",
        "budget": budget,
        "budget_role": "primary_proportional_intervention",
    }

    output = Path(
        "results/e012/"
        f"e012_g{args.generation_seed}_"
        f"seed{args.model_seed}_"
        "q0125_sign_disrupting_matched_retraining_attempt01.json"
    )
    checkpoint_dir = Path("results/e012")

    training_module.MODEL_SEED = args.model_seed
    training_module.CHECKPOINT_DIR = checkpoint_dir
    training_module.selection_module = selection_module

    clean_checkpoint = (
        checkpoint_dir
        / (
            f"msgnn_k1_seed{args.model_seed}_"
            f"{clean_scenario['name']}_attempt01.pt"
        )
    )
    perturbed_checkpoint = (
        checkpoint_dir
        / (
            f"msgnn_k1_seed{args.model_seed}_"
            f"{perturbed_scenario['name']}_attempt01.pt"
        )
    )

    require(not output.exists(), f"Refusing to overwrite: {output}")
    require(
        not clean_checkpoint.exists(),
        f"Refusing to overwrite: {clean_checkpoint}",
    )
    require(
        not perturbed_checkpoint.exists(),
        f"Refusing to overwrite: {perturbed_checkpoint}",
    )

    print("E012_CONFIGURATION: PASS")
    print("GENERATION_SEED:", args.generation_seed)
    print("SPLIT_SEED:", SPLIT_SEED)
    print("MODEL_SEED:", args.model_seed)
    print("FIXED_Q:", FIXED_Q)
    print("TRAINABLE_Q:", TRAINABLE_Q)
    print("BUNDLE_FINGERPRINT:", shared.bundle_fingerprint)
    print(
        "TRAINING_GRAPH_EDGES:",
        manifest["eligibility"]["training_graph_edge_count"],
    )
    print(
        "SIGNAL_DISRUPTING_ELIGIBLE:",
        manifest["selection"]["signal_disrupting_eligible_count"],
    )
    print("PROPORTIONAL_BUDGET:", budget)
    print(
        "REALIZED_BUDGET_FRACTION:",
        budget
        / manifest["eligibility"]["training_graph_edge_count"],
    )

    if args.preflight:
        frozen_after = training_module.frozen_hashes(base, shared)
        require(
            frozen_after == frozen_before,
            "Frozen tensors changed during preflight",
        )
        print("E012_PREFLIGHT: PASS")
        print("TRAINING_PERFORMED: False")
        print("PERSISTENT_OUTPUTS: none")
        return 0

    clean_run = training_module.train_one(
        base,
        edit_module,
        shared,
        manifest,
        ZERO_METRICS,
        config,
        clean_scenario,
    )
    require(clean_run["passed"] is True, "Clean training failed")
    require(
        clean_run["learning_signal_satisfied"] is True,
        "Clean learning signal failed",
    )

    clean_evaluation = checkpoint_predictions(
        base,
        edit_module,
        shared,
        config,
        Path(clean_run["checkpoint"]),
    )
    clean_metrics = clean_evaluation["metrics"]

    for metric_name in ZERO_METRICS:
        require(
            abs(
                clean_metrics[metric_name]
                - clean_run["test_metrics"][metric_name]
            )
            <= 1e-7,
            f"Clean metric mismatch: {metric_name}",
        )

    references = metric_references(
        shared.test_labels,
        base.LABEL_DIM,
    )
    clean_gate = {
        "accuracy_above_strict_floor": (
            clean_metrics["accuracy"]
            > references["accuracy_floor"]
        ),
        "macro_f1_above_strict_floor": (
            clean_metrics["macro_f1"]
            > references["macro_f1_floor"]
        ),
        "all_prediction_classes_present": all(
            count > 0
            for count in clean_evaluation["prediction_counts"]
        ),
        "all_class_recalls_positive": all(
            value > 0.0
            for value in clean_evaluation["per_class_recall"]
        ),
    }
    require(all(clean_gate.values()), "Clean inclusion gate failed")

    perturbed_run = training_module.train_one(
        base,
        edit_module,
        shared,
        manifest,
        clean_metrics,
        config,
        perturbed_scenario,
    )
    require(
        perturbed_run["passed"] is True,
        "Perturbed training failed",
    )
    require(
        perturbed_run["learning_signal_satisfied"] is True,
        "Perturbed learning signal failed",
    )

    require(
        len(perturbed_run["selected_edges"]) == budget,
        "Perturbed selection-count mismatch",
    )
    require(
        [
            record["edge_index_position"]
            for record in perturbed_run["selected_edges"]
        ]
        == manifest["selection"]["selected_edge_positions"],
        "Perturbed selection mismatch",
    )

    frozen_after = training_module.frozen_hashes(base, shared)
    require(
        frozen_after == frozen_before,
        "Frozen tensors changed",
    )

    for source_string, expected_hash in SOURCE_HASHES.items():
        require(
            sha256_file(Path(source_string)) == expected_hash,
            f"Source changed during execution: {source_string}",
        )

    result = {
        "artifact": (
            "E012 q=0.125 cross-graph paired "
            "matched-retraining observation"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "generation_seed": args.generation_seed,
            "split_seed": SPLIT_SEED,
            "model_seed": args.model_seed,
            "fixed_q": FIXED_Q,
            "trainable_q": TRAINABLE_Q,
            "budget_fraction_target": BUDGET_FRACTION,
            "realized_budget": budget,
            "realized_budget_fraction": (
                budget
                / manifest["eligibility"]["training_graph_edge_count"]
            ),
            "primary_metric": "macro_f1",
            "estimand": (
                "Clean-relative paired retraining change for "
                "signal-disrupting sign reversal"
            ),
        },
        "sources": {
            path_string: {
                "sha256": expected_hash,
            }
            for path_string, expected_hash in SOURCE_HASHES.items()
        },
        "shared_data": {
            "bundle_fingerprint": shared.bundle_fingerprint,
            "num_nodes": shared.num_nodes,
            "training_graph_edges": (
                manifest["eligibility"]["training_graph_edge_count"]
            ),
            "train_queries": int(shared.train_queries.shape[0]),
            "validation_queries": int(shared.val_queries.shape[0]),
            "test_queries": int(shared.test_queries.shape[0]),
        },
        "manifest": manifest,
        "configuration": config,
        "references": references,
        "clean_inclusion_gate": clean_gate,
        "clean_evaluation": clean_evaluation,
        "clean_run": clean_run,
        "perturbed_run": perturbed_run,
        "paired_clean_relative_changes": (
            perturbed_run["clean_relative_changes"]
        ),
        "integrity": {
            "all_sources_hash_locked": True,
            "clean_checkpoint_hash_verified": (
                sha256_file(Path(clean_run["checkpoint"]))
                == clean_run["checkpoint_sha256"]
            ),
            "perturbed_checkpoint_hash_verified": (
                sha256_file(Path(perturbed_run["checkpoint"]))
                == perturbed_run["checkpoint_sha256"]
            ),
            "both_checkpoints_restored": (
                clean_run["checkpoint_restored"]
                and perturbed_run["checkpoint_restored"]
            ),
            "all_frozen_tensors_unchanged": (
                frozen_after == frozen_before
            ),
            "all_selected_edges_match_manifest": True,
            "clean_inclusion_gate_passed": all(clean_gate.values()),
            "all_values_finite": (
                clean_run["finite_value_checks"]["all_finite"]
                and perturbed_run["finite_value_checks"]["all_finite"]
            ),
        },
        "limitations": [
            (
                "This artifact is one graph-seed/model-seed pair; "
                "cross-graph inference requires the E012 aggregate."
            ),
            (
                "The split seed is fixed at zero to isolate graph "
                "generation variability."
            ),
            (
                "Features are computed from the clean graph and frozen "
                "in both training arms."
            ),
            (
                "The deterministic perturbation selection does not "
                "estimate variability across perturbation samples."
            ),
            (
                "This result does not establish real-world or "
                "architecture-level generalization."
            ),
        ],
    }

    boolean_integrity = [
        value
        for value in result["integrity"].values()
        if isinstance(value, bool)
    ]
    require(all(boolean_integrity), "E012 integrity failure")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

    print("E012_MULTIGRAPH_MATCHED_RETRAINING: PASS")
    print("OUTPUT:", output)
    print("CLEAN_CHECKPOINT:", clean_run["checkpoint"])
    print("PERTURBED_CHECKPOINT:", perturbed_run["checkpoint"])
    print("CLEAN_METRICS:", clean_metrics)
    print(
        "PAIRED_CLEAN_RELATIVE_CHANGES:",
        perturbed_run["clean_relative_changes"],
    )
    print("CLEAN_INCLUSION_GATE:", clean_gate)
    print("FROZEN_TENSORS_UNCHANGED:", frozen_after == frozen_before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
