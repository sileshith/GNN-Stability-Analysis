"""Run the E009 scaled fixed-checkpoint perturbation pilot."""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


TRAINING_SOURCE = Path("scripts/e005_charge_comparison_replication.py")
TRAINING_SOURCE_SHA256 = (
    "5c252bbb56c543e56a5ac43c0f7163df96393ee69604a7e867e03e0e93721f82"
)
OPERATOR_SOURCE = Path(
    "scripts/e002_q025_fixed_checkpoint_sign_reversal.py"
)
OPERATOR_SOURCE_SHA256 = (
    "4b7f3946447220fb14e0a414c13db39fa671caebf740c96f474b333179c76013"
)
MANIFEST = Path(
    "results/e008/"
    "e008_expanded_perturbation_manifest_attempt01.json"
)
MANIFEST_SHA256 = (
    "6d1dd9ebb5e5702df31549fb88fa2a2be6080e8102bf7530e4e01a485beb7a6b"
)
PARENT_SOURCE = Path(
    "scripts/e006_fixed_checkpoint_stratified_pilot.py"
)
PARENT_SOURCE_SHA256 = (
    "300fb65268fe61de546bbee6d656638fe4ccc455d4ad0b7879d8b1afc946198f"
)
OUTPUT = Path(
    "results/e009/"
    "e009_scaled_fixed_checkpoint_pilot_attempt01.json"
)

EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc"
)
EXPECTED_PYGSD_VERSION = "1.2.0"
ATOL = 1e-12
SELECTION_SEED = 20260920

CONDITIONS = {
    "q0125": {
        "q": 0.125,
        "clean_result": Path(
            "results/e005/"
            "e005_k1_q0125_seed0_clean_attempt01.json"
        ),
        "expected_gate": "E005",
    },
    "q025": {
        "q": 0.25,
        "clean_result": Path(
            "results/e002/"
            "e002_k1_q025_seed0_clean_attempt01.json"
        ),
        "expected_gate": "E002",
    },
}

BUDGETS = {
    "sign_reversal": {
        "signal_disrupting": [15, 29, 58, 146],
        "signal_restoring": [15, 29, 58, 146],
    },
    "direction_reversal": {
        "signal_disrupting": [15, 29, 58, 146],
        "signal_restoring": [15, 29, 58, 146],
        "direction_neutral": [15, 29, 58, 146],
    },
    "deletion": {
        "signal_removing": [15, 29, 58, 146],
        "noise_removing": [15, 29, 58, 146],
    },
}

STRATUM_FIELD = {
    "sign_reversal": "sign_reversal_stratum",
    "direction_reversal": "direction_reversal_stratum",
    "deletion": "deletion_stratum",
}


def selection_key(
    edit_type: str,
    stratum: str,
    record: dict,
) -> str:
    payload = (
        f"E009|{SELECTION_SEED}|{edit_type}|{stratum}|"
        f"{record['edge_index_position']}|"
        f"{record['source']}|{record['target']}|"
        f"{record['weight']}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    require(path.is_file(), f"Missing source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"Cannot import source: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_hashes(module, shared) -> dict[str, str]:
    tensors = {
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
    return {
        name: module.compute_tensor_hash(name, tensor)
        for name, tensor in tensors.items()
    }


def task_metrics(module, output, labels) -> dict[str, float]:
    accuracy, macro_f1, micro_f1 = module.compute_metrics(
        output,
        labels,
    )
    predictions = output.argmax(dim=1)

    true_sign = torch.where(
        torch.isin(labels, torch.tensor([0, 2])),
        0,
        1,
    )
    predicted_sign = torch.where(
        torch.isin(predictions, torch.tensor([0, 2])),
        0,
        1,
    )
    true_direction = torch.where(labels < 2, 0, 1)
    predicted_direction = torch.where(predictions < 2, 0, 1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "sign_accuracy": float(
            (true_sign == predicted_sign).float().mean().item()
        ),
        "direction_accuracy": float(
            (true_direction == predicted_direction).float().mean().item()
        ),
    }


def forward_capture(model, shared, graph, weights):
    linear_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda layer, inputs, output: linear_outputs.append(
                output.detach().clone()
            )
        )
        for layer in model.modules()
        if isinstance(layer, torch.nn.Linear)
    ]

    with torch.no_grad():
        output = model(
            shared.features,
            shared.features.clone(),
            graph,
            shared.test_queries,
            weights,
        )
        representation = model.z.detach().clone()

    for hook in hooks:
        hook.remove()

    require(linear_outputs, "No linear output was captured")
    logits = linear_outputs[-1]

    for name, tensor in {
        "output": output,
        "representation": representation,
        "logits": logits,
    }.items():
        require(
            torch.isfinite(tensor).all().item(),
            f"Non-finite {name}",
        )

    return {
        "output": output.detach().clone(),
        "representation": representation,
        "logits": logits,
    }


def apply_edit(
    edit_type: str,
    records: list[dict],
    graph: torch.Tensor,
    weights: torch.Tensor,
):
    indices = [record["edge_index_position"] for record in records]
    require(len(indices) == len(set(indices)), "Duplicate edit index")

    edited_graph = graph.clone()
    edited_weights = weights.clone()

    if edit_type == "sign_reversal":
        edited_weights[indices] *= -1

    elif edit_type == "direction_reversal":
        original_sources = graph[0, indices].clone()
        original_targets = graph[1, indices].clone()
        edited_graph[0, indices] = original_targets
        edited_graph[1, indices] = original_sources

        edited_pairs = [
            (
                int(edited_graph[0, index].item()),
                int(edited_graph[1, index].item()),
            )
            for index in range(edited_graph.shape[1])
        ]
        require(
            len(edited_pairs) == len(set(edited_pairs)),
            "Direction reversal created an ordered-pair collision",
        )

    elif edit_type == "deletion":
        keep = torch.ones(
            graph.shape[1],
            dtype=torch.bool,
            device=graph.device,
        )
        keep[indices] = False
        edited_graph = graph[:, keep].clone()
        edited_weights = weights[keep].clone()

    else:
        raise RuntimeError(f"Unsupported edit type: {edit_type}")

    return edited_graph, edited_weights


def delta_fro(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(left - right, ord="fro"))


def relative_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.matrix_norm(left - right, ord="fro")
    denominator = torch.linalg.matrix_norm(right, ord="fro")
    require(denominator.item() > 0, "Zero clean norm")
    return float(numerator / denominator)


def scenario_result(
    module,
    operator_module,
    model,
    shared,
    clean,
    clean_h,
    clean_l,
    q,
    edit_type,
    stratum,
    budget,
    pool_size,
    selected,
):
    graph, weights = apply_edit(
        edit_type,
        selected,
        shared.train_graph,
        shared.train_weights,
    )

    perturbed = forward_capture(
        model,
        shared,
        graph,
        weights,
    )

    perturbed_h = operator_module.magnetic_adjacency(
        graph,
        weights,
        q,
        shared.num_nodes,
    )
    perturbed_l = operator_module.normalized_laplacian(
        graph,
        weights,
        q,
        shared.num_nodes,
    )

    clean_predictions = clean["output"].argmax(dim=1)
    perturbed_predictions = perturbed["output"].argmax(dim=1)
    flip_count = int(
        (clean_predictions != perturbed_predictions).sum().item()
    )

    clean_metrics = task_metrics(
        module,
        clean["output"],
        shared.test_labels,
    )
    perturbed_metrics = task_metrics(
        module,
        perturbed["output"],
        shared.test_labels,
    )

    metric_changes = {
        key: perturbed_metrics[key] - clean_metrics[key]
        for key in clean_metrics
    }

    return {
        "edit_type": edit_type,
        "stratum": stratum,
        "budget": budget,
        "selection_rule": (
            "First m records after deterministic SHA-256 ordering "
            "within the frozen edit-type/stratum pool"
        ),
        "selection_seed": SELECTION_SEED,
        "available_stratum_count": pool_size,
        "fraction_of_training_graph": (
            budget / shared.train_graph.shape[1]
        ),
        "fraction_of_available_stratum": budget / pool_size,
        "selection_prefix_sha256": hashlib.sha256(
            json.dumps(
                selected,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "selected_edges": [
            {
                "edge_index_position": record["edge_index_position"],
                "source": record["source"],
                "target": record["target"],
                "weight": record["weight"],
                "source_community": record["source_community"],
                "target_community": record["target_community"],
            }
            for record in selected
        ],
        "operator_diagnostics": {
            "delta_h_fro": delta_fro(perturbed_h, clean_h),
            "delta_l_fro": delta_fro(perturbed_l, clean_l),
        },
        "representation_diagnostics": {
            "delta_fro": delta_fro(
                perturbed["representation"],
                clean["representation"],
            ),
            "relative_delta_fro": relative_delta(
                perturbed["representation"],
                clean["representation"],
            ),
        },
        "logit_diagnostics": {
            "delta_fro": delta_fro(
                perturbed["logits"],
                clean["logits"],
            ),
            "max_abs_delta": float(
                torch.max(
                    torch.abs(
                        perturbed["logits"] - clean["logits"]
                    )
                ).item()
            ),
        },
        "output_diagnostics": {
            "delta_fro": delta_fro(
                perturbed["output"],
                clean["output"],
            ),
            "max_abs_delta": float(
                torch.max(
                    torch.abs(
                        perturbed["output"] - clean["output"]
                    )
                ).item()
            ),
            "prediction_flip_count": flip_count,
            "prediction_flip_rate": (
                flip_count / shared.test_labels.numel()
            ),
        },
        "task_metrics": {
            "clean": clean_metrics,
            "perturbed": perturbed_metrics,
            "changes": metric_changes,
        },
        "finite": True,
    }


def run_condition(
    module,
    operator_module,
    shared,
    manifest,
    condition_name,
    condition,
):
    clean_path = condition["clean_result"]
    require(clean_path.is_file(), f"Missing clean result: {clean_path}")

    clean_record = json.loads(clean_path.read_text())
    require(
        clean_record["gate"] == condition["expected_gate"],
        "Clean gate mismatch",
    )
    require(
        clean_record["summary"]["overall_pass"] is True,
        "Clean run did not pass",
    )

    run = clean_record["results"][0]
    require(run["seed"] == 0, "Pilot requires model seed 0")
    require(
        run["bundle_fingerprint"] == EXPECTED_BUNDLE,
        "Clean bundle mismatch",
    )

    config = dict(run["config"])
    require(
        abs(float(config["q"]) - condition["q"]) <= ATOL,
        "Clean charge mismatch",
    )
    require(config["K"] == 1, "Unexpected filter order")
    require(config["trainable_q"] is False, "q is trainable")
    require(config["cached"] is False, "Cached convolution found")

    checkpoint_path = Path(run["checkpoint_path"])
    checkpoint_hash_before = sha256_file(checkpoint_path)
    require(
        checkpoint_hash_before == run["checkpoint_sha256"],
        "Checkpoint hash mismatch",
    )

    payload = torch.load(
        checkpoint_path,
        map_location=module.DEVICE,
        weights_only=True,
    )
    model = module.MSGNN_link_prediction(**config).to(module.DEVICE)
    model.load_state_dict(payload["model_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "A model parameter remains trainable",
    )

    model_hash_before = operator_module.state_hash(model)
    clean = forward_capture(
        model,
        shared,
        shared.train_graph,
        shared.train_weights,
    )

    clean_metrics = task_metrics(
        module,
        clean["output"],
        shared.test_labels,
    )
    require(
        abs(clean_metrics["accuracy"] - run["test_accuracy"]) <= 1e-12,
        "Clean accuracy was not reproduced",
    )
    require(
        abs(clean_metrics["macro_f1"] - run["test_macro_f1"]) <= 1e-12,
        "Clean macro-F1 was not reproduced",
    )
    require(
        abs(clean_metrics["micro_f1"] - run["test_micro_f1"]) <= 1e-12,
        "Clean micro-F1 was not reproduced",
    )

    q = float(config["q"])
    clean_h = operator_module.magnetic_adjacency(
        shared.train_graph,
        shared.train_weights,
        q,
        shared.num_nodes,
    )
    clean_l = operator_module.normalized_laplacian(
        shared.train_graph,
        shared.train_weights,
        q,
        shared.num_nodes,
    )

    scenarios = []

    for edit_type, stratum_budgets in BUDGETS.items():
        records = manifest[edit_type]["records"]
        field = STRATUM_FIELD[edit_type]

        for stratum, budgets in stratum_budgets.items():
            stratum_records = [
                record
                for record in records
                if record[field] == stratum
            ]

            require(
                stratum_records,
                f"Empty stratum: {edit_type}/{stratum}",
            )

            ordered_records = sorted(
                stratum_records,
                key=lambda record: selection_key(
                    edit_type,
                    stratum,
                    record,
                ),
            )

            previous_positions = set()

            for budget in budgets:
                require(
                    budget <= len(stratum_records),
                    (
                        f"Budget {budget} exceeds "
                        f"{edit_type}/{stratum} population"
                    ),
                )
                selected = ordered_records[:budget]
                selected_positions = {
                    record["edge_index_position"]
                    for record in selected
                }
                require(
                    len(selected_positions) == budget,
                    "Duplicate selected edge position",
                )
                require(
                    previous_positions.issubset(selected_positions),
                    "Perturbation budgets are not nested",
                )
                previous_positions = selected_positions

                scenarios.append(
                    scenario_result(
                        module,
                        operator_module,
                        model,
                        shared,
                        clean,
                        clean_h,
                        clean_l,
                        q,
                        edit_type,
                        stratum,
                        budget,
                        len(stratum_records),
                        selected,
                    )
                )

    model_hash_after = operator_module.state_hash(model)
    checkpoint_hash_after = sha256_file(checkpoint_path)

    require(
        model_hash_after == model_hash_before,
        "Model state changed during pilot",
    )
    require(
        checkpoint_hash_after == checkpoint_hash_before,
        "Checkpoint changed during pilot",
    )

    return {
        "condition": condition_name,
        "q": q,
        "model_seed": 0,
        "clean_result": str(clean_path),
        "clean_result_sha256": sha256_file(clean_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "config": config,
        "clean_metrics_reproduced": True,
        "clean_metrics": clean_metrics,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "integrity": {
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
            "parameters_require_grad": any(
                parameter.requires_grad
                for parameter in model.parameters()
            ),
        },
    }


def main() -> int:
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite pilot output: {OUTPUT}",
    )
    require(
        metadata.version("torch-geometric-signed-directed")
        == EXPECTED_PYGSD_VERSION,
        "Unexpected PyGSD version",
    )
    require(
        sha256_file(TRAINING_SOURCE) == TRAINING_SOURCE_SHA256,
        "Training-source hash mismatch",
    )
    require(
        sha256_file(OPERATOR_SOURCE) == OPERATOR_SOURCE_SHA256,
        "Operator-source hash mismatch",
    )
    require(
        sha256_file(MANIFEST) == MANIFEST_SHA256,
        "Manifest hash mismatch",
    )
    require(
        sha256_file(PARENT_SOURCE) == PARENT_SOURCE_SHA256,
        "Parent-source hash mismatch",
    )

    module = load_module("e009_training_source", TRAINING_SOURCE)
    operator_module = load_module(
        "e009_operator_source",
        OPERATOR_SOURCE,
    )
    manifest = json.loads(MANIFEST.read_text())

    require(manifest["status"] == "PASS", "Manifest did not pass")
    require(
        manifest["source"]["bundle_fingerprint"] == EXPECTED_BUNDLE,
        "Manifest bundle mismatch",
    )

    shared = module.create_shared_data()
    require(
        shared.bundle_fingerprint == EXPECTED_BUNDLE,
        "Regenerated bundle mismatch",
    )

    frozen_before = frozen_hashes(module, shared)

    conditions = {
        name: run_condition(
            module,
            operator_module,
            shared,
            manifest,
            name,
            condition,
        )
        for name, condition in CONDITIONS.items()
    }

    frozen_after = frozen_hashes(module, shared)
    require(
        frozen_after == frozen_before,
        "A frozen shared tensor changed",
    )

    expected_scenarios_per_condition = sum(
        len(budgets)
        for strata in BUDGETS.values()
        for budgets in strata.values()
    )
    require(
        expected_scenarios_per_condition == 28,
        "Unexpected scenario-plan size",
    )
    require(
        all(
            condition["scenario_count"] == 28
            for condition in conditions.values()
        ),
        "Condition scenario count mismatch",
    )

    result = {
        "artifact": "E009 scaled fixed-checkpoint screening pilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "conditions": ["q0125", "q025"],
            "model_seed": 0,
            "checkpoint_count": 2,
            "scenarios_per_checkpoint": 28,
            "total_scenarios": 56,
            "training_performed": False,
            "estimand": (
                "Frozen-checkpoint sensitivity to deterministic "
                "hash-ordered nested perturbations at scaled budgets"
            ),
            "not_estimated": [
                "retraining stability",
                "between-model-seed variability",
                "between-graph variability",
                "between-split variability",
                "robustness",
                "generalization",
                "architecture superiority",
            ],
        },
        "sources": {
            "training_script": str(TRAINING_SOURCE),
            "training_script_sha256": TRAINING_SOURCE_SHA256,
            "operator_script": str(OPERATOR_SOURCE),
            "operator_script_sha256": OPERATOR_SOURCE_SHA256,
            "manifest": str(MANIFEST),
            "manifest_sha256": MANIFEST_SHA256,
            "parent_script": str(PARENT_SOURCE),
            "parent_script_sha256": PARENT_SOURCE_SHA256,
            "bundle_fingerprint": EXPECTED_BUNDLE,
        },
        "budget_plan": BUDGETS,
        "selection_plan": {
            "selection_seed": SELECTION_SEED,
            "ordering": (
                "SHA-256 over edit type, stratum, edge position, "
                "endpoints, and weight"
            ),
            "nested_prefixes": True,
            "primary_budgets": [15, 29, 58],
            "stress_test_budget": 146,
        },
        "conditions": conditions,
        "integrity": {
            "frozen_tensor_hashes_before": frozen_before,
            "frozen_tensor_hashes_after": frozen_after,
            "frozen_tensors_unchanged": (
                frozen_after == frozen_before
            ),
            "all_models_unchanged": all(
                condition["integrity"]["model_state_unchanged"]
                for condition in conditions.values()
            ),
            "all_checkpoints_unchanged": all(
                condition["integrity"]["checkpoint_unchanged"]
                for condition in conditions.values()
            ),
            "all_parameters_frozen": all(
                not condition["integrity"]["parameters_require_grad"]
                for condition in conditions.values()
            ),
            "all_clean_metrics_reproduced": all(
                condition["clean_metrics_reproduced"]
                for condition in conditions.values()
            ),
            "all_scenarios_finite": all(
                scenario["finite"]
                for condition in conditions.values()
                for scenario in condition["scenarios"]
            ),
        },
        "limitations": [
            (
                "This is a deterministic seed-0 fixed-checkpoint "
                "screening study, not a random perturbation sample."
            ),
            (
                "Budgets 15, 29, and 58 are primary graded comparisons; "
                "budget 146 is a high-intensity stress test."
            ),
            (
                "Equal absolute budgets imply different fractions of "
                "each available stratum; both graph and stratum "
                "fractions are reported."
            ),
            (
                "All scenarios use one generated graph and one split."
            ),
            (
                "No retraining, robustness, or stability conclusion "
                "is supported."
            ),
        ],
    }

    require(
        all(result["integrity"].values()),
        "An integrity gate failed",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

    print("E009_SCALED_FIXED_CHECKPOINT_PILOT: PASS")
    print("OUTPUT:", OUTPUT)
    print("TOTAL_SCENARIOS:", result["scope"]["total_scenarios"])

    for condition_name, condition in conditions.items():
        print("CONDITION:", condition_name)
        print("Q:", condition["q"])
        print("CLEAN_METRICS:", condition["clean_metrics"])

        for edit_type in BUDGETS:
            selected = [
                scenario
                for scenario in condition["scenarios"]
                if scenario["edit_type"] == edit_type
            ]
            print(
                "EDIT_SUMMARY:",
                edit_type,
                {
                    "scenario_count": len(selected),
                    "max_flip_count": max(
                        item["output_diagnostics"][
                            "prediction_flip_count"
                        ]
                        for item in selected
                    ),
                    "max_abs_macro_f1_change": max(
                        abs(
                            item["task_metrics"]["changes"][
                                "macro_f1"
                            ]
                        )
                        for item in selected
                    ),
                    "max_delta_h_fro": max(
                        item["operator_diagnostics"]["delta_h_fro"]
                        for item in selected
                    ),
                },
            )

    print(
        "ALL_CLEAN_METRICS_REPRODUCED:",
        result["integrity"]["all_clean_metrics_reproduced"],
    )
    print(
        "ALL_MODELS_UNCHANGED:",
        result["integrity"]["all_models_unchanged"],
    )
    print(
        "ALL_CHECKPOINTS_UNCHANGED:",
        result["integrity"]["all_checkpoints_unchanged"],
    )
    print(
        "FROZEN_TENSORS_UNCHANGED:",
        result["integrity"]["frozen_tensors_unchanged"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
