"""E007 q=1/4 seed-0 matched clean-versus-perturbed retraining pilot."""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


TRAINING_SOURCE = Path("scripts/e005_charge_comparison_replication.py")
TRAINING_SOURCE_SHA256 = (
    "5c252bbb56c543e56a5ac43c0f7163df96393ee69604a7e867e03e0e93721f82"
)
EDIT_SOURCE = Path("scripts/e006_fixed_checkpoint_stratified_pilot.py")
EDIT_SOURCE_SHA256 = (
    "300fb65268fe61de546bbee6d656638fe4ccc455d4ad0b7879d8b1afc946198f"
)
MANIFEST = Path(
    "results/e006/e006_stratified_perturbation_manifest_attempt01.json"
)
MANIFEST_SHA256 = (
    "ef1381415cb4f49fbfac3dbf58ab8023e39eb9035d2ae9593cba93a2ef9027d2"
)
CLEAN_RESULT = Path(
    "results/e002/e002_k1_q025_seed0_clean_attempt01.json"
)
OUTPUT = Path(
    "results/e007/"
    "e007_q025_seed0_matched_retraining_pilot_attempt01.json"
)
CHECKPOINT_DIR = Path("results/e007")

EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc"
)
EXPECTED_PYGSD_VERSION = "1.2.0"
MODEL_SEED = 0
ATOL = 1e-12

SCENARIOS = [
    {
        "name": "sign_reversal_signal_disrupting_m8",
        "edit_type": "sign_reversal",
        "stratum": "signal_disrupting",
        "budget": 8,
    },
    {
        "name": "sign_reversal_signal_restoring_m1",
        "edit_type": "sign_reversal",
        "stratum": "signal_restoring",
        "budget": 1,
        "case_study_only": True,
    },
    {
        "name": "direction_reversal_signal_disrupting_m4",
        "edit_type": "direction_reversal",
        "stratum": "signal_disrupting",
        "budget": 4,
    },
    {
        "name": "direction_reversal_signal_restoring_m4",
        "edit_type": "direction_reversal",
        "stratum": "signal_restoring",
        "budget": 4,
    },
    {
        "name": "direction_reversal_direction_neutral_m4",
        "edit_type": "direction_reversal",
        "stratum": "direction_neutral",
        "budget": 4,
    },
    {
        "name": "deletion_signal_removing_m8",
        "edit_type": "deletion",
        "stratum": "signal_removing",
        "budget": 8,
    },
    {
        "name": "deletion_noise_removing_m4",
        "edit_type": "deletion",
        "stratum": "noise_removing",
        "budget": 4,
    },
]

STRATUM_FIELD = {
    "sign_reversal": "sign_reversal_stratum",
    "direction_reversal": "direction_reversal_stratum",
    "deletion": "deletion_stratum",
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
        f"Cannot import: {path}",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(module, name, tensor):
    return module.compute_tensor_hash(name, tensor)


def frozen_hashes(module, shared):
    return {
        name: tensor_hash(module, name, getattr(shared, name))
        for name in [
            "features",
            "train_graph",
            "train_weights",
            "train_queries",
            "train_labels",
            "val_queries",
            "val_labels",
            "test_queries",
            "test_labels",
        ]
    }


def perturbed_fingerprint(module, base_bundle, scenario, graph, weights):
    digest = hashlib.sha256()
    digest.update(base_bundle.encode())
    digest.update(scenario["name"].encode())
    digest.update(tensor_hash(module, "train_graph", graph).encode())
    digest.update(tensor_hash(module, "train_weights", weights).encode())
    return digest.hexdigest()


def select_records(manifest, scenario):
    edit_type = scenario["edit_type"]
    field = STRATUM_FIELD[edit_type]
    records = [
        record
        for record in manifest[edit_type]["records"]
        if record[field] == scenario["stratum"]
    ]
    require(
        scenario["budget"] <= len(records),
        f"Insufficient records for {scenario['name']}",
    )
    return records[:scenario["budget"]]


def checkpoint_path_for(scenario):
    return CHECKPOINT_DIR / (
        f"msgnn_k1_q025_seed0_{scenario['name']}_attempt01.pt"
    )


def validate_no_collisions():
    require(not OUTPUT.exists(), f"Refusing to overwrite: {OUTPUT}")
    for scenario in SCENARIOS:
        checkpoint = checkpoint_path_for(scenario)
        require(
            not checkpoint.exists(),
            f"Refusing to overwrite: {checkpoint}",
        )


def train_one(
    module,
    edit_module,
    shared,
    manifest,
    clean_metrics,
    config,
    scenario,
):
    selected = select_records(manifest, scenario)
    graph, weights = edit_module.apply_edit(
        scenario["edit_type"],
        selected,
        shared.train_graph,
        shared.train_weights,
    )

    graph_hash = tensor_hash(module, "train_graph", graph)
    weights_hash = tensor_hash(module, "train_weights", weights)
    scenario_bundle = perturbed_fingerprint(
        module,
        shared.bundle_fingerprint,
        scenario,
        graph,
        weights,
    )

    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    random.seed(MODEL_SEED)

    model = module.MSGNN_link_prediction(**config).to(module.DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=module.LR,
        weight_decay=module.WEIGHT_DECAY,
    )
    criterion = nn.NLLLoss()

    def forward(queries):
        return model(
            shared.features,
            shared.features.clone(),
            graph,
            queries,
            weights,
        )

    model.eval()
    with torch.no_grad():
        initial_val_output = forward(shared.val_queries)
    module.validate_output(
        initial_val_output,
        (shared.val_queries.shape[0], module.LABEL_DIM),
        "Initial validation",
    )
    initial_val_loss = criterion(
        initial_val_output,
        shared.val_labels,
    )
    module.validate_loss(initial_val_loss, "Initial validation loss")

    model.train()
    initial_train_output = forward(shared.train_queries)
    module.validate_output(
        initial_train_output,
        (shared.train_queries.shape[0], module.LABEL_DIM),
        "Initial training",
    )
    initial_train_loss = criterion(
        initial_train_output,
        shared.train_labels,
    )
    module.validate_loss(initial_train_loss, "Initial training loss")

    best_val_loss = float(initial_val_loss.item())
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    train_losses = []
    val_losses = []
    epochs_without_improvement = 0
    stopping_epoch = module.MAX_EPOCHS
    stopping_reason = "max epochs"
    gradients_finite = True
    parameters_finite = True

    for epoch in range(1, module.MAX_EPOCHS + 1):
        model.train()
        output = forward(shared.train_queries)
        module.validate_output(
            output,
            (shared.train_queries.shape[0], module.LABEL_DIM),
            f"Epoch {epoch}",
        )
        loss = criterion(output, shared.train_labels)
        module.validate_loss(loss, f"Epoch {epoch} loss")

        optimizer.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                require(
                    torch.isfinite(parameter.grad).all().item(),
                    f"Non-finite gradient at epoch {epoch}",
                )

        optimizer.step()

        for parameter in model.parameters():
            require(
                torch.isfinite(parameter).all().item(),
                f"Non-finite parameter at epoch {epoch}",
            )

        model.eval()
        with torch.no_grad():
            val_output = forward(shared.val_queries)
        module.validate_output(
            val_output,
            (shared.val_queries.shape[0], module.LABEL_DIM),
            f"Epoch {epoch} validation",
        )
        val_loss = criterion(val_output, shared.val_labels)
        module.validate_loss(
            val_loss,
            f"Epoch {epoch} validation loss",
        )

        train_losses.append(float(loss.item()))
        val_losses.append(float(val_loss.item()))

        if val_loss.item() < best_val_loss - module.MIN_IMPROVEMENT:
            best_val_loss = float(val_loss.item())
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= module.MIN_EPOCHS
            and epochs_without_improvement >= module.PATIENCE
        ):
            stopping_epoch = epoch
            stopping_reason = "early stopping"
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        reference_val_output = forward(
            shared.val_queries
        ).detach().clone()

    checkpoint_path = checkpoint_path_for(scenario)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config,
            "seed": MODEL_SEED,
            "base_bundle_fingerprint": shared.bundle_fingerprint,
            "perturbed_bundle_fingerprint": scenario_bundle,
            "scenario": scenario,
            "selected_edges": selected,
            "train_graph_sha256": graph_hash,
            "train_weights_sha256": weights_hash,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)

    payload = torch.load(
        checkpoint_path,
        map_location=module.DEVICE,
        weights_only=True,
    )
    fresh_model = module.MSGNN_link_prediction(**config).to(module.DEVICE)
    fresh_model.load_state_dict(payload["model_state_dict"])
    model = fresh_model
    model.eval()

    with torch.no_grad():
        restored_val_output = forward(shared.val_queries)
    reload_max_abs_diff = float(
        torch.max(
            torch.abs(restored_val_output - reference_val_output)
        ).item()
    )
    require(
        reload_max_abs_diff <= 1e-6,
        "Checkpoint reload output mismatch",
    )

    restored_val_loss = criterion(
        restored_val_output,
        shared.val_labels,
    )
    require(
        abs(restored_val_loss.item() - best_val_loss) < 1e-6,
        "Restored validation loss mismatch",
    )

    with torch.no_grad():
        test_output = forward(shared.test_queries)
    module.validate_output(
        test_output,
        (shared.test_queries.shape[0], module.LABEL_DIM),
        "Test",
    )
    test_loss = criterion(test_output, shared.test_labels)
    module.validate_loss(test_loss, "Test loss")

    metrics = edit_module.task_metrics(
        module,
        test_output,
        shared.test_labels,
    )
    changes = {
        key: metrics[key] - clean_metrics[key]
        for key in clean_metrics
    }

    return {
        "scenario": scenario,
        "selected_edges": selected,
        "train_graph_edge_count": int(graph.shape[1]),
        "train_graph_sha256": graph_hash,
        "train_weights_sha256": weights_hash,
        "base_bundle_fingerprint": shared.bundle_fingerprint,
        "perturbed_bundle_fingerprint": scenario_bundle,
        "initial_train_loss": float(initial_train_loss.item()),
        "initial_val_loss": float(initial_val_loss.item()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stopping_epoch": stopping_epoch,
        "stopping_reason": stopping_reason,
        "total_epochs": len(train_losses),
        "per_epoch_train_loss": train_losses,
        "per_epoch_val_loss": val_losses,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_restored": True,
        "reload_max_abs_diff": reload_max_abs_diff,
        "test_loss": float(test_loss.item()),
        "test_metrics": metrics,
        "clean_metrics": clean_metrics,
        "clean_relative_changes": changes,
        "learning_signal_satisfied": (
            best_val_loss <= initial_val_loss.item() - 0.001
        ),
        "finite_value_checks": {
            "gradients_finite": gradients_finite,
            "parameters_finite": parameters_finite,
            "losses_finite": True,
            "outputs_finite": bool(torch.isfinite(test_output).all()),
            "all_finite": True,
        },
        "passed": True,
    }


def main():
    validate_no_collisions()

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
        sha256_file(EDIT_SOURCE) == EDIT_SOURCE_SHA256,
        "Edit-source hash mismatch",
    )
    require(
        sha256_file(MANIFEST) == MANIFEST_SHA256,
        "Manifest hash mismatch",
    )

    module = load_module("e007_training", TRAINING_SOURCE)
    edit_module = load_module("e007_edits", EDIT_SOURCE)
    manifest = json.loads(MANIFEST.read_text())
    clean_record = json.loads(CLEAN_RESULT.read_text())

    require(manifest["status"] == "PASS", "Manifest did not pass")
    require(clean_record["gate"] == "E002", "Clean gate mismatch")
    require(
        clean_record["summary"]["overall_pass"] is True,
        "Clean result did not pass",
    )

    clean_run = clean_record["results"][0]
    require(clean_run["seed"] == MODEL_SEED, "Clean seed mismatch")
    require(
        clean_run["bundle_fingerprint"] == EXPECTED_BUNDLE,
        "Clean bundle mismatch",
    )

    config = dict(clean_run["config"])
    require(abs(config["q"] - 0.25) <= ATOL, "Expected q=0.25")
    require(config["K"] == 1, "Expected filter order K=1")
    require(config["trainable_q"] is False, "q must be fixed")
    require(config["cached"] is False, "Caching must be disabled")

    shared = module.create_shared_data()
    require(
        shared.bundle_fingerprint == EXPECTED_BUNDLE,
        "Regenerated bundle mismatch",
    )

    frozen_before = frozen_hashes(module, shared)

    clean_checkpoint = Path(clean_run["checkpoint_path"])
    clean_checkpoint_hash_before = sha256_file(clean_checkpoint)
    require(
        clean_checkpoint_hash_before
        == clean_run["checkpoint_sha256"],
        "Clean checkpoint hash mismatch",
    )

    clean_payload = torch.load(
        clean_checkpoint,
        map_location=module.DEVICE,
        weights_only=True,
    )
    clean_model = module.MSGNN_link_prediction(**config).to(module.DEVICE)
    clean_model.load_state_dict(clean_payload["model_state_dict"])
    clean_model.eval()
    with torch.no_grad():
        clean_output = clean_model(
            shared.features,
            shared.features.clone(),
            shared.train_graph,
            shared.test_queries,
            shared.train_weights,
        )

    clean_metrics = edit_module.task_metrics(
        module,
        clean_output,
        shared.test_labels,
    )
    require(
        abs(clean_metrics["accuracy"] - clean_run["test_accuracy"])
        <= 1e-12,
        "Clean accuracy mismatch",
    )
    require(
        abs(clean_metrics["macro_f1"] - clean_run["test_macro_f1"])
        <= 1e-12,
        "Clean macro-F1 mismatch",
    )

    runs = []
    for scenario in SCENARIOS:
        print("=" * 70)
        print("E007 RETRAINING:", scenario["name"])
        print("=" * 70)
        run = train_one(
            module,
            edit_module,
            shared,
            manifest,
            clean_metrics,
            config,
            scenario,
        )
        runs.append(run)
        print("BEST_EPOCH:", run["best_epoch"])
        print("STOPPING_EPOCH:", run["stopping_epoch"])
        print("TEST_METRICS:", run["test_metrics"])
        print("CLEAN_RELATIVE_CHANGES:", run["clean_relative_changes"])

    frozen_after = frozen_hashes(module, shared)
    require(frozen_after == frozen_before, "Frozen tensors changed")
    require(
        sha256_file(clean_checkpoint)
        == clean_checkpoint_hash_before,
        "Clean checkpoint changed",
    )
    require(all(run["passed"] for run in runs), "A retraining run failed")
    require(
        all(run["learning_signal_satisfied"] for run in runs),
        "A retraining run lacked learning signal",
    )
    require(
        all(
            run["finite_value_checks"]["all_finite"]
            for run in runs
        ),
        "A retraining run was non-finite",
    )

    result = {
        "artifact": "E007 q=1/4 seed-0 matched retraining pilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "model": "MSGNN_link_prediction",
            "q": 0.25,
            "model_seed": MODEL_SEED,
            "clean_comparator_count": 1,
            "perturbed_retraining_run_count": len(runs),
            "same_initialization_seed": True,
            "same_graph_generation_seed": True,
            "same_split_seed": True,
            "features_frozen_from_clean_graph": True,
            "queries_and_labels_frozen": True,
            "estimand": (
                "Clean-relative performance after matched retraining "
                "on deterministically perturbed training graphs"
            ),
            "not_estimated": [
                "between-model-seed variability",
                "between-graph variability",
                "between-split variability",
                "architecture superiority",
                "publication-level generalization",
            ],
        },
        "sources": {
            "training_script": str(TRAINING_SOURCE),
            "training_script_sha256": TRAINING_SOURCE_SHA256,
            "edit_script": str(EDIT_SOURCE),
            "edit_script_sha256": EDIT_SOURCE_SHA256,
            "manifest": str(MANIFEST),
            "manifest_sha256": MANIFEST_SHA256,
            "clean_result": str(CLEAN_RESULT),
            "clean_result_sha256": sha256_file(CLEAN_RESULT),
            "clean_checkpoint": str(clean_checkpoint),
            "clean_checkpoint_sha256": clean_checkpoint_hash_before,
            "base_bundle_fingerprint": shared.bundle_fingerprint,
        },
        "configuration": {
            "config": config,
            "optimizer": {
                "name": "Adam",
                "lr": module.LR,
                "weight_decay": module.WEIGHT_DECAY,
            },
            "max_epochs": module.MAX_EPOCHS,
            "patience": module.PATIENCE,
            "min_epochs": module.MIN_EPOCHS,
            "min_improvement": module.MIN_IMPROVEMENT,
        },
        "clean_comparator": {
            "metrics": clean_metrics,
            "best_epoch": clean_run["best_epoch"],
            "stopping_epoch": clean_run["stopping_epoch"],
        },
        "runs": runs,
        "integrity": {
            "scenario_count": len(runs),
            "all_runs_passed": all(run["passed"] for run in runs),
            "all_learning_signals_satisfied": all(
                run["learning_signal_satisfied"] for run in runs
            ),
            "all_finite": all(
                run["finite_value_checks"]["all_finite"]
                for run in runs
            ),
            "all_checkpoints_restored": all(
                run["checkpoint_restored"] for run in runs
            ),
            "all_checkpoint_reload_diffs_within_tolerance": all(
                run["reload_max_abs_diff"] <= 1e-6
                for run in runs
            ),
            "frozen_tensor_hashes_before": frozen_before,
            "frozen_tensor_hashes_after": frozen_after,
            "frozen_tensors_unchanged": frozen_before == frozen_after,
            "clean_checkpoint_unchanged": (
                sha256_file(clean_checkpoint)
                == clean_checkpoint_hash_before
            ),
        },
        "limitations": [
            (
                "This is a seed-0 pilot; multi-seed variability is "
                "not estimated."
            ),
            (
                "The sign-restoring scenario is a one-edge case study."
            ),
            (
                "Features are frozen from the clean graph to isolate "
                "message-graph retraining effects."
            ),
            (
                "All runs use one generated graph and one split."
            ),
            (
                "Passing the pilot validates the retraining pipeline "
                "but does not alone establish publication-level stability."
            ),
        ],
    }

    boolean_integrity = [
        value
        for value in result["integrity"].values()
        if isinstance(value, bool)
    ]
    require(all(boolean_integrity), "An integrity gate failed")
    require(len(runs) == 7, "Unexpected retraining-run count")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

    print("=" * 70)
    print("E007_MATCHED_RETRAINING_PILOT: PASS")
    print("OUTPUT:", OUTPUT)
    print("PERTURBED_RETRAINING_RUNS:", len(runs))
    print("CLEAN_METRICS:", clean_metrics)
    print("ALL_RUNS_PASSED:", result["integrity"]["all_runs_passed"])
    print(
        "ALL_CHECKPOINTS_RESTORED:",
        result["integrity"]["all_checkpoints_restored"],
    )
    print(
        "FROZEN_TENSORS_UNCHANGED:",
        result["integrity"]["frozen_tensors_unchanged"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
