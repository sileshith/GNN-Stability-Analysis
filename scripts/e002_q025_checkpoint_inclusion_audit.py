"""Audit the E002 q=1/4 checkpoint without retraining or altering E002 artifacts."""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


SOURCE_SCRIPT = Path("scripts/e002_q025_clean_baseline.py")
SOURCE_RESULT = Path(
    "results/e002/e002_k1_q025_seed0_clean_attempt01.json"
)
AUDIT_OUTPUT = Path(
    "results/e002/e002_k1_q025_seed0_inclusion_audit_attempt01.json"
)
EXPECTED_PYGSD_VERSION = "1.2.0"
EXPECTED_Q = 0.25
EXPECTED_K = 1
LABEL_DIM = 4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def class_metrics(confusion: torch.Tensor) -> tuple[list[float], list[float]]:
    confusion_float = confusion.float()
    true_counts = confusion_float.sum(dim=1)
    predicted_counts = confusion_float.sum(dim=0)
    diagonal = confusion_float.diag()

    recalls = torch.where(
        true_counts > 0,
        diagonal / true_counts,
        torch.zeros_like(diagonal),
    )
    precisions = torch.where(
        predicted_counts > 0,
        diagonal / predicted_counts,
        torch.zeros_like(diagonal),
    )

    return precisions.tolist(), recalls.tolist()


def main() -> int:
    require(SOURCE_SCRIPT.is_file(), f"Missing source script: {SOURCE_SCRIPT}")
    require(SOURCE_RESULT.is_file(), f"Missing source result: {SOURCE_RESULT}")
    require(
        not AUDIT_OUTPUT.exists(),
        f"Refusing to overwrite audit output: {AUDIT_OUTPUT}",
    )

    spec = importlib.util.spec_from_file_location(
        "e002_checkpoint_audit_source",
        SOURCE_SCRIPT,
    )
    require(spec is not None and spec.loader is not None, "Cannot load source script")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pygsd_version = metadata.version("torch-geometric-signed-directed")
    require(
        pygsd_version == EXPECTED_PYGSD_VERSION,
        f"Expected PyGSD {EXPECTED_PYGSD_VERSION}, found {pygsd_version}",
    )

    source_data = json.loads(SOURCE_RESULT.read_text())
    require(source_data["gate"] == "E002", "Source result is not E002")
    require(
        source_data["configuration"]["msgnn_config"]["q"] == EXPECTED_Q,
        "Source result q mismatch",
    )

    saved_result = source_data["results"][0]
    checkpoint_path = Path(saved_result["checkpoint_path"])

    require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    require(
        checkpoint_sha256 == saved_result["checkpoint_sha256"],
        "Checkpoint SHA-256 mismatch",
    )

    shared = module.create_shared_data()
    expected_bundle = source_data["shared_data"]["bundle_fingerprint"]

    require(
        shared.bundle_fingerprint == expected_bundle,
        "Recreated data bundle fingerprint mismatch",
    )
    require(
        saved_result["bundle_fingerprint"] == expected_bundle,
        "Saved result bundle fingerprint mismatch",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=module.DEVICE,
        weights_only=True,
    )

    require(
        checkpoint["bundle_fingerprint"] == expected_bundle,
        "Checkpoint bundle fingerprint mismatch",
    )
    require(checkpoint["config"]["q"] == EXPECTED_Q, "Checkpoint q mismatch")
    require(checkpoint["config"]["K"] == EXPECTED_K, "Checkpoint K mismatch")
    require(
        checkpoint["config"]["trainable_q"] is False,
        "Checkpoint unexpectedly uses trainable q",
    )
    require(
        checkpoint["config"]["cached"] is False,
        "Checkpoint unexpectedly enables caching",
    )

    model = module.MSGNN_link_prediction(
        **checkpoint["config"]
    ).to(module.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(
            shared.features,
            shared.features.clone(),
            shared.train_graph,
            shared.test_queries,
            shared.train_weights,
        )

    module.validate_output(
        output,
        (shared.test_queries.shape[0], LABEL_DIM),
        "E002 checkpoint inclusion audit",
    )

    predictions = output.argmax(dim=1)
    labels = shared.test_labels

    prediction_counts = torch.bincount(predictions, minlength=LABEL_DIM)
    label_counts = torch.bincount(labels, minlength=LABEL_DIM)

    confusion = torch.bincount(
        labels * LABEL_DIM + predictions,
        minlength=LABEL_DIM ** 2,
    ).reshape(LABEL_DIM, LABEL_DIM)

    per_class_precision, per_class_recall = class_metrics(confusion)

    accuracy, macro_f1, micro_f1 = module.compute_metrics(output, labels)

    majority_class = int(label_counts.argmax().item())
    majority_predictions = torch.full_like(labels, majority_class)
    majority_confusion = torch.bincount(
        labels * LABEL_DIM + majority_predictions,
        minlength=LABEL_DIM ** 2,
    ).reshape(LABEL_DIM, LABEL_DIM)

    majority_precision, majority_recall = class_metrics(majority_confusion)

    majority_f1 = []
    for precision, recall in zip(majority_precision, majority_recall):
        denominator = precision + recall
        majority_f1.append(
            0.0 if denominator == 0.0
            else 2.0 * precision * recall / denominator
        )

    majority_accuracy = label_counts.max().item() / labels.numel()
    majority_macro_f1 = sum(majority_f1) / LABEL_DIM
    predicted_class_count = int((prediction_counts > 0).sum().item())
    no_zero_recall = all(value > 0.0 for value in per_class_recall)

    audit = {
        "audit": "E002 checkpoint clean-baseline inclusion audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": {
            "script": str(SOURCE_SCRIPT),
            "script_sha256": sha256_file(SOURCE_SCRIPT),
            "result": str(SOURCE_RESULT),
            "result_sha256": sha256_file(SOURCE_RESULT),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "bundle_fingerprint": expected_bundle,
        },
        "environment": {
            "device": str(module.DEVICE),
            "pygsd_version": pygsd_version,
            "torch_version": torch.__version__,
        },
        "configuration": {
            "q": checkpoint["config"]["q"],
            "K": checkpoint["config"]["K"],
            "trainable_q": checkpoint["config"]["trainable_q"],
            "cached": checkpoint["config"]["cached"],
            "model_seed": checkpoint["seed"],
        },
        "test_split": {
            "sample_count": int(labels.numel()),
            "label_counts": label_counts.tolist(),
            "prediction_counts": prediction_counts.tolist(),
            "confusion_rows_true_columns_predicted": confusion.tolist(),
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "majority_class": majority_class,
            "majority_accuracy": majority_accuracy,
            "majority_macro_f1": majority_macro_f1,
            "accuracy_margin_over_majority": accuracy - majority_accuracy,
            "macro_f1_margin_over_majority": macro_f1 - majority_macro_f1,
        },
        "gates": {
            "more_than_one_class_predicted": predicted_class_count > 1,
            "no_zero_recall": no_zero_recall,
            "accuracy_above_majority": accuracy > majority_accuracy,
            "macro_f1_above_majority": macro_f1 > majority_macro_f1,
            "class_behavior_pass": (
                predicted_class_count > 1 and no_zero_recall
            ),
            "nontrivial_margin_epsilon": None,
            "full_clean_baseline_gate": "pending mentor-approved epsilon",
        },
        "warnings": [
            "This is a single-seed clean checkpoint audit.",
            "Low nonzero class recall may still indicate weak minority-class behavior.",
            "Passing this audit does not establish robustness or publication-level generalization.",
        ],
    }

    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )

    print("E002_INCLUSION_AUDIT: PASS")
    print("AUDIT_OUTPUT:", AUDIT_OUTPUT)
    print("PREDICTION_COUNTS:", prediction_counts.tolist())
    print("PER_CLASS_RECALL:", per_class_recall)
    print("ACCURACY:", accuracy)
    print("MACRO_F1:", macro_f1)
    print("MAJORITY_ACCURACY:", majority_accuracy)
    print("MAJORITY_MACRO_F1:", majority_macro_f1)
    print("CLASS_BEHAVIOR_PASS:", audit["gates"]["class_behavior_pass"])
    print("FULL_CLEAN_BASELINE_GATE:", audit["gates"]["full_clean_baseline_gate"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
