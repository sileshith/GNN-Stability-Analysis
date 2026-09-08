"""Aggregate the validated q=1/4 model-seed 0, 1, and 2 evidence."""

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path(
    "results/e003/"
    "e003_k1_q025_seeds0-2_three_run_aggregate_attempt01.json"
)

RUNS = {
    0: {
        "clean": Path(
            "results/e002/"
            "e002_k1_q025_seed0_clean_attempt01.json"
        ),
        "audit": Path(
            "results/e002/"
            "e002_k1_q025_seed0_inclusion_audit_attempt01.json"
        ),
        "expected_gate": "E002",
    },
    1: {
        "clean": Path(
            "results/e003/"
            "e003_k1_q025_seed1_clean_attempt01.json"
        ),
        "audit": Path(
            "results/e003/"
            "e003_k1_q025_seed1_inclusion_audit_attempt01.json"
        ),
        "expected_gate": "E003",
    },
    2: {
        "clean": Path(
            "results/e003/"
            "e003_k1_q025_seed2_clean_attempt01.json"
        ),
        "audit": Path(
            "results/e003/"
            "e003_k1_q025_seed2_inclusion_audit_attempt01.json"
        ),
        "expected_gate": "E003",
    },
}

EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc"
)
EXPECTED_LABEL_COUNTS = [238, 377, 238, 377]
EXPECTED_SAMPLE_COUNT = 1230
EXPECTED_Q = 0.25
EXPECTED_FILTER_K = 1
EXPECTED_GRAPH_K = 4
EXPECTED_GENERATION_SEED = 0
EXPECTED_SPLIT_SEED = 0
METRIC_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_matrices(left, right):
    return [
        [
            left[row][column] + right[row][column]
            for column in range(len(left[row]))
        ]
        for row in range(len(left))
    ]


def block_sum(matrix, rows, columns):
    return sum(
        matrix[row][column]
        for row in rows
        for column in columns
    )


def sign_confusion(matrix):
    positive = (0, 2)
    negative = (1, 3)

    return [
        [
            block_sum(matrix, positive, positive),
            block_sum(matrix, positive, negative),
        ],
        [
            block_sum(matrix, negative, positive),
            block_sum(matrix, negative, negative),
        ],
    ]


def direction_confusion(matrix):
    queried = (0, 1)
    reverse = (2, 3)

    return [
        [
            block_sum(matrix, queried, queried),
            block_sum(matrix, queried, reverse),
        ],
        [
            block_sum(matrix, reverse, queried),
            block_sum(matrix, reverse, reverse),
        ],
    ]


def matrix_accuracy(matrix):
    correct = sum(
        matrix[index][index]
        for index in range(len(matrix))
    )
    total = sum(map(sum, matrix))
    require(total > 0, "Cannot calculate accuracy from an empty matrix")
    return correct / total


def class_metrics(confusion):
    class_count = len(confusion)
    precision = []
    recall = []
    f1 = []

    for class_index in range(class_count):
        true_positive = confusion[class_index][class_index]
        true_count = sum(confusion[class_index])
        predicted_count = sum(
            confusion[row][class_index]
            for row in range(class_count)
        )

        class_precision = (
            true_positive / predicted_count
            if predicted_count > 0
            else 0.0
        )
        class_recall = (
            true_positive / true_count
            if true_count > 0
            else 0.0
        )
        denominator = class_precision + class_recall
        class_f1 = (
            2.0 * class_precision * class_recall / denominator
            if denominator > 0
            else 0.0
        )

        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": statistics.mean(f1),
    }


def descriptive(values):
    require(
        len(values) == len(EXPECTED_SEEDS),
        "Descriptive summary requires exactly three values",
    )

    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def main() -> int:
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite aggregate output: {OUTPUT}",
    )
    require(
        sorted(RUNS) == EXPECTED_SEEDS,
        "Run manifest must contain model seeds 0, 1, and 2",
    )

    per_seed = []
    aggregate_confusion = [
        [0, 0, 0, 0]
        for _ in range(4)
    ]

    for seed in EXPECTED_SEEDS:
        manifest = RUNS[seed]
        clean_path = manifest["clean"]
        audit_path = manifest["audit"]

        require(
            clean_path.is_file(),
            f"Missing clean result for seed {seed}: {clean_path}",
        )
        require(
            audit_path.is_file(),
            f"Missing inclusion audit for seed {seed}: {audit_path}",
        )

        clean_sha = sha256_file(clean_path)
        audit_sha = sha256_file(audit_path)
        clean = json.loads(clean_path.read_text())
        audit = json.loads(audit_path.read_text())

        require(
            clean["gate"] == manifest["expected_gate"],
            f"Unexpected clean gate for seed {seed}",
        )
        require(
            clean["environment"]["pygsd_version"] == "1.2.0",
            f"PyGSD version mismatch for seed {seed}",
        )
        require(
            clean["environment"]["device"] == "cpu",
            f"Device mismatch for seed {seed}",
        )

        configuration = clean["configuration"]
        model_configuration = configuration["msgnn_config"]
        result = clean["results"][0]

        require(
            configuration["generation_seed"]
            == EXPECTED_GENERATION_SEED,
            f"Generation seed mismatch for model seed {seed}",
        )
        require(
            configuration["split_seed"] == EXPECTED_SPLIT_SEED,
            f"Split seed mismatch for model seed {seed}",
        )
        require(
            configuration["model_seeds"] == [seed],
            f"Model-seed declaration mismatch for seed {seed}",
        )
        require(
            configuration["K"] == EXPECTED_GRAPH_K,
            f"SDSBM community K mismatch for seed {seed}",
        )
        require(
            model_configuration["q"] == EXPECTED_Q,
            f"q mismatch for seed {seed}",
        )
        require(
            model_configuration["K"] == EXPECTED_FILTER_K,
            f"MSGNN filter K mismatch for seed {seed}",
        )
        require(
            model_configuration["trainable_q"] is False,
            f"trainable_q mismatch for seed {seed}",
        )
        require(
            model_configuration["cached"] is False,
            f"cached mismatch for seed {seed}",
        )

        require(
            len(clean["results"]) == 1,
            f"Unexpected result count for seed {seed}",
        )
        require(
            result["seed"] == seed,
            f"Saved-result seed mismatch for seed {seed}",
        )
        require(
            result["passed"] is True,
            f"Clean result failed for seed {seed}",
        )
        require(
            result["learning_signal_satisfied"] is True,
            f"Learning signal failed for seed {seed}",
        )
        require(
            result["checkpoint_restored"] is True,
            f"Checkpoint was not restored for seed {seed}",
        )
        require(
            result["reload_max_abs_diff"] == 0.0,
            f"Checkpoint reload mismatch for seed {seed}",
        )
        require(
            all(result["finite_value_checks"].values()),
            f"Nonfinite value detected for seed {seed}",
        )
        require(
            clean["summary"]["overall_pass"] is True,
            f"Overall clean result failed for seed {seed}",
        )

        bundle = clean["shared_data"]["bundle_fingerprint"]
        require(
            bundle == EXPECTED_BUNDLE,
            f"Bundle mismatch for seed {seed}",
        )
        require(
            result["bundle_fingerprint"] == EXPECTED_BUNDLE,
            f"Result bundle mismatch for seed {seed}",
        )

        checkpoint_path = Path(result["checkpoint_path"])
        require(
            checkpoint_path.is_file(),
            f"Missing checkpoint for seed {seed}: {checkpoint_path}",
        )
        checkpoint_sha = sha256_file(checkpoint_path)
        require(
            checkpoint_sha == result["checkpoint_sha256"],
            f"Checkpoint hash mismatch for seed {seed}",
        )

        require(
            audit["configuration"]["model_seed"] == seed,
            f"Audit seed mismatch for seed {seed}",
        )
        require(
            audit["configuration"]["q"] == EXPECTED_Q,
            f"Audit q mismatch for seed {seed}",
        )
        require(
            audit["configuration"]["K"] == EXPECTED_FILTER_K,
            f"Audit K mismatch for seed {seed}",
        )
        require(
            audit["source"]["result"] == str(clean_path),
            f"Audit source-result path mismatch for seed {seed}",
        )
        require(
            audit["source"]["result_sha256"] == clean_sha,
            f"Audit source-result hash mismatch for seed {seed}",
        )
        require(
            audit["source"]["checkpoint"] == str(checkpoint_path),
            f"Audit checkpoint path mismatch for seed {seed}",
        )
        require(
            audit["source"]["checkpoint_sha256"] == checkpoint_sha,
            f"Audit checkpoint hash mismatch for seed {seed}",
        )
        require(
            audit["source"]["bundle_fingerprint"] == EXPECTED_BUNDLE,
            f"Audit bundle mismatch for seed {seed}",
        )

        source_script = Path(audit["source"]["script"])
        require(
            source_script.is_file(),
            f"Missing source script for seed {seed}: {source_script}",
        )
        require(
            sha256_file(source_script)
            == audit["source"]["script_sha256"],
            f"Source-script hash mismatch for seed {seed}",
        )

        split = audit["test_split"]
        confusion = split[
            "confusion_rows_true_columns_predicted"
        ]

        require(
            split["sample_count"] == EXPECTED_SAMPLE_COUNT,
            f"Test sample count mismatch for seed {seed}",
        )
        require(
            split["label_counts"] == EXPECTED_LABEL_COUNTS,
            f"Label counts mismatch for seed {seed}",
        )
        require(
            len(confusion) == 4
            and all(len(row) == 4 for row in confusion),
            f"Confusion-matrix shape mismatch for seed {seed}",
        )
        require(
            sum(map(sum, confusion)) == EXPECTED_SAMPLE_COUNT,
            f"Confusion-matrix total mismatch for seed {seed}",
        )

        require(
            abs(
                audit["metrics"]["accuracy"]
                - result["test_accuracy"]
            )
            <= METRIC_TOLERANCE,
            f"Accuracy mismatch for seed {seed}",
        )
        require(
            abs(
                audit["metrics"]["macro_f1"]
                - result["test_macro_f1"]
            )
            <= METRIC_TOLERANCE,
            f"Macro-F1 mismatch for seed {seed}",
        )
        require(
            audit["gates"]["class_behavior_pass"] is True,
            f"Class-behavior gate failed for seed {seed}",
        )
        require(
            audit["gates"]["accuracy_above_majority"] is True,
            f"Accuracy baseline gate failed for seed {seed}",
        )
        require(
            audit["gates"]["macro_f1_above_majority"] is True,
            f"Macro-F1 baseline gate failed for seed {seed}",
        )

        seed_sign_confusion = sign_confusion(confusion)
        seed_direction_confusion = direction_confusion(confusion)

        per_seed.append(
            {
                "model_seed": seed,
                "clean_result": str(clean_path),
                "clean_result_sha256": clean_sha,
                "inclusion_audit": str(audit_path),
                "inclusion_audit_sha256": audit_sha,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "bundle_fingerprint": bundle,
                "best_epoch": result["best_epoch"],
                "stopping_epoch": result["stopping_epoch"],
                "stopping_reason": result["stopping_reason"],
                "accuracy": audit["metrics"]["accuracy"],
                "macro_f1": audit["metrics"]["macro_f1"],
                "micro_f1": audit["metrics"]["micro_f1"],
                "per_class_recall": split["per_class_recall"],
                "prediction_counts": split["prediction_counts"],
                "confusion_rows_true_columns_predicted": confusion,
                "sign_confusion_positive_negative": (
                    seed_sign_confusion
                ),
                "sign_accuracy": matrix_accuracy(
                    seed_sign_confusion
                ),
                "direction_confusion_queried_reverse": (
                    seed_direction_confusion
                ),
                "direction_accuracy": matrix_accuracy(
                    seed_direction_confusion
                ),
                "class_behavior_pass": True,
            }
        )

        aggregate_confusion = add_matrices(
            aggregate_confusion,
            confusion,
        )

    aggregate_sign = sign_confusion(aggregate_confusion)
    aggregate_direction = direction_confusion(
        aggregate_confusion
    )
    aggregate_class_metrics = class_metrics(
        aggregate_confusion
    )

    accuracies = [
        run["accuracy"]
        for run in per_seed
    ]
    macro_f1_values = [
        run["macro_f1"]
        for run in per_seed
    ]
    sign_accuracies = [
        run["sign_accuracy"]
        for run in per_seed
    ]
    direction_accuracies = [
        run["direction_accuracy"]
        for run in per_seed
    ]
    best_epochs = [
        run["best_epoch"]
        for run in per_seed
    ]
    stopping_epochs = [
        run["stopping_epoch"]
        for run in per_seed
    ]

    aggregate_prediction_counts = [
        sum(
            aggregate_confusion[row][column]
            for row in range(4)
        )
        for column in range(4)
    ]

    aggregate = {
        "artifact": "E003 q=1/4 three-model-seed aggregate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "model": "MSGNN_link_prediction",
            "model_seeds": EXPECTED_SEEDS,
            "total_model_runs": 3,
            "generation_seed": EXPECTED_GENERATION_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "independent_graph_count": 1,
            "independent_split_count": 1,
            "interpretation": (
                "Three model-initialization/training replicates "
                "on one fixed graph and one fixed data split"
            ),
            "not_estimated": [
                "between-graph variability",
                "between-split variability",
                "robustness",
                "perturbation tolerance",
                "architecture superiority",
                "publication-level generalization",
            ],
        },
        "configuration": {
            "pygsd_version": "1.2.0",
            "device": "cpu",
            "q": EXPECTED_Q,
            "msgnn_filter_order_K": EXPECTED_FILTER_K,
            "sdsbm_community_count_K": EXPECTED_GRAPH_K,
            "trainable_q": False,
            "cached": False,
            "bundle_fingerprint": EXPECTED_BUNDLE,
            "test_sample_count_per_run": EXPECTED_SAMPLE_COUNT,
            "test_label_counts_per_run": EXPECTED_LABEL_COUNTS,
        },
        "per_seed": per_seed,
        "descriptive_statistics": {
            "accuracy": descriptive(accuracies),
            "macro_f1": descriptive(macro_f1_values),
            "sign_accuracy": descriptive(sign_accuracies),
            "direction_accuracy": descriptive(
                direction_accuracies
            ),
            "best_epoch": descriptive(best_epochs),
            "stopping_epoch": descriptive(stopping_epochs),
        },
        "pooled_descriptive_counts": {
            "warning": (
                "Counts pool repeated predictions on the same "
                "1,230 test queries across three trained models; "
                "they are descriptive and not 3,690 independent data cases"
            ),
            "total_prediction_records": (
                EXPECTED_SAMPLE_COUNT * len(EXPECTED_SEEDS)
            ),
            "confusion_rows_true_columns_predicted": (
                aggregate_confusion
            ),
            "prediction_counts": aggregate_prediction_counts,
            "per_class_precision": (
                aggregate_class_metrics["precision"]
            ),
            "per_class_recall": (
                aggregate_class_metrics["recall"]
            ),
            "macro_f1_from_pooled_confusion": (
                aggregate_class_metrics["macro_f1"]
            ),
            "sign_confusion_positive_negative": aggregate_sign,
            "sign_accuracy": matrix_accuracy(aggregate_sign),
            "direction_confusion_queried_reverse": (
                aggregate_direction
            ),
            "direction_accuracy": matrix_accuracy(
                aggregate_direction
            ),
        },
        "gates": {
            "all_three_clean_runs_passed": True,
            "all_three_class_behavior_audits_passed": True,
            "all_bundle_fingerprints_match": True,
            "all_checkpoint_hashes_match": True,
            "all_source_hashes_match": True,
            "nontrivial_margin_epsilon": None,
            "full_clean_baseline_gate": (
                "pending frozen inclusion margin and "
                "semantic-channel criterion"
            ),
            "three_run_milestone": "PASS",
        },
        "scientific_interpretation": [
            (
                "The three runs are consistent across model "
                "initialization and training randomness."
            ),
            (
                "All three runs exceed the declared joint-class "
                "majority baselines."
            ),
            (
                "All three runs predict every class and have "
                "nonzero recall for every class."
            ),
            (
                "All three runs learn direction substantially "
                "better than sign."
            ),
            (
                "Low positive-sign recall remains a repeated "
                "limitation."
            ),
            (
                "The q=1/4 restricted sign degeneracy is a "
                "mechanistically relevant hypothesis, not a "
                "causal conclusion from these three runs."
            ),
        ],
        "next_milestones": [
            (
                "Mentor progress update after repository and "
                "manuscript synchronization"
            ),
            (
                "Expand retained q=1/4 condition to five total "
                "model seeds if scientifically retained"
            ),
            (
                "Run a controlled q=1/8 comparison because it "
                "retains all four restricted semantic states"
            ),
            (
                "Evaluate a training-label-derived class-weighted "
                "loss as a separately declared condition"
            ),
            (
                "Create corrected architecture-native SGCN and "
                "MagNet clean baselines"
            ),
            (
                "Treat SSSNET and DIMPA as conditional extensions"
            ),
            (
                "Introduce graph-generation and split variability "
                "for publication-oriented replication"
            ),
            (
                "Expand retained conditions toward ten total runs "
                "only under a frozen analysis plan"
            ),
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            aggregate,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    print("E003_THREE_RUN_AGGREGATE: PASS")
    print("OUTPUT:", OUTPUT)
    print("MODEL_SEEDS:", EXPECTED_SEEDS)
    print("TOTAL_MODEL_RUNS: 3")
    print(
        "ACCURACY_MEAN:",
        aggregate["descriptive_statistics"]["accuracy"]["mean"],
    )
    print(
        "ACCURACY_SAMPLE_SD:",
        aggregate["descriptive_statistics"]["accuracy"][
            "sample_standard_deviation"
        ],
    )
    print(
        "MACRO_F1_MEAN:",
        aggregate["descriptive_statistics"]["macro_f1"]["mean"],
    )
    print(
        "MACRO_F1_SAMPLE_SD:",
        aggregate["descriptive_statistics"]["macro_f1"][
            "sample_standard_deviation"
        ],
    )
    print(
        "SIGN_ACCURACY_MEAN:",
        aggregate["descriptive_statistics"]["sign_accuracy"]["mean"],
    )
    print(
        "DIRECTION_ACCURACY_MEAN:",
        aggregate["descriptive_statistics"][
            "direction_accuracy"
        ]["mean"],
    )
    print(
        "POOLED_PER_CLASS_RECALL:",
        aggregate["pooled_descriptive_counts"][
            "per_class_recall"
        ],
    )
    print(
        "THREE_RUN_MILESTONE:",
        aggregate["gates"]["three_run_milestone"],
    )
    print(
        "FULL_CLEAN_BASELINE_GATE:",
        aggregate["gates"]["full_clean_baseline_gate"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
