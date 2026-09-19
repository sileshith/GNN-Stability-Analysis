"""Aggregate the validated five-seed q=0, q=1/8, and q=1/4 evidence."""

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path(
    "results/e005/"
    "e005_k1_three_charge_seeds0-4_aggregate_attempt01.json"
)

EXPECTED_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc"
)
EXPECTED_LABEL_COUNTS = [238, 377, 238, 377]
EXPECTED_SAMPLE_COUNT = 1230
EXPECTED_FILTER_K = 1
EXPECTED_GRAPH_K = 4
EXPECTED_GENERATION_SEED = 0
EXPECTED_SPLIT_SEED = 0
METRIC_TOLERANCE = 1e-6

ACCURACY_FLOOR = 377 / 1230
MACRO_F1_FLOOR = 0.2468
SIGN_ACCURACY_FLOOR = 754 / 1230
DIRECTION_ACCURACY_FLOOR = 0.5

CONDITIONS = {
    "q000": {
        "q": 0.0,
        "runs": {
            seed: {
                "clean": Path(
                    f"results/e005/e005_k1_q000_"
                    f"seed{seed}_clean_attempt01.json"
                ),
                "audit": Path(
                    f"results/e005/e005_k1_q000_"
                    f"seed{seed}_inclusion_audit_attempt01.json"
                ),
                "expected_gate": "E005",
            }
            for seed in EXPECTED_SEEDS
        },
    },
    "q0125": {
        "q": 0.125,
        "runs": {
            seed: {
                "clean": Path(
                    f"results/e005/e005_k1_q0125_"
                    f"seed{seed}_clean_attempt01.json"
                ),
                "audit": Path(
                    f"results/e005/e005_k1_q0125_"
                    f"seed{seed}_inclusion_audit_attempt01.json"
                ),
                "expected_gate": "E005",
            }
            for seed in EXPECTED_SEEDS
        },
    },
    "q025": {
        "q": 0.25,
        "runs": {
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
            3: {
                "clean": Path(
                    "results/e004/"
                    "e004_k1_q025_seed3_clean_attempt01.json"
                ),
                "audit": Path(
                    "results/e004/"
                    "e004_k1_q025_seed3_inclusion_audit_attempt01.json"
                ),
                "expected_gate": "E004",
            },
            4: {
                "clean": Path(
                    "results/e004/"
                    "e004_k1_q025_seed4_clean_attempt01.json"
                ),
                "audit": Path(
                    "results/e004/"
                    "e004_k1_q025_seed4_inclusion_audit_attempt01.json"
                ),
                "expected_gate": "E004",
            },
        },
    },
}


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


def sign_confusion(confusion):
    return [
        [
            block_sum(confusion, [0, 2], [0, 2]),
            block_sum(confusion, [0, 2], [1, 3]),
        ],
        [
            block_sum(confusion, [1, 3], [0, 2]),
            block_sum(confusion, [1, 3], [1, 3]),
        ],
    ]


def direction_confusion(confusion):
    return [
        [
            block_sum(confusion, [0, 1], [0, 1]),
            block_sum(confusion, [0, 1], [2, 3]),
        ],
        [
            block_sum(confusion, [2, 3], [0, 1]),
            block_sum(confusion, [2, 3], [2, 3]),
        ],
    ]


def matrix_accuracy(matrix):
    total = sum(map(sum, matrix))
    require(total > 0, "Cannot calculate accuracy from empty matrix")
    return sum(
        matrix[index][index]
        for index in range(len(matrix))
    ) / total


def pooled_class_metrics(confusion):
    recalls = []
    precisions = []
    f1_values = []

    for class_index in range(4):
        true_count = sum(confusion[class_index])
        predicted_count = sum(
            confusion[row][class_index]
            for row in range(4)
        )
        true_positive = confusion[class_index][class_index]

        recall = true_positive / true_count if true_count else 0.0
        precision = (
            true_positive / predicted_count
            if predicted_count
            else 0.0
        )
        denominator = precision + recall
        f1 = (
            2.0 * precision * recall / denominator
            if denominator
            else 0.0
        )

        recalls.append(recall)
        precisions.append(precision)
        f1_values.append(f1)

    return {
        "precision": precisions,
        "recall": recalls,
        "macro_f1": statistics.mean(f1_values),
    }


def descriptive(values):
    require(
        len(values) == len(EXPECTED_SEEDS),
        "Expected exactly five values",
    )
    return {
        "values": values,
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def validate_and_load_run(condition, expected_q, seed, manifest):
    clean_path = manifest["clean"]
    audit_path = manifest["audit"]

    require(clean_path.is_file(), f"Missing clean result: {clean_path}")
    require(audit_path.is_file(), f"Missing audit: {audit_path}")

    clean_sha = sha256_file(clean_path)
    audit_sha = sha256_file(audit_path)
    clean = json.loads(clean_path.read_text())
    audit = json.loads(audit_path.read_text())

    require(
        clean["gate"] == manifest["expected_gate"],
        f"Unexpected gate for {condition} seed {seed}",
    )
    require(
        clean["environment"]["pygsd_version"] == "1.2.0",
        f"PyGSD mismatch for {condition} seed {seed}",
    )
    require(
        clean["environment"]["device"] == "cpu",
        f"Device mismatch for {condition} seed {seed}",
    )

    configuration = clean["configuration"]
    model_configuration = configuration["msgnn_config"]
    result = clean["results"][0]

    require(
        configuration["generation_seed"] == EXPECTED_GENERATION_SEED,
        f"Generation seed mismatch for {condition} seed {seed}",
    )
    require(
        configuration["split_seed"] == EXPECTED_SPLIT_SEED,
        f"Split seed mismatch for {condition} seed {seed}",
    )
    require(
        configuration["model_seeds"] == [seed],
        f"Model seed mismatch for {condition} seed {seed}",
    )
    require(
        configuration["K"] == EXPECTED_GRAPH_K,
        f"Graph K mismatch for {condition} seed {seed}",
    )
    require(
        model_configuration["q"] == expected_q,
        f"q mismatch for {condition} seed {seed}",
    )
    require(
        model_configuration["K"] == EXPECTED_FILTER_K,
        f"Filter K mismatch for {condition} seed {seed}",
    )
    require(
        model_configuration["trainable_q"] is False,
        f"trainable_q mismatch for {condition} seed {seed}",
    )
    require(
        model_configuration["cached"] is False,
        f"cached mismatch for {condition} seed {seed}",
    )

    if condition in {"q000", "q0125"}:
        require(
            configuration["charge_condition"] == condition,
            f"Charge label mismatch for {condition} seed {seed}",
        )
        require(
            configuration["fixed_q"] == expected_q,
            f"Fixed q mismatch for {condition} seed {seed}",
        )

    require(result["seed"] == seed, "Saved seed mismatch")
    require(result["passed"] is True, "Clean run failed")
    require(
        result["learning_signal_satisfied"] is True,
        "Learning signal failed",
    )
    require(
        result["checkpoint_restored"] is True,
        "Checkpoint restore failed",
    )
    require(
        result["reload_max_abs_diff"] == 0.0,
        "Checkpoint reload mismatch",
    )
    require(
        result["finite_value_checks"]["all_finite"] is True,
        "Nonfinite values detected",
    )
    require(
        clean["summary"]["overall_pass"] is True,
        "Clean summary failed",
    )

    bundle = clean["shared_data"]["bundle_fingerprint"]
    require(bundle == EXPECTED_BUNDLE, "Bundle mismatch")
    require(
        result["bundle_fingerprint"] == EXPECTED_BUNDLE,
        "Result bundle mismatch",
    )

    checkpoint_path = Path(result["checkpoint_path"])
    require(checkpoint_path.is_file(), "Missing checkpoint")
    checkpoint_sha = sha256_file(checkpoint_path)
    require(
        checkpoint_sha == result["checkpoint_sha256"],
        "Checkpoint hash mismatch",
    )

    require(
        audit["configuration"]["model_seed"] == seed,
        "Audit seed mismatch",
    )
    require(
        audit["configuration"]["q"] == expected_q,
        "Audit q mismatch",
    )
    require(
        audit["configuration"]["K"] == EXPECTED_FILTER_K,
        "Audit K mismatch",
    )

    if condition in {"q000", "q0125"}:
        require(
            audit["configuration"]["charge_condition"] == condition,
            "Audit charge mismatch",
        )

    require(
        audit["source"]["result"] == str(clean_path),
        "Audit source-result path mismatch",
    )
    require(
        audit["source"]["result_sha256"] == clean_sha,
        "Audit result hash mismatch",
    )
    require(
        audit["source"]["checkpoint"] == str(checkpoint_path),
        "Audit checkpoint path mismatch",
    )
    require(
        audit["source"]["checkpoint_sha256"] == checkpoint_sha,
        "Audit checkpoint hash mismatch",
    )
    require(
        audit["source"]["bundle_fingerprint"] == EXPECTED_BUNDLE,
        "Audit bundle mismatch",
    )

    source_script = Path(audit["source"]["script"])
    require(source_script.is_file(), "Missing audit source script")
    require(
        sha256_file(source_script)
        == audit["source"]["script_sha256"],
        "Audit source-script hash mismatch",
    )

    split = audit["test_split"]
    confusion = split[
        "confusion_rows_true_columns_predicted"
    ]
    require(
        split["sample_count"] == EXPECTED_SAMPLE_COUNT,
        "Test sample-count mismatch",
    )
    require(
        split["label_counts"] == EXPECTED_LABEL_COUNTS,
        "Label-count mismatch",
    )
    require(
        len(confusion) == 4
        and all(len(row) == 4 for row in confusion),
        "Confusion shape mismatch",
    )
    require(
        sum(map(sum, confusion)) == EXPECTED_SAMPLE_COUNT,
        "Confusion total mismatch",
    )

    require(
        abs(
            audit["metrics"]["accuracy"]
            - result["test_accuracy"]
        )
        <= METRIC_TOLERANCE,
        "Accuracy mismatch",
    )
    require(
        abs(
            audit["metrics"]["macro_f1"]
            - result["test_macro_f1"]
        )
        <= METRIC_TOLERANCE,
        "Macro-F1 mismatch",
    )
    require(
        audit["gates"]["class_behavior_pass"] is True,
        "Class-behavior gate failed",
    )

    seed_sign = sign_confusion(confusion)
    seed_direction = direction_confusion(confusion)
    sign_accuracy = matrix_accuracy(seed_sign)
    direction_accuracy = matrix_accuracy(seed_direction)

    return {
        "condition": condition,
        "q": expected_q,
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
        "accuracy": audit["metrics"]["accuracy"],
        "macro_f1": audit["metrics"]["macro_f1"],
        "micro_f1": audit["metrics"]["micro_f1"],
        "prediction_counts": split["prediction_counts"],
        "per_class_recall": split["per_class_recall"],
        "confusion_rows_true_columns_predicted": confusion,
        "sign_confusion_positive_negative": seed_sign,
        "sign_accuracy": sign_accuracy,
        "direction_confusion_queried_reverse": seed_direction,
        "direction_accuracy": direction_accuracy,
        "gates": {
            "class_behavior": True,
            "accuracy": (
                audit["metrics"]["accuracy"] > ACCURACY_FLOOR
            ),
            "macro_f1": (
                audit["metrics"]["macro_f1"] > MACRO_F1_FLOOR
            ),
            "sign_reference": (
                sign_accuracy > SIGN_ACCURACY_FLOOR
            ),
            "direction_reference": (
                direction_accuracy > DIRECTION_ACCURACY_FLOOR
            ),
        },
    }


def summarize_condition(condition, condition_data):
    expected_q = condition_data["q"]
    manifests = condition_data["runs"]
    require(
        sorted(manifests) == EXPECTED_SEEDS,
        f"Seed manifest mismatch for {condition}",
    )

    runs = [
        validate_and_load_run(
            condition,
            expected_q,
            seed,
            manifests[seed],
        )
        for seed in EXPECTED_SEEDS
    ]

    pooled_confusion = [[0, 0, 0, 0] for _ in range(4)]
    for run in runs:
        pooled_confusion = add_matrices(
            pooled_confusion,
            run["confusion_rows_true_columns_predicted"],
        )

    pooled_sign = sign_confusion(pooled_confusion)
    pooled_direction = direction_confusion(pooled_confusion)
    pooled_metrics = pooled_class_metrics(pooled_confusion)

    metric_names = [
        "accuracy",
        "macro_f1",
        "sign_accuracy",
        "direction_accuracy",
    ]
    descriptive_statistics = {
        metric: descriptive([run[metric] for run in runs])
        for metric in metric_names
    }

    gate_names = [
        "class_behavior",
        "accuracy",
        "macro_f1",
        "sign_reference",
        "direction_reference",
    ]
    gate_summary = {
        gate: {
            "pass_count": sum(
                run["gates"][gate]
                for run in runs
            ),
            "total_count": len(runs),
            "all_pass": all(
                run["gates"][gate]
                for run in runs
            ),
        }
        for gate in gate_names
    }

    return {
        "condition": condition,
        "q": expected_q,
        "runs": runs,
        "descriptive_statistics": descriptive_statistics,
        "pooled_descriptive_counts": {
            "warning": (
                "These are repeated predictions on the same "
                "1,230 test queries across five trained models, "
                "not 6,150 independent cases."
            ),
            "total_prediction_records": 6150,
            "confusion_rows_true_columns_predicted": pooled_confusion,
            "per_class_precision": pooled_metrics["precision"],
            "per_class_recall": pooled_metrics["recall"],
            "macro_f1_from_pooled_confusion": (
                pooled_metrics["macro_f1"]
            ),
            "sign_confusion_positive_negative": pooled_sign,
            "sign_accuracy": matrix_accuracy(pooled_sign),
            "direction_confusion_queried_reverse": pooled_direction,
            "direction_accuracy": matrix_accuracy(pooled_direction),
        },
        "gates": gate_summary,
    }


def paired_comparison(left_name, right_name, summaries):
    left_runs = summaries[left_name]["runs"]
    right_runs = summaries[right_name]["runs"]

    require(
        [run["model_seed"] for run in left_runs]
        == [run["model_seed"] for run in right_runs]
        == EXPECTED_SEEDS,
        "Paired seed mismatch",
    )

    metrics = [
        "accuracy",
        "macro_f1",
        "sign_accuracy",
        "direction_accuracy",
    ]
    result = {
        "left_condition": left_name,
        "right_condition": right_name,
        "difference_definition": f"{left_name} minus {right_name}",
        "not_an_independent_sample_test": True,
        "metrics": {},
    }

    for metric in metrics:
        differences = [
            left_runs[index][metric] - right_runs[index][metric]
            for index in range(len(EXPECTED_SEEDS))
        ]
        result["metrics"][metric] = descriptive(differences)

    return result


def main() -> int:
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite aggregate output: {OUTPUT}",
    )
    require(
        sorted(CONDITIONS) == ["q000", "q0125", "q025"],
        "Charge-condition manifest mismatch",
    )

    summaries = {
        condition: summarize_condition(
            condition,
            CONDITIONS[condition],
        )
        for condition in ["q000", "q0125", "q025"]
    }

    comparisons = {
        "q0125_minus_q000": paired_comparison(
            "q0125", "q000", summaries
        ),
        "q0125_minus_q025": paired_comparison(
            "q0125", "q025", summaries
        ),
        "q000_minus_q025": paired_comparison(
            "q000", "q025", summaries
        ),
    }

    all_integrity_checks_passed = all(
        all(
            run["bundle_fingerprint"] == EXPECTED_BUNDLE
            for run in summary["runs"]
        )
        for summary in summaries.values()
    )

    aggregate = {
        "artifact": "E005 three-charge five-model-seed aggregate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS" if all_integrity_checks_passed else "FAIL"
        ),
        "scope": {
            "model": "MSGNN_link_prediction",
            "charge_conditions": ["q000", "q0125", "q025"],
            "q_values": [0.0, 0.125, 0.25],
            "model_seeds_per_condition": EXPECTED_SEEDS,
            "total_model_runs": 15,
            "independent_graph_count": 1,
            "independent_split_count": 1,
            "interpretation": (
                "Matched model-initialization replicates across "
                "three fixed charge values on one graph and split."
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
            "trainable_q": False,
            "msgnn_filter_order_K": EXPECTED_FILTER_K,
            "sdsbm_community_count_K": EXPECTED_GRAPH_K,
            "generation_seed": EXPECTED_GENERATION_SEED,
            "split_seed": EXPECTED_SPLIT_SEED,
            "bundle_fingerprint": EXPECTED_BUNDLE,
            "test_sample_count_per_run": EXPECTED_SAMPLE_COUNT,
            "test_label_counts_per_run": EXPECTED_LABEL_COUNTS,
        },
        "frozen_references": {
            "strict_inequality_required": True,
            "accuracy": ACCURACY_FLOOR,
            "macro_f1": MACRO_F1_FLOOR,
            "sign_accuracy": SIGN_ACCURACY_FLOOR,
            "direction_accuracy": DIRECTION_ACCURACY_FLOOR,
            "all_classes_predicted": True,
            "no_zero_class_recall": True,
        },
        "conditions": summaries,
        "paired_seed_comparisons": comparisons,
        "gates": {
            "all_integrity_checks_passed": (
                all_integrity_checks_passed
            ),
            "all_15_class_behavior_audits_passed": all(
                summary["gates"]["class_behavior"]["all_pass"]
                for summary in summaries.values()
            ),
            "all_15_accuracy_references_cleared": all(
                summary["gates"]["accuracy"]["all_pass"]
                for summary in summaries.values()
            ),
            "all_15_macro_f1_references_cleared": all(
                summary["gates"]["macro_f1"]["all_pass"]
                for summary in summaries.values()
            ),
            "all_15_sign_references_cleared": all(
                summary["gates"]["sign_reference"]["all_pass"]
                for summary in summaries.values()
            ),
            "all_15_direction_references_cleared": all(
                summary["gates"]["direction_reference"]["all_pass"]
                for summary in summaries.values()
            ),
            "three_charge_clean_comparison_milestone": "PASS",
        },
        "scientific_interpretation": [
            (
                "q=1/8 has the highest descriptive mean accuracy "
                "and macro-F1 in this fixed graph/split experiment."
            ),
            (
                "The macro-F1 improvement at q=1/8 reflects more "
                "balanced four-class behavior, not a large binary "
                "sign-accuracy improvement over q=1/4."
            ),
            (
                "q=1/4 has the smallest initialization variability "
                "and the highest descriptive direction accuracy."
            ),
            (
                "Matched-seed differences are descriptive with "
                "only five model initializations."
            ),
            (
                "No condition is established as generally superior."
            ),
        ],
        "next_milestones": [
            (
                "Freeze signal-related and signal-unrelated "
                "perturbation manifests and edit budgets."
            ),
            (
                "Run fixed-checkpoint perturbation budget curves."
            ),
            (
                "Run matched clean-versus-perturbed retraining "
                "for the actual stability estimand."
            ),
            (
                "Synchronize the verified charge comparison into "
                "the manuscript and mentor update."
            ),
        ],
    }

    require(
        aggregate["gates"]["all_15_class_behavior_audits_passed"],
        "A class-behavior audit failed",
    )
    require(
        aggregate["gates"]["all_15_accuracy_references_cleared"],
        "An accuracy reference failed",
    )
    require(
        aggregate["gates"]["all_15_macro_f1_references_cleared"],
        "A macro-F1 reference failed",
    )
    require(
        aggregate["gates"]["all_15_sign_references_cleared"],
        "A sign reference failed",
    )
    require(
        aggregate["gates"]["all_15_direction_references_cleared"],
        "A direction reference failed",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + "\n"
    )

    print("E005_THREE_CHARGE_AGGREGATE: PASS")
    print("OUTPUT:", OUTPUT)
    print("TOTAL_MODEL_RUNS:", 15)

    for condition in ["q000", "q0125", "q025"]:
        stats = summaries[condition]["descriptive_statistics"]
        print("CONDITION:", condition)
        print("Q:", summaries[condition]["q"])
        print("ACCURACY_MEAN:", stats["accuracy"]["mean"])
        print("ACCURACY_SAMPLE_SD:", stats["accuracy"]["sample_standard_deviation"])
        print("MACRO_F1_MEAN:", stats["macro_f1"]["mean"])
        print("MACRO_F1_SAMPLE_SD:", stats["macro_f1"]["sample_standard_deviation"])
        print("SIGN_ACCURACY_MEAN:", stats["sign_accuracy"]["mean"])
        print("DIRECTION_ACCURACY_MEAN:", stats["direction_accuracy"]["mean"])

    for name, comparison in comparisons.items():
        print("PAIRED_COMPARISON:", name)
        for metric, stats in comparison["metrics"].items():
            print(metric, "MEAN_DIFFERENCE:", stats["mean"])

    print(
        "THREE_CHARGE_MILESTONE:",
        aggregate["gates"]["three_charge_clean_comparison_milestone"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
