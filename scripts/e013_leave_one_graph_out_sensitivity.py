"""Leave-one-graph-out sensitivity analysis for the E012 hierarchy."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t as student_t


SOURCE_AGGREGATE = Path(
    "results/e012/"
    "e012_hierarchical_five_graph_aggregate_attempt01.json"
)
OUTPUT = Path(
    "results/e013/"
    "e013_leave_one_graph_out_sensitivity_attempt01.json"
)

EXPECTED_GRAPH_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_MODEL_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_FIXED_Q = 0.125
EXPECTED_SPLIT_SEED = 0
EXPECTED_PRIMARY_METRIC = "macro_f1"

METRICS = (
    "accuracy",
    "macro_f1",
    "micro_f1",
    "sign_accuracy",
    "direction_accuracy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_counts(values: list[float]) -> dict[str, int]:
    return {
        "negative": sum(value < 0.0 for value in values),
        "positive": sum(value > 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
    }


def summarize(values: list[float]) -> dict:
    require(len(values) >= 2, "At least two graph means are required")
    require(
        all(math.isfinite(value) for value in values),
        "Non-finite graph-level value",
    )

    count = len(values)
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(count)
    degrees_of_freedom = count - 1
    critical_value = float(
        student_t.ppf(0.975, degrees_of_freedom)
    )
    half_width = critical_value * standard_error
    lower = mean - half_width
    upper = mean + half_width

    return {
        "count": count,
        "values": values,
        "mean": mean,
        "median": statistics.median(values),
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "minimum": min(values),
        "maximum": max(values),
        "sign_counts": sign_counts(values),
        "confidence_interval_95_t": {
            "degrees_of_freedom": degrees_of_freedom,
            "critical_value": critical_value,
            "half_width": half_width,
            "lower": lower,
            "upper": upper,
            "unit_of_analysis": "graph_generation_seed",
        },
        "interval_excludes_zero": lower > 0.0 or upper < 0.0,
        "interval_includes_zero": lower <= 0.0 <= upper,
    }


def main() -> int:
    require(
        SOURCE_AGGREGATE.is_file(),
        f"Missing source aggregate: {SOURCE_AGGREGATE}",
    )
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite existing output: {OUTPUT}",
    )

    source_hash_before = sha256_file(SOURCE_AGGREGATE)
    source = json.loads(SOURCE_AGGREGATE.read_text())

    require(source.get("status") == "PASS", "E012 source did not pass")

    scope = source["scope"]
    require(
        scope["graph_generation_seeds"] == EXPECTED_GRAPH_SEEDS,
        "Unexpected graph-generation seeds",
    )
    require(
        scope["model_seeds_within_graph"] == EXPECTED_MODEL_SEEDS,
        "Unexpected nested model seeds",
    )
    require(
        scope["graph_count"] == 5,
        "Expected exactly five graph units",
    )
    require(
        scope["model_seed_count_per_graph"] == 5,
        "Expected five model seeds within each graph",
    )
    require(
        scope["paired_result_count"] == 25,
        "Expected 25 paired results",
    )
    require(
        scope["fixed_q"] == EXPECTED_FIXED_Q,
        "Expected fixed q = 0.125",
    )
    require(
        scope["split_seed"] == EXPECTED_SPLIT_SEED,
        "Expected split seed zero",
    )
    require(
        scope["primary_metric"] == EXPECTED_PRIMARY_METRIC,
        "Unexpected primary metric",
    )
    require(
        scope["primary_unit_of_analysis"]
        == "graph_generation_seed",
        "Graph seed is not the declared primary unit",
    )

    source_integrity = source["integrity"]
    required_integrity_gates = (
        "all_25_source_artifacts_hash_locked",
        "all_clean_inclusion_gates_passed",
        "all_referenced_checkpoints_hash_verified",
        "all_source_artifacts_passed",
        "all_source_boolean_integrity_gates_passed",
        "all_values_finite",
        "five_graph_units_present",
        "five_model_seeds_present_per_graph",
        "graph0_aggregate_hash_locked",
        "pooled_25_run_analysis_marked_secondary",
        "primary_analysis_uses_graph_means",
    )
    for gate in required_integrity_gates:
        require(
            source_integrity.get(gate) is True,
            f"Required E012 integrity gate failed: {gate}",
        )

    within_graph = source["within_graph_summaries"]
    require(
        sorted(int(seed) for seed in within_graph)
        == EXPECTED_GRAPH_SEEDS,
        "Within-graph summaries do not cover graph seeds 0--4",
    )

    graph_means: dict[str, dict[int, float]] = {}
    for metric in METRICS:
        graph_means[metric] = {}
        for graph_seed in EXPECTED_GRAPH_SEEDS:
            summary = within_graph[str(graph_seed)][metric]
            require(
                summary["count"] == 5,
                f"Graph {graph_seed}, metric {metric}: expected five runs",
            )
            values = [float(value) for value in summary["values"]]
            require(
                len(values) == 5,
                f"Graph {graph_seed}, metric {metric}: value count mismatch",
            )
            require(
                all(math.isfinite(value) for value in values),
                f"Graph {graph_seed}, metric {metric}: non-finite value",
            )
            recomputed_mean = statistics.fmean(values)
            stored_mean = float(summary["mean"])
            require(
                math.isclose(
                    recomputed_mean,
                    stored_mean,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                f"Graph {graph_seed}, metric {metric}: mean mismatch",
            )
            graph_means[metric][graph_seed] = stored_mean

    full_summaries = {}
    for metric in METRICS:
        values = [
            graph_means[metric][seed]
            for seed in EXPECTED_GRAPH_SEEDS
        ]
        recomputed = summarize(values)
        stored = source["primary_between_graph_summary"][metric]

        require(
            math.isclose(
                recomputed["mean"],
                float(stored["mean"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"Full-sample {metric} mean does not reproduce E012",
        )
        require(
            math.isclose(
                recomputed["sample_sd"],
                float(stored["sample_sd"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"Full-sample {metric} SD does not reproduce E012",
        )
        full_summaries[metric] = recomputed

    full_primary = full_summaries[EXPECTED_PRIMARY_METRIC]
    leave_one_out = []

    for omitted_seed in EXPECTED_GRAPH_SEEDS:
        retained_seeds = [
            seed
            for seed in EXPECTED_GRAPH_SEEDS
            if seed != omitted_seed
        ]

        metric_summaries = {}
        for metric in METRICS:
            retained_values = [
                graph_means[metric][seed]
                for seed in retained_seeds
            ]
            metric_summaries[metric] = summarize(retained_values)

        primary = metric_summaries[EXPECTED_PRIMARY_METRIC]
        mean_shift = primary["mean"] - full_primary["mean"]

        leave_one_out.append(
            {
                "omitted_graph_seed": omitted_seed,
                "retained_graph_seeds": retained_seeds,
                "retained_graph_count": len(retained_seeds),
                "metric_summaries": metric_summaries,
                "primary_macro_f1": {
                    "mean": primary["mean"],
                    "sample_sd": primary["sample_sd"],
                    "standard_error": primary["standard_error"],
                    "confidence_interval_95_t": (
                        primary["confidence_interval_95_t"]
                    ),
                    "sign_counts": primary["sign_counts"],
                    "interval_excludes_zero": (
                        primary["interval_excludes_zero"]
                    ),
                    "interval_includes_zero": (
                        primary["interval_includes_zero"]
                    ),
                    "mean_shift_from_full_five_graph_estimate": mean_shift,
                    "absolute_mean_shift_from_full": abs(mean_shift),
                    "direction": (
                        "negative"
                        if primary["mean"] < 0.0
                        else "positive"
                        if primary["mean"] > 0.0
                        else "zero"
                    ),
                    "conclusion_matches_full_interval_status": (
                        primary["interval_excludes_zero"]
                        == full_primary["interval_excludes_zero"]
                    ),
                },
            }
        )

    primary_results = [
        entry["primary_macro_f1"]
        for entry in leave_one_out
    ]
    loo_means = [entry["mean"] for entry in primary_results]
    loo_lower = [
        entry["confidence_interval_95_t"]["lower"]
        for entry in primary_results
    ]
    loo_upper = [
        entry["confidence_interval_95_t"]["upper"]
        for entry in primary_results
    ]

    conclusion_change_seeds = [
        entry["omitted_graph_seed"]
        for entry in leave_one_out
        if not entry["primary_macro_f1"][
            "conclusion_matches_full_interval_status"
        ]
    ]

    source_hash_after = sha256_file(SOURCE_AGGREGATE)
    require(
        source_hash_before == source_hash_after,
        "E012 source aggregate changed during analysis",
    )

    output = {
        "artifact": (
            "E013 leave-one-graph-out sensitivity analysis "
            "of E012 hierarchical graph means"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source": {
            "path": str(SOURCE_AGGREGATE),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "unchanged": source_hash_before == source_hash_after,
            "source_status": source["status"],
        },
        "scope": {
            "model": scope["model"],
            "scenario": scope["scenario"],
            "fixed_q": scope["fixed_q"],
            "trainable_q": scope["trainable_q"],
            "split_seed": scope["split_seed"],
            "graph_generation_seeds": EXPECTED_GRAPH_SEEDS,
            "model_seeds_nested_within_graph": EXPECTED_MODEL_SEEDS,
            "primary_metric": EXPECTED_PRIMARY_METRIC,
            "primary_unit_of_analysis": "graph_generation_seed",
            "analysis_type": "leave_one_graph_out",
            "training_performed": False,
            "new_checkpoints_created": False,
        },
        "full_five_graph_reference": {
            "macro_f1": full_primary,
            "stored_interpretation": (
                source["primary_macro_f1_interpretation"]
            ),
        },
        "leave_one_graph_out": leave_one_out,
        "sensitivity_summary": {
            "leave_one_out_count": len(leave_one_out),
            "all_four_graph_means_negative": all(
                mean < 0.0 for mean in loo_means
            ),
            "leave_one_out_mean_range": {
                "minimum": min(loo_means),
                "maximum": max(loo_means),
            },
            "leave_one_out_interval_envelope": {
                "minimum_lower_bound": min(loo_lower),
                "maximum_upper_bound": max(loo_upper),
            },
            "maximum_absolute_mean_shift_from_full": max(
                entry["absolute_mean_shift_from_full"]
                for entry in primary_results
            ),
            "all_interval_statuses_match_full": (
                len(conclusion_change_seeds) == 0
            ),
            "omitted_seeds_changing_interval_status": (
                conclusion_change_seeds
            ),
            "interpretation": (
                "The direction of the mean is stable across all "
                "leave-one-graph-out analyses."
                if all(mean < 0.0 for mean in loo_means)
                else
                "The direction of the mean changes in at least one "
                "leave-one-graph-out analysis."
            ),
        },
        "integrity": {
            "source_hash_unchanged": source_hash_before == source_hash_after,
            "source_e012_status_passed": source["status"] == "PASS",
            "source_integrity_gates_passed": all(
                source_integrity.get(gate) is True
                for gate in required_integrity_gates
            ),
            "five_graph_units_present": (
                sorted(int(seed) for seed in within_graph)
                == EXPECTED_GRAPH_SEEDS
            ),
            "five_model_seed_values_per_graph": all(
                len(
                    within_graph[str(seed)][metric]["values"]
                ) == 5
                for seed in EXPECTED_GRAPH_SEEDS
                for metric in METRICS
            ),
            "full_primary_statistics_reproduced": True,
            "all_values_finite": all(
                math.isfinite(value)
                for metric in METRICS
                for value in graph_means[metric].values()
            ),
            "no_training_performed": True,
            "no_new_checkpoints_created": True,
        },
        "limitations": [
            (
                "Each leave-one-out estimate contains only four "
                "independent generated-graph units."
            ),
            (
                "The analysis diagnoses dependence on individual graph "
                "realizations; it does not create new replication."
            ),
            (
                "The split seed remains fixed at zero, so between-split "
                "variability is not estimated."
            ),
            (
                "The result remains restricted to MSGNN, SDSBM, "
                "q = 0.125, and the selected sign-reversal condition."
            ),
            (
                "A change in confidence-interval status under omission "
                "must be reported as small-sample sensitivity, not as a "
                "new confirmatory significance result."
            ),
        ],
    }

    require(
        output["integrity"]["source_hash_unchanged"],
        "Source hash integrity failed",
    )
    require(
        output["integrity"]["source_integrity_gates_passed"],
        "Inherited E012 integrity gate failed",
    )
    require(
        output["integrity"]["full_primary_statistics_reproduced"],
        "Full E012 primary statistics were not reproduced",
    )
    require(
        output["integrity"]["all_values_finite"],
        "Non-finite output detected",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )

    print("E013_LEAVE_ONE_GRAPH_OUT_SENSITIVITY: PASS")
    print("OUTPUT:", OUTPUT)
    print("SOURCE_SHA256:", source_hash_before)
    print("FULL_GRAPH_MEAN:", full_primary["mean"])
    print(
        "FULL_INTERVAL:",
        [
            full_primary["confidence_interval_95_t"]["lower"],
            full_primary["confidence_interval_95_t"]["upper"],
        ],
    )

    for entry in leave_one_out:
        primary = entry["primary_macro_f1"]
        interval = primary["confidence_interval_95_t"]
        print(
            "OMIT_GRAPH:",
            entry["omitted_graph_seed"],
            "MEAN:",
            primary["mean"],
            "SD:",
            primary["sample_sd"],
            "SE:",
            primary["standard_error"],
            "CI:",
            [interval["lower"], interval["upper"]],
            "NEGATIVE_GRAPHS:",
            primary["sign_counts"]["negative"],
            "INTERVAL_INCLUDES_ZERO:",
            primary["interval_includes_zero"],
            "MEAN_SHIFT:",
            primary["mean_shift_from_full_five_graph_estimate"],
        )

    print(
        "ALL_LOO_MEANS_NEGATIVE:",
        output["sensitivity_summary"]["all_four_graph_means_negative"],
    )
    print(
        "OMITTED_SEEDS_CHANGING_INTERVAL_STATUS:",
        output["sensitivity_summary"][
            "omitted_seeds_changing_interval_status"
        ],
    )
    print("TRAINING_PERFORMED: False")
    print("NEW_CHECKPOINTS_CREATED: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
