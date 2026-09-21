"""Aggregate five-seed primary-budget matched-retraining evidence."""

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path(
    "results/e011/"
    "e011_primary_budget_five_seed_aggregate_attempt01.json"
)

RUN_SOURCES = {
    0: {
        "path": Path(
            "results/e010/"
            "e010_seed0_reduced_matched_retraining_attempt01.json"
        ),
        "sha256": (
            "9cc1dc22a5e0d2be626d186c26f9afdf"
            "1e2478b07dafbcf9a79444e4cd95f714"
        ),
        "expected_artifact_prefix": "E010",
    },
    1: {
        "path": Path(
            "results/e011/"
            "e011_seed1_primary_budget_matched_retraining_attempt01.json"
        ),
        "sha256": (
            "b486d04f1eb2983c6cacd9fb15134ce5"
            "c049141831e3cdfb67591082c9649a41"
        ),
        "expected_artifact_prefix": "E011",
    },
    2: {
        "path": Path(
            "results/e011/"
            "e011_seed2_primary_budget_matched_retraining_attempt01.json"
        ),
        "sha256": (
            "b219b4850d1f04563e4578784bc1496f"
            "244a3ebd30399287f90c21834df21311"
        ),
        "expected_artifact_prefix": "E011",
    },
    3: {
        "path": Path(
            "results/e011/"
            "e011_seed3_primary_budget_matched_retraining_attempt01.json"
        ),
        "sha256": (
            "08c2abb72ce51d6c485d9c1486b4e712"
            "cb29d27aaddcc40a553d95b8e662c63b"
        ),
        "expected_artifact_prefix": "E011",
    },
    4: {
        "path": Path(
            "results/e011/"
            "e011_seed4_primary_budget_matched_retraining_attempt01.json"
        ),
        "sha256": (
            "2049289b687da44da229eba1347bdeebc"
            "38ad30b4fc902852e9fa63d4395d7d9"
        ),
        "expected_artifact_prefix": "E011",
    },
}

EXPECTED_SCENARIOS = [
    "q0125_sign_reversal_signal_disrupting_m58",
    "q025_direction_reversal_signal_disrupting_m58",
    "q025_direction_reversal_direction_neutral_m58",
    "q025_deletion_signal_removing_m58",
    "q025_deletion_noise_removing_m58",
]

METRICS = [
    "accuracy",
    "macro_f1",
    "micro_f1",
    "sign_accuracy",
    "direction_accuracy",
]

PAIRED_CONTRASTS = {
    "direction_disrupting_minus_direction_neutral": (
        "q025_direction_reversal_signal_disrupting_m58",
        "q025_direction_reversal_direction_neutral_m58",
    ),
    "deletion_signal_removing_minus_noise_removing": (
        "q025_deletion_signal_removing_m58",
        "q025_deletion_noise_removing_m58",
    ),
}

EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c812"
    "70444c2cea2aa8481d73a2b344094dc"
)
MODEL_SEEDS = [0, 1, 2, 3, 4]
PRIMARY_BUDGET = 58
T_CRITICAL_DF4_975 = 2.7764451051977987
ZERO_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    require(len(values) == 5, "Expected five paired values")
    mean_value = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = (
        T_CRITICAL_DF4_975
        * sample_sd
        / math.sqrt(len(values))
    )
    negative = sum(value < -ZERO_TOLERANCE for value in values)
    positive = sum(value > ZERO_TOLERANCE for value in values)
    zero = len(values) - negative - positive

    return {
        "values_by_seed": {
            str(seed): value
            for seed, value in zip(MODEL_SEEDS, values)
        },
        "mean": mean_value,
        "sample_sd": sample_sd,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "confidence_interval_95_t": {
            "lower": mean_value - half_width,
            "upper": mean_value + half_width,
            "half_width": half_width,
            "degrees_of_freedom": 4,
            "critical_value": T_CRITICAL_DF4_975,
        },
        "sign_counts": {
            "negative": negative,
            "zero": zero,
            "positive": positive,
        },
    }


def main():
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite aggregate: {OUTPUT}",
    )

    artifacts = {}
    scenario_runs = {
        name: {}
        for name in EXPECTED_SCENARIOS
    }
    source_records = {}

    for seed in MODEL_SEEDS:
        declaration = RUN_SOURCES[seed]
        source_path = declaration["path"]

        require(
            source_path.is_file(),
            f"Missing source artifact: {source_path}",
        )
        actual_hash = sha256_file(source_path)
        require(
            actual_hash == declaration["sha256"],
            f"Source artifact hash mismatch for seed {seed}",
        )

        artifact = json.loads(source_path.read_text())
        require(
            artifact["status"] == "PASS",
            f"Source artifact failed for seed {seed}",
        )
        require(
            artifact["scope"]["model_seed"] == seed,
            f"Source model-seed mismatch for seed {seed}",
        )
        require(
            artifact["sources"]["base_bundle_fingerprint"]
            == EXPECTED_BUNDLE,
            f"Bundle mismatch for seed {seed}",
        )
        require(
            artifact["artifact"].startswith(
                declaration["expected_artifact_prefix"]
            ),
            f"Artifact identity mismatch for seed {seed}",
        )

        runs_by_name = {
            run["scenario"]["name"]: run
            for run in artifact["runs"]
            if run["scenario"]["name"] in EXPECTED_SCENARIOS
        }
        require(
            sorted(runs_by_name) == sorted(EXPECTED_SCENARIOS),
            f"Scenario set mismatch for seed {seed}",
        )

        for name, run in runs_by_name.items():
            scenario = run["scenario"]
            require(
                scenario["budget"] == PRIMARY_BUDGET,
                f"Budget mismatch: seed {seed}/{name}",
            )
            require(run["passed"] is True, f"Run failed: {seed}/{name}")
            require(
                run["checkpoint_restored"] is True,
                f"Checkpoint not restored: {seed}/{name}",
            )
            checkpoint = Path(run["checkpoint"])
            require(
                checkpoint.is_file(),
                f"Missing checkpoint: {checkpoint}",
            )
            require(
                sha256_file(checkpoint)
                == run["checkpoint_sha256"],
                f"Checkpoint hash mismatch: {seed}/{name}",
            )
            require(
                run["finite_value_checks"]["all_finite"] is True,
                f"Non-finite run: {seed}/{name}",
            )
            scenario_runs[name][seed] = run

        boolean_integrity = [
            value
            for value in artifact["integrity"].values()
            if isinstance(value, bool)
        ]
        require(
            all(boolean_integrity),
            f"Stored integrity failure for seed {seed}",
        )

        artifacts[seed] = artifact
        source_records[str(seed)] = {
            "path": str(source_path),
            "sha256": actual_hash,
            "artifact": artifact["artifact"],
        }

    reference_selections = {}
    for name in EXPECTED_SCENARIOS:
        for seed in MODEL_SEEDS:
            run = scenario_runs[name][seed]
            positions = [
                record["edge_index_position"]
                for record in run["selected_edges"]
            ]
            require(
                len(positions) == PRIMARY_BUDGET,
                f"Selection-count mismatch: {seed}/{name}",
            )
            require(
                len(set(positions)) == PRIMARY_BUDGET,
                f"Duplicate selected position: {seed}/{name}",
            )
            if name not in reference_selections:
                reference_selections[name] = positions
            else:
                require(
                    positions == reference_selections[name],
                    f"Cross-seed selection mismatch: {name}",
                )

    scenario_summaries = {}
    for name in EXPECTED_SCENARIOS:
        metric_summaries = {}
        for metric in METRICS:
            values = [
                scenario_runs[name][seed][
                    "clean_relative_changes"
                ][metric]
                for seed in MODEL_SEEDS
            ]
            metric_summaries[metric] = summarize(values)

        scenario_summaries[name] = {
            "scenario": scenario_runs[name][0]["scenario"],
            "metrics": metric_summaries,
            "all_selections_identical_across_seeds": True,
            "selected_edge_positions": reference_selections[name],
        }

    paired_contrasts = {}
    for contrast_name, (left_name, right_name) in (
        PAIRED_CONTRASTS.items()
    ):
        metric_summaries = {}
        for metric in METRICS:
            values = [
                (
                    scenario_runs[left_name][seed][
                        "clean_relative_changes"
                    ][metric]
                    - scenario_runs[right_name][seed][
                        "clean_relative_changes"
                    ][metric]
                )
                for seed in MODEL_SEEDS
            ]
            metric_summaries[metric] = summarize(values)

        paired_contrasts[contrast_name] = {
            "left_scenario": left_name,
            "right_scenario": right_name,
            "interpretation": (
                "Negative values indicate a larger clean-relative "
                "decline for the left semantic stratum."
            ),
            "metrics": metric_summaries,
        }

    result = {
        "artifact": (
            "E011 five-seed primary-budget matched-retraining aggregate"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "model": "MSGNN_link_prediction",
            "model_seeds": MODEL_SEEDS,
            "model_seed_count": len(MODEL_SEEDS),
            "scenario_count": len(EXPECTED_SCENARIOS),
            "total_scenario_seed_results": (
                len(MODEL_SEEDS) * len(EXPECTED_SCENARIOS)
            ),
            "primary_budget": PRIMARY_BUDGET,
            "graph_generation_seed": 0,
            "split_seed": 0,
            "estimand": (
                "Clean-relative matched-retraining change across "
                "five model-initialization seeds"
            ),
        },
        "sources": source_records,
        "scenario_summaries": scenario_summaries,
        "paired_contrasts": paired_contrasts,
        "integrity": {
            "all_source_artifacts_passed": True,
            "all_source_hashes_match": True,
            "all_checkpoint_hashes_match": True,
            "all_source_integrity_gates_passed": True,
            "all_scenarios_present_for_all_seeds": True,
            "all_selections_identical_across_seeds": True,
            "total_checkpoint_count_verified": 25,
            "total_scenario_seed_results": 25,
        },
        "gates": {
            "five_seed_primary_budget_milestone": "PASS",
            "publication_level_generalization": (
                "not established"
            ),
        },
        "limitations": [
            (
                "The five observations per scenario are paired model-"
                "initialization seeds on one generated graph and split."
            ),
            (
                "The confidence intervals describe between-model-seed "
                "variation only and do not estimate between-graph or "
                "between-split variation."
            ),
            (
                "With five seeds, confidence intervals are descriptive "
                "and low-powered; they are not multiplicity-adjusted."
            ),
            (
                "Deterministic edge selections test one fixed intervention "
                "set rather than variability across perturbation samples."
            ),
            (
                "Fixed clean-derived features isolate message-graph effects."
            ),
            (
                "No architecture-superiority claim is supported."
            ),
        ],
    }

    boolean_integrity = [
        value
        for value in result["integrity"].values()
        if isinstance(value, bool)
    ]
    require(all(boolean_integrity), "Aggregate integrity failure")
    require(
        result["integrity"]["total_checkpoint_count_verified"] == 25,
        "Unexpected verified-checkpoint count",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )

    print("E011_FIVE_SEED_AGGREGATE: PASS")
    print("OUTPUT:", OUTPUT)
    print("MODEL_SEEDS:", MODEL_SEEDS)
    print("SCENARIOS:", len(EXPECTED_SCENARIOS))
    print("TOTAL_SCENARIO_SEED_RESULTS:", 25)

    for name in EXPECTED_SCENARIOS:
        summary = scenario_summaries[name]["metrics"]["macro_f1"]
        print("SCENARIO:", name)
        print("MACRO_F1_CHANGE_MEAN:", summary["mean"])
        print("MACRO_F1_CHANGE_SAMPLE_SD:", summary["sample_sd"])
        print(
            "MACRO_F1_CHANGE_95CI:",
            summary["confidence_interval_95_t"],
        )
        print("MACRO_F1_SIGN_COUNTS:", summary["sign_counts"])

    for name, contrast in paired_contrasts.items():
        summary = contrast["metrics"]["macro_f1"]
        print("PAIRED_CONTRAST:", name)
        print("MACRO_F1_MEAN:", summary["mean"])
        print("MACRO_F1_SAMPLE_SD:", summary["sample_sd"])
        print("MACRO_F1_95CI:", summary["confidence_interval_95_t"])

    print(
        "FIVE_SEED_PRIMARY_BUDGET_MILESTONE:",
        result["gates"]["five_seed_primary_budget_milestone"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
