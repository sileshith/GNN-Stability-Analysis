"""Aggregate E012 matched-retraining results using graphs as primary units."""

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path(
    "results/e012/"
    "e012_hierarchical_five_graph_aggregate_attempt01.json"
)

GRAPH_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEEDS = [0, 1, 2, 3, 4]
METRICS = [
    "accuracy",
    "macro_f1",
    "micro_f1",
    "sign_accuracy",
    "direction_accuracy",
]
PRIMARY_METRIC = "macro_f1"
SCENARIO_NAME = "q0125_sign_reversal_signal_disrupting_m58"
T_CRITICAL_DF4 = 2.7764451051977987
ZERO_TOLERANCE = 1e-12

GRAPH0_SOURCES = {
    0: (
        Path(
            "results/e010/"
            "e010_seed0_reduced_matched_retraining_attempt01.json"
        ),
        "9cc1dc22a5e0d2be626d186c26f9afdf"
        "1e2478b07dafbcf9a79444e4cd95f714",
    ),
    1: (
        Path(
            "results/e011/"
            "e011_seed1_primary_budget_matched_retraining_attempt01.json"
        ),
        "b486d04f1eb2983c6cacd9fb15134ce5"
        "c049141831e3cdfb67591082c9649a41",
    ),
    2: (
        Path(
            "results/e011/"
            "e011_seed2_primary_budget_matched_retraining_attempt01.json"
        ),
        "b219b4850d1f04563e4578784bc1496f"
        "244a3ebd30399287f90c21834df21311",
    ),
    3: (
        Path(
            "results/e011/"
            "e011_seed3_primary_budget_matched_retraining_attempt01.json"
        ),
        "08c2abb72ce51d6c485d9c1486b4e712"
        "cb29d27aaddcc40a553d95b8e662c63b",
    ),
    4: (
        Path(
            "results/e011/"
            "e011_seed4_primary_budget_matched_retraining_attempt01.json"
        ),
        "2049289b687da44da229eba1347bdeebc"
        "38ad30b4fc902852e9fa63d4395d7d9",
    ),
}

GRAPH0_AGGREGATE = (
    Path(
        "results/e011/"
        "e011_primary_budget_five_seed_aggregate_attempt01.json"
    ),
    "978d6d5decd5778739a7ca4b2d8ff3"
    "a90908b96557d9526a675fd9f4d7aa5ded",
)

E012_ARTIFACT_HASHES = {
    1: {
        0: "5b4fbb8cf239491e6168cbf3c81633159047f384bb5f9375486e059adec8615c",
        1: "3f5a758bdc97363c18c62614fbdc6760aba626b68e47ea04a24347e9eac767d6",
        2: "2e7fce8ac23b0cd0b680118901a49965554369fc81792591bcde188928424bc0",
        3: "f1242f621b288b439536cd6f8a9400104eaf6a8fd26c274d6049f9698e83b9a4",
        4: "3bb7088bd2676634e6a78f632943e530d39f9d80a3d80ee171838cd835000889",
    },
    2: {
        0: "97c1340ff756e8656658ac36fe7b0d2f9315f2aea0a80f0393aa985bf6ea22e4",
        1: "ac25f72feeea32d6d1f63b0da761c341f10363fb6e9b5a5f205e8e89c5ac6d21",
        2: "6aefb960d3420f22a65a4d68a11f4a23dc8cd410901caa27044eb9bdea0001fe",
        3: "f0282e8db66a7ffabf168b489756913cbdc50fdc19cdbfbeeb9c876d0b12392a",
        4: "eb6e406f83519df7ce036904bd911f3677b91e23af57c55ee7eee82f6a052032",
    },
    3: {
        0: "472f8d737091f13f2f1cfa5aaaccf491e283c753253d6a2d61fde3bd80734181",
        1: "e61f0cb08800d0acddad12b0d32ddff8f68bbf19c52deee4331e0103d7f98d1a",
        2: "37c8887826fbd5470eea94711c8aedebf80dcf8ca6eddb5c4abbaa9530f83e13",
        3: "2922462dd8f0012c7e1afde77b1456e0666901322c4b53812527bc724e1d8546",
        4: "5a628a7c46ec3046989c97e87e08244efac3b8022d6e78c60d31e8481cb603fa",
    },
    4: {
        0: "aae92c637590db09c8db6db4448f642e6dfebfd3cd9fc5c9ab0961744c22c717",
        1: "c00a34efcffa43d403a45008ceb8500fa7905a1286a8281d11a52719a89b5aae",
        2: "39c761cdb588a78b56d17a283323611a492b6a582179d5bd03b36146e1f5fb53",
        3: "feaf51d5354b6cbbeb9373e5dda3d947552ac74df80d81e45a02ba8adb7dbe87",
        4: "76d8ad758659d60f9e4a9499a556d764c4e1418001f0208189cab106714c83ed",
    },
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_locked_json(path, expected_hash):
    require(path.is_file(), f"Missing source artifact: {path}")
    observed_hash = sha256_file(path)
    require(
        observed_hash == expected_hash,
        f"Source-artifact hash mismatch: {path}",
    )
    data = json.loads(path.read_text())
    require(data["status"] == "PASS", f"Source artifact failed: {path}")
    return data, observed_hash


def verify_boolean_integrity(data, label):
    integrity = data.get("integrity", {})
    for name, value in integrity.items():
        if isinstance(value, bool):
            require(value is True, f"Integrity failure: {label}/{name}")


def verify_checkpoint(run, label):
    checkpoint = Path(run["checkpoint"])
    require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    observed_hash = sha256_file(checkpoint)
    require(
        observed_hash == run["checkpoint_sha256"],
        f"Checkpoint hash mismatch: {label}",
    )
    require(run["passed"] is True, f"Run failed: {label}")
    require(
        run["checkpoint_restored"] is True,
        f"Checkpoint not restored: {label}",
    )
    require(
        math.isfinite(float(run["reload_max_abs_diff"])),
        f"Non-finite reload difference: {label}",
    )
    require(
        float(run["reload_max_abs_diff"]) <= 1e-7,
        f"Reload difference too large: {label}",
    )
    return {
        "path": str(checkpoint),
        "sha256": observed_hash,
    }


def summarize(values, include_primary_ci=False):
    require(len(values) >= 2, "At least two values are required")
    require(all(math.isfinite(float(value)) for value in values), "Non-finite value")

    result = {
        "values": values,
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sign_counts": {
            "negative": sum(value < -ZERO_TOLERANCE for value in values),
            "zero": sum(abs(value) <= ZERO_TOLERANCE for value in values),
            "positive": sum(value > ZERO_TOLERANCE for value in values),
        },
    }

    if include_primary_ci:
        require(len(values) == 5, "Primary CI requires five graph means")
        half_width = (
            T_CRITICAL_DF4
            * result["sample_sd"]
            / math.sqrt(len(values))
        )
        result["confidence_interval_95_t"] = {
            "lower": result["mean"] - half_width,
            "upper": result["mean"] + half_width,
            "half_width": half_width,
            "degrees_of_freedom": 4,
            "unit_of_analysis": "graph_generation_seed",
        }

    return result


def find_graph0_run(data, model_seed):
    require(
        int(data["scope"]["model_seed"]) == model_seed,
        f"Graph-0 model-seed mismatch: {model_seed}",
    )
    matches = [
        run
        for run in data["runs"]
        if run["scenario"]["name"] == SCENARIO_NAME
    ]
    require(
        len(matches) == 1,
        f"Expected one graph-0 scenario for model seed {model_seed}",
    )
    return matches[0]


def e012_artifact_path(graph_seed, model_seed):
    return Path(
        "results/e012/"
        f"e012_g{graph_seed}_seed{model_seed}_q0125_"
        "sign_disrupting_matched_retraining_attempt01.json"
    )


def main():
    require(not OUTPUT.exists(), f"Refusing to overwrite {OUTPUT}")

    graph0_aggregate_path, graph0_aggregate_hash = GRAPH0_AGGREGATE
    graph0_aggregate, observed_graph0_aggregate_hash = load_locked_json(
        graph0_aggregate_path,
        graph0_aggregate_hash,
    )
    verify_boolean_integrity(graph0_aggregate, "graph0_aggregate")

    values_by_graph = {
        graph_seed: {metric: [] for metric in METRICS}
        for graph_seed in GRAPH_SEEDS
    }
    source_records = []
    checkpoint_records = []

    for model_seed in MODEL_SEEDS:
        path, expected_hash = GRAPH0_SOURCES[model_seed]
        data, observed_hash = load_locked_json(path, expected_hash)
        verify_boolean_integrity(data, f"g0_seed{model_seed}")

        run = find_graph0_run(data, model_seed)
        checkpoint = verify_checkpoint(run, f"g0_seed{model_seed}")

        for metric in METRICS:
            value = float(run["clean_relative_changes"][metric])
            require(math.isfinite(value), "Non-finite graph-0 change")
            values_by_graph[0][metric].append(value)

        source_records.append(
            {
                "generation_seed": 0,
                "model_seed": model_seed,
                "path": str(path),
                "sha256": observed_hash,
                "source_experiment": "E010" if model_seed == 0 else "E011",
            }
        )
        checkpoint_records.append(
            {
                "generation_seed": 0,
                "model_seed": model_seed,
                "arm": "perturbed",
                **checkpoint,
            }
        )

    for graph_seed in [1, 2, 3, 4]:
        for model_seed in MODEL_SEEDS:
            path = e012_artifact_path(graph_seed, model_seed)
            expected_hash = E012_ARTIFACT_HASHES[graph_seed][model_seed]
            data, observed_hash = load_locked_json(path, expected_hash)
            verify_boolean_integrity(data, f"g{graph_seed}_seed{model_seed}")

            scope = data["scope"]
            require(
                int(scope["generation_seed"]) == graph_seed,
                "Generation-seed mismatch",
            )
            require(
                int(scope["model_seed"]) == model_seed,
                "Model-seed mismatch",
            )
            require(float(scope["fixed_q"]) == 0.125, "Fixed-q mismatch")
            require(scope["trainable_q"] is False, "q must remain fixed")
            require(int(scope["split_seed"]) == 0, "Split-seed mismatch")

            for name, passed in data["clean_inclusion_gate"].items():
                require(
                    passed is True,
                    f"Clean inclusion gate failed: "
                    f"g{graph_seed}/seed{model_seed}/{name}",
                )

            clean_checkpoint = verify_checkpoint(
                data["clean_run"],
                f"g{graph_seed}_seed{model_seed}_clean",
            )
            perturbed_checkpoint = verify_checkpoint(
                data["perturbed_run"],
                f"g{graph_seed}_seed{model_seed}_perturbed",
            )

            for metric in METRICS:
                value = float(
                    data["paired_clean_relative_changes"][metric]
                )
                require(math.isfinite(value), "Non-finite paired change")
                values_by_graph[graph_seed][metric].append(value)

            source_records.append(
                {
                    "generation_seed": graph_seed,
                    "model_seed": model_seed,
                    "path": str(path),
                    "sha256": observed_hash,
                    "source_experiment": "E012",
                }
            )
            checkpoint_records.extend(
                [
                    {
                        "generation_seed": graph_seed,
                        "model_seed": model_seed,
                        "arm": "clean",
                        **clean_checkpoint,
                    },
                    {
                        "generation_seed": graph_seed,
                        "model_seed": model_seed,
                        "arm": "perturbed",
                        **perturbed_checkpoint,
                    },
                ]
            )

    graph_summaries = {}
    graph_means = {metric: [] for metric in METRICS}
    pooled_values = {metric: [] for metric in METRICS}

    for graph_seed in GRAPH_SEEDS:
        graph_summaries[str(graph_seed)] = {}
        for metric in METRICS:
            values = values_by_graph[graph_seed][metric]
            require(
                len(values) == 5,
                f"Expected five model seeds: graph {graph_seed}/{metric}",
            )
            summary = summarize(values)
            graph_summaries[str(graph_seed)][metric] = summary
            graph_means[metric].append(summary["mean"])
            pooled_values[metric].extend(values)

    primary_graph_level = {
        metric: summarize(graph_means[metric], include_primary_ci=True)
        for metric in METRICS
    }
    secondary_pooled_model_seed = {
        metric: {
            **summarize(pooled_values[metric]),
            "inferential_status": (
                "descriptive_only; these 25 observations are nested within "
                "five generated graphs and are not treated as 25 independent "
                "experimental units"
            ),
        }
        for metric in METRICS
    }

    primary = primary_graph_level[PRIMARY_METRIC]
    primary_ci = primary["confidence_interval_95_t"]

    result = {
        "artifact": "E012 hierarchical five-graph aggregate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "model": "MSGNN_link_prediction",
            "fixed_q": 0.125,
            "trainable_q": False,
            "scenario": SCENARIO_NAME,
            "graph_generation_seeds": GRAPH_SEEDS,
            "model_seeds_within_graph": MODEL_SEEDS,
            "split_seed": 0,
            "graph_count": 5,
            "model_seed_count_per_graph": 5,
            "paired_result_count": 25,
            "primary_metric": PRIMARY_METRIC,
            "primary_unit_of_analysis": "graph_generation_seed",
            "secondary_unit": "model_seed_nested_within_graph",
            "estimand": (
                "paired change after matched retraining on sign-reversed "
                "signal-disrupting edges at approximately two percent of "
                "the training graph"
            ),
        },
        "sources": {
            "graph0_e011_aggregate": {
                "path": str(graph0_aggregate_path),
                "sha256": observed_graph0_aggregate_hash,
            },
            "individual_artifacts": source_records,
            "verified_checkpoints": checkpoint_records,
        },
        "within_graph_summaries": graph_summaries,
        "primary_between_graph_summary": primary_graph_level,
        "secondary_pooled_model_seed_summary": secondary_pooled_model_seed,
        "primary_macro_f1_interpretation": {
            "graph_mean_values": primary["values"],
            "mean_change": primary["mean"],
            "sample_sd_across_graph_means": primary["sample_sd"],
            "confidence_interval_95_t": primary_ci,
            "graphs_negative": primary["sign_counts"]["negative"],
            "graphs_zero": primary["sign_counts"]["zero"],
            "graphs_positive": primary["sign_counts"]["positive"],
            "interval_excludes_zero": (
                primary_ci["lower"] > 0.0
                or primary_ci["upper"] < 0.0
            ),
            "direction": (
                "negative"
                if primary["mean"] < 0.0
                else "positive"
                if primary["mean"] > 0.0
                else "zero"
            ),
        },
        "integrity": {
            "all_25_source_artifacts_hash_locked": len(source_records) == 25,
            "graph0_aggregate_hash_locked": True,
            "all_referenced_checkpoints_hash_verified": True,
            "all_source_artifacts_passed": True,
            "all_source_boolean_integrity_gates_passed": True,
            "all_clean_inclusion_gates_passed": True,
            "all_values_finite": True,
            "five_graph_units_present": len(graph_summaries) == 5,
            "five_model_seeds_present_per_graph": all(
                len(values_by_graph[graph_seed][PRIMARY_METRIC]) == 5
                for graph_seed in GRAPH_SEEDS
            ),
            "primary_analysis_uses_graph_means": True,
            "pooled_25_run_analysis_marked_secondary": True,
        },
        "limitations": [
            (
                "Only five generated graphs are available, so the "
                "between-graph confidence interval is necessarily imprecise."
            ),
            (
                "The split seed is fixed at zero; between-split variability "
                "is not estimated."
            ),
            (
                "The perturbation selection is deterministic within each "
                "generated graph."
            ),
            (
                "The result is limited to the locked SDSBM configuration, "
                "MSGNN, q = 0.125, and the selected perturbation condition."
            ),
            (
                "The secondary 25-run pooled summary must not be interpreted "
                "as 25 independent graph-level replications."
            ),
            (
                "This experiment does not establish real-world, alternative-"
                "architecture, or publication-wide generalization."
            ),
        ],
    }

    for name, passed in result["integrity"].items():
        require(passed is True, f"Final integrity gate failed: {name}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("E012_HIERARCHICAL_FIVE_GRAPH_AGGREGATE: PASS")
    print("OUTPUT:", OUTPUT)
    print("PRIMARY_UNIT:", "graph_generation_seed")
    print("GRAPH_SEEDS:", GRAPH_SEEDS)
    print("MODEL_SEEDS_WITHIN_GRAPH:", MODEL_SEEDS)
    print("TOTAL_PAIRED_RESULTS:", 25)
    print("GRAPH_MACRO_F1_MEANS:", primary["values"])
    print("MACRO_F1_GRAPH_MEAN:", primary["mean"])
    print("MACRO_F1_GRAPH_MEAN_SAMPLE_SD:", primary["sample_sd"])
    print("MACRO_F1_GRAPH_LEVEL_95_T_CI:", primary_ci)
    print("MACRO_F1_GRAPH_SIGN_COUNTS:", primary["sign_counts"])
    print(
        "MACRO_F1_GRAPH_LEVEL_INTERVAL_EXCLUDES_ZERO:",
        result["primary_macro_f1_interpretation"][
            "interval_excludes_zero"
        ],
    )
    print(
        "SECONDARY_POOLED_25_RUN_MEAN:",
        secondary_pooled_model_seed[PRIMARY_METRIC]["mean"],
    )
    print("SOURCE_ARTIFACTS_VERIFIED:", len(source_records))
    print("CHECKPOINTS_VERIFIED:", len(checkpoint_records))
    print("HIERARCHICAL_MILESTONE: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
