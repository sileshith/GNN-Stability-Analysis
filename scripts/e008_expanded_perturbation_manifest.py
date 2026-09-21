"""Build the E008 expanded SDSBM-aware perturbation manifest without training."""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


SOURCE_SCRIPT = Path("scripts/e005_charge_comparison_replication.py")
SOURCE_SCRIPT_SHA256 = (
    "5c252bbb56c543e56a5ac43c0f7163df96393ee69604a7e867e03e0e93721f82"
)
OUTPUT = Path(
    "results/e008/"
    "e008_expanded_perturbation_manifest_attempt01.json"
)

EXPECTED_PYGSD_VERSION = "1.2.0"
EXPECTED_BUNDLE = (
    "fcd161442a591368153c9c039d228c81270444c2cea2aa8481d73a2b344094dc"
)
ATOL = 1e-12

META_GRAPH_F = np.array(
    [
        [0.5, 0.1, -0.1, 0.1],
        [0.9, 0.5, -0.1, -0.5],
        [-0.9, -0.9, 0.5, -0.9],
        [-0.9, -0.5, -0.1, 0.5],
    ],
    dtype=float,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_module():
    require(SOURCE_SCRIPT.is_file(), f"Missing source script: {SOURCE_SCRIPT}")
    require(
        sha256_file(SOURCE_SCRIPT) == SOURCE_SCRIPT_SHA256,
        "E005 source-script SHA-256 mismatch",
    )

    spec = importlib.util.spec_from_file_location(
        "e008_manifest_source",
        SOURCE_SCRIPT,
    )
    require(
        spec is not None and spec.loader is not None,
        "Cannot load E005 source script",
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(module, name: str, tensor: torch.Tensor) -> str:
    return module.compute_tensor_hash(name, tensor)


def frozen_hashes(module, shared) -> dict[str, str]:
    return {
        "clean_graph": tensor_hash(
            module, "clean_graph", shared.clean_graph
        ),
        "clean_weights": tensor_hash(
            module, "clean_weights", shared.clean_weights
        ),
        "train_graph": tensor_hash(
            module, "train_graph", shared.train_graph
        ),
        "train_weights": tensor_hash(
            module, "train_weights", shared.train_weights
        ),
        "train_queries": tensor_hash(
            module, "train_queries", shared.train_queries
        ),
        "train_labels": tensor_hash(
            module, "train_labels", shared.train_labels
        ),
        "val_queries": tensor_hash(
            module, "val_queries", shared.val_queries
        ),
        "val_labels": tensor_hash(
            module, "val_labels", shared.val_labels
        ),
        "test_queries": tensor_hash(
            module, "test_queries", shared.test_queries
        ),
        "test_labels": tensor_hash(
            module, "test_labels", shared.test_labels
        ),
        "features": tensor_hash(
            module, "features", shared.features
        ),
    }


def regenerate_communities(module, shared) -> torch.Tensor:
    random.seed(module.GENERATION_SEED)
    np.random.seed(module.GENERATION_SEED)
    torch.manual_seed(module.GENERATION_SEED)

    adjacency, communities = module.SDSBM(
        module.N,
        module.K,
        module.P,
        META_GRAPH_F,
        module.SIZE_RATIO,
        module.ETA,
    )
    adjacency, communities = module.extract_network(
        adjacency,
        communities,
    )

    regenerated = module.SignedData(
        A=adjacency,
        y=torch.as_tensor(communities, dtype=torch.long),
    )

    require(
        tensor_hash(
            module,
            "clean_graph",
            regenerated.edge_index,
        )
        == shared.clean_graph_fingerprint,
        "Regenerated clean-graph fingerprint mismatch",
    )
    require(
        tensor_hash(
            module,
            "clean_weights",
            regenerated.edge_weight,
        )
        == shared.clean_weights_fingerprint,
        "Regenerated clean-weight fingerprint mismatch",
    )

    labels = torch.as_tensor(
        communities,
        dtype=torch.long,
    ).detach().cpu().clone()

    require(
        labels.ndim == 1,
        "Community labels are not one-dimensional",
    )
    require(
        labels.shape[0] == shared.num_nodes,
        "Community-label count mismatch",
    )
    require(
        set(labels.tolist()) == {0, 1, 2, 3},
        "Unexpected community-label support",
    )

    return labels


def evaluation_forbidden_pairs(
    shared,
) -> set[tuple[int, int]]:
    """Protect validation/test pairs while permitting training-query edges."""
    forbidden = set()

    for queries in [
        shared.val_queries,
        shared.test_queries,
    ]:
        for source, target in queries.detach().cpu().tolist():
            source = int(source)
            target = int(target)
            forbidden.add((source, target))
            forbidden.add((target, source))

    return forbidden


def absolute_degree(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.float64)
    half_weight = edge_weight.abs().to(torch.float64) / 2.0

    degree.index_add_(0, edge_index[0], half_weight)
    degree.index_add_(0, edge_index[1], half_weight)

    require(
        torch.all(degree > 0).item(),
        "Training graph contains a nonpositive absolute degree",
    )
    return degree


def classify_edge(
    source: int,
    target: int,
    index: int,
    weight: float,
    communities: torch.Tensor,
) -> dict:
    source_community = int(communities[source].item())
    target_community = int(communities[target].item())

    forward_value = float(
        META_GRAPH_F[source_community, target_community]
    )
    reverse_value = float(
        META_GRAPH_F[target_community, source_community]
    )

    require(
        abs(forward_value) > ATOL,
        "Encountered a zero planted meta-graph value",
    )

    observed_sign = 1 if weight > 0 else -1
    planted_sign = 1 if forward_value > 0 else -1
    sign_consistent = observed_sign == planted_sign

    forward_magnitude = abs(forward_value)
    reverse_magnitude = abs(reverse_value)

    if forward_magnitude > reverse_magnitude + ATOL:
        direction_relation = "meta_preferred"
        direction_reversal_stratum = "signal_disrupting"
    elif forward_magnitude + ATOL < reverse_magnitude:
        direction_relation = "meta_counterpreferred"
        direction_reversal_stratum = "signal_restoring"
    else:
        direction_relation = "meta_tied"
        direction_reversal_stratum = "direction_neutral"

    sign_reversal_stratum = (
        "signal_disrupting"
        if sign_consistent
        else "signal_restoring"
    )

    deletion_stratum = (
        "signal_removing"
        if sign_consistent
        and direction_relation in {"meta_preferred", "meta_tied"}
        else "noise_removing"
    )

    return {
        "edge_index_position": index,
        "source": source,
        "target": target,
        "weight": weight,
        "source_community": source_community,
        "target_community": target_community,
        "meta_graph_forward_value": forward_value,
        "meta_graph_reverse_value": reverse_value,
        "observed_sign": observed_sign,
        "planted_sign": planted_sign,
        "sign_consistent": sign_consistent,
        "direction_relation": direction_relation,
        "sign_reversal_stratum": sign_reversal_stratum,
        "direction_reversal_stratum": direction_reversal_stratum,
        "deletion_stratum": deletion_stratum,
    }


def count_by(records: list[dict], field: str) -> dict[str, int]:
    counts = Counter(record[field] for record in records)
    return dict(sorted(counts.items()))


def main() -> int:
    require(
        not OUTPUT.exists(),
        f"Refusing to overwrite manifest output: {OUTPUT}",
    )

    pygsd_version = metadata.version(
        "torch-geometric-signed-directed"
    )
    require(
        pygsd_version == EXPECTED_PYGSD_VERSION,
        (
            f"Expected PyGSD {EXPECTED_PYGSD_VERSION}, "
            f"found {pygsd_version}"
        ),
    )

    module = load_source_module()
    shared = module.create_shared_data()

    require(
        shared.bundle_fingerprint == EXPECTED_BUNDLE,
        "Locked E005 bundle fingerprint mismatch",
    )
    require(
        module.GENERATION_SEED == 0,
        "Unexpected generation seed",
    )
    require(
        module.SPLIT_SEED == 0,
        "Unexpected split seed",
    )

    hashes_before = frozen_hashes(module, shared)
    communities = regenerate_communities(module, shared)

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
    forbidden = evaluation_forbidden_pairs(shared)
    degree = absolute_degree(
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
        if abs(abs(weight) - 1.0) > ATOL:
            exclusions["non_unit_weight"] += 1
            continue
        if ordered_pair_counts[(source, target)] != 1:
            exclusions["ordered_pair_not_unique"] += 1
            continue
        if (target, source) in pair_set:
            exclusions["reciprocated"] += 1
            continue
        if (source, target) in forbidden:
            exclusions["validation_test_query_or_reverse_query"] += 1
            continue

        record = classify_edge(
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

    require(
        len(common_records) > 0,
        "No sign/direction eligible edges found",
    )
    require(
        len(deletion_records) > 0,
        "No deletion-eligible edges found",
    )

    hashes_after = frozen_hashes(module, shared)
    require(
        hashes_after == hashes_before,
        "Frozen E005 tensors changed during manifest construction",
    )

    manifest = {
        "artifact": "E008 expanded SDSBM-aware perturbation manifest",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": {
            "purpose": (
                "Inventory and classify eligible training-graph "
                "edges before selecting perturbation budgets"
            ),
            "training_performed": False,
            "perturbations_applied": False,
            "model_outputs_computed": False,
            "training_query_edges_permitted": True,
            "validation_test_query_pairs_protected": True,
            "estimand": (
                "Controlled perturbation of the training message graph "
                "with frozen labels and held-out evaluation pairs"
            ),
            "robustness_estimated": False,
            "stability_estimated": False,
        },
        "source": {
            "script": str(SOURCE_SCRIPT),
            "script_sha256": SOURCE_SCRIPT_SHA256,
            "bundle_fingerprint": shared.bundle_fingerprint,
            "generation_seed": module.GENERATION_SEED,
            "split_seed": module.SPLIT_SEED,
            "pygsd_version": pygsd_version,
        },
        "sdsbm": {
            "N": module.N,
            "community_count": module.K,
            "p": module.P,
            "size_ratio": module.SIZE_RATIO,
            "eta_sign_flip_probability": module.ETA,
            "meta_graph_F": META_GRAPH_F.tolist(),
            "community_labels_sha256": tensor_hash(
                module,
                "community_labels",
                communities,
            ),
            "community_counts": {
                str(community): int(
                    (communities == community).sum().item()
                )
                for community in sorted(set(communities.tolist()))
            },
        },
        "eligibility": {
            "shared_rules": [
                "non-self-loop",
                "unit absolute weight",
                "ordered pair unique",
                "reverse edge absent",
                (
                    "validation/test query and reverse-query leakage "
                    "excluded; training-query edges permitted"
                ),
            ],
            "deletion_additional_rule": (
                "both endpoint absolute degrees remain positive"
            ),
            "training_graph_edge_count": edge_index.shape[1],
            "training_query_edges_permitted": True,
            "validation_test_query_pairs_forbidden": True,
            "evaluation_forbidden_ordered_pair_count": len(forbidden),
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "sign_reversal": {
            "eligible_count": len(common_records),
            "stratum_counts": count_by(
                common_records,
                "sign_reversal_stratum",
            ),
            "records": common_records,
        },
        "direction_reversal": {
            "eligible_count": len(common_records),
            "stratum_counts": count_by(
                common_records,
                "direction_reversal_stratum",
            ),
            "direction_relation_counts": count_by(
                common_records,
                "direction_relation",
            ),
            "records": common_records,
        },
        "deletion": {
            "eligible_count": len(deletion_records),
            "stratum_counts": count_by(
                deletion_records,
                "deletion_stratum",
            ),
            "records": deletion_records,
        },
        "semantic_definitions": {
            "sign_reversal": {
                "signal_disrupting": (
                    "Observed edge sign agrees with sign(F_ab); "
                    "reversal destroys planted sign signal"
                ),
                "signal_restoring": (
                    "Observed edge sign disagrees with sign(F_ab); "
                    "reversal corrects a generator sign flip"
                ),
            },
            "direction_reversal": {
                "signal_disrupting": (
                    "|F_ab| > |F_ba| before reversal"
                ),
                "signal_restoring": (
                    "|F_ab| < |F_ba| before reversal"
                ),
                "direction_neutral": (
                    "|F_ab| equals |F_ba|"
                ),
            },
            "deletion": {
                "signal_removing": (
                    "Planted-consistent sign and preferred or tied direction"
                ),
                "noise_removing": (
                    "Generator-flipped sign or counter-preferred direction"
                ),
            },
        },
        "integrity": {
            "frozen_tensor_hashes_before": hashes_before,
            "frozen_tensor_hashes_after": hashes_after,
            "frozen_tensors_unchanged": (
                hashes_before == hashes_after
            ),
            "source_script_unchanged": (
                sha256_file(SOURCE_SCRIPT)
                == SOURCE_SCRIPT_SHA256
            ),
        },
        "limitations": [
            (
                "This manifest uses one fixed generated graph and split; "
                "it does not estimate between-graph or between-split variability."
            ),
            (
                "Strata describe the SDSBM generator semantics and do not "
                "by themselves establish causal importance to the trained model."
            ),
            (
                "No perturbation budget is selected until expanded "
                "stratum counts are reviewed."
            ),
            (
                "Training-query edges are deliberately eligible because "
                "the matched-retraining estimand perturbs the training "
                "message graph while freezing supervised labels."
            ),
            (
                "Validation and test query pairs, including their reverse "
                "orientations, remain excluded from perturbation."
            ),
            (
                "No robustness or stability conclusion is authorized."
            ),
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )

    print("E008_EXPANDED_PERTURBATION_MANIFEST: PASS")
    print("OUTPUT:", OUTPUT)
    print(
        "COMMUNITY_COUNTS:",
        manifest["sdsbm"]["community_counts"],
    )
    print(
        "SIGN_ELIGIBLE:",
        manifest["sign_reversal"]["eligible_count"],
    )
    print(
        "SIGN_STRATA:",
        manifest["sign_reversal"]["stratum_counts"],
    )
    print(
        "DIRECTION_ELIGIBLE:",
        manifest["direction_reversal"]["eligible_count"],
    )
    print(
        "DIRECTION_STRATA:",
        manifest["direction_reversal"]["stratum_counts"],
    )
    print(
        "DELETION_ELIGIBLE:",
        manifest["deletion"]["eligible_count"],
    )
    print(
        "DELETION_STRATA:",
        manifest["deletion"]["stratum_counts"],
    )
    print(
        "FROZEN_TENSORS_UNCHANGED:",
        manifest["integrity"]["frozen_tensors_unchanged"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
