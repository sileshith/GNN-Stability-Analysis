"""Exact tests for v7 Theorem 5.3 under its restricted assumptions."""

import math
import unittest

import torch


DTYPE = torch.float64
ATOL = 1e-10
TARGET = 1


def clean_graph():
    edge_index = torch.tensor(
        [[0, 1, 2, 2], [1, 2, 0, 3]],
        dtype=torch.long,
    )
    edge_weight = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        dtype=DTYPE,
    )
    return edge_index, edge_weight


def delete_target(edge_index, edge_weight):
    keep = torch.ones(edge_weight.numel(), dtype=torch.bool)
    keep[TARGET] = False
    return edge_index[:, keep].clone(), edge_weight[keep].clone()


def reverse_target_sign(edge_index, edge_weight):
    result_index = edge_index.clone()
    result_weight = edge_weight.clone()
    result_weight[TARGET] *= -1
    return result_index, result_weight


def reverse_target_direction(edge_index, edge_weight):
    result_index = edge_index.clone()
    result_weight = edge_weight.clone()
    source = result_index[0, TARGET].item()
    target = result_index[1, TARGET].item()
    result_index[0, TARGET] = target
    result_index[1, TARGET] = source
    return result_index, result_weight


def magnetic_adjacency(edge_index, edge_weight, q, num_nodes=4):
    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=DTYPE,
    )
    adjacency.index_put_(
        (edge_index[0], edge_index[1]),
        edge_weight,
        accumulate=True,
    )

    amplitude = (adjacency + adjacency.T) / 2
    theta = 2 * math.pi * q * (adjacency - adjacency.T)

    return amplitude.to(torch.complex128) * torch.exp(
        1j * theta.to(torch.complex128)
    )


def frobenius_drift(clean, edited):
    return torch.linalg.matrix_norm(
        edited - clean,
        ord="fro",
    ).item()


class TestRestrictedExactIdentities(unittest.TestCase):
    def test_v7_theorem_5_3_for_one_eligible_edge(self):
        edge_index, edge_weight = clean_graph()

        deleted = delete_target(edge_index, edge_weight)
        sign_reversed = reverse_target_sign(edge_index, edge_weight)
        direction_reversed = reverse_target_direction(
            edge_index,
            edge_weight,
        )

        m = 1

        for q in (0.0, 0.125, 0.25):
            clean_h = magnetic_adjacency(
                edge_index,
                edge_weight,
                q,
            )

            deletion_drift = frobenius_drift(
                clean_h,
                magnetic_adjacency(*deleted, q),
            )
            sign_drift = frobenius_drift(
                clean_h,
                magnetic_adjacency(*sign_reversed, q),
            )
            direction_drift = frobenius_drift(
                clean_h,
                magnetic_adjacency(*direction_reversed, q),
            )

            expected_deletion = math.sqrt(m / 2)
            expected_sign = (
                math.sqrt(2 * m)
                * abs(math.cos(2 * math.pi * q))
            )
            expected_direction = (
                math.sqrt(2 * m)
                * abs(math.sin(2 * math.pi * q))
            )

            self.assertAlmostEqual(
                deletion_drift,
                expected_deletion,
                delta=ATOL,
            )
            self.assertAlmostEqual(
                sign_drift,
                expected_sign,
                delta=ATOL,
            )
            self.assertAlmostEqual(
                direction_drift,
                expected_direction,
                delta=ATOL,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
