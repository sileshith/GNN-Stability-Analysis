"""Gate 3 controlled structural-edit and magnetic-operator tests.

This is a small, deterministic, CPU-only diagnostic. It does not train a
model, alter a dataset, or write experiment results.
"""

import unittest

import torch
from torch_geometric_signed_directed.utils.general import (
    get_magnetic_signed_Laplacian,
)


DTYPE = torch.float64
ATOL = 1e-10
NUM_NODES = 4
TARGET = 1


def clean_graph():
    """Four unique, non-self-loop, directed signed unit edges."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2],
            [1, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=DTYPE)
    return edge_index, edge_weight


def delete_target(edge_index, edge_weight):
    keep = torch.ones(edge_weight.numel(), dtype=torch.bool)
    keep[TARGET] = False
    return edge_index[:, keep].clone(), edge_weight[keep].clone()


def reverse_target_sign(edge_index, edge_weight):
    edited_index = edge_index.clone()
    edited_weight = edge_weight.clone()
    edited_weight[TARGET] *= -1
    return edited_index, edited_weight


def reverse_target_direction(edge_index, edge_weight):
    edited_index = edge_index.clone()
    edited_weight = edge_weight.clone()
    source = edited_index[0, TARGET].item()
    target = edited_index[1, TARGET].item()
    edited_index[0, TARGET] = target
    edited_index[1, TARGET] = source
    return edited_index, edited_weight


def dense_laplacian(edge_index, edge_weight, q):
    lap_index, lap_real, lap_imag = get_magnetic_signed_Laplacian(
        edge_index=edge_index,
        edge_weight=edge_weight,
        normalization="sym",
        dtype=DTYPE,
        num_nodes=NUM_NODES,
        q=q,
        return_lambda_max=False,
        absolute_degree=True,
    )

    values = torch.complex(lap_real, lap_imag)
    dense = torch.zeros((NUM_NODES, NUM_NODES), dtype=values.dtype)
    dense.index_put_(
        (lap_index[0], lap_index[1]),
        values,
        accumulate=True,
    )
    return dense


class TestStructuralEditSemantics(unittest.TestCase):
    def test_zero_edit_identity(self):
        edge_index, edge_weight = clean_graph()
        copied_index = edge_index.clone()
        copied_weight = edge_weight.clone()

        self.assertTrue(torch.equal(edge_index, copied_index))
        self.assertTrue(torch.equal(edge_weight, copied_weight))

        for q in (0.0, 0.125, 0.25):
            clean = dense_laplacian(edge_index, edge_weight, q)
            copied = dense_laplacian(copied_index, copied_weight, q)
            self.assertTrue(torch.equal(clean, copied))

    def test_deletion_changes_only_edge_count(self):
        edge_index, edge_weight = clean_graph()
        edited_index, edited_weight = delete_target(edge_index, edge_weight)

        self.assertEqual(edited_weight.numel(), edge_weight.numel() - 1)
        self.assertTrue(torch.equal(edited_index, edge_index[:, [0, 2, 3]]))
        self.assertTrue(torch.equal(edited_weight, edge_weight[[0, 2, 3]]))

    def test_sign_reversal_preserves_endpoints(self):
        edge_index, edge_weight = clean_graph()
        edited_index, edited_weight = reverse_target_sign(
            edge_index, edge_weight
        )

        expected_weight = edge_weight.clone()
        expected_weight[TARGET] *= -1

        self.assertTrue(torch.equal(edited_index, edge_index))
        self.assertTrue(torch.equal(edited_weight, expected_weight))
        self.assertEqual(
            edited_weight[TARGET].item(),
            -edge_weight[TARGET].item(),
        )

    def test_direction_reversal_preserves_sign(self):
        edge_index, edge_weight = clean_graph()
        edited_index, edited_weight = reverse_target_direction(
            edge_index, edge_weight
        )

        self.assertTrue(torch.equal(edited_weight, edge_weight))
        self.assertEqual(
            edited_index[0, TARGET].item(),
            edge_index[1, TARGET].item(),
        )
        self.assertEqual(
            edited_index[1, TARGET].item(),
            edge_index[0, TARGET].item(),
        )

        unchanged = [0, 2, 3]
        self.assertTrue(
            torch.equal(edited_index[:, unchanged], edge_index[:, unchanged])
        )


class TestMagneticOperatorBehavior(unittest.TestCase):
    def setUp(self):
        self.edge_index, self.edge_weight = clean_graph()
        self.deleted = delete_target(self.edge_index, self.edge_weight)
        self.sign_reversed = reverse_target_sign(
            self.edge_index, self.edge_weight
        )
        self.direction_reversed = reverse_target_direction(
            self.edge_index, self.edge_weight
        )

    def assert_hermitian(self, matrix):
        self.assertTrue(
            torch.allclose(matrix, matrix.conj().T, atol=ATOL, rtol=0)
        )

    def test_all_constructed_operators_are_hermitian(self):
        graphs = (
            (self.edge_index, self.edge_weight),
            self.deleted,
            self.sign_reversed,
            self.direction_reversed,
        )
        for q in (0.0, 0.125, 0.25):
            for edge_index, edge_weight in graphs:
                self.assert_hermitian(
                    dense_laplacian(edge_index, edge_weight, q)
                )

    def test_deletion_changes_operator(self):
        for q in (0.0, 0.125, 0.25):
            clean = dense_laplacian(
                self.edge_index, self.edge_weight, q
            )
            edited = dense_laplacian(*self.deleted, q)
            self.assertFalse(
                torch.allclose(clean, edited, atol=ATOL, rtol=0)
            )

    def test_charge_controls_edit_visibility(self):
        clean_q0 = dense_laplacian(
            self.edge_index, self.edge_weight, 0.0
        )
        sign_q0 = dense_laplacian(*self.sign_reversed, 0.0)
        direction_q0 = dense_laplacian(*self.direction_reversed, 0.0)

        self.assertFalse(
            torch.allclose(clean_q0, sign_q0, atol=ATOL, rtol=0)
        )
        self.assertTrue(
            torch.allclose(clean_q0, direction_q0, atol=ATOL, rtol=0)
        )

        clean_q18 = dense_laplacian(
            self.edge_index, self.edge_weight, 0.125
        )
        sign_q18 = dense_laplacian(*self.sign_reversed, 0.125)
        direction_q18 = dense_laplacian(*self.direction_reversed, 0.125)

        self.assertFalse(
            torch.allclose(clean_q18, sign_q18, atol=ATOL, rtol=0)
        )
        self.assertFalse(
            torch.allclose(clean_q18, direction_q18, atol=ATOL, rtol=0)
        )

        clean_q14 = dense_laplacian(
            self.edge_index, self.edge_weight, 0.25
        )
        sign_q14 = dense_laplacian(*self.sign_reversed, 0.25)
        direction_q14 = dense_laplacian(*self.direction_reversed, 0.25)

        self.assertTrue(
            torch.allclose(clean_q14, sign_q14, atol=ATOL, rtol=0)
        )
        self.assertFalse(
            torch.allclose(clean_q14, direction_q14, atol=ATOL, rtol=0)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
