"""
Gate R2 — PyGSD 1.2.0 MSConv validation checks.

These tests validate the corrected PyGSD 1.2.0 release in an isolated
environment. They are separate from the historical PyGSD 1.1.1 audit and do
not modify either installed package.

Validated behaviors:
- the installed package version is exactly 1.2.0;
- MSConv uses the corrected target_to_source flow;
- MSGNN contains the expected MSConv layer inventory;
- the q=0 complex-computation invariant is satisfied;
- an MSGNN q=1/4 forward pass using X+iX is finite and normalized.
"""

from importlib.metadata import version
import unittest

import torch
from torch_geometric_signed_directed.nn.general.MSConv import MSConv
from torch_geometric_signed_directed.nn.general.MSGNN import (
    MSGNN_link_prediction,
)


PYGSD_VERSION = version("torch-geometric-signed-directed")


@unittest.skipUnless(
    PYGSD_VERSION == "1.2.0",
    f"PyGSD 1.2.0 validation only; installed version is {PYGSD_VERSION}",
)
class TestPyGSD120MSConvValidation(unittest.TestCase):
    """Validate project-relevant behavior of the PyGSD 1.2.0 release."""

    @staticmethod
    def make_first_order_layer(q: float) -> MSConv:
        """Create a deterministic layer with only its first-order term active."""
        layer = MSConv(
            in_channels=1,
            out_channels=1,
            K=1,
            q=q,
            trainable_q=False,
            normalization="sym",
            bias=False,
            cached=False,
            absolute_degree=True,
        )

        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[1, 0, 0] = 1.0

        layer.eval()
        return layer

    def test_installed_release_is_120(self):
        """Record the exact PyGSD release used by this validation."""
        self.assertEqual(PYGSD_VERSION, "1.2.0")

    def test_effective_flow_is_target_to_source(self):
        """Confirm that the corrected flow reaches MessagePassing."""
        layer = self.make_first_order_layer(q=0.25)
        self.assertEqual(layer.flow, "target_to_source")

    def test_msgnn_q_quarter_layer_inventory(self):
        """Check the MSGNN layers for the primary synthetic q configuration."""
        model = MSGNN_link_prediction(
            num_features=2,
            hidden=16,
            q=0.25,
            K=1,
            label_dim=4,
            activation=True,
            trainable_q=False,
            layer=2,
            dropout=0.5,
            normalization="sym",
            cached=False,
            conv_bias=True,
            absolute_degree=True,
        )

        self.assertEqual(len(model.Chebs), 2)
        self.assertTrue(all(isinstance(cheb, MSConv) for cheb in model.Chebs))
        self.assertTrue(
            all(cheb.flow == "target_to_source" for cheb in model.Chebs)
        )
        self.assertEqual(tuple(model.Chebs[0].weight.shape), (2, 2, 16))
        self.assertEqual(tuple(model.Chebs[1].weight.shape), (2, 16, 16))

    def test_q_zero_real_input_has_zero_imaginary_output(self):
        """
        Confirm ordinary complex multiplication at q=0.

        At q=0 the magnetic operator is real. A purely real input and real
        learned weights must therefore produce a zero imaginary output.
        """
        layer = self.make_first_order_layer(q=0.0)

        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float)

        x_real = torch.tensor([[1.0], [0.0]])
        x_imag = torch.zeros_like(x_real)

        with torch.no_grad():
            out_real, out_imag = layer(
                x_real,
                x_imag,
                edge_index,
                edge_weight,
            )

        self.assertTrue(bool(torch.isfinite(out_real).all()))
        self.assertTrue(bool(torch.isfinite(out_imag).all()))
        self.assertGreater(
            torch.max(torch.abs(out_real)).item(),
            0.0,
            msg="The isolated first-order real output should be nonzero.",
        )
        self.assertTrue(
            torch.allclose(
                out_imag,
                torch.zeros_like(out_imag),
                atol=1e-7,
                rtol=0.0,
            ),
            msg="At q=0, a purely real input produced imaginary output.",
        )

    def test_msgnn_q_quarter_x_plus_ix_forward_is_finite(self):
        """Exercise the mentor-confirmed X+iX feature convention at q=1/4."""
        torch.manual_seed(0)

        model = MSGNN_link_prediction(
            num_features=2,
            hidden=16,
            q=0.25,
            K=1,
            label_dim=4,
            activation=True,
            trainable_q=False,
            layer=2,
            dropout=0.5,
            normalization="sym",
            cached=False,
            conv_bias=True,
            absolute_degree=True,
        )
        model.eval()

        x_real = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float,
        )
        x_imag = x_real.clone()
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_weight = torch.tensor([1.0], dtype=torch.float)
        query_edges = torch.tensor([[0, 1]], dtype=torch.long)

        with torch.no_grad():
            log_prob = model(
                x_real,
                x_imag,
                edge_index,
                query_edges,
                edge_weight,
            )

        self.assertEqual(tuple(log_prob.shape), (1, 4))
        self.assertTrue(bool(torch.isfinite(log_prob).all()))
        self.assertTrue(
            torch.allclose(
                log_prob.exp().sum(dim=1),
                torch.ones(1),
                atol=1e-6,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
