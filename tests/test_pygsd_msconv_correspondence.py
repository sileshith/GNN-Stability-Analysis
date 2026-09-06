"""
Gate R1 — Installed PyGSD MSConv correspondence checks.

These tests document the behavior of the installed PyGSD 1.1.1 package used
for E001. They do not modify PyGSD and do not assert that an upstream defect
has been confirmed.

The q=0 invariant check is marked as an expected failure while the intended
MSConv semantics remain unresolved. If a future implementation satisfies the
invariant, unittest will report an unexpected success, prompting review of
this audit record.
"""

import unittest
from importlib.metadata import version

import torch
from torch_geometric_signed_directed.nn.general.MSConv import MSConv
from torch_geometric_signed_directed.nn.general.MSGNN import (
    MSGNN_link_prediction,
)


@unittest.skipUnless(version("torch-geometric-signed-directed") == "1.1.1", "Historical PyGSD 1.1.1 audit")
class TestInstalledMSConvCorrespondence(unittest.TestCase):
    """Document installed MSConv behavior relevant to E001."""

    @staticmethod
    def make_first_order_layer(q: float) -> MSConv:
        """Create a deterministic layer with only the first-order term active."""
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

    def test_effective_flow_is_source_to_target(self):
        """Record the effective MessagePassing flow in the installed package."""
        layer = self.make_first_order_layer(q=0.125)
        self.assertEqual(layer.flow, "source_to_target")

    def test_e001_layer_inventory(self):
        """Confirm that the accepted E001 model uses two installed MSConv layers."""
        model = MSGNN_link_prediction(
            num_features=2,
            hidden=16,
            q=0.125,
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
            all(cheb.flow == "source_to_target" for cheb in model.Chebs)
        )
        self.assertEqual(tuple(model.Chebs[0].weight.shape), (2, 2, 16))
        self.assertEqual(tuple(model.Chebs[1].weight.shape), (2, 16, 16))

    @unittest.expectedFailure
    def test_q_zero_real_input_should_have_zero_imaginary_output(self):
        """
        Test the ordinary complex-operator invariant at q=0.

        At q=0 the magnetic operator is real. With a purely real input and
        real learned weights, ordinary complex multiplication requires the
        imaginary output to be zero.

        This is an expected failure for the installed PyGSD 1.1.1 behavior
        observed during Gate R1.
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
            msg=(
                "At q=0, a purely real input unexpectedly produced a "
                "nonzero imaginary output."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
