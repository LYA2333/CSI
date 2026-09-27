"""Module-level tests for the current CSI Conv1D + graph prototype.

Run with:

    python -m unittest -v test_csi.py

These tests validate the neural update blocks only. They do not measure
code-generation pass@1 because the current prototype has no LLM encoder,
code decoder, trained CSI checkpoint, or production runner yet.
"""

from __future__ import annotations

import unittest

import torch

from conv_graph_mas_demo import (
    CSIGCN,
    GraphConvolutionUpdate,
    IntraAgentConv,
    SharedLatentAdapter,
)


class SharedLatentAdapterTests(unittest.TestCase):
    def test_heterogeneous_agents_share_one_latent_space(self) -> None:
        torch.manual_seed(1)
        adapter = SharedLatentAdapter([12, 16, 20], d_model=8)
        hidden_states = [
            torch.randn(2, 5, 12),
            torch.randn(2, 5, 16),
            torch.randn(2, 5, 20),
        ]

        shared = adapter.encode(hidden_states)
        decoded = adapter.decode(shared.mean(dim=2))

        self.assertEqual(tuple(shared.shape), (2, 3, 5, 8))
        self.assertEqual([tuple(x.shape) for x in decoded], [(2, 12), (2, 16), (2, 20)])
        self.assertTrue(torch.isfinite(shared).all().item())


class CSIUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.features = torch.randn(2, 3, 4, 32, 16)
        self.adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

    def test_dilated_conv_stack_runs_over_five_feature_types(self) -> None:
        intra = IntraAgentConv(
            d_model=16,
            num_feature_types=5,
            kernel_size=3,
            dilations=(1, 2),
        )

        self.assertEqual(len(intra.conv_blocks), 2)
        self.assertEqual(intra.dilations, (1, 2))
        self.assertEqual(intra.conv_axis, "feature_types")
        self.assertEqual(intra.theoretical_receptive_field, 7)
        self.assertEqual(intra.receptive_field, 5)
        self.assertEqual(
            [block.depthwise.dilation for block in intra.conv_blocks],
            [(1,), (2,)],
        )

        first_block_input: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs) -> None:
            first_block_input.append(tuple(inputs[0].shape))

        handle = intra.conv_blocks[0].register_forward_pre_hook(capture_shape)
        output = intra(torch.randn(2, 3, 5, 32, 16))
        handle.remove()

        # B*N*S sequences, each containing the K=5 feature representations.
        self.assertEqual(first_block_input, [(2 * 3 * 32, 5, 16)])
        self.assertEqual(tuple(output.shape), (2, 3, 16))

    def test_dilation_cannot_skip_beyond_feature_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature-axis length"):
            IntraAgentConv(
                d_model=16,
                num_feature_types=5,
                kernel_size=3,
                dilations=(1, 5),
            )

    def test_forward_shapes_and_symmetric_gcn_normalization(self) -> None:
        model = CSIGCN(d_model=16)
        model.train()

        output = model(self.features, max_steps=3, adjacency=self.adjacency)
        messages = output["messages"]
        message_history = output["message_history"]
        normalized = output["adjacency"]

        self.assertEqual(tuple(messages.shape), (2, 3, 16))
        self.assertEqual(len(message_history), 3)
        self.assertTrue(all(tuple(item.shape) == (2, 3, 16) for item in message_history))
        torch.testing.assert_close(messages, message_history[-1])
        self.assertEqual(tuple(normalized.shape), (2, 3, 3))
        self.assertEqual(output["steps"], 3)

        adjacency_with_self_loops = self.adjacency + torch.eye(3)
        degree = adjacency_with_self_loops.sum(dim=-1)
        expected = (
            degree.rsqrt().unsqueeze(-1)
            * adjacency_with_self_loops
            * degree.rsqrt().unsqueeze(-2)
        )
        torch.testing.assert_close(
            normalized,
            expected.unsqueeze(0).expand(2, -1, -1),
        )

    def test_backward_produces_finite_gradients(self) -> None:
        model = CSIGCN(d_model=16)
        output = model(self.features, max_steps=2, adjacency=self.adjacency)
        loss = output["messages"].square().mean()

        loss.backward()

        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(gradients), 0)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))

    def test_missing_edges_remain_zero_after_normalization(self) -> None:
        normalized = GraphConvolutionUpdate.normalize_adjacency(
            self.adjacency,
            batch_size=2,
            num_agents=3,
            device=self.features.device,
            dtype=self.features.dtype,
        )

        self.assertTrue((normalized[:, 0, 2] == 0).all().item())
        self.assertTrue((normalized[:, 2, 0] == 0).all().item())

    def test_eval_can_stop_early(self) -> None:
        model = CSIGCN(d_model=16)
        model.eval()

        with torch.no_grad():
            output = model(
                self.features,
                max_steps=8,
                tolerance=1e20,
                adjacency=self.adjacency,
            )

        self.assertEqual(output["steps"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
