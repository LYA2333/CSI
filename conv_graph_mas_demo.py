"""A small prototype for CSI intra-agent Conv1D + inter-agent GCN updates.

Tensor conventions
------------------
B: batch size
N: number of agents
K: number of feature types (task, role, experience, prompt, message)
S: latent tokens per feature
D: shared latent dimension

This file intentionally does not depend on a particular LLM. In a real system,
the feature encoders would turn text/KV/hidden states into tensors before this
module, and ``SharedLatentAdapter`` would bridge each LLM's hidden dimension.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SharedLatentAdapter(nn.Module):
    """Project heterogeneous agent hidden states into one shared latent space."""

    def __init__(self, agent_hidden_dims: Sequence[int], d_model: int) -> None:
        super().__init__()
        self.to_shared = nn.ModuleList(
            nn.Linear(hidden_dim, d_model) for hidden_dim in agent_hidden_dims
        )
        self.from_shared = nn.ModuleList(
            nn.Linear(d_model, hidden_dim) for hidden_dim in agent_hidden_dims
        )

    def encode(self, hidden_states: Sequence[Tensor]) -> Tensor:
        """Return shared tokens with shape [B, N, S, D]."""
        if len(hidden_states) != len(self.to_shared):
            raise ValueError("Need one hidden-state tensor for every agent")
        projected = [layer(h) for layer, h in zip(self.to_shared, hidden_states)]
        return torch.stack(projected, dim=1)

    def decode(self, messages: Tensor) -> list[Tensor]:
        """Map [B, N, D] messages back to each agent's input dimension."""
        return [layer(messages[:, i]) for i, layer in enumerate(self.from_shared)]


class DilatedDepthwiseConvBlock(nn.Module):
    """One residual depthwise-separable Conv1D block over an ordered axis."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        if dilation < 1:
            raise ValueError("dilation must be positive")

        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        """Update a sequence with shape [batch, length, D]."""
        conv_input = sequence.transpose(1, 2)
        conv_output = self.pointwise(F.gelu(self.depthwise(conv_input)))
        return self.norm(sequence + conv_output.transpose(1, 2))


class IntraAgentConv(nn.Module):
    """Fuse typed features inside each agent.

    For every latent-token position, Conv1D runs along the ordered feature axis
    K: task -> role -> experience -> prompt -> message. This makes feature
    interaction an explicit part of every convolution block. The resulting
    feature-token grid is fused afterwards by gated pooling.
    """

    def __init__(
        self,
        d_model: int,
        num_feature_types: int = 5,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__()
        if num_feature_types < 1:
            raise ValueError("num_feature_types must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve feature length")
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        if any(dilation >= num_feature_types for dilation in dilations):
            raise ValueError(
                "Each dilation must be smaller than the feature-axis length "
                f"({num_feature_types}); larger values only convolve padding"
            )

        self.num_feature_types = num_feature_types
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.theoretical_receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.receptive_field = min(
            num_feature_types,
            self.theoretical_receptive_field,
        )
        self.conv_axis = "feature_types"
        self.type_embedding = nn.Parameter(
            torch.randn(num_feature_types, 1, d_model) * 0.02
        )
        self.conv_blocks = nn.ModuleList(
            DilatedDepthwiseConvBlock(d_model, kernel_size, dilation)
            for dilation in self.dilations
        )
        self.pool_gate = nn.Linear(d_model, 1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Convert [B, N, K, S, D] features to [B, N, D] local states."""
        batch, agents, feature_types, tokens, d_model = features.shape
        if feature_types != self.num_feature_types:
            raise ValueError(
                f"Expected {self.num_feature_types} feature types, got {feature_types}"
            )

        x = features + self.type_embedding.view(1, 1, feature_types, 1, d_model)
        # Preserve each latent-token index and expose the five feature types as
        # the Conv1D sequence: [B,N,K,S,D] -> [B*N*S,K,D].
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch * agents * tokens, feature_types, d_model)

        # Each block now mixes neighbouring feature types rather than
        # neighbouring latent-token positions.
        for block in self.conv_blocks:
            x = block(x)

        x = x.reshape(batch, agents, tokens, feature_types, d_model)
        x = x.permute(0, 1, 3, 2, 4).contiguous()

        # Learn which mixed feature/token positions matter for the current
        # agent and task.
        x = x.reshape(batch, agents, feature_types * tokens, d_model)
        weights = torch.softmax(self.pool_gate(x).squeeze(-1), dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=2)
        return pooled + self.output(pooled)


class GraphConvolutionUpdate(nn.Module):
    """Exchange messages with a standard symmetrically normalized GCN layer.

    ``adjacency[i, j]`` means that agent i aggregates a message from agent j.
    The graph is fixed for a forward pass and may be supplied by the caller.
    When it is omitted, every agent communicates with every other agent.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def normalize_adjacency(
        adjacency: Tensor | None,
        *,
        batch_size: int,
        num_agents: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return D^-1/2 (A + I) D^-1/2 with shape [B, N, N]."""
        if adjacency is None:
            # The diagonal is removed here because self-loops are added below.
            adjacency = torch.ones(num_agents, num_agents, device=device, dtype=dtype)
            adjacency.fill_diagonal_(0)
        else:
            adjacency = adjacency.to(device=device, dtype=dtype)

        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        elif adjacency.ndim == 3 and adjacency.shape[0] == 1:
            adjacency = adjacency.expand(batch_size, -1, -1)

        expected_shape = (batch_size, num_agents, num_agents)
        if tuple(adjacency.shape) != expected_shape:
            raise ValueError(
                f"Expected adjacency shape {expected_shape}, got {tuple(adjacency.shape)}"
            )
        if not torch.isfinite(adjacency).all().item() or (adjacency < 0).any().item():
            raise ValueError("adjacency must contain finite non-negative weights")

        identity = torch.eye(num_agents, device=device, dtype=dtype).unsqueeze(0)
        adjacency_with_self_loops = adjacency + identity
        degree = adjacency_with_self_loops.sum(dim=-1).clamp_min(1e-12)
        inverse_sqrt_degree = degree.rsqrt()
        return (
            inverse_sqrt_degree.unsqueeze(-1)
            * adjacency_with_self_loops
            * inverse_sqrt_degree.unsqueeze(-2)
        )

    def forward(
        self,
        node_states: Tensor,
        adjacency: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update [B, N, D] node states and return normalized [B, N, N] A."""
        batch_size, num_agents, _ = node_states.shape
        normalized_adjacency = self.normalize_adjacency(
            adjacency,
            batch_size=batch_size,
            num_agents=num_agents,
            device=node_states.device,
            dtype=node_states.dtype,
        )

        transformed = self.projection(node_states)
        propagated = torch.matmul(normalized_adjacency, transformed) + self.bias

        # A residual connection mitigates over-smoothing across repeated rounds.
        updated = self.norm(node_states + F.gelu(propagated))
        return updated, normalized_adjacency


class CSIGCN(nn.Module):
    """Alternate CSI intra-agent Conv1D and inter-agent GCN updates."""

    def __init__(
        self,
        d_model: int,
        num_static_features: int = 4,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__()
        # Static features + the evolving message feature.
        self.num_static_features = num_static_features
        self.intra = IntraAgentConv(
            d_model,
            num_static_features + 1,
            kernel_size=kernel_size,
            dilations=dilations,
        )
        self.inter = GraphConvolutionUpdate(d_model)

    def forward(
        self,
        static_features: Tensor,
        initial_messages: Tensor | None = None,
        max_steps: int = 4,
        tolerance: float = 1e-3,
        adjacency: Tensor | None = None,
    ) -> dict[str, Tensor | int | list[Tensor]]:
        """Run recursive updates.

        ``static_features`` has shape [B, N, 4, S, D]. During training all
        ``max_steps`` are unrolled. During evaluation, inference stops early
        when every message changes less than ``tolerance``.
        """
        batch, agents, feature_types, latent_tokens, d_model = static_features.shape
        if feature_types != self.num_static_features:
            raise ValueError(
                f"Expected {self.num_static_features} static features, got {feature_types}"
            )
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if initial_messages is None:
            messages = static_features.new_zeros(batch, agents, d_model)
        else:
            messages = initial_messages

        normalized_adjacency = static_features.new_zeros(batch, agents, agents)
        relative_change = static_features.new_full((batch, agents), float("inf"))
        message_history: list[Tensor] = []

        for step in range(1, max_steps + 1):
            message_tokens = messages[:, :, None, None, :].expand(
                -1, -1, 1, latent_tokens, -1
            )
            all_features = torch.cat([static_features, message_tokens], dim=2)
            local_states = self.intra(all_features)
            updated, normalized_adjacency = self.inter(local_states, adjacency)

            relative_change = (updated - messages).norm(dim=-1) / (
                messages.norm(dim=-1) + 1e-6
            )
            messages = updated
            message_history.append(messages)

            # The hard stopping decision is used only for inference. Training
            # keeps a fixed differentiable unroll; ACT could replace this later.
            if not self.training and torch.all(relative_change < tolerance).item():
                break

        return {
            "messages": messages,
            "message_history": message_history,
            "adjacency": normalized_adjacency,
            "relative_change": relative_change,
            "steps": step,
        }


# Backward-compatible alias for earlier experiments that imported this name.
ConvGraphMAS = CSIGCN


def demo() -> None:
    """Run one optimization step and one adaptive inference pass."""
    torch.manual_seed(7)
    batch, agents, features, tokens, d_model = 3, 3, 4, 32, 32

    # These stand in for encoded task, role, experience, and prompt tokens.
    static_features = torch.randn(batch, agents, features, tokens, d_model)
    targets = torch.tensor([0, 2, 1])
    collaboration_graph = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    model = CSIGCN(d_model=d_model)
    answer_head = nn.Linear(d_model, 3)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(answer_head.parameters()), lr=3e-4
    )

    model.train()
    output = model(static_features, max_steps=4, adjacency=collaboration_graph)
    executor_message = output["messages"][:, -1]  # Agent 3 is the executor.
    loss = F.cross_entropy(answer_head(executor_message), targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"training loss: {loss.item():.4f}")
    print(
        "intra-agent feature-axis dilations / effective receptive field: "
        f"{model.intra.dilations} / {model.intra.receptive_field}"
    )
    print("normalized GCN graph for sample 0 (row=receiver, col=sender):")
    print(output["adjacency"][0].detach().round(decimals=3))

    model.eval()
    with torch.no_grad():
        inference = model(
            static_features,
            max_steps=8,
            tolerance=5e-2,
            adjacency=collaboration_graph,
        )
    print(f"inference steps: {inference['steps']}")
    print(f"max relative change: {inference['relative_change'].max().item():.4f}")


if __name__ == "__main__":
    demo()
