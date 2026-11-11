"""ConvRSI v3: configurable convolution padding and optional broadcast residual."""
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def parse_conv_padding(value):
    """CLI/config value: 'same' or a nonnegative symmetric integer padding."""
    if isinstance(value, str) and value.strip().lower() == 'same':
        return 'same'
    if isinstance(value, bool):
        raise ValueError('conv-padding must be same or a nonnegative integer')
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str):
        try:
            number = int(value)
        except ValueError:
            number = -1
        if number >= 0:
            return number
    raise ValueError('conv-padding must be same or a nonnegative integer')


def conv_padding_sides(kernel_size, dilation, padding):
    padding = parse_conv_padding(padding)
    if kernel_size < 1 or dilation < 1:
        raise ValueError('kernel_size and dilation must be positive')
    if padding == 'same':
        total = dilation * (kernel_size-1)
        return total//2, total-total//2
    return padding, padding


def validate_conv_layout(num_features, kernel_size, dilations, padding='same', residual=True):
    """Validate each block's actual input length before loading the backbones."""
    if not isinstance(residual, bool):
        raise ValueError('conv-residual must be a boolean')
    if num_features < 1 or kernel_size < 1:
        raise ValueError('num_feature_types and kernel_size must be positive')
    if not dilations or any(d < 1 for d in dilations):
        raise ValueError('dilations must contain positive integers')
    if any(d >= num_features for d in dilations):
        raise ValueError(f'Each dilation must be smaller than the feature-axis length ({num_features})')
    lengths = []
    input_length = num_features
    for index,dilation in enumerate(dilations):
        left,right = conv_padding_sides(kernel_size,dilation,padding)
        effective_kernel = dilation*(kernel_size-1)+1
        length = input_length+left+right-effective_kernel+1
        if length < 1:
            raise ValueError(f'Conv block {index+1}: effective kernel {effective_kernel} exceeds padded '
                             f'feature length {input_length+left+right}; adjust --kernel-size, --dilations or --conv-padding')
        if residual and length not in (1,input_length):
            raise ValueError(f'Conv block {index+1}: padding={padding!r}, kernel_size={kernel_size}, '
                             f'dilation={dilation} gives {length} positions for {input_length} input features. '
                             f'The preserved broadcast residual requires 1 or {input_length} positions')
        lengths.append(length)
        if not residual:
            input_length = length
    return lengths


class DilatedDepthwiseConvBlock(nn.Module):
    """Depthwise Conv1D, GELU and LayerNorm, with an optional input skip."""

    def __init__(self, d_model: int, kernel_size: int, dilation: int, padding='same',
                 residual=True, *, pointwise=False) -> None:
        super().__init__()
        if not isinstance(residual, bool):
            raise ValueError('conv-residual must be a boolean')
        if not isinstance(pointwise, bool):
            raise ValueError('conv_pointwise must be a boolean')
        self.residual = residual
        self.padding_mode = parse_conv_padding(padding)
        left,right = conv_padding_sides(kernel_size,dilation,self.padding_mode)
        self.padding_sides = (left,right)
        self.explicit_padding = (left,right) if left != right else None
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=0 if self.explicit_padding is not None else left,
            dilation=dilation,
            groups=d_model,
        )
        # Legacy checkpoints used an additional 1x1 channel projection.
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1) if pointwise else None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        """Update a sequence with shape [batch, length, D]."""
        conv_input = sequence.transpose(1, 2)
        if self.explicit_padding is not None:
            conv_input = F.pad(conv_input,self.explicit_padding)
        conv_output = F.gelu(self.depthwise(conv_input))
        if self.pointwise is not None:
            conv_output = self.pointwise(conv_output)
        if self.residual and conv_output.shape[-1] not in (1,sequence.shape[1]):
            raise ValueError('Convolution output must have one position or match the input length '
                             'for the preserved broadcast residual')
        # A singleton convolution is intentionally broadcast to every original
        # feature. Keep its original mask and gated pooling afterward.
        output = conv_output.transpose(1, 2)
        return self.norm(sequence + output if self.residual else output)

    def output_mask(self, mask: Tensor) -> Tensor:
        """A non-residual output is valid when its window reads any valid input.

        Missing features and padding are not observations. In the residual path,
        output positions retain the original feature identities and masks.
        """
        if self.residual:
            return mask
        left,right = self.padding_sides
        windows = F.pad(mask[:, None].float(), (left,right))
        kernel = windows.new_ones(1,1,self.depthwise.kernel_size[0])
        counts = F.conv1d(windows, kernel, dilation=self.depthwise.dilation[0])
        return counts[:, 0].gt(0)


class IntraAgentConv(nn.Module):
    """Produce one agent's next message from K representations.

    Each feature keeps its identity until Conv1D mixes the ordered feature axis
    K (experience CSI): task -> role -> system -> experience -> message;
    legacy four-part CSI omits experience. With return_sequence,
    the gated pool reduces K only, retaining masked messages [B,L,D]. The
    legacy standalone demo defaults to pooling both K and L into [B,D].

    The module also accepts [B, N, K, S, D] for standalone experiments, but a
    :class:`CSIGCN` owns one independent instance per agent and calls each one
    with [B, K, S, D].
    """

    def __init__(
        self,
        d_model: int,
        num_feature_types: int = 5,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        padding='same',
        residual=True,
        cross_feature_fusion=False,
        conv_pointwise=False,
    ) -> None:
        super().__init__()
        self.conv_padding = parse_conv_padding(padding)
        self.conv_residual = residual
        self.conv_output_lengths = validate_conv_layout(num_feature_types,kernel_size,dilations,self.conv_padding,residual)

        self.num_feature_types = num_feature_types
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.theoretical_receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.receptive_field = min(
            num_feature_types,
            self.theoretical_receptive_field,
        )
        self.conv_axis = "feature_types"
        if not isinstance(cross_feature_fusion, bool):
            raise ValueError('cross_feature_fusion must be a boolean')
        self.cross_feature_fusion = cross_feature_fusion
        # Only historical checkpoints own the flattened all-feature FC.
        # Fresh models have neither its parameters nor its forward computation.
        if cross_feature_fusion:
            fusion_width = num_feature_types * d_model
            self.fusion_norm = nn.LayerNorm(fusion_width)
            self.feature_fusion = nn.Linear(fusion_width, fusion_width)
        self.type_embedding = nn.Parameter(
            torch.randn(num_feature_types, 1, d_model) * 0.02
        )
        self.conv_blocks = nn.ModuleList(
            DilatedDepthwiseConvBlock(d_model, kernel_size, dilation, self.conv_padding, residual,
                                      pointwise=conv_pointwise)
            for dilation in self.dilations
        )
        self.pool_gate = nn.Linear(d_model, 1)
        self.output = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self, features: Tensor, *, return_sequence: bool = False,
        token_mask: Tensor | None = None,
        feature_mask: Tensor | None = None,
    ) -> Tensor:
        """Mix feature types; optionally retain the masked token sequence."""
        single_agent = features.ndim == 4
        if single_agent:
            features = features.unsqueeze(1)
        if features.ndim != 5:
            raise ValueError(
                "features must have shape [B,K,S,D] or [B,N,K,S,D]"
            )
        batch, agents, feature_types, tokens, d_model = features.shape
        if feature_types != self.num_feature_types:
            raise ValueError(
                f"Expected {self.num_feature_types} feature types, got {feature_types}"
            )

        valid_features = None
        if feature_mask is not None:
            valid_features = feature_mask.unsqueeze(1) if single_agent else feature_mask
            if tuple(valid_features.shape) != (batch, agents, feature_types, tokens):
                raise ValueError("feature_mask must match [B,K,L] or [B,N,K,L]")
            valid_features = valid_features.to(device=features.device, dtype=torch.bool)
            features = features * valid_features.unsqueeze(-1)

        x = features
        if self.cross_feature_fusion:
            # Restore the original dense mixing exactly for old checkpoints.
            flattened = features.permute(0, 1, 3, 2, 4).contiguous().reshape(
                batch, agents, tokens, feature_types * d_model)
            fused = flattened + F.gelu(self.feature_fusion(self.fusion_norm(flattened)))
            x = fused.reshape(batch, agents, tokens, feature_types, d_model)
            x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x + self.type_embedding.view(1, 1, feature_types, 1, d_model)
        if valid_features is not None:
            x = x * valid_features.unsqueeze(-1)
        # Preserve each latent-token index and expose the K feature types as
        # the Conv1D sequence: [B,N,K,S,D] -> [B*N*S,K,D].
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch * agents * tokens, feature_types, d_model)
        conv_mask = (None if valid_features is None else
                     valid_features.permute(0, 1, 3, 2).reshape(-1, feature_types))
        if conv_mask is None and not self.conv_residual:
            conv_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        # Each block now mixes neighbouring feature types rather than
        # neighbouring latent-token positions.
        for block in self.conv_blocks:
            x = block(x)
            if conv_mask is not None:
                conv_mask = block.output_mask(conv_mask)
                x = x * conv_mask.unsqueeze(-1)

        output_features = x.shape[1]
        x = x.reshape(batch, agents, tokens, output_features, d_model)
        pool_mask = (None if conv_mask is None else
                     conv_mask.reshape(batch, agents, tokens, output_features))
        if return_sequence:
            # Pool only K. Never pool away the L axis of a real message.
            scores = self.pool_gate(x).squeeze(-1)
            if pool_mask is not None:
                scores = scores.masked_fill(~pool_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            if pool_mask is not None:
                weights = weights * pool_mask
            pooled = (weights.unsqueeze(-1) * x).sum(dim=3)
            message = pooled + self.output(pooled)
            if pool_mask is not None:
                message = message * pool_mask.any(dim=-1).unsqueeze(-1)
            if token_mask is not None:
                mask = token_mask.unsqueeze(1) if single_agent else token_mask
                if tuple(mask.shape) != (batch, agents, tokens):
                    raise ValueError("token_mask must match the message token axes")
                message = message * mask.to(message.device).unsqueeze(-1)
            return message[:, 0] if single_agent else message
        x = x.permute(0, 1, 3, 2, 4).contiguous()

        # Learn which mixed feature/token positions matter for the current
        # agent and task.
        x = x.reshape(batch, agents, output_features * tokens, d_model)
        scores = self.pool_gate(x).squeeze(-1)
        flat_mask = (None if pool_mask is None else
                     pool_mask.permute(0,1,3,2).reshape(batch, agents, -1))
        if flat_mask is not None:
            scores = scores.masked_fill(~flat_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if flat_mask is not None:
            weights = weights * flat_mask
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=2)
        message = pooled + self.output(pooled)
        if flat_mask is not None:
            message = message * flat_mask.any(-1).unsqueeze(-1)
        return message[:, 0] if single_agent else message


class GraphConvolutionUpdate(nn.Module):
    """Exchange messages with a standard symmetrically normalized GCN layer.

    ``adjacency[i, j]`` means that agent i aggregates a message from agent j.
    The graph is fixed for a forward pass and may be supplied by the caller.
    When it is omitted, every agent communicates with every other agent.
    """

    def __init__(self, d_model: int, *, residual: bool = False) -> None:
        super().__init__()
        if not isinstance(residual, bool):
            raise ValueError('gcn_residual must be a boolean')
        # The input skip is restored only when loading a legacy configuration.
        self.residual = residual
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
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Return normalized [B,N,N], or active-token graphs [B,L,N,N]."""
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
        if node_mask is not None:
            if node_mask.ndim != 3 or tuple(node_mask.shape[:2]) != (batch_size, num_agents):
                raise ValueError("node_mask must have shape [B,N,L]")
            valid = node_mask.to(device=device, dtype=dtype).transpose(1, 2)
            # [B,L,N,N]: padded senders/receivers are not graph nodes, and
            # normalization uses the active graph at each token position.
            adjacency_with_self_loops = (
                adjacency_with_self_loops[:, None]
                * valid.unsqueeze(-1) * valid.unsqueeze(-2)
            )
        # Valid nodes have a self-loop (degree >= 1); invalid nodes have degree
        # zero. Clamp to 1 so even float16 avoids inf * 0 on padded positions.
        degree = adjacency_with_self_loops.sum(dim=-1).clamp_min(1)
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
        *,
        node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update vectors or masked [B,N,L,D] sequences at each token position."""
        if node_states.ndim not in (3, 4):
            raise ValueError("node_states must have shape [B,N,D] or [B,N,L,D]")
        batch_size, num_agents = node_states.shape[:2]
        if node_states.ndim == 4:
            if node_mask is None:
                node_mask = torch.ones(node_states.shape[:-1], device=node_states.device, dtype=torch.bool)
            if tuple(node_mask.shape) != tuple(node_states.shape[:-1]):
                raise ValueError("node_mask must match [B,N,L]")
        elif node_mask is not None:
            raise ValueError("node_mask is only used by sequence node_states")
        normalized_adjacency = self.normalize_adjacency(
            adjacency,
            batch_size=batch_size,
            num_agents=num_agents,
            device=node_states.device,
            dtype=node_states.dtype,
            node_mask=node_mask,
        )

        transformed = self.projection(node_states)
        if node_states.ndim == 4:
            transformed = transformed * node_mask.to(node_states.device).unsqueeze(-1)
            propagated = torch.matmul(normalized_adjacency, transformed.transpose(1, 2)).transpose(1, 2) + self.bias
        else:
            propagated = torch.matmul(normalized_adjacency, transformed) + self.bias

        activated = F.gelu(propagated)
        updated = self.norm(node_states + activated if self.residual else activated)
        if node_mask is not None:
            updated = updated * node_mask.to(node_states.device).unsqueeze(-1)
        return updated, normalized_adjacency

