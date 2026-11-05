"""6.1 MoE 层（评测来自仓库的 `mixture-of-experts` 题）。讲义见在线教程第 6 节。
核验：python study/board.py check 6.1
注意：辅助损失按 Switch Transformer 的写法 aux = N·Σ f_i·P_i（均衡时为 1），讲义里有推导。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    d_model: int = 64
    d_ff: int = 128
    aux_loss_weight: float = 0.01


class Expert(nn.Module):
    """A simple feed-forward expert network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO: Create a two-layer FFN: Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pass x through the FFN
        ...


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k gating and load balancing."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # TODO: Create a list of Expert modules (use nn.ModuleList)

        # TODO: Create the router: a Linear layer (d_model -> num_experts)

        ...

    def forward(self, x: torch.Tensor):
        """
        Forward pass with top-k expert routing.

        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: scalar load balancing loss
        """
        batch_size, seq_len, d_model = x.shape
        # Flatten to (num_tokens, d_model) for routing
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.size(0)

        # TODO: Step 1 - Compute router logits: (num_tokens, num_experts)

        # TODO: Step 2 - Get top-k experts per token
        #       Use torch.topk on gate_logits to get top_k_values and top_k_indices

        # TODO: Step 3 - Normalize gate values (softmax over the top-k values only)

        # TODO: Step 4 - Dispatch tokens to experts and combine outputs
        #       Initialize output = zeros_like(x_flat)
        #       For each expert i:
        #         - Find which (token, slot) pairs selected expert i
        #         - Get those tokens, run through expert i
        #         - Multiply by gate weight, add to output

        # TODO: Step 5 - Compute auxiliary load balancing loss
        #       router_probs = softmax(gate_logits)  -- over ALL experts
        #       f_i = fraction of tokens routed to expert i (from top_k_indices)
        #       p_i = mean router probability for expert i
        #       aux_loss = num_experts * sum(f_i * p_i)

        # TODO: Reshape output back to (batch_size, seq_len, d_model)

        ...
