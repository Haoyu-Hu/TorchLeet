"""5.4 投机解码（评测来自仓库的 `speculative-decoding` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DraftModel(nn.Module):
    """
    Small, fast model that proposes candidate tokens.
    Simple architecture: Embedding -> Linear -> ReLU -> Linear -> logits
    """
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # TODO: Define layers
        # - Embedding(vocab_size, d_model)
        # - Linear(d_model, d_model)
        # - Linear(d_model, vocab_size)  (output logits)
        ...
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) token indices
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # TODO: Forward pass
        ...


class TargetModel(nn.Module):
    """
    Larger, slower model that produces the "true" distribution.
    Architecture: Embedding -> 3x(Linear -> ReLU) -> Linear -> logits
    """
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # TODO: Define layers (larger/deeper than DraftModel)
        ...
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) token indices
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # TODO: Forward pass
        ...


def standard_decode(model, input_ids, num_tokens, temperature=1.0):
    """
    Standard autoregressive decoding from the target model.
    Generates one token at a time.
    
    Args:
        model: The target model
        input_ids: (1, prompt_len) initial token indices
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        generated_ids: (1, prompt_len + num_tokens) full sequence
    """
    # TODO: Implement standard autoregressive decoding
    # For each step:
    #   1. Forward pass through model
    #   2. Get logits for last position
    #   3. Apply temperature and softmax to get probabilities
    #   4. Sample next token
    #   5. Append to sequence
    ...


def speculative_decode(draft_model, target_model, input_ids, K=5, temperature=1.0):
    """
    Speculative decoding: draft model proposes K tokens, target model verifies.
    
    Args:
        draft_model: Small, fast model for proposing tokens
        target_model: Large, accurate model for verification
        input_ids: (1, prompt_len) initial token indices
        K: Number of speculative tokens to propose
        temperature: Sampling temperature
        
    Returns:
        accepted_tokens: List of accepted token ids
        num_accepted: Number of draft tokens accepted (out of K)
    """
    # TODO: Implement speculative decoding
    #
    # Step 1: Draft phase
    #   - Use draft_model to autoregressively generate K tokens
    #   - Store each draft token and its probability under the draft model
    #
    # Step 2: Verification phase
    #   - Run target_model on the full sequence (prompt + K draft tokens) in ONE forward pass
    #   - Extract target probabilities for each draft token position
    #
    # Step 3: Rejection sampling
    #   For each draft token i (in order):
    #     - p_draft = draft probability of chosen token
    #     - p_target = target probability of chosen token
    #     - Draw r ~ Uniform(0, 1)
    #     - If r < min(1, p_target / p_draft): ACCEPT
    #     - Else: REJECT
    #       - Sample replacement from: normalize(max(0, p_target - p_draft))
    #       - Return accepted tokens so far + replacement token
    #
    # Step 4: If all K tokens accepted, sample one bonus token from target at position K+1
    #
    ...
