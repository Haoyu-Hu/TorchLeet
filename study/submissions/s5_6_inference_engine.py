"""5.6 推理引擎（评测来自仓库的 `inference-engine` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class SimpleTokenizer:
    """
    Character-level tokenizer.
    
    Special tokens:
        0 = <pad>
        1 = <eos>
    Regular characters: ord(c) + 2
    """
    
    def __init__(self):
        # TODO: Define pad_id, eos_id, and compute vocab_size
        # vocab_size should cover all printable ASCII + 2 special tokens
        ...
    
    def encode(self, text):
        """Convert text to list of token IDs."""
        # TODO: Map each character to ord(c) + 2
        ...
    
    def decode(self, ids):
        """Convert list of token IDs back to text. Skip pad/eos."""
        # TODO: Map each ID back to chr(id - 2), skip special tokens
        ...


class KVCache:
    """KV Cache for a single attention layer."""
    
    def __init__(self):
        # TODO: Initialize empty cache
        ...
    
    def update(self, new_k, new_v):
        """Append new K/V and return full cached K/V."""
        # TODO
        ...
    
    def reset(self):
        # TODO
        ...


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV cache support."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Initialize Q, K, V, output projections
        # Store d_model, num_heads, d_head
        ...
    
    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (B, S, d_model)
            kv_cache: Optional KVCache
        Returns:
            output: (B, S, d_model)
        """
        # TODO: Implement attention with optional KV cache
        # Same logic as Question 12
        ...


class FeedForward(nn.Module):
    """Simple feed-forward network: Linear -> GELU -> Linear."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Two linear layers with GELU activation in between
        ...
    
    def forward(self, x):
        # TODO
        ...


class TransformerBlock(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm FFN."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # TODO: Initialize LayerNorms, MultiHeadAttention, FeedForward
        ...
    
    def forward(self, x, kv_cache=None):
        """
        Pre-norm transformer block:
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))
        """
        # TODO
        ...


class MiniTransformer(nn.Module):
    """
    Minimal 2-layer transformer for text generation.
    
    Components:
    - Token embedding + positional embedding
    - N transformer blocks (each with attention + FFN)
    - Final layer norm
    - Output projection to vocab logits
    """
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        # TODO: Initialize embeddings, transformer blocks, final norm, output projection
        ...
    
    def forward(self, input_ids, kv_caches=None):
        """
        Args:
            input_ids: (B, S) token indices
            kv_caches: Optional list of KVCache (one per layer)
        Returns:
            logits: (B, S, vocab_size)
        """
        # TODO:
        # 1. Token embedding + positional embedding
        #    - For cached generation, position offset = cache length
        # 2. Pass through each transformer block (with its KV cache)
        # 3. Final layer norm
        # 4. Output projection
        ...


def top_p_sample(logits, temperature=1.0, top_p=0.9):
    """
    Top-p (nucleus) sampling.
    
    Args:
        logits: (B, vocab_size) raw logits for next token
        temperature: Sampling temperature
        top_p: Cumulative probability threshold
    
    Returns:
        sampled_tokens: (B, 1) sampled token IDs
    """
    # TODO:
    # 1. Apply temperature scaling
    # 2. Compute softmax probabilities
    # 3. Sort probabilities in descending order
    # 4. Compute cumulative sum
    # 5. Create mask: keep tokens where (cumsum - current_prob) < top_p
    # 6. Zero out masked probabilities and renormalize
    # 7. Sample from the filtered distribution
    ...


class InferenceEngine:
    """
    Complete inference engine tying together:
    - Tokenizer
    - Transformer model with KV cache
    - Top-p sampling
    - Basic batching
    """
    
    def __init__(self, model, tokenizer, device="cpu"):
        # TODO: Store model, tokenizer, device
        ...
    
    def generate(self, prompts, max_new_tokens=50, temperature=0.8, top_p=0.9):
        """
        Generate completions for a list of text prompts.
        
        Args:
            prompts: List of text strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            List of generated text strings (prompt + completion)
        """
        # TODO:
        # 1. Tokenize all prompts
        # 2. Pad to same length (left-pad or right-pad with pad_id)
        # 3. Create KV caches (one per layer)
        # 4. Prefill: run model on full prompt to populate KV cache
        # 5. Decode loop:
        #    a. Get logits for the last token position
        #    b. Apply top-p sampling
        #    c. Append new tokens
        #    d. Feed only the new token through the model (using KV cache)
        #    e. Stop if all sequences have generated EOS or hit max length
        # 6. Decode token IDs back to text
        ...
