"""5.2 Top-k 采样（评测来自仓库的 `top-k-sampling` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.2
"""
import torch
import torch.nn.functional as F


def top_k_sample(logits, k=50, temperature=1.0):
    """
    Perform top-k sampling on logits.
    
    Args:
        logits (Tensor): Raw logits of shape (1, vocab_size)
        k (int): Number of top tokens to keep
        temperature (float): Temperature for scaling logits
        
    Returns:
        int: Sampled token index
    """
    # Step 1: Apply temperature scaling
    # TODO: Divide logits by temperature
    ...
    
    # Step 2: Find the top-k logits
    # TODO: Use torch.topk to get top-k values and indices
    ...
    
    # Step 3: Find the threshold (minimum of top-k values)
    # TODO: Get the k-th largest logit value
    ...
    
    # Step 4: Mask all logits below the threshold to -inf
    # TODO: Set logits below threshold to float('-inf')
    ...
    
    # Step 5: Apply softmax to get probabilities
    # TODO: Convert masked logits to probability distribution
    ...
    
    # Step 6: Sample from the distribution
    # TODO: Use torch.multinomial to sample a token
    ...
