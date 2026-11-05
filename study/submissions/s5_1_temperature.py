"""5.1 温度采样（评测来自仓库的 `temperature-sampling` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.1
"""
import torch
import torch.nn.functional as F


def temperature_sample(logits, temperature=1.0):
    """
    Perform temperature-scaled sampling on logits.
    
    Args:
        logits (Tensor): Raw logits of shape (1, vocab_size)
        temperature (float): Temperature for scaling (must be > 0)
        
    Returns:
        int: Sampled token index
    """
    # Step 1: Validate temperature
    # TODO: Assert temperature is positive
    ...
    
    # Step 2: Apply temperature scaling
    # TODO: Divide logits by temperature
    ...
    
    # Step 3: Apply softmax to get probabilities
    # TODO: Convert scaled logits to probability distribution
    ...
    
    # Step 4: Sample from the distribution
    # TODO: Use torch.multinomial to sample a token
    ...
