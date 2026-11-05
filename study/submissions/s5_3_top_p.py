"""5.3 Top-p（核）采样（评测来自仓库的 `top-p-sampling` 题）。讲义见在线教程第 5 节。
核验：python study/board.py check 5.3
"""
import torch
import torch.nn.functional as F


def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Perform top-p (nucleus) sampling on logits.
    
    Args:
        logits (Tensor): Raw logits of shape (1, vocab_size)
        p (float): Cumulative probability threshold (0.0 to 1.0)
        temperature (float): Temperature for scaling logits
        
    Returns:
        int: Sampled token index
    """
    # Step 1: Apply temperature scaling
    # TODO: Divide logits by temperature
    ...
    
    # Step 2: Convert to probabilities using softmax
    # TODO: Apply softmax to get probabilities
    ...
    
    # Step 3: Sort probabilities in descending order
    # TODO: Use torch.sort to get sorted probs and original indices
    ...
    
    # Step 4: Compute cumulative sum
    # TODO: Use torch.cumsum on sorted probabilities
    ...
    
    # Step 5: Create nucleus mask and zero out tokens outside the nucleus
    # TODO: Mask where cumsum - current_prob >= p, then zero those out
    ...
    
    # Step 6: Renormalize the remaining probabilities
    # TODO: Divide by sum so probabilities sum to 1
    ...
    
    # Step 7: Sample from the renormalized distribution
    # TODO: Use torch.multinomial to sample, then map back to original index
    ...
