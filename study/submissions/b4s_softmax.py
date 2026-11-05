"""B.4s Softmax（评测来自仓库的 `softmax` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.4s
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute numerically stable softmax along the specified dimension.

    Args:
        x (Tensor): Input tensor of any shape.
        dim (int): Dimension along which to compute softmax. Default: -1.

    Returns:
        Tensor: Softmax probabilities with the same shape as input.
    """
    # TODO: Step 1 - Subtract the max along `dim` for numerical stability (use keepdim=True)
    # TODO: Step 2 - Exponentiate the shifted values
    # TODO: Step 3 - Normalize by dividing by the sum along `dim` (use keepdim=True)
    ...
