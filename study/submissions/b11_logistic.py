"""B.11 逻辑回归 (PyTorch)（评测来自仓库的 `logistic-regression` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.11
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Compute the sigmoid function."""
    # TODO: Implement sigmoid: 1 / (1 + exp(-z))
    ...


def bce_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute binary cross-entropy loss."""
    # TODO: Implement BCE = -mean(y * log(p) + (1-y) * log(1-p))
    # Use eps inside log() to prevent log(0)
    ...


def train_logistic_regression(X: torch.Tensor, y: torch.Tensor,
                               lr: float = 0.1, n_epochs: int = 1000):
    """
    Train logistic regression with manual gradient descent.

    Args:
        X (Tensor): Features of shape (N, D).
        y (Tensor): Binary labels of shape (N,).
        lr (float): Learning rate.
        n_epochs (int): Number of training epochs.

    Returns:
        w (Tensor): Learned weights of shape (D,).
        b (Tensor): Learned bias (scalar).
        losses (list): Loss at each epoch.
    """
    N, D = X.shape

    # TODO: Initialize weights and bias to zeros

    losses = []

    for epoch in range(n_epochs):
        # TODO: Step 1 - Compute linear output: z = X @ w + b

        # TODO: Step 2 - Apply sigmoid to get predictions: p = sigmoid(z)

        # TODO: Step 3 - Compute loss

        # TODO: Step 4 - Compute gradients manually
        # dw = (1/N) * X^T @ (p - y)
        # db = (1/N) * sum(p - y)

        # TODO: Step 5 - Update weights and bias
        # w -= lr * dw
        # b -= lr * db

        ...

    return w, b, losses
