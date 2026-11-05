"""B.11k K-Means（评测来自仓库的 `kmeans` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.11k
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch


def kmeans(data: torch.Tensor, k: int, max_iters: int = 100, tol: float = 1e-4):
    """
    K-Means clustering using PyTorch.

    Args:
        data (Tensor): Input data of shape (N, D).
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance for centroid movement.

    Returns:
        centroids (Tensor): Final centroids of shape (K, D).
        assignments (Tensor): Cluster assignment for each point, shape (N,).
    """
    N, D = data.shape

    # TODO: Step 1 - Initialize centroids by randomly selecting K data points
    # Hint: use torch.randperm(N)[:k] to pick random indices

    for i in range(max_iters):
        # TODO: Step 2 - Compute distances from each point to each centroid
        # Hint: use torch.cdist(data, centroids) -> shape (N, K)

        # TODO: Step 3 - Assign each point to the nearest centroid
        # Hint: use torch.argmin on the distance matrix

        # TODO: Step 4 - Update centroids as the mean of assigned points
        # Hint: loop over range(k), mask = (assignments == j), new_centroids[j] = data[mask].mean(dim=0)

        # TODO: Step 5 - Check for convergence
        # Hint: if torch.allclose(old_centroids, new_centroids, atol=tol): break

        ...

    return centroids, assignments
