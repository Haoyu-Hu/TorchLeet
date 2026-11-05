"""B.11n KNN（评测来自仓库的 `knn` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.11n
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch


def knn_predict(X_train: torch.Tensor, y_train: torch.Tensor,
                X_test: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    K-Nearest Neighbors classifier.

    Args:
        X_train (Tensor): Training features of shape (N_train, D).
        y_train (Tensor): Training labels of shape (N_train,).
        X_test (Tensor): Test features of shape (N_test, D).
        k (int): Number of neighbors.

    Returns:
        Tensor: Predicted labels of shape (N_test,).
    """
    # TODO: Step 1 - Compute pairwise Euclidean distances between test and train points
    # Hint: use torch.cdist(X_test, X_train) -> shape (N_test, N_train)

    # TODO: Step 2 - Find the K nearest neighbors (smallest distances)
    # Hint: use torch.topk(distances, k, largest=False) to get indices

    # TODO: Step 3 - Gather the labels of the K nearest neighbors
    # Hint: y_train[indices] -> shape (N_test, K)

    # TODO: Step 4 - Majority vote to determine predicted labels
    # Hint: use torch.mode(neighbor_labels, dim=1).values

    ...
