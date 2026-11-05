"""B.10 基准测试（评测来自仓库的 `benchmark` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.10
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time


# Define a simple neural network model
# TODO: Add layers to the model
class SimpleNN(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
