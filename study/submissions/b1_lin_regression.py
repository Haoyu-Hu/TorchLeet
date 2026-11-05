"""B.1 线性回归与训练循环（评测来自仓库的 `lin-regression` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.1
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Define the Linear Regression Model
#TODO: Add the layer and forward implementation
class LinearRegressionModel(nn.Module):
    def __init__(self):
        layer = torch.nn.layer()

    def forward(self, x):
        ...
