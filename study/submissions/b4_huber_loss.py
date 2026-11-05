"""B.4 Huber 损失（评测来自仓库的 `custom-loss` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.4
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


#TODO: Define the nn.Module for the Huber Loss
class HuberLoss(nn.Module):
    ...


# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)
