"""B.5 深层全连接网络（评测来自仓库的 `custom-dnn` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.5
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Define the Deep Neural Network Model
class DNNModel(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
        return x
