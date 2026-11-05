"""B.6 从零实现卷积与池化（评测来自仓库的 `cnn-scratch` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.6
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Conv2dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        ...

    def forward(self, x):
        ...


class MaxPool2dCustom(nn.Module):
    def __init__(self, kernel_size, stride=None):
        ...

    def forward(self, x):
        ...
