"""B.9 自编码器（评测来自仓库的 `autoencoder` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.9
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define an Autoencoder model
# TODO: Implement the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        # init
        ...
        # Encoder
        self.encoder = ...
        # Decoder
        self.decoder = ...

    def forward(self, x):
        ...
