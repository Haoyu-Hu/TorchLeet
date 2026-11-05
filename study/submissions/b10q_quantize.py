"""B.10q 动态量化（评测来自仓库的 `quantize-lm` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.10q
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import quantize_dynamic


# TODO: Define a simple Language Model (an LSTM-based model)
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        ...

    def forward(self, x):
        ...
