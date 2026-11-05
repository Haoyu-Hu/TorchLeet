"""B.7 RNN（评测来自仓库的 `rnn` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.7
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)


# Define the RNN Model
# TODO: Add RNN layer, fully connected layer, and forward implementation
class RNNModel(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
