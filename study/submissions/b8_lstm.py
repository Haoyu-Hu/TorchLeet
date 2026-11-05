"""B.8 LSTM（评测来自仓库的 `lstm` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.8
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)


class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        ...
        
    def forward(self, inputs, H_C=None):
        ...


# Define the LSTM Model
# TODO: Add LSTM layer, forward implementation
class LSTMModel(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        ...
