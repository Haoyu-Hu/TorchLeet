"""B.10t TensorBoard 记录（评测来自仓库的 `tensorboard` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.10t
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # 记录部分不计分，需要先 pip install tensorboard，然后在这里写 SummaryWriter 训练循环：
    # from torch.utils.tensorboard import SummaryWriter
    # TODO: 训练模型，每个 epoch 用 writer.add_scalar("Loss/train", loss, epoch) 记录
    pass
