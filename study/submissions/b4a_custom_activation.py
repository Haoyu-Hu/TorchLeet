"""B.4a 自定义激活（评测来自仓库的 `custom-activation` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.4a
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Define the Linear Regression Model
class CustomActivationModel(nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    # TODO: Implement the forward pass
    def custom_activation(self, x):
        ...
            
    # TODO: Implement the forward pass
    def forward(self, x):
        ...
