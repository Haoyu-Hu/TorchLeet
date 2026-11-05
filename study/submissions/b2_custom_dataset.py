"""B.2 自定义 Dataset（评测来自仓库的 `custom-dataset` 题）。讲义见在线教程 Basics 部分。
核验：python study/board.py check B.2
只保留定义和 import；训练/演示代码放进 if __name__ == "__main__":
"""
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# TODO: Add the missing code
class LinearRegressionDataset(Dataset):
    ...
