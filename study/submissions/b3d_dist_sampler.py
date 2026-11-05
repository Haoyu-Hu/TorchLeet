"""B.3d 分布式采样器（自写核验 b-dist-sampler）。讲义见在线教程 Basics · B.3。
核验：python study/board.py check B.3d

实现 SimpleDistributedSampler(dataset, num_replicas, rank, shuffle=True, seed=0)，
行为需与 torch.utils.data.DistributedSampler（drop_last=False）逐元素一致：
- __len__：每个 rank 拿到 ceil(N / num_replicas) 个索引。
- 索引总数不足 num_replicas 的倍数时，从头部循环补齐。
- shuffle=True 时用 torch.Generator 以 seed + epoch 为种子做 randperm。
- set_epoch(epoch)：改变下一轮的打乱方式；同一 epoch 结果确定。
- 各 rank 取索引的方式与 PyTorch 相同（提示：不是连续切块）。
"""
import math
import torch
from torch.utils.data import Sampler


class SimpleDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=True, seed=0):
        # TODO
        ...

    def set_epoch(self, epoch):
        # TODO
        ...

    def __len__(self):
        # TODO
        ...

    def __iter__(self):
        # TODO
        ...
