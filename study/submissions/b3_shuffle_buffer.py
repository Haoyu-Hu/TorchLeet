"""B.3 Shuffle Buffer（自写核验 b-shuffle-buffer）。讲义见在线教程 Basics · B.3。
核验：python study/board.py check B.3

要求：
- ShuffleBufferDataset(base_dataset, buffer_size) 继承 IterableDataset，包装一个可迭代数据集。
- 维护一个最多 buffer_size 个元素的缓冲区；缓冲区满后，每读入一个新元素就随机吐出缓冲区里的一个。
- 源数据耗尽后，把缓冲区剩余元素打乱后全部吐出。
- 性质：每个元素恰好出现一次；buffer_size=1 时顺序不变；
  任意时刻「已读入 − 已吐出」不超过 buffer_size（不能先把全部数据读进内存）。
"""
import random
from torch.utils.data import IterableDataset


class ShuffleBufferDataset(IterableDataset):
    def __init__(self, base_dataset, buffer_size):
        super().__init__()
        # TODO
        ...

    def __iter__(self):
        # TODO
        ...
