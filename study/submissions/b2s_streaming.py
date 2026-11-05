"""B.2s 流式数据集 StreamingDataset（自写核验 b-streaming）。讲义见在线教程 Basics · B.2。
核验：python study/board.py check B.2s

要求：
- 继承 torch.utils.data.IterableDataset；构造参数 filepath（无表头的 CSV，每行全是数字）。
- 逐行惰性读取：__iter__ 必须是生成器（不要先把整个文件读进列表）。
- 每行产出 (x, y)：x 为除最后一列外的所有列组成的 1-D float tensor，y 为最后一列的标量 tensor。
- 多 worker（DataLoader num_workers>1）时每行恰好只被一个 worker 产出：
  用 torch.utils.data.get_worker_info() 做分片。
"""
import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingDataset(IterableDataset):
    def __init__(self, filepath):
        super().__init__()
        # TODO
        ...

    def __iter__(self):
        # TODO: 取得 worker 信息 → 打开文件 → 逐行解析 → 只产出属于本 worker 的行
        ...
