"""4.2 LoRA（评测来自仓库的 `lora` 题）"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """包住一个已有的 nn.Linear：output = linear(x) + (alpha / rank) * (x @ A) @ B
    A: (d_in, rank)，B: (rank, d_out)。原层的权重要冻结（requires_grad=False）。
    初始化：A 随机（如 kaiming_uniform_），B 全 0，这样刚开始时增量为 0。"""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        ...

    def forward(self, x):
        ...


def apply_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0, target_modules: list = None) -> nn.Module:
    """把名字里含有 target_modules 中任一子串的 nn.Linear 换成 LoRALinear（用 setattr 替换到父模块上）。返回 model。"""
    ...


def merge_lora(model: nn.Module) -> nn.Module:
    """把每个 LoRALinear 合并回普通 nn.Linear：W_new = W + scaling * (A @ B).T，偏置不变。返回 model。"""
    ...
