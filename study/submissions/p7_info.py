"""P.7 信息论：熵、交叉熵、互信息、困惑度（numpy / math）"""
import math
import numpy as np


def entropy(p, base=2):
    """H(p) = −Σ p log p，约定 0·log 0 = 0。"""
    ...


def cross_entropy(p, q, base=2):
    """H(p, q) = −Σ p log q。"""
    ...


def mutual_information(joint, base=2):
    """joint：二维联合分布表 p(x, y)（总和为 1）。返回 I(X; Y)。"""
    ...


def perplexity(token_logprobs):
    """token_logprobs：每个 token 的自然对数概率。返回 exp(−平均对数概率)。"""
    ...
