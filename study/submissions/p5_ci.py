"""P.5 置信区间与 bootstrap（numpy / math）"""
import math
import numpy as np


def norm_ppf(p):
    """标准正态分布的分位数 Φ⁻¹(p)，误差小于 1e-6。"""
    ...


def mean_ci(x, conf=0.95):
    """均值的正态近似置信区间 (lo, hi)：x̄ ± z·s/√n，s 用样本标准差（ddof=1）。"""
    ...


def bootstrap_ci(x, stat_fn, n_boot=2000, conf=0.95, seed=0):
    """百分位 bootstrap 区间 (lo, hi)。用 np.random.default_rng(seed)，同一个 seed 结果必须相同。"""
    ...
