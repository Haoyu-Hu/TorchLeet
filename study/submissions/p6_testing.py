"""P.6 假设检验与多重检验（numpy / math）"""
import math
import numpy as np


def two_proportion_ztest(x1, n1, x2, n2):
    """两比例 z 检验（合并比例）。返回 (z, 双侧 p 值)。"""
    ...


def welch_t(a, b):
    """Welch t 检验的统计量与自由度，返回 (t, df)。"""
    ...


def bonferroni(pvals, alpha=0.05):
    """返回布尔数组：哪些假设被拒绝。"""
    ...


def benjamini_hochberg(pvals, alpha=0.05):
    """BH 方法控制错误发现率（FDR）。返回布尔数组，顺序与输入一致。"""
    ...
