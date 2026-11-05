"""P.3 均值、方差、协方差与相关（numpy）"""
import numpy as np


def mean_var(x, ddof=0):
    """返回 (均值, 方差)，方差分母为 n − ddof。"""
    ...


def covariance_matrix(X, ddof=1):
    """X：(n, D)，每行一个样本。返回 (D, D) 的协方差矩阵（应与 np.cov(X, rowvar=False) 一致）。"""
    ...


def correlation_matrix(X):
    """返回 (D, D) 的相关系数矩阵。"""
    ...


def welford(stream):
    """一遍扫描任意可迭代对象（可能是生成器，只能读一次），返回 (均值, 总体方差)。数值要稳定。"""
    ...
