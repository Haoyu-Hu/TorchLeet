"""A.2 数据预处理（只用 numpy，按列处理，不要修改输入）"""
import numpy as np


def standardize(X):
    """每列减均值、除以总体标准差 (ddof=0)。常数列结果为 0。"""
    ...


def min_max_scale(X):
    """每列缩放到 [0, 1]。常数列结果为 0。"""
    ...


def one_hot(labels, categories=None):
    """labels: 一维数组。categories 缺省时取排序后的唯一值。返回 (n, 类别数) 的 0/1 数组。"""
    ...


def impute_mean(X):
    """把每列的 NaN 填成该列（忽略 NaN 的）均值。"""
    ...
