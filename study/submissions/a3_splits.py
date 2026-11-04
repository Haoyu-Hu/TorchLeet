"""A.3 划分数据集与 K 折交叉验证（只用 numpy）"""
import numpy as np


def train_test_split(X, y, test_size=0.25, seed=0):
    """随机划分，测试集大小 = ceil(n*test_size)。同一 seed 结果相同。
    返回 X_train, X_test, y_train, y_test。"""
    ...


def k_fold_indices(n, k, seed=0):
    """返回长度为 k 的列表，每项 (train_idx, val_idx)。各折验证集互不重叠且覆盖所有样本。"""
    ...
