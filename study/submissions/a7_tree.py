"""A.7 决策树的一次划分（CART，基尼不纯度）"""
import numpy as np


def gini(y):
    """1 - sum(p_c^2)。"""
    ...


def best_split(X, y):
    """遍历所有特征与候选阈值（相邻不同取值的中点），规则 x <= thr 进左子树。
    返回 (feature, threshold, gain)，gain = gini(父) - 加权 gini(子)。
    节点已纯或无法划分时返回 (None, None, 0.0)。"""
    ...
