"""A.5 逻辑回归（只用 numpy）"""
import numpy as np


def sigmoid(z):
    """数值稳定：z=±1000 不能溢出。"""
    ...


def log_loss(y, p):
    """二元交叉熵的均值；p 要 clip，避免 log(0)。"""
    ...


def predict_proba(X, w, b):
    """返回正类概率，形状 (n,)。"""
    ...


def fit_logistic(X, y, lr=0.1, n_iter=2000, l2=0.0):
    """批量梯度下降，l2 为 L2 系数（不惩罚 b）。返回 (w, b)。"""
    ...
