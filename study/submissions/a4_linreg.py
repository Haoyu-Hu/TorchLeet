"""A.4 线性回归 / 岭回归 / 梯度下降（只用 numpy）。w 形状 (d,)，b 为标量。"""
import numpy as np


def mse(y_true, y_pred):
    ...


def linear_regression(X, y):
    """最小二乘（带截距），返回 (w, b)。"""
    ...


def ridge_regression(X, y, alpha):
    """岭回归，截距不惩罚。返回 (w, b)。"""
    ...


def gradient_descent_linreg(X, y, lr=0.1, n_iter=500):
    """用批量梯度下降最小化 MSE。返回 (w, b)。"""
    ...
