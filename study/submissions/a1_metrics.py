"""A.1 评估指标（只用 numpy）"""
import numpy as np


def confusion_matrix(y_true, y_pred):
    """二分类，返回 2x2 数组 [[tn, fp], [fn, tp]]。"""
    ...


def precision_recall_f1(y_true, y_pred):
    """返回 (precision, recall, f1)。分母为 0 时该项返回 0.0。"""
    ...


def roc_auc(y_true, y_score):
    """ROC-AUC。得分并列要算 0.5（用平均秩）。不要调用 sklearn。"""
    ...
