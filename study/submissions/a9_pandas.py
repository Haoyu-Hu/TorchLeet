"""A.9 pandas 数据处理（不要修改传入的 DataFrame）"""
import pandas as pd


def top_k_per_group(df, group_col, value_col, k):
    """每个分组里 value_col 最大的 k 行。"""
    ...


def fill_group_mean(df, group_col, value_col):
    """用同组均值填充 value_col 的缺失值，返回新 DataFrame。"""
    ...


def merge_and_aggregate(orders, users):
    """orders(user_id, amount) 左连接 users(user_id, country)，
    按 country 汇总 amount 的 sum 和 count。返回 DataFrame：列 country, sum, count。"""
    ...
