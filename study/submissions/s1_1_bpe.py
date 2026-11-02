"""1.1 Byte Pair Encoding — 在 ... 处填写实现。讲义: study/lessons/01-foundations.md"""
from collections import defaultdict, Counter


def get_vocab(corpus):
    """词 -> 频次。每个词拆成字符元组并加结尾符 '</w>'。
    例: ["test"] -> {('t','e','s','t','</w>'): 1}"""
    ...


def get_stats(vocab):
    """相邻符号对 -> 加权频次（按词频加权）。"""
    ...


def merge_vocab(pair, vocab):
    """把所有相邻的 pair 合并成一个符号；词的拼写不能变。"""
    ...


def byte_pair_encoding(corpus, num_merges=10):
    """贪心循环：每次合并最高频的 pair。返回 (vocab, merges)。"""
    ...
