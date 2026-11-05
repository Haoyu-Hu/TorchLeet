"""P.2 常见分布（只用 math / numpy，不用 scipy）"""
import math
import numpy as np


def binomial_pmf(k, n, p):
    """P(X = k)，X ~ Binomial(n, p)。n 很大（如 2000）时不能溢出或下溢成 0。"""
    ...


def poisson_pmf(k, lam):
    """P(X = k)，X ~ Poisson(λ)。"""
    ...


def normal_pdf(x, mu, sigma):
    """N(μ, σ²) 的密度，x 可以是 numpy 数组。"""
    ...


def normal_cdf(x, mu, sigma):
    """N(μ, σ²) 的分布函数，x 可以是 numpy 数组。用 math.erf。"""
    ...
