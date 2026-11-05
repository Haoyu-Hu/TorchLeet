"""6.2 MoE 的路由、负载均衡与稀疏计算（本教程自建判题）"""
import torch


def topk_route(router_logits, k):
    """router_logits: (T, N)。返回 (weights (T, k), indices (T, k))：
    indices 是每个 token logits 最大的 k 个专家；weights 只对这 k 个 logits 做 softmax（每行和为 1）。"""
    ...


def load_balancing_loss(router_logits, indices, num_experts):
    """Switch Transformer 的辅助损失：loss = N · Σ_i f_i · P_i。
      P_i：所有 token 在专家 i 上的平均路由概率（对全部 N 个专家做 softmax 后求平均），可导；
      f_i：indices (T, k) 里专家 i 出现的次数 / (T·k)，不可导。
    完全均衡时 loss = 1。返回标量张量。"""
    ...


def moe_combine(x, router_logits, experts, k):
    """x: (T, D)；experts：长度为 N 的列表，experts[e](x_sub) 把 (m, D) 映射到 (m, D)。
    输出第 t 行 = Σ_j w_tj · experts[idx_tj](x_t)。
    每个专家只能处理分配给它的那些 token（不要把全部 token 都送进每个专家）。返回 (T, D)。"""
    ...
