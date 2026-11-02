# 第 1 节：Foundations（从文本到 Attention）

学习顺序对应 README 的 LLM Learning Path。每课：**为什么 → 直觉 → 数学 → 你要想清楚的问题**。
学完一课就去 `study/submissions/` 写代码，然后 `python study/board.py check <编号>`。

整条流水线先有个全局图：

```
文本 --BPE--> token id --Embedding--> (B,T,d) --+位置信息--> [RMSNorm -> Attention -> ...] x N
```

---

## 1.1 Byte Pair Encoding（分词）

**为什么**：模型只吃数字。按词切 → 词表爆炸且遇到生词（OOV）就废；按字符切 → 序列太长。
BPE 折中：高频的字符组合合成一个 token，生词也能拆成已知碎片，永远不会 OOV。

**算法（训练）**
1. 每个词拆成字符，末尾加 `</w>`（让模型分得清 "est" 在词尾还是词中）。统计词频。
2. 统计所有**相邻符号对**的出现次数，**按词频加权**（"low" 出现 4 次，它的每个 pair 就算 4 次）。
3. 取最高频 pair，在所有词里把它合成一个新符号；记录这次 merge。
4. 重复 N 次。N 次 merge ≈ 词表增加 N 个符号。

**例**：`l o w </w>` 出现很多次 → 先合并 `(l,o)` → `lo w </w>` → 再 `(lo,w)` → `low </w>`。

**要想清楚**
- 合并之后，词的"拼写"变了吗？（不变，只有切分方式变）
- 合并 `(a,a)` 时，在 `a a a` 里该怎么扫？从左到右贪心，别重复使用同一个字符。
- 频次相同怎么办？任选但要确定性（固定的 tie-break）。
- 推理时怎么编码新词？按**训练时的 merge 顺序**依次应用。

> 工业界用 byte-level BPE（GPT-2 起）：底层符号是 256 个字节，天然覆盖任何语言和 emoji。

---

## 1.2 Sinusoidal 位置编码

**为什么**：Attention 对输入是**置换不变**的——打乱 token 顺序，没有位置信息的话输出只是跟着打乱。
所以要把"我是第几个"显式注入。

**公式**（Attention Is All You Need）
```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```
**直觉**：每一对维度 (2i, 2i+1) 是一个以角频率 ω_i 转动的"指针"。
i 小 → 频率高 → 像秒针；i 大 → 频率低 → 像时针。多个转速组合 = 给每个位置一个唯一的"时钟读数"，
并且 `PE[pos+k]` 是 `PE[pos]` 的**线性变换**（旋转），所以相对位移容易学。

**要想清楚**
- 频率下标 i 是 0..d/2-1，**不是**原始维度下标——sin 和 cos 共用同一个 ω。
- 为什么用 `register_buffer` 而不是 `nn.Parameter`？（固定函数，不训练，但要跟着 `.to(device)` / 存进 state_dict）
- 数值上 `10000^(-2i/d)` 常用 `exp(-2i/d * ln 10000)` 算，更稳。
- `forward` 只看 `x.shape[1]`：返回前 `seq_len` 行，形状 `(1, seq_len, d)`，之后由外部做 `x + pe`。

---

## 1.3 RoPE（旋转位置编码）

**为什么**：绝对位置编码是"加"在输入上的，注意力分数里相对距离不明显、外推长度也差。
RoPE 改成**旋转 q 和 k**，让 `q_m · k_n` 只依赖相对距离 `m-n`。LLaMA/Mistral/Qwen 都用它。

**数学**：把向量两两配对成二维平面上的点，位置 m 处整体转 mθ 角：
```
[x1']   [cos mθ  -sin mθ] [x1]
[x2'] = [sin mθ   cos mθ] [x2]
```
旋转是正交变换：`(R_m q)·(R_n k) = q · R_{n-m} k`，只剩相对位移。

**实现上的关键恒等式**（"half" 配对法）：
```
q' = q * cos + rotate_half(q) * sin
rotate_half([x1, x2]) = [-x2, x1]      # 最后一维切两半
```
- `θ_i = 10000^(-2i/d)`，和 sinusoidal 同一套频率。cos/sin 表按位置预先算好缓存。
- 只作用在 **q 和 k**，不作用在 v。
- cos/sin 形状要能广播到 `(batch, heads, seq, head_dim)`。

**要想清楚**
- 为什么旋转不改变向量长度？（所以不影响 attention 的数值尺度）
- `rotate_half` 连用两次得到什么？（`-x`，判题器会检查）
- cos=1, sin=0 时应该什么都不变。

---

## 1.4 RMSNorm

**为什么**：深层网络需要归一化稳定激活。LayerNorm = 减均值 + 除标准差。
RMSNorm 发现**去掉减均值**效果几乎不变、更快（少一次 reduce），LLaMA 等全面采用。

**公式**
```
RMSNorm(x) = x / sqrt(mean(x², dim=-1) + eps) * γ
```
- 沿**最后一维**（特征维）；`γ` 形状 `(dim,)`，初始化为 1，可学习。
- `eps` 防止全零输入除以 0。

**要想清楚**
- `keepdim=True` 为什么必要？（让 (…,1) 能广播回 (…,dim)）
- 它和 LayerNorm 差在哪一步？输出的均值还是 0 吗？（不是）
- 低精度（fp16/bf16）下通常先转 float32 算 RMS 再转回——为什么？

---

## 1.5 Scaled Dot-Product Attention

**为什么**：让每个 token 按"相关度"从别的 token 那里取信息。

**直觉**：q = 我在找什么，k = 我有什么标签，v = 我的内容。q·k 越大越相关，softmax 变成权重，对 v 加权平均。

**公式**
```
Attention(Q,K,V) = softmax( Q Kᵀ / sqrt(d_k) ) V
```
形状：`Q:(…,Tq,d)  K:(…,Tk,d)  V:(…,Tk,dv)` → 分数 `(…,Tq,Tk)` → 输出 `(…,Tq,dv)`。

**要想清楚**
- 为什么除 `sqrt(d_k)`？假设 q、k 各分量均值 0 方差 1，则 q·k 的方差是 `d_k`；不缩放则 d_k 大时 softmax 饱和、梯度几乎为 0。
- Mask 必须在 softmax **之前**把屏蔽位置设为 `-inf`（softmax 后变 0 且其余行仍归一）；softmax 之后再乘 0 会让每行和不为 1。
- 因果 mask = 下三角：token t 只能看 ≤ t 的位置。
- softmax 对哪一维做？（Tk，也就是最后一维，每个 query 对所有 key 归一）

---

## 做题顺序与自检

| 编号 | 提交文件 | 核验命令 |
|---|---|---|
| 1.1 | `submissions/s1_1_bpe.py` | `python study/board.py check 1.1` |
| 1.2 | `submissions/s1_2_sinusoidal.py` | `python study/board.py check 1.2` |
| 1.3 | `submissions/s1_3_rope.py` | `python study/board.py check 1.3` |
| 1.4 | `submissions/s1_4_rmsnorm.py` | `python study/board.py check 1.4` |
| 1.5 | `submissions/s1_5_attention.py` | `python study/board.py check 1.5` |

卡住了先 `python study/board.py hint 1.x`，再来问我——我会用提问带你找，不直接给答案。
