# TorchLeet 学习看板

按仓库 README 的 **LLM Learning Path** 顺序推进：先读讲义，再在 `submissions/` 写代码，最后用看板核验。

```bash
pip install torch numpy pandas   # A.x 题只需 numpy/pandas
python study/board.py        # 看板
python study/board.py check 1.1
python study/board.py hint 1.1
```

> 请用 `python study/board.py` 运行，不要在仓库根目录里 `python -c "import torch"`——根目录有个 `torch/` 题目文件夹会遮住真正的 PyTorch。

| 节 | 内容 | 状态 |
|---|---|---|
| 1 Foundations | BPE · Sinusoidal · RoPE · RMSNorm · Attention | ✅ 讲义+看板已就绪 → `lessons/01-foundations.md` |
| 应用 ML（笔试向） | 指标 · 预处理 · CV · 回归 · KNN · 决策树 · PCA · pandas，编号 A.1~A.9，只需 numpy/pandas | ✅ 看板就绪（讲义在线文档里） |
| 2 Core Transformer | MHA · GQA · KV Cache · Sliding Window | 待开讲 |
| 3 Full Model | SmolLM | 待开讲 |
| 4 Alignment | KL · LoRA · SFT · DPO · PPO · GRPO | 待开讲 |
| 5 Decoding & Inference | 采样 · 投机解码 · 连续批处理 · 推理引擎 | 待开讲 |
| 6 Systems | MoE | 待开讲 |
| Basics | 线性回归 → CNN/RNN/LSTM → 经典 ML | 待开讲 |
