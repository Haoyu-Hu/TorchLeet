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
| 2 Core Transformer | MHA · GQA · KV Cache · Sliding Window | ✅ 看板就绪（提交桩 `s2_*.py`） |
| 3 Full Model | 架构总览 · RoPE+GQA 注意力块 · 组装 SmolLM（`s3_*.py`） | ✅ 看板就绪 |
| 4 Alignment | KL · LoRA · SFT · DPO · PPO · GRPO | 待开讲 |
| 5 Decoding & Inference | 采样 · 投机解码 · 连续批处理 · 推理引擎 | 待开讲 |
| 6 Systems | MoE | 待开讲 |
| Basics | 线性回归 → CNN/RNN/LSTM → 经典 ML | 待开讲 |

## TorchLeet 全题库

`study/problem_bank.json` 收录全部 68 题（题面、起始代码、判题项、分级提示），由 `python study/build_bank.py` 从 `problems.json`、题目 notebook 和 `python/torchleet` 判题规格生成（不含 SOLN 答案）。

```bash
python study/board.py tl list basics
python study/board.py tl init  lin-regression
python study/board.py tl check lin-regression
```

每次核验会打印一行 `TL-RESULT <id> <通过>/<总数>`，粘贴到在线教程的“核验结果”框里即可同步进度。

## 部署一套自己的在线教程 + 提交流水线

见 [`DEPLOY.md`](DEPLOY.md)：组成、步骤、限制与踩过的坑。快速自检：

```bash
bash study/setup_env.sh      # 装 numpy / pandas / torch 并跑自检
python study/selftest.py     # 只做自检
```
