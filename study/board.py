#!/usr/bin/env python3
"""学习看板: 查看进度 / 提交核验。

    python study/board.py            # 看板（进度总览）
    python study/board.py check 1.1  # 核验 submissions/ 里对应的代码
    python study/board.py hint 1.1   # 逐级提示（不会给答案）
    编号 A.1 ~ A.9 是应用 ML 题（numpy/pandas，不需要 PyTorch）。

    python study/board.py tl list [llm-path|basics|advanced]   # TorchLeet 全题库
    python study/board.py tl init  <id>   # 生成桩文件 study/submissions/torchleet/<id>.py
    python study/board.py tl check <id>   # 自动核验
    python study/board.py tl hint  <id> [n]
每次核验都会打印一行 TL-RESULT，整行粘贴到在线教程的“核验结果”框即可记录进度。

核验使用仓库自带的判题器 python/torchleet（检查性质，不对比标准答案，
所以写法和官方不同也能通过）。
"""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "python"))  # 使用仓库内的 torchleet

# id: (章节, 标题, 题目id(torchleet), 提交文件, 需要提交的名字)
TASKS = {
    "1.1": ("1 Foundations", "Byte Pair Encoding", "byte-pair-encoder", "s1_1_bpe.py",
            ["get_vocab", "get_stats", "merge_vocab", "byte_pair_encoding"]),
    "1.2": ("1 Foundations", "Sinusoidal Embeddings", "sinusoidal-positional-embedding",
            "s1_2_sinusoidal.py", ["SinusoidalPositionalEmbedding"]),
    "1.3": ("1 Foundations", "RoPE", "rotary-positional-embedding", "s1_3_rope.py",
            ["rotate_half", "apply_rotary_pos_emb"]),
    "1.4": ("1 Foundations", "RMSNorm", "rms-norm", "s1_4_rmsnorm.py", ["RMSNorm"]),
    "1.5": ("1 Foundations", "Attention", "implement-attention-from-scratch",
            "s1_5_attention.py", ["scaled_dot_product_attention"]),
    "2.1": ("2 Core Transformer", "Multi-Head Attention", "multi-head-attention",
            "s2_1_mha.py", ["multi_head_attention"]),
    "2.2": ("2 Core Transformer", "Grouped Query Attention", "grouped-query-attention",
            "s2_2_gqa.py", ["grouped_query_attention"]),
    "2.3": ("2 Core Transformer", "KV Cache", "kv-cache",
            "s2_3_kvcache.py", ["KVCache", "CachedAttention"]),
    "2.4": ("2 Core Transformer", "Sliding Window Attention", "sliding-window-attention",
            "s2_4_swa.py", ["create_sliding_window_mask", "full_attention", "sliding_window_attention"]),
    "3.2": ("3 Full Model", "RoPE + GQA 注意力块", "smollm", "s3_2_rope_gqa_block.py",
            ["rotate_half", "apply_rotary_pos_emb", "repeat_kv", "RotaryEmbedder", "RopeAttention"]),
    "3.3": ("3 Full Model", "组装完整 SmolLM", "s3-smollm-model", "s3_3_smollm_model.py",
            ["RMSNorm", "RopeAttention", "SwiGLU", "SmolLM", "greedy_generate"]),
    "4.1": ("4 Alignment", "KL 散度", "s4-kl", "s4_1_kl.py", ["kl_divergence", "kl_estimators"]),
    "4.2": ("4 Alignment", "LoRA", "lora", "s4_2_lora.py", ["LoRALinear", "apply_lora", "merge_lora"]),
    "4.3": ("4 Alignment", "SFT 数据与损失", "s4-sft", "s4_3_sft.py", ["build_sft_example", "collate", "sft_loss"]),
    "4.4": ("4 Alignment", "DPO", "dpo", "s4_4_dpo.py", ["get_batch_logps", "dpo_loss"]),
    "4.5": ("4 Alignment", "PPO-RLHF", "ppo-rlhf", "s4_5_ppo.py", ["generate_sequences", "compute_gae", "ppo_step"]),
    "4.6": ("4 Alignment", "GRPO", "grpo", "s4_6_grpo.py",
            ["generate_group_completions", "compute_group_advantages", "grpo_loss"]),
    "5.1": ("5 Decoding & Inference", "温度采样", "temperature-sampling", "s5_1_temperature.py", ["temperature_sample"]),
    "5.2": ("5 Decoding & Inference", "Top-k 采样", "top-k-sampling", "s5_2_top_k.py", ["top_k_sample"]),
    "5.3": ("5 Decoding & Inference", "Top-p 采样", "top-p-sampling", "s5_3_top_p.py", ["top_p_sample"]),
    "5.4": ("5 Decoding & Inference", "投机解码", "speculative-decoding", "s5_4_speculative.py",
            ["DraftModel", "TargetModel", "standard_decode", "speculative_decode"]),
    "5.5": ("5 Decoding & Inference", "连续批处理", "continuous-batching", "s5_5_continuous_batching.py",
            ["Request", "ContinuousBatchScheduler"]),
    "5.6": ("5 Decoding & Inference", "推理引擎", "inference-engine", "s5_6_inference_engine.py",
            ["SimpleTokenizer", "KVCache", "MultiHeadAttention", "FeedForward", "TransformerBlock",
             "MiniTransformer", "top_p_sample", "InferenceEngine"]),
    "6.1": ("6 Mixture of Experts", "MoE 层", "mixture-of-experts", "s6_1_moe.py", ["Expert", "MoELayer"]),
    "6.2": ("6 Mixture of Experts", "路由与负载均衡", "s6-moe-routing", "s6_2_moe_routing.py",
            ["topk_route", "load_balancing_loss", "moe_combine"]),
    "P.1": ("P 概率与统计", "贝叶斯与朴素贝叶斯", "p1-bayes", "p1_bayes.py",
            ["posterior", "sequential_update", "bernoulli_nb_fit", "bernoulli_nb_predict"]),
    "P.2": ("P 概率与统计", "常见分布", "p2-distributions", "p2_distributions.py",
            ["binomial_pmf", "poisson_pmf", "normal_pdf", "normal_cdf"]),
    "P.3": ("P 概率与统计", "均值方差与协方差", "p3-moments", "p3_moments.py",
            ["mean_var", "covariance_matrix", "correlation_matrix", "welford"]),
    "P.4": ("P 概率与统计", "极大似然与 MAP", "p4-mle", "p4_mle.py",
            ["gaussian_mle", "gaussian_nll", "bernoulli_map", "categorical_mle"]),
    "P.5": ("P 概率与统计", "置信区间与 bootstrap", "p5-ci", "p5_ci.py", ["norm_ppf", "mean_ci", "bootstrap_ci"]),
    "P.6": ("P 概率与统计", "假设检验", "p6-testing", "p6_testing.py",
            ["two_proportion_ztest", "welch_t", "bonferroni", "benjamini_hochberg"]),
    "P.7": ("P 概率与统计", "信息论", "p7-info", "p7_info.py",
            ["entropy", "cross_entropy", "mutual_information", "perplexity"]),
    "P.8": ("P 概率与统计", "采样方法", "p8-sampling", "p8_sampling.py",
            ["sample_exponential", "rejection_sample", "importance_estimate"]),
    "B.1": ("B Basics", "线性回归与训练循环", "lin-regression", "b1_lin_regression.py", ["LinearRegressionModel"]),
    "B.2": ("B Basics", "自定义 Dataset", "custom-dataset", "b2_custom_dataset.py", ["LinearRegressionDataset"]),
    "B.2s": ("B Basics", "流式 IterableDataset", "b-streaming", "b2s_streaming.py", ["StreamingDataset"]),
    "B.3": ("B Basics", "Shuffle Buffer", "b-shuffle-buffer", "b3_shuffle_buffer.py", ["ShuffleBufferDataset"]),
    "B.3d": ("B Basics", "分布式采样器", "b-dist-sampler", "b3d_dist_sampler.py", ["SimpleDistributedSampler"]),
    "B.4": ("B Basics", "Huber 损失", "custom-loss", "b4_huber_loss.py", ["HuberLoss"]),
    "B.4a": ("B Basics", "自定义激活", "custom-activation", "b4a_custom_activation.py", ["CustomActivationModel"]),
    "B.4s": ("B Basics", "Softmax", "softmax", "b4s_softmax.py", ["softmax"]),
    "B.5": ("B Basics", "深层全连接网络", "custom-dnn", "b5_dnn.py", ["DNNModel"]),
    "B.5i": ("B Basics", "参数初始化", "cnn-param-init", "b5i_param_init.py", ["config_init"]),
    "B.6": ("B Basics", "从零实现卷积与池化", "cnn-scratch", "b6_cnn_scratch.py", ["Conv2dCustom", "MaxPool2dCustom"]),
    "B.6c": ("B Basics", "CNN 分类器", "cnn", "b6c_cnn.py", ["CNNModel"]),
    "B.7": ("B Basics", "RNN", "rnn", "b7_rnn.py", ["RNNModel"]),
    "B.8": ("B Basics", "LSTM", "lstm", "b8_lstm.py", ["CustomLSTMModel", "LSTMModel"]),
    "B.9": ("B Basics", "自编码器", "autoencoder", "b9_autoencoder.py", ["Autoencoder"]),
    "B.10": ("B Basics", "基准测试", "benchmark", "b10_benchmark.py", ["SimpleNN"]),
    "B.10q": ("B Basics", "动态量化", "quantize-lm", "b10q_quantize.py", ["LanguageModel"]),
    "B.10t": ("B Basics", "TensorBoard 记录", "tensorboard", "b10t_tensorboard.py", ["LinearRegressionModel"]),
    "B.11": ("B Basics", "逻辑回归 (PyTorch)", "logistic-regression", "b11_logistic.py", ["sigmoid", "bce_loss", "train_logistic_regression"]),
    "B.11k": ("B Basics", "K-Means", "kmeans", "b11k_kmeans.py", ["kmeans"]),
    "B.11n": ("B Basics", "KNN", "knn", "b11n_knn.py", ["knn_predict"]),
}
APPLIED = {
    "A.1": ("评估指标", "a1-metrics", "a1_metrics.py"),
    "A.2": ("数据预处理", "a2-preprocess", "a2_preprocess.py"),
    "A.3": ("划分与交叉验证", "a3-splits", "a3_splits.py"),
    "A.4": ("线性/岭回归", "a4-linreg", "a4_linreg.py"),
    "A.5": ("逻辑回归", "a5-logreg", "a5_logreg.py"),
    "A.6": ("KNN", "a6-knn", "a6_knn.py"),
    "A.7": ("决策树划分", "a7-tree", "a7_tree.py"),
    "A.8": ("PCA", "a8-pca", "a8_pca.py"),
    "A.9": ("pandas 数据处理", "a9-pandas", "a9_pandas.py"),
}
PROGRESS = Path.home() / ".torchleet" / "progress.json"


def passed():
    try:
        d = json.loads(PROGRESS.read_text())
    except Exception:
        return {}
    return d


def show():
    d, sec = passed(), None
    print("\n  TorchLeet 学习看板\n")
    for tid, (s, title, pid, f, _) in TASKS.items():
        if s != sec:
            sec = s; print(f"  ── 第 {s} ──")
        v = d.get(pid)
        mark = "⬜" if v is None else ("◐" if v.get("partial") else "✅")
        print(f"   {mark} {tid}  {title:<24} 提交: study/submissions/{f}")
    print("  ── 应用 ML（CodeSignal 风格，只需 numpy/pandas）──")
    for tid, (title, pid, f) in APPLIED.items():
        mark = "✅" if pid in d else "⬜"
        print(f"   {mark} {tid}  {title:<24} 提交: study/submissions/{f}")
    n = sum(1 for t in TASKS.values() if t[2] in d) + sum(1 for t in APPLIED.values() if t[1] in d)
    print(f"\n  完成 {n}/{len(TASKS) + len(APPLIED)}   核验: python study/board.py check <编号>\n")


def load(path, names):
    spec = importlib.util.spec_from_file_location("submission", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {n: getattr(m, n) for n in names if hasattr(m, n)}


def emit(pid, passed_n, total, skipped=0):
    """打印一行结果，可整行粘贴到在线教程的“核验结果”框里自动记录进度。"""
    extra = f" skip={skipped}" if skipped else ""
    print(f"  TL-RESULT {pid} {passed_n}/{total}{extra}")


def run_torchleet(pid, ns):
    from torchleet.runner import run, report
    results = run(pid, ns)
    ok = report(pid, results)
    n_pass = sum(1 for r in results if r[1] == "pass")
    n_skip = sum(1 for r in results if r[1] == "skip")
    n_fail = sum(1 for r in results if r[1] == "fail")
    emit(pid, n_pass, n_pass + n_fail + n_skip, n_skip)
    return ok


def tl(a):
    """TorchLeet 全题库: list / init / check / hint。"""
    bank = json.loads((ROOT / "problem_bank.json").read_text())
    by = {b["id"]: b for b in bank}
    sub = a[0] if a else "list"
    d = passed()
    if sub == "list":
        flt = a[1] if len(a) > 1 else None
        for b in bank:
            if flt and flt not in b["tracks"] and flt != b["difficulty"]:
                continue
            mark = "✅" if b["id"] in d else ("⬜" if b["graded"] else "·")
            print(f"  {mark} {b['id']:<36} {b['difficulty']:<7} {','.join(b['tracks'])}{'' if b['graded'] else '  (无自动判题)'}")
        return print("\n  tracks: llm-path / basics / advanced；用法: python study/board.py tl init|check|hint <id>\n")
    pid = a[1] if len(a) > 1 else None
    if pid not in by:
        sys.exit(f"未知题目 id: {pid}。先运行: python study/board.py tl list")
    b = by[pid]
    f = ROOT / "submissions" / "torchleet" / f"{pid}.py"
    if sub == "init":
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            return print(f"  已存在: {f.relative_to(ROOT.parent)}（不会覆盖）")
        f.write_text(b["starter"])
        return print(f"  已生成 {f.relative_to(ROOT.parent)}，填写后: python study/board.py tl check {pid}")
    if sub == "hint":
        hs = b["hints"]
        if not hs:
            return print("  这题还没有提示")
        n = int(a[2]) if len(a) > 2 else 1
        return print(f"  hint {min(max(n, 1), len(hs))}/{len(hs)}: {hs[min(max(n, 1), len(hs)) - 1]}")
    if sub != "check":
        sys.exit(__doc__)
    if not b["graded"]:
        sys.exit("这题没有自动判题。请对照 notebook 里的测试单元自查，再看 SOLN 解答。")
    sys.path.insert(0, str(ROOT))
    import applied_checks as _A
    if pid in _A.CHECKS:                       # 本教程自建的题：直接用对应编号的提交桩
        tid = next(t for t, v in list(TASKS.items()) + list(APPLIED.items()) if (v[2] if len(v) == 5 else v[1]) == pid)
        return main_check(tid)
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(b["starter"])
        sys.exit(f"已生成桩文件 {f.relative_to(ROOT.parent)}。先填写实现，再运行本命令。")
    try:
        ns = load(f, b["entries"])
    except Exception as e:
        sys.exit(f"导入 {f.name} 失败: {type(e).__name__}: {e}")
    import torchleet
    print(f"\n  核验 {b['title']}")
    ok = run_torchleet(pid, ns)
    print("  🎉 通过！\n" if ok else f"  还没通过；需要提示: python study/board.py tl hint {pid}\n")


def applied(cmd, tid, a, spec=None):
    sys.path.insert(0, str(ROOT))
    import applied_checks as A
    title, pid, f = spec or APPLIED[tid]
    spec = A.CHECKS[pid]
    if cmd == "hint":
        hs = spec["hints"]
        n = int(a[2]) if len(a) > 2 else 1
        return print(f"  hint {n}/{len(hs)}: {hs[min(max(n, 1), len(hs)) - 1]}")
    if cmd != "check":
        sys.exit(__doc__)
    try:
        ns = load(ROOT / "submissions" / f, spec["entries"])
    except Exception as e:
        sys.exit(f"导入 {f} 失败: {type(e).__name__}: {e}")
    missing = [n for n in spec["entries"] if n not in ns]
    if missing:
        sys.exit(f"{f} 里缺少: {', '.join(missing)}")
    import types
    ns = types.SimpleNamespace(**ns)
    print(f"\n  核验 {tid} {title}\n")
    bad = 0
    for fn in spec["checks"]:
        name = fn.__name__.replace("check_", "").replace("_", " ")
        try:
            fn(ns)
            print(f"   ✅ {name}")
        except Exception as e:
            bad += 1
            msg = str(e) if isinstance(e, AssertionError) and str(e) else f"{type(e).__name__}: {e}"
            print(f"   ❌ {name} — {msg}")
    emit(pid, len(spec["checks"]) - bad, len(spec["checks"]))
    if bad:
        print(f"\n  {bad} 项未通过。需要提示: python study/board.py hint {tid} [1-{len(spec['hints'])}]\n")
        return
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    d = passed(); d[pid] = {"partial": False}
    PROGRESS.write_text(json.dumps(d))
    print("\n  🎉 全部通过！\n")


def main_check(tid):
    sys.argv = [sys.argv[0], "check", tid]
    return main()


def main():
    a = sys.argv[1:]
    if not a:
        return show()
    if a[0] == "tl":
        return tl(a[1:])
    cmd, tid = a[0], (a[1] if len(a) > 1 else None)
    if tid in APPLIED:
        return applied(cmd, tid, a)
    if tid not in TASKS:
        sys.exit(f"未知编号 {tid}，可选: {', '.join(list(TASKS) + list(APPLIED))}")
    _, title, pid, f, names = TASKS[tid]
    sys.path.insert(0, str(ROOT))
    import applied_checks as _A
    if pid in _A.CHECKS:                       # 本仓库自带的判题（非 torchleet 包）
        return applied(cmd, tid, a, spec=(title, pid, f))
    import torchleet
    if cmd == "hint":
        return torchleet.hint(pid, int(a[2]) if len(a) > 2 else None)
    if cmd != "check":
        sys.exit(__doc__)
    try:
        ns = load(ROOT / "submissions" / f, names)
    except Exception as e:
        sys.exit(f"导入 {f} 失败: {type(e).__name__}: {e}")
    missing = [n for n in names if n not in ns]
    if missing:
        sys.exit(f"{f} 里缺少: {', '.join(missing)}")
    print(f"\n  核验 {tid} {title}")
    ok = run_torchleet(pid, ns)
    print("  🎉 通过！" if ok else "  还没通过，看上面的失败信息再试；需要提示: python study/board.py hint " + tid)


if __name__ == "__main__":
    main()
