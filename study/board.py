#!/usr/bin/env python3
"""学习看板: 查看进度 / 提交核验。

    python study/board.py            # 看板（进度总览）
    python study/board.py check 1.1  # 核验 submissions/ 里对应的代码
    python study/board.py hint 1.1   # 逐级提示（不会给答案）
    编号 A.1 ~ A.9 是应用 ML 题（numpy/pandas，不需要 PyTorch）。

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


def applied(cmd, tid, a):
    sys.path.insert(0, str(ROOT))
    import applied_checks as A
    title, pid, f = APPLIED[tid]
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
    if bad:
        print(f"\n  {bad} 项未通过。需要提示: python study/board.py hint {tid} [1-{len(spec['hints'])}]\n")
        return
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    d = passed(); d[pid] = {"partial": False}
    PROGRESS.write_text(json.dumps(d))
    print("\n  🎉 全部通过！\n")


def main():
    a = sys.argv[1:]
    if not a:
        return show()
    cmd, tid = a[0], (a[1] if len(a) > 1 else None)
    if tid in APPLIED:
        return applied(cmd, tid, a)
    if tid not in TASKS:
        sys.exit(f"未知编号 {tid}，可选: {', '.join(list(TASKS) + list(APPLIED))}")
    _, title, pid, f, names = TASKS[tid]
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
    ok = torchleet.check(pid, **ns)
    print("  🎉 通过！" if ok else "  还没通过，看上面的失败信息再试；需要提示: python study/board.py hint " + tid)


if __name__ == "__main__":
    main()
