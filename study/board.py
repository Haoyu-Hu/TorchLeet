#!/usr/bin/env python3
"""学习看板: 查看进度 / 提交核验。

    python study/board.py            # 看板（进度总览）
    python study/board.py check 1.1  # 核验 submissions/ 里对应的代码
    python study/board.py hint 1.1   # 逐级提示（不会给答案）

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
    n = sum(1 for t in TASKS.values() if t[2] in d)
    print(f"\n  完成 {n}/{len(TASKS)}   核验: python study/board.py check <编号>\n")


def load(path, names):
    spec = importlib.util.spec_from_file_location("submission", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {n: getattr(m, n) for n in names if hasattr(m, n)}


def main():
    a = sys.argv[1:]
    if not a:
        return show()
    cmd, tid = a[0], (a[1] if len(a) > 1 else None)
    if tid not in TASKS:
        sys.exit(f"未知编号 {tid}，可选: {', '.join(TASKS)}")
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
