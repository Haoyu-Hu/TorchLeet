#!/usr/bin/env python3
"""部署自检：不需要任何真实题目的答案，验证提交流水线本身是否可用。

    python study/selftest.py

检查：正确提交能通过、错误提交会失败、死循环会超时、提交代码连不上网、
看不到环境变量里的令牌、PyTorch/numpy/pandas 是否装好。
退出码非 0 表示有项目失败；“网络隔离不可用”只给警告。
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_submission.py"
rows, bad = [], 0


def submit(code, timeout=60, problem="selftest-add"):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "sub.py"
        f.write_text(code)
        p = subprocess.run([sys.executable, str(RUN), "--problem", problem, "--file", str(f),
                            "--id", "selftest", "--timeout", str(timeout)], capture_output=True, text=True)
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"status": "error", "error": (p.stdout + p.stderr)[-500:]}


def report(name, ok, detail="", warn=False):
    global bad
    mark = "✅" if ok else ("⚠️ " if warn else "❌")
    if not ok and not warn:
        bad += 1
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


GOOD = "def add(a, b):\n    return a + b\n"
r = submit(GOOD)
report("正确的提交能通过", r.get("status") == "pass" and r.get("passed") == r.get("total") == 2, f"status={r.get('status')} {r.get('error') or ''}")

r = submit("def add(a, b):\n    return a - b\n")
report("错误的提交会失败", r.get("status") == "fail", f"status={r.get('status')}")

r = submit("while True:\n    pass\n" + GOOD, timeout=3)
report("死循环会被超时终止", r.get("status") == "timeout", f"status={r.get('status')}")

r = submit(GOOD + "\nimport socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('NETWORK_OPEN')\nexcept Exception as e:\n    print('NET_BLOCKED')\n")
iso = r.get("isolated_network")
log = r.get("log", "")
if "NETWORK_OPEN" in log:
    report("提交代码不能联网", False, "提交代码能访问外网：请确认 `unshare -rn true` 可用，或在隔离的容器里运行", warn=True)
else:
    report("提交代码不能联网", True, f"isolated_network={iso}")
if not iso:
    print("     ⚠️  没有网络隔离（unshare -rn 不可用）：只应运行你自己的提交。")

os.environ["SELFTEST_FAKE_TOKEN"] = "should-not-leak"
r = submit(GOOD + "\nimport os\nprint('LEAK' if 'SELFTEST_FAKE_TOKEN' in os.environ else 'CLEAN')\n")
report("提交代码看不到当前环境变量", "CLEAN" in r.get("log", "") and "LEAK" not in r.get("log", ""), r.get("log", "").strip()[:60])

for mod in ("numpy", "pandas", "torch"):
    p = subprocess.run([sys.executable, "-I", "-c", f"import {mod};print({mod}.__version__)"], capture_output=True, text=True, cwd="/")
    ok = p.returncode == 0
    report(f"已安装 {mod}", ok, p.stdout.strip() if ok else "没装：见 study/setup_env.sh", warn=(mod == "torch" and not ok))

print("\n  自检通过" if not bad else f"\n  有 {bad} 项失败")
sys.exit(1 if bad else 0)
