#!/usr/bin/env python3
"""流水线执行器：把一份提交放进隔离子进程里判题，打印一个 JSON。

    python study/run_submission.py --problem lin-regression --file path/to/code.py [--id sub_xxx] [--timeout 120]

隔离措施：临时目录为 cwd/HOME、清空环境变量（不带任何令牌）、有墙钟超时与 CPU 上限；
若系统支持，则放进没有网络的新 user+net 命名空间。提交代码仍然是“任意代码”，
只应该运行你自己的提交。
"""
import argparse, datetime, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "@@RESULT@@"


def can_unshare():
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--id", default="local")
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="tl-sub-"))
    try:
        shutil.copy(a.file, tmp / "submission.py")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp), "TMPDIR": str(tmp),
               "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
        base = [sys.executable, "-I", str(HERE / "_worker.py"), a.problem, str(tmp / "submission.py")]
        cmd = ["prlimit", f"--cpu={a.timeout * 2}"] + base
        isolated = can_unshare()
        if isolated:
            cmd = ["unshare", "-rn"] + cmd
        t0 = time.time()
        res = {"problem": a.problem, "status": "error", "passed": 0, "total": 0, "skipped": 0, "checks": []}
        try:
            p = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=a.timeout)
            line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(MARK)), None)
            if line:
                res = json.loads(line[len(MARK):])
            else:
                res["error"] = "判题进程异常退出（可能是内存不足、被强制终止或代码调用了 exit）"
                res["log"] = (p.stdout + p.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            res.update(status="timeout", error=f"超过 {a.timeout} 秒仍未结束：请删掉模块顶层的训练循环/演示代码，只保留定义。")
        res.update(id=a.id, seconds=round(time.time() - t0, 1), isolated_network=isolated,
                   ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
        print(json.dumps(res, ensure_ascii=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
