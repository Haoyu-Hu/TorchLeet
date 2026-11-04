"""沙箱子进程：加载一份提交并跑判题，把结果以一行 JSON 打到 stdout。
由 run_submission.py 调用；不要直接信任被判代码，调用方负责隔离与超时。"""
import contextlib, importlib.util, io, json, sys, traceback, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "study")]   # 不放仓库根目录，避免 torch/ 题目文件夹遮住 PyTorch

MARK = "@@RESULT@@"


def out(d):
    sys.stdout = sys.__stdout__
    print(MARK + json.dumps(d, ensure_ascii=False))
    sys.exit(0)


def main(pid, path):
    buf = io.StringIO()
    res = {"problem": pid, "status": "error", "passed": 0, "total": 0, "skipped": 0, "checks": [], "error": None}
    try:
        import applied_checks as A
        if pid in A.CHECKS:
            entries, kind = A.CHECKS[pid]["entries"], "applied"
        else:
            bank = {b["id"]: b for b in json.loads((ROOT / "study" / "problem_bank.json").read_text())}
            if pid not in bank or not bank[pid]["graded"]:
                res["error"] = f"未知或没有自动判题的题目: {pid}"
                out(res)
            entries, kind = bank[pid]["entries"], "torchleet"
        spec = importlib.util.spec_from_file_location("submission", path)
        mod = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                spec.loader.exec_module(mod)
            except BaseException as e:
                res["error"] = f"导入失败: {type(e).__name__}: {e}"
                res["log"] = buf.getvalue()[-1500:]
                out(res)
            ns = {n: getattr(mod, n) for n in entries if hasattr(mod, n)}
            missing = [n for n in entries if n not in ns]
            if missing:
                res["error"] = "缺少需要实现的名字: " + ", ".join(missing)
                res["log"] = buf.getvalue()[-1500:]
                out(res)
            rows = []
            if kind == "applied":
                sn = types.SimpleNamespace(**ns)
                for fn in A.CHECKS[pid]["checks"]:
                    name = fn.__name__.replace("check_", "").replace("_", " ")
                    try:
                        fn(sn); rows.append((name, "pass", ""))
                    except AssertionError as e:
                        rows.append((name, "fail", str(e) or "assertion failed"))
                    except Exception as e:
                        rows.append((name, "fail", f"{type(e).__name__}: {e}"))
            else:
                from torchleet.runner import run
                rows = run(pid, ns)
        res["checks"] = [{"name": n, "status": s, "detail": d[:400]} for n, s, d in rows]
        p = sum(1 for r in rows if r[1] == "pass"); f = sum(1 for r in rows if r[1] == "fail"); k = sum(1 for r in rows if r[1] == "skip")
        res.update(passed=p, total=p + f + k, skipped=k, log=buf.getvalue()[-1500:])
        res["status"] = "fail" if f else ("pass" if p and not k else ("partial" if p else "error"))
        if not rows:
            res["error"] = "没有可运行的检查"
    except SystemExit:
        raise
    except BaseException:
        res["error"] = traceback.format_exc()[-1500:]
    out(res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
