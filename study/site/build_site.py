#!/usr/bin/env python3
"""生成可发布的教程页面 study/site/dist/course.html。

    python study/site/build_site.py --session-id session_xxxxxxxx

模板里有两个占位符：
  __SESSION_ID__          提交流水线要把代码发往的 Claude Code 会话 ID（"从哪里取"见 study/DEPLOY.md）
  __PROBLEM_BANK_JSON__   study/problem_bank.json 的内容（先运行 python study/build_bank.py 更新它）
"""
import argparse, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True, help="形如 session_01AbC...")
    ap.add_argument("--out", default=str(HERE / "dist" / "course.html"))
    a = ap.parse_args()
    if not re.fullmatch(r"session_[A-Za-z0-9]+", a.session_id):
        raise SystemExit("session id 格式不对，应该形如 session_01AbCdEf...")
    tpl = (HERE / "course.template.html").read_text()
    for k in ("__SESSION_ID__", "__PROBLEM_BANK_JSON__"):
        assert tpl.count(k) == 1, f"模板里应恰好有一个 {k}"
    bank = json.loads((ROOT / "problem_bank.json").read_text())
    data = json.dumps(bank, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("__PROBLEM_BANK_JSON__", data).replace("__SESSION_ID__", a.session_id)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"已生成 {out}（{len(html) // 1024} KB，{len(bank)} 道题）")


if __name__ == "__main__":
    main()
