#!/usr/bin/env python3
"""从 problems.json + 题目 notebook + python/torchleet 判题规格 生成 study/problem_bank.json。

只读取"题目"notebook（不含 _SOLN 答案），也不需要安装 PyTorch。
    python study/build_bank.py
"""
import ast, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "python" / "torchleet" / "problems"


def module_name(pid):
    m = pid.replace("-", "_")
    return f"_{m}" if m[:1].isdigit() else m


def read_spec(pid):
    f = SPECS / f"{module_name(pid)}.py"
    if not f.exists():
        return None
    tree = ast.parse(f.read_text())
    out = {"entries": [], "hints": [], "checks": []}
    funcs = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("check_"):
            funcs[n.name] = (ast.get_docstring(n) or "").strip()
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            k = n.targets[0].id
            try:
                if k == "ENTRIES":
                    out["entries"] = list(ast.literal_eval(n.value))
                elif k == "HINTS":
                    out["hints"] = list(ast.literal_eval(n.value))
                elif k == "DEVICE":
                    out["device"] = ast.literal_eval(n.value)
                elif k == "EXTRAS":
                    out["extras"] = list(ast.literal_eval(n.value))
            except Exception:
                pass
    common = {}
    cf = ROOT / "python" / "torchleet" / "checks" / "common.py"
    for n in ast.parse(cf.read_text()).body:
        if isinstance(n, ast.FunctionDef):
            common[n.name] = (ast.get_docstring(n) or "").strip()
    items = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CHECKS" for t in n.targets):
            for e in getattr(n.value, "elts", []):
                if isinstance(e, ast.Name):                       # 直接写的 check_xxx
                    items.append((e.id.replace("check_", "").replace("_", " "), funcs.get(e.id, "")))
                elif isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute):   # 工厂: c.output_shape(...)
                    items.append((e.func.attr.replace("_", " "), common.get(e.func.attr, "")))
    if not items:
        items = [(c.replace("check_", "").replace("_", " "), d) for c, d in funcs.items()]
    out["checks"] = [{"name": a, "doc": b} for a, b in items]
    return out


def clean_md(src):
    src = re.sub(r"<details.*?</details>", "", src, flags=re.S)       # 提示用我们自己的分级提示
    src = re.sub(r"<!--.*?-->", "", src, flags=re.S)
    src = re.sub(r"<img[^>]*>", "", src)
    src = re.sub(r"</?(?:p|div|span|br|summary|center|b|i|u|sub|sup)[^>]*>", "", src)
    return src.strip()


def split_cells(nb_path):
    cells = json.loads(Path(nb_path).read_text())["cells"]
    md, code = [], []
    for c in cells:
        s = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if "torchleet:colab" in s or not s.strip():
                continue
            s = clean_md(s)
            if s:
                md.append(s)
        elif s.strip():
            code.append(s)
    return md, code


def starter(pid, title, entries, code):
    defs = lambda e: re.compile(rf"^\s*(?:class|def)\s+{re.escape(e)}\b|^{re.escape(e)}\s*=", re.M)
    last = -1
    found = set()
    for i, c in enumerate(code):
        for e in entries:
            if defs(e).search(c):
                last, _ = i, found.add(e)
    body = code[: last + 1] if last >= 0 else code
    miss = [e for e in entries if e not in found]
    text = f'"""{title}  ({pid})\n\n把 ... / TODO 填完后运行:  python study/board.py tl check {pid}\n只保留定义和 import；演示、训练循环、画图请删掉或放进 if __name__ == "__main__":\n"""\n'
    text += "\n\n".join(body)
    for e in miss:
        text += f"\n\n# TODO: 题目要求提供 `{e}`，请在这里定义\n"
    return text.rstrip() + "\n"


def main():
    P = json.loads((ROOT / "problems.json").read_text())["problems"]
    bank = []
    for p in P:
        spec = read_spec(p["id"])
        md, code = split_cells(ROOT / p["question"])
        entries = (spec or {}).get("entries") or p.get("entries") or []
        bank.append({
            "id": p["id"], "title": p["title"], "description": p["description"],
            "difficulty": p["difficulty"], "tracks": p["tracks"], "category": p.get("category", ""),
            "companies": p.get("companies", []), "order": p.get("llm_path_order"),
            "question": p["question"], "colab": p.get("colab", ""),
            "graded": spec is not None, "entries": entries,
            "device": (spec or {}).get("device", "cpu"), "extras": (spec or {}).get("extras", []),
            "hints": (spec or {}).get("hints") or p.get("hints", []),
            "checks": (spec or {}).get("checks", []),
            "statement": "\n\n".join(md),
            "starter": starter(p["id"], p["title"], entries, code),
        })
    (ROOT / "study" / "problem_bank.json").write_text(json.dumps(bank, ensure_ascii=False, indent=1))
    g = sum(b["graded"] for b in bank)
    print(f"{len(bank)} 题，其中 {g} 题有自动判题 → study/problem_bank.json")


if __name__ == "__main__":
    main()
