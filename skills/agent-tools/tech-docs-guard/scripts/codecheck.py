"""P2:可量化对照代码(信得过的「事实闸」)。

输入 agent 从教程抽出的一个「具体物」(命令/flag/路径/配置键/符号),输出:
  triage(分诊) + 证据等级(强/中/弱/无) + 命中位置 + 近似变体 + **建议三态**。

**纪律(贯穿全文件)**:本工具**绝不自己下 `CONFIRMED_MISMATCH`**——grep 不到 ≠ 编造。
它只给证据 + 近似变体;「确认对不上」由 agent 看到强反证(如近似变体存在、文档写的那个没有)后自己升。
建议三态只在 {CONSISTENT, SUSPECTED, NO_STATIC_EVIDENCE} 里挑。

用法:python -m scripts.codecheck --code-root <path> --token '<x>' --kind <command|flag|path|config_key|symbol|link> [--json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

_EXCLUDE_DIRS = [".git", "build", "build_out", "dist", "third_party", "3rd",
                 "__pycache__", "node_modules", ".github"]
_EXTERNAL_CMDS = {"cmake", "bash", "sh", "pip", "pip3", "git", "python", "python3",
                  "make", "gcc", "g++", "sudo", "source", "cd", "ls", "echo", "export",
                  "mkdir", "rm", "cp", "mv", "tar", "wget", "curl", "conda", "apt", "yum"}


def triage(token: str, kind: str) -> str:
    """先分诊:占位符 / 外部命令 / 生成产物 / 仓内对象。只有仓内对象才去取证。"""
    t = token.strip()
    if re.search(r"<[^>]+>|\$\{?\w+\}?|\{[a-z_]+\}|your[_-]|xxx|/path/to/", t, re.IGNORECASE):
        return "placeholder"
    if kind in ("path", "link") and re.search(r"(^|/)(build|build_out|dist|out)(/|$)", t):
        return "generated"
    if kind == "command":
        head = t.split()[0] if t.split() else t
        if head in _EXTERNAL_CMDS:
            return "external"
    return "repo_object"


def _is_test(path: str) -> bool:
    parts = path.lower().split("/")
    base = parts[-1]
    # 只认「目录组件恰为 test/tests」或「文件名 test_*/*_test.*」,不被父路径里的 test_ 误伤
    return "test" in parts or "tests" in parts or base.startswith("test_") or "_test." in base


def _is_doc(path: str) -> bool:
    return path.lower().endswith((".md", ".rst", ".txt"))


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith(("#", "//", "*", "<!--", "/*", '"""', "'''"))


def _is_def(line: str, token: str, kind: str) -> bool:
    """这行是不是「定义/入口」(强证据),而非引用。"""
    esc = re.escape(token)
    if kind in ("flag", "command"):
        # argparse 的 --flag、shell case 的 "--flag")、getopts
        return bool(re.search(r"add_argument\(|getopts|\)\s*$|case\s|;;|\"" + esc + r"\"\)|'" + esc + r"'\)", line)) \
            or bool(re.search(r"add_argument\(\s*['\"]" + esc, line))
    if kind == "config_key":
        return bool(re.search(esc + r"\s*[:=]", line)) or bool(re.search(r"add_argument\(\s*['\"]", line))
    if kind == "symbol":
        return bool(re.search(r"\bdef\s+" + esc + r"\s*\(", line)) \
            or bool(re.search(r"\w[\w:<>\*&\s]*\b" + esc + r"\s*\([^;]*\)\s*\{?\s*$", line))
    return False


def _grep(code_root: str, token: str) -> list[tuple[str, int, str]]:
    cmd = ["grep", "-rnIF"] + [f"--exclude-dir={d}" for d in _EXCLUDE_DIRS] + ["--", token, code_root]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    hits = []
    for line in res.stdout.splitlines()[:400]:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            hits.append((parts[0], int(parts[1]), parts[2]))
    return hits


def _near_variants(code_root: str, token: str, kind: str) -> list[str]:
    """文档写的 token 没强命中时,找代码里**近似**的真实物(如 --opkernel_test → --opkernel)。"""
    # 注意:grep -E 用 POSIX 字符类(`\w` 是 GNU 扩展,BSD/macOS grep 不认)
    if kind in ("flag", "command"):
        if not token.startswith("--"):
            return []
        pat, harvest = "--[[:alnum:]_][[:alnum:]_-]+", token
    elif kind == "symbol":
        pat, harvest = "[[:alpha:]_][[:alnum:]_]{3,}", token
    else:
        return []
    found = set()
    cmd = ["grep", "-rhoIE"] + [f"--exclude-dir={d}" for d in _EXCLUDE_DIRS] + ["--", pat, code_root]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        for tok in res.stdout.split():
            found.add(tok)
    except (OSError, subprocess.TimeoutExpired):
        return []
    found.discard(harvest)
    return difflib.get_close_matches(harvest, sorted(found), n=3, cutoff=0.7)


def check(code_root: str, token: str, kind: str) -> dict:
    tri = triage(token, kind)
    base = {"token": token, "kind": kind, "triage": tri,
            "grade": "none", "code_location": None, "locations": [],
            "near_variants": [], "suggested_verdict": "NO_STATIC_EVIDENCE", "note": ""}

    if tri == "placeholder":
        base["note"] = "占位符——不按字面 grep;改查文档有没有解释取值来源"
        return base
    if tri == "external":
        base["note"] = "外部命令——按环境依赖,grep 不到≠缺陷"
        return base
    if tri == "generated":
        base["note"] = "疑似生成/安装产物——不直接判对不上"
        return base

    # 仓内对象
    if kind in ("path", "link"):
        p = (Path(code_root) / token).expanduser()
        if p.exists():
            base.update(grade="strong", code_location=str(p), suggested_verdict="CONSISTENT",
                        note="仓内路径存在")
        else:
            base.update(suggested_verdict="SUSPECTED",
                        note="仓内静态路径未找到(疑似缺失;也可能是生成/安装产物,agent 定)")
        return base

    hits = _grep(code_root, token)
    if not hits:
        nv = _near_variants(code_root, token, kind)
        base.update(near_variants=nv, suggested_verdict="SUSPECTED",
                    note=("代码未命中" + (f";但有近似变体 {nv} → 很可能对不上,agent 核实后可升 CONFIRMED" if nv
                                        else ";无近似变体 → 疑似缺失/可能动态生成")))
        return base

    # 给命中分级:def(强) > 引用(中) > 注释/测试/doc(弱),取最高
    order = {"weak": 0, "medium": 1, "strong": 2}
    grade, loc, best = "weak", None, -1
    for path, ln, content in hits:
        if _is_comment(content) or _is_test(path) or _is_doc(path):
            ctx = "weak"
        elif _is_def(content, token, kind):
            ctx = "strong"
        else:
            ctx = "medium"
        if order[ctx] > best:
            best, grade, loc = order[ctx], ctx, f"{path}:{ln}"
    base.update(grade=grade, code_location=loc,
                locations=[f"{p}:{l}" for p, l, _ in hits[:5]])
    if grade == "strong":
        base.update(suggested_verdict="CONSISTENT", note="代码里有真定义/入口 → 一致")
    else:
        base.update(suggested_verdict="SUSPECTED",
                    note=f"只在{'引用' if grade == 'medium' else '注释/测试/文档'}命中,非定义 → 疑似,需核")
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="可量化对照代码(事实闸)")
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--kind", required=True,
                    choices=["command", "flag", "path", "config_key", "symbol", "link"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = check(args.code_root, args.token, args.kind)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f"[{r['triage']}] {r['token']}  grade={r['grade']}  →建议 {r['suggested_verdict']}")
        print(f"  位置: {r['code_location']}  近似: {r['near_variants']}")
        print(f"  {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
