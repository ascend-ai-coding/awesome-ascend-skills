"""T0 脚本检查:全文档内链/锚点可达性(category C1)。确定性、不调 LLM、~0 token。

产出 skill 标准 finding(category=C1,cls=quantifiable,axis=findable,
verdict=CONFIRMED_MISMATCH,带 impact)。这是「脚本铺底」策略的 T0 之一:机械类不进 LLM。

用法:cd skills/tech-docs-guard && python -m scripts.linkcheck <repo_root> [--json out.json]
"""
from __future__ import annotations

import os
import re
import sys
import json
import urllib.parse
from pathlib import Path

_EXCLUDE = ("/build/", "/third_party/", "/.git/", "/node_modules/", "/dist/", "/build_out/")
_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.M)


def _slug(text: str) -> str:
    t = re.sub(r"[`*_~]", "", text.strip().lower())
    t = re.sub(r"[^\w一-鿿 \-]", "", t)
    return t.replace(" ", "-")


def _heading_slugs(p: Path) -> set:
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    slugs, seen = set(), {}
    for h in _HEADING.findall(txt):
        s = _slug(h)
        n = seen.get(s, 0)
        slugs.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return slugs


def _is_external(t: str) -> bool:
    return t.startswith(("http://", "https://", "mailto:", "ftp://", "tel:"))


def find_broken_links(repo_root: str, under: str | None = None) -> list[dict]:
    """返回 skill 标准 finding 列表(category C1)。

    under: 限定只扫 `<repo_root>/<under>` 子树(与 P0 的 `--under docs` 范围对齐);None=全仓。
    链接仍按各文档自身位置相对解析,under 只缩小「被检文档」集合。
    """
    root = Path(repo_root).resolve()
    base = (root / under) if under else root
    mds = [p for p in base.rglob("*.md") if not any(x in str(p).replace(os.sep, "/") for x in _EXCLUDE)]
    out, slug_cache = [], {}
    for p in mds:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root))
        for m in _LINK.finditer(txt):
            is_img = m.group(0).startswith("!")
            tgt = re.sub(r"\s*\n\s*", "", m.group(2)).strip()        # 折叠换行
            if " " in tgt:
                tgt = tgt.split(" ", 1)[0]
            if not tgt or tgt.startswith("#") or _is_external(tgt):
                continue
            if tgt.startswith("<") and tgt.endswith(">"):
                tgt = tgt[1:-1]
            path_part, _, anchor = tgt.partition("#")
            path_part = urllib.parse.unquote(path_part)
            anchor = urllib.parse.unquote(anchor)
            # 只查像文件引用的目标(排除表格/公式里的 [文本](x))
            if not (path_part.endswith("/") or path_part.startswith(("./", "../"))
                    or re.search(r"\.[A-Za-z0-9]{1,6}($|/)", path_part)):
                continue
            try:
                resolved = (p.parent / path_part).resolve() if path_part else p
            except (OSError, ValueError):
                continue
            line = txt[:m.start()].count("\n") + 1
            quote = m.group(0)[:120]
            if not resolved.exists():
                out.append({
                    "category": "C1", "cls": "quantifiable", "axis": "findable",
                    "form": "错", "source": "code_mismatch",
                    "verdict": "CONFIRMED_MISMATCH", "evidence_grade": "strong",
                    "quote": quote,
                    "code_location": f"{rel}:{line} -> 目标不存在: {tgt}",
                    "improvement": ("图片缺失,补图或删链接。" if is_img
                                    else f"链接目标 `{tgt}` 在仓内不存在,修正路径或补文件/删链接。"),
                    "impact": "minor" if is_img else "misleading",
                    "root_cause": None,
                })
            elif anchor and resolved.suffix.lower() == ".md":
                key = str(resolved)
                if key not in slug_cache:
                    slug_cache[key] = _heading_slugs(resolved)
                if _slug(anchor) not in slug_cache[key] and anchor.lower() not in slug_cache[key]:
                    out.append({
                        "category": "C1", "cls": "quantifiable", "axis": "findable",
                        "form": "错", "source": "code_mismatch",
                        "verdict": "CONFIRMED_MISMATCH", "evidence_grade": "strong",
                        "quote": quote,
                        "code_location": f"{rel}:{line} -> 目标文件存在但锚点不存在: #{anchor}",
                        "improvement": f"目标文件无 `#{anchor}` 对应标题,修正锚点或补该小节。",
                        "impact": "misleading",
                        "root_cause": None,
                    })
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 -m scripts.linkcheck <repo_root> [--under docs] [--json out.json]", file=sys.stderr)
        return 2
    under = None
    if "--under" in sys.argv:
        j = sys.argv.index("--under") + 1
        if j >= len(sys.argv):
            print("--under 需要一个子目录参数", file=sys.stderr)
            return 2
        under = sys.argv[j]
    fs = find_broken_links(sys.argv[1], under)
    from collections import Counter
    anchors = sum(1 for f in fs if "锚点不存在" in (f.get("code_location") or ""))
    print(f"linkcheck: 扫到死链/死锚 {len(fs)} 条(C1;文件 {len(fs)-anchors} / 锚点 {anchors})")
    print("  impact:", dict(Counter(f["impact"] for f in fs)))
    if "--json" in sys.argv:
        i = sys.argv.index("--json") + 1
        if i >= len(sys.argv):
            print("--json 需要一个输出路径参数", file=sys.stderr)
            return 2
        out = sys.argv[i]
        Path(out).write_text(json.dumps(fs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
