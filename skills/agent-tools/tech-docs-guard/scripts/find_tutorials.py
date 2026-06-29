"""P0:发现待评文档,供 agent 让用户确认评哪些。两种模式:

- **`discover_docs(repo, under="docs")`【范围默认】**:枚举 `<repo>/docs/` 下**全部**技术 .md
  (含 context/install/invocation,非只教程),严格限定在 docs/ 子树。对应 SKILL.md「范围默认 docs/」。
- `find_tutorials(repo)`:**全仓**按启发式找「进阶教程」(文件名/标题像开发指南/教程);旧逻辑,
  非默认范围(会越出 docs/、又漏掉 docs/ 里的非教程文档)。

**零硬编码**:不写死 `docs/zh/develop`。只发现,不评质量。
用法:`python -m scripts.find_tutorials <repo_root> --under docs --json`(默认范围)/ 无 `--under` 走教程启发式。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 文件名命中(小写):进阶教程的典型命名
NAME_INCLUDE = ["develop_guide", "develop-guide", "dev_guide", "devguide",
                "tutorial", "进阶", "开发指南", "开发教程", "教程", "develop_tutorial"]
# 文件名排除:这些不是进阶教程
NAME_EXCLUDE = ["quickstart", "quick_start", "quick-start", "快速入门", "快速开始",
                "readme", "op_list", "op_api_list", "api_list", "changelog",
                "contributing", "security", "license", "faq", "release", "notice"]
# 标题命中(内容):标题里出现这些 → 候选(低优先于文件名命中)
TITLE_INCLUDE = re.compile(r"^#{1,2}\s.*(开发指南|开发教程|进阶|develop\s*guide|tutorial)",
                           re.IGNORECASE | re.MULTILINE)
# 噪声目录 + 明显的 how-to/参考目录名(按目录名启发,不写死具体路径)
SKIP_DIR_NAMES = {".git", "node_modules", "build", "build_out", "dist",
                  "third_party", "3rd", "__pycache__", ".github", "figures", "images"}


def _skip(p: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in p.parts)


def _title(text: str) -> str:
    m = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def find_tutorials(repo_root: str) -> list[dict]:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repo root not found: {root}")
    out, seen = [], set()
    for p in root.rglob("*.md"):
        if _skip(p):
            continue
        low = p.name.lower()
        if any(k in low for k in NAME_EXCLUDE):
            continue
        rel = str(p.relative_to(root))
        match = None
        if any(k in low for k in NAME_INCLUDE):
            match = "filename"
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if TITLE_INCLUDE.search(text):
                match = "title"
        if match and rel not in seen:
            seen.add(rel)
            try:
                title = _title(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                title = ""
            out.append({"path": rel, "abs_path": str(p), "match": match, "title": title})
    out.sort(key=lambda d: (0 if d["match"] == "filename" else 1, d["path"]))
    return out


def discover_docs(repo_root: str, under: str = "docs") -> list[dict]:
    """范围默认:枚举 `<repo>/<under>` 下**全部**技术 .md(只排噪声目录),不做教程启发式。

    对应 SKILL.md P0「范围默认只取根 docs/ 中央文档」——含 context/install/invocation 等
    **非教程**文档,且严格限定在 `<repo>/<under>` 子树内(不像 find_tutorials 那样 rglob 全仓)。
    """
    root = Path(repo_root).resolve()
    base = root / under
    if not base.is_dir():
        raise FileNotFoundError(f"docs dir not found: {base}")
    out = []
    for p in sorted(base.rglob("*.md")):
        if _skip(p):
            continue
        try:
            title = _title(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            title = ""
        out.append({"path": str(p.relative_to(root)), "abs_path": str(p), "match": "docs", "title": title})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="发现待评文档:默认 --under docs 取 docs/ 全量;无 --under 则全仓找进阶教程")
    ap.add_argument("repo_root")
    ap.add_argument("--under", help="只取该子目录下全部技术 .md(范围默认 'docs');不传则全仓找进阶教程启发式")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        docs = discover_docs(args.repo_root, args.under) if args.under else find_tutorials(args.repo_root)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(docs, ensure_ascii=False, indent=2))
    else:
        if not docs:
            print("(未发现任何进阶教程——覆盖度缺口,属「找得到」轴缺陷)")
        for d in docs:
            print(f"[{d['match']:8s}] {d['path']}    {d['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
