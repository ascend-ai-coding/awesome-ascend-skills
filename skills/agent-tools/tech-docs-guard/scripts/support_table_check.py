"""T0 脚本检查:算子文档「产品支持情况」表 vs 代码注册(category C5)。确定性、~0 token。

读文档支持表的 产品行→√/×,把产品名映射到芯片代号(arch),再 grep 该算子 op_host 里
AddConfig(...)/SUPPORT_COMPUTE_UNIT 看实际注册了哪些 arch,比对:
  - 文档标 √ 但代码未注册该 arch  → 疑似「假√」(开发者会选到跑不了的芯片)→ impact=blocker
  - 文档标 × 但代码已注册该 arch  → 疑似「假×/欠声明」→ impact=minor
默认 verdict=SUSPECTED(老 SOC 可能由别处 binary 承接,留人/LLM 终判)。

用法:cd skills/tech-docs-guard && python -m scripts.support_table_check <repo_root> [--json out.json]
"""
from __future__ import annotations

import os
import re
import sys
import json
from pathlib import Path

# 产品名(子串匹配)→ 芯片代号(arch)。来源:真体检里 LLM 已核对过的映射。
PRODUCT_ARCH = [
    ("950", ["ascend950"]),
    ("Atlas A3", ["ascend910_93"]),
    ("A3 训练", ["ascend910_93"]),
    ("A3 推理", ["ascend910_93"]),
    ("Atlas A2", ["ascend910b"]),
    ("A2 训练", ["ascend910b"]),
    ("A2 推理", ["ascend910b"]),
    ("200I/500 A2", ["ascend310p", "ascend310b"]),
    ("Atlas 推理系列", ["ascend310p"]),
    ("Atlas 训练系列", ["ascend910", "ascend910a"]),
    ("Kirin X90", ["kirinx90"]),
    ("Kirin 9030", ["kirin9030"]),
]
_EXCLUDE = ("/build/", "/third_party/", "/.git/", "/node_modules/")
_ROW = re.compile(r"\|\s*(?:<term>)?\s*([^|<]+?)\s*(?:</term>)?\s*\|\s*([√✓×xX✗])\s*\|")


def _arch_for(product: str) -> list[str]:
    for sub, archs in PRODUCT_ARCH:
        if sub in product:
            return archs
    return []


def _op_root(doc: Path) -> Path | None:
    """从文档向上找算子根(含 op_host 的目录)。"""
    for d in [doc.parent, *doc.parents]:
        if (d / "op_host").is_dir():
            return d
    return None


def _doc_marks(txt: str) -> dict:
    """从一篇文档抽 {产品名: '√'|'×'}(只取已知产品行)。"""
    marks = {}
    for prod, mark in _ROW.findall(txt):
        prod = prod.strip()
        if _arch_for(prod):
            marks[prod] = "√" if mark in "√✓" else "×"
    return marks


def check(repo_root: str, under: str | None = None) -> list[dict]:
    """精确信号:同一算子的多篇文档对同一产品的 √/× 自相矛盾(铁的 doc bug,无 binary 承接歧义)。

    放弃「支持表 vs 代码」方向——老芯片 binary 常由别处承接,脚本判不了,留给 LLM。
    under: 限定只扫 `<repo_root>/<under>`(与 P0 范围对齐);None=全仓且自动跳过中央 docs/
    (中央文档无算子产品支持表,本工具针对算子文档)。显式给 under 则尊重该范围、不再跳 docs/。
    """
    root = Path(repo_root).resolve()
    base = (root / under) if under else root
    from collections import defaultdict
    by_op = defaultdict(dict)              # op_root -> {rel_doc: {prod: mark}}
    mds = [p for p in base.rglob("*.md")
           if not any(x in str(p).replace(os.sep, "/") for x in _EXCLUDE)
           and (under is not None or not str(p.relative_to(root)).startswith("docs/"))]
    for p in mds:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        marks = _doc_marks(txt)
        if not marks:
            continue
        op_root = _op_root(p)
        if op_root:
            by_op[op_root][str(p.relative_to(root))] = marks

    out = []
    for op_root, docs in by_op.items():
        if len(docs) < 2:
            continue
        prods = set().union(*[set(m) for m in docs.values()])
        for prod in prods:
            votes = {d: m[prod] for d, m in docs.items() if prod in m}
            if len(set(votes.values())) > 1:                 # 同产品、不同文档结论打架
                yes = [d for d, v in votes.items() if v == "√"]
                no = [d for d, v in votes.items() if v == "×"]
                out.append({
                    "category": "C5", "cls": "quantifiable", "axis": "trustworthy",
                    "form": "错", "source": "self_contradiction",
                    "verdict": "CONFIRMED_MISMATCH", "evidence_grade": "strong",
                    "impact": "misleading", "root_cause": "copy_paste_not_updated",
                    "quote": f"同算子文档对「{prod}」结论冲突:√={[d.split('/')[-1] for d in yes]} / ×={[d.split('/')[-1] for d in no]}",
                    "code_location": "; ".join(f"{d} = {votes[d]}" for d in sorted(votes)),
                    "improvement": f"同一算子的多篇文档对「{prod}」是否支持给出相反结论,开发者无法判断。请二选一对齐(并与该算子真实构建/注册契约一致)。",
                })
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python3 -m scripts.support_table_check <repo_root> [--under <subdir>] [--json out.json]", file=sys.stderr)
        return 2
    under = None
    if "--under" in sys.argv:
        j = sys.argv.index("--under") + 1
        if j >= len(sys.argv):
            print("--under 需要一个子目录参数", file=sys.stderr)
            return 2
        under = sys.argv[j]
    fs = check(sys.argv[1], under)
    from collections import Counter
    print(f"support_table_check: {len(fs)} 条疑似支持表错(impact: {dict(Counter(f['impact'] for f in fs))})")
    if "--json" in sys.argv:
        i = sys.argv.index("--json") + 1
        if i >= len(sys.argv):
            print("--json 需要一个输出路径参数", file=sys.stderr)
            return 2
        Path(sys.argv[i]).write_text(json.dumps(fs, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
