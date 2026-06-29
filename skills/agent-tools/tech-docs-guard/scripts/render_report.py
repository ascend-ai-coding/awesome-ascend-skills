"""P4:从 findings.json + doc_meta.json 渲染体检报告(Markdown + HTML,纯 stdlib)。

强制**自校验闸**:每条缺陷先过 `_state.self_check_finding`——可量化必带代码位置、不可量化
必带判例+steelman、「确认讲错」必带外部反证。**不过的不准进正文**,落到「待补」段。
报告分两段:事实问题(可量化,先列,可计数) / 教学判断(不可量化,后列,只定性)。

三项约定:
- **TE-1 未评≠合格**:`doc_meta.axes_evaluated`(本轮真评过的轴列表)若存在,不在其中的轴标
  「本轮未评」而非「合格」,避免把覆盖缺口伪装成通过;字段缺省=向后兼容(视作全评过)。
- **TE-2 跨轴去重**:多个 finder 在同一文档行命中同一处(如信得过+读得懂各报一遍)时,按
  `(cls, 文档行号)` 折叠,保留证据最强一条,其余记 `also_hit`,五轴计数不再虚高。
- **TE-3 双格式**:默认 `--format both`,同目录产 REPORT.md + REPORT.html(卡片式/三态色标/折叠)。

用法:python -m scripts.render_report --repo R [--out <...>/REPORT.md] [--format md|html|both]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _state  # noqa: E402

_CONFIRMED = {"CONFIRMED_MISMATCH", "CONFIRMED_CONCEPT_WRONG"}
_AX_ORDER = ["findable", "trustworthy", "learnable", "operable", "readable"]

# 默认舍弃 minor(瑕疵):问题量可控、聚焦阻断+误导。env 或 CLI --with-minor 可保留。
WITH_MINOR = os.environ.get("TECH_DOCS_GUARD_WITH_MINOR", "") == "1"

_VERDICT_ZH = {"CONFIRMED_MISMATCH": "确认对不上", "SUSPECTED": "疑似",
               "CONSISTENT": "一致", "NO_STATIC_EVIDENCE": "未找到静态证据",
               "TEACHING_JUDGMENT": "教学判断", "SUSPECTED_CONCEPT_RISK": "疑似概念风险",
               "CONFIRMED_CONCEPT_WRONG": "确认讲错"}


def _axis_grade(axis_findings: list) -> str:
    if any(f.get("verdict") in _CONFIRMED for f in axis_findings):
        return "不合格"
    if any(f.get("verdict") in _state.DEFECT_VERDICTS for f in axis_findings):
        return "有缺陷"
    return "合格"


def _score(f: dict) -> int:
    s = 10 if f.get("verdict") in _CONFIRMED else (5 if f.get("verdict") in _state.DEFECT_VERDICTS else 0)
    return s + {"trustworthy": 3, "findable": 2, "operable": 2}.get(f.get("axis"), 1)


def _doc_line(f: dict, docbase: str):
    """从 code_location 提取教程自身的行号(只认指向被评教程文件的 `<docbase>:NNN`)。"""
    if not docbase:
        return None
    m = re.search(re.escape(docbase) + r":(\d+)", f.get("code_location") or "")
    return int(m.group(1)) if m else None


def _dedup(findings: list, docbase: str) -> list:
    """TE-2:按 (cls, 教程行号) 折叠跨 finder/跨轴重复,保留证据最强一条,其余记 also_hit。
    无教程行号的(多为教学判断)原样保留——不强行合并不同处的判断。"""
    winners, seq = {}, []
    for f in findings:
        ln = _doc_line(f, docbase)
        if ln is None:
            seq.append(("F", f))
            continue
        key = (f.get("cls"), ln)
        if key not in winners:
            winners[key] = dict(f)            # 复制,避免改到 findings.json 原对象
            seq.append(("K", key))
        else:
            cur = winners[key]
            keeper, folded = (f, cur) if _score(f) > _score(cur) else (cur, f)  # 留证据最强,折叠另一条
            merged = dict(keeper)
            merged["also_hit"] = (keeper.get("also_hit", []) + folded.get("also_hit", [])
                                  + [{"axis": folded.get("axis"), "verdict": folded.get("verdict")}])
            winners[key] = merged
    return [winners[tok[1]] if tok[0] == "K" else tok[1] for tok in seq]


def _imp_rank(f: dict) -> int:
    """v4:按开发者影响排序,阻断(0)→误导(1)→瑕疵(2);未标按误导处理。"""
    return _state.IMPACT_ORDER.get(f.get("impact"), 1)


_IMP_LABEL = {"blocker": "🔴阻断", "misleading": "🟠误导", "minor": "⚪瑕疵"}


def _imp_label(f: dict) -> str:
    return _IMP_LABEL.get(f.get("impact"), "—")


def _partition(repo: str):
    """共享:读 meta/findings → 自校验闸分 valid/needs_fix → 跨轴去重 → 分可量化/不可量化。"""
    meta = _state.load_meta(repo)
    raw = _state.load_findings(repo)
    docbase = (meta.get("doc") or "").split("/")[-1]

    valid_raw, needs_fix = [], []
    for f in raw:
        probs = _state.self_check_finding(f)
        if probs:
            needs_fix.append((f, probs))
        else:
            valid_raw.append(f)

    valid = _dedup(valid_raw, docbase)
    defects = [f for f in valid if f.get("verdict") in _state.DEFECT_VERDICTS]
    if not WITH_MINOR:                                  # 默认舍弃 minor(瑕疵);--with-minor / env 可保留
        defects = [f for f in defects if f.get("impact") != "minor"]
    quant = [f for f in defects if f.get("cls") == "quantifiable"]
    nonquant = [f for f in defects if f.get("cls") == "non_quantifiable"]
    quant.sort(key=lambda f: (_imp_rank(f), f.get("verdict") != "CONFIRMED_MISMATCH", f.get("idx", 0)))
    return meta, valid, needs_fix, quant, nonquant


def _axis_cells(meta: dict, valid: list):
    """逐轴出 (中文名, 定性档, 可量化缺陷数显示)。TE-1:未评过的轴 → 本轮未评 / —。"""
    evaluated = meta.get("axes_evaluated")  # None=向后兼容(视作全评)
    for ax in _AX_ORDER:
        af = [f for f in valid if f.get("axis") == ax]
        if evaluated is not None and ax not in evaluated:
            yield _state.AXIS_ZH[ax], "本轮未评", "—"
        else:
            nq = len([f for f in af if f.get("cls") == "quantifiable" and f.get("verdict") in _state.DEFECT_VERDICTS])
            yield _state.AXIS_ZH[ax], _axis_grade(af), str(nq)


def _also(f: dict) -> str:
    hits = f.get("also_hit") or []
    return "、".join(f"{_state.AXIS_ZH.get(h.get('axis'), '?')}" for h in hits)


# ============================== Markdown ==============================

def _cell(s) -> str:
    return (str(s) if s is not None else "—").replace("|", "\\|").replace("\n", " ").strip() or "—"


def render(repo: str) -> str:
    meta, valid, needs_fix, quant, nonquant = _partition(repo)

    L = []
    L.append(f"# 进阶教程体检报告 — {repo}\n")
    L.append(f"> 教程:`{meta.get('doc', '(未记录)')}` · 类型:{meta.get('type', '?')} · "
             f"受众:{meta.get('audience', '?')} · 代码根:`{meta.get('code_root', '?')}`")
    L.append(f"> 生成:{datetime.now().isoformat(timespec='seconds')} · "
             "skill `cann-ops:tech-docs-guard` —— 通读+对照代码**静态**评(默认不跑);"
             "**不打玄学分、grep 不到≠编造**。\n")

    # 一、五轴总评
    L.append("## 一、五轴总评\n")
    L.append("| 轴 | 定性档 | 可量化缺陷数 |")
    L.append("|---|---|---|")
    for name, grade, cnt in _axis_cells(meta, valid):
        L.append(f"| {name} | {grade} | {cnt} |")
    L.append("")
    L.append(f"- **可量化缺陷**(高置信,可计数):{len(quant)}  ·  **教学判断**(只定性,不计分):{len(nonquant)}")
    _alldef = quant + nonquant
    _ic = lambda k: sum(1 for f in _alldef if (f.get("impact") or "") == k)
    _none = sum(1 for f in _alldef if not f.get("impact"))
    L.append(f"- **开发者影响**:🔴阻断 {_ic('blocker')} · 🟠误导 {_ic('misleading')} · ⚪瑕疵 {_ic('minor')}"
             + (f" · 未标 {_none}" if _none else "") + "(阻断优先修;瑕疵可缓)")
    if needs_fix:
        L.append(f"- ⚠ **{len(needs_fix)} 条未过自校验闸**(无证据/无判例),见文末「待补」,**不计入结论**")
    L.append("")

    # 二、事实问题（可量化）
    L.append("## 二、事实问题（可量化 · 对照代码 · 高置信）\n")
    if not quant:
        L.append("（未发现可量化事实问题。）\n")
    else:
        L.append("| # | 影响 | 文档原文 | 轴·形态·来源 | 证据等级 | 三态 | 代码位置 | 改进建议 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for f in quant:
            tag = f"{_state.AXIS_ZH.get(f.get('axis'),'?')}·{_cell(f.get('form'))}·{_cell(f.get('source'))}"
            L.append(f"| {f.get('idx')} | {_imp_label(f)} | {_cell(f.get('quote'))} | {tag} | {_cell(f.get('evidence_grade'))} | "
                     f"**{_cell(f.get('verdict'))}** | `{_cell(f.get('code_location'))}` | {_cell(f.get('improvement'))} |")
        L.append("")
        for f in quant:
            if f.get("open_question"):
                L.append(f"> #{f.get('idx')} 开放问题:{f['open_question']}")
            if f.get("also_hit"):
                L.append(f"> #{f.get('idx')} 另命中此处的轴:{_also(f)}")
        L.append("")

    # 三、教学判断（不可量化）
    L.append("## 三、教学判断（不可量化 · steelman 已过 · 只定性）\n")
    if not nonquant:
        L.append("（未发现教学判断类缺陷。）\n")
    for f in nonquant:
        L.append(f"### #{f.get('idx')} — {_state.AXIS_ZH.get(f.get('axis'),'?')}（{f.get('verdict')}）`教学判断`")
        L.append(f"- **文档原文**:{f.get('quote')}")
        L.append(f"- **判例(读者会卡在哪)**:{f.get('precedent')}")
        L.append(f"- **steelman(已打反论)**:{f.get('steelman')}")
        if f.get("external_evidence"):
            L.append(f"- **外部反证**:{f.get('external_evidence')}")
        L.append(f"- **改进建议**:{f.get('improvement')}")
        L.append(f"- 下一责任人:{f.get('next_owner', '文档作者')}")
        L.append("")

    # 四、待补（未过自校验闸）
    if needs_fix:
        L.append("## 四、待补（未过自校验闸,不计入结论）\n")
        for f, probs in needs_fix:
            L.append(f"- 原文「{_cell(f.get('quote'))}」({f.get('cls')}/{f.get('verdict')}):缺 {'; '.join(probs)}")
        L.append("")
    return "\n".join(L)


# ================================ HTML（华为风设计引擎 + DATA 注入）================================
# 产物 = templates/report-engine.html（自包含 CSS+JS 引擎，勿改）+ 从 findings 映射的 DATA 数组。
# 汇总条/数字/阻断横幅/按文件表/筛选/分组/修改清单全部由引擎 JS 从 DATA 现算。

_TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "report-engine.html"

_CODE_CAT = ("C2", "C3", "C4", "C6", "C7", "C9")        # → 须查代码,其余查文档
_MISSING_RE = re.compile(r"漏列|未列|没列|缺(?!省)|应补|补上|补入|少列|未声明|欠声明|漏写|漏掉|未提供|未列出")
_OVER_RE = re.compile(r"多列|多写|假\s*√|标.{0,4}√|误标.{0,4}支持")
_READABLE_RE = re.compile(r"术语|措辞|风格不一致|命名不一致|写法不一致|大小写|前后不一致|表述不一致|统一为|拼写|错别字|锚文本|可读性")


def _esc(s) -> str:
    return html.escape(html.unescape(str(s) if s is not None else ""), quote=False)


def _raw(s) -> str:
    """还原 HTML 实体为原文(DATA 存原文,引擎运行时再 esc 显示)。"""
    return html.unescape(str(s) if s is not None else "")


def _short(s, n: int) -> str:
    s = _raw(s).strip().replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def _design_type(f: dict) -> str:
    """映射到设计的三维缺陷类型(缺失/不可信/易读)。finder 显式给了就用,否则启发式。"""
    if f.get("type") in ("untrust", "missing", "readable"):
        return f["type"]
    if f.get("axis") == "readable":
        return "readable"
    txt = (f.get("improvement") or "") + " " + (f.get("quote") or "")
    if _READABLE_RE.search(txt) and "矛盾" not in txt and "冲突" not in txt:
        return "readable"
    if _MISSING_RE.search(txt) and not _OVER_RE.search(txt):
        return "missing"
    return "untrust"


def _design_check(f: dict) -> str:
    if f.get("check") in ("code", "doc", "rule"):
        return f["check"]
    return "code" if (f.get("category") or "")[:2] in _CODE_CAT else "doc"


def _design_prob(f: dict) -> str:
    if f.get("prob"):
        return _short(f["prob"], 60)
    head = re.split(r"[。;；\n]", _raw(f.get("improvement")), 1)[0]
    return _short(head, 50) or _short(f.get("quote"), 50) or "(未填问题)"


_CONSEQ_DEFAULT = {"blocker": "照做会失败或选到跑不通的目标", "misleading": "会误导,但通常有兜底可恢复", "minor": "对开发影响轻微"}

# 旧叶子码 → 8 类(C1–C6/J1–J2);新 finder 已直出 8 类,此映射兜底老数据。
_CAT8 = {"C1": "C1", "C2": "C1", "C3": "C2", "C6": "C2", "C4": "C3", "C5": "C4",
         "C7": "C5", "C8": "C6", "C9": "C6", "J1": "J1", "J2": "J2", "J3": "J2"}


_CAT8_SET = {"C1", "C2", "C3", "C4", "C5", "C6", "J1", "J2"}


def _design_cat(f: dict) -> str:
    c = (f.get("category") or "").upper().strip()
    if c in _CAT8_SET:                          # 新 finder 已直出 8 类裸码 → 原样保留
        return c
    return _CAT8.get(c.split(".")[0], "C2")     # 老叶子码(带 . 或 C7–C9/J3)→ 映射到 8 类


def _to_data(findings: list, default_file: str) -> list:
    """skill finding → 设计 DATA 契约(report-engine.html 字段名)。finder 给了 prob/conseq/fig 就直接用,否则降级派生。"""
    out = []
    for i, f in enumerate(findings, 1):
        confirmed = bool(f["conf"]) if isinstance(f.get("conf"), bool) else str(f.get("verdict", "")).startswith("CONFIRMED")
        d = {
            "idx": f.get("idx", i),
            "doc": f.get("doc") or f.get("file") or default_file or "(未记录)",
            "cat": _design_cat(f),
            "impact": f.get("impact") if f.get("impact") in _state.IMPACT else "misleading",
            "type": _design_type(f),
            "check": _design_check(f),
            "suspected": not confirmed,
            "prob": _design_prob(f),
            "conseq": f.get("conseq") or _CONSEQ_DEFAULT.get(f.get("impact"), ""),
            "conseqBad": f.get("impact") == "blocker",
            "fix": _short(f.get("fix") or f.get("improvement"), 220),
            "quote": _raw(f.get("quote")),
            "code": _raw(f.get("code_location") or f.get("precedent")),
            "rc": f.get("root_cause") or "",
        }
        if isinstance(f.get("fig"), dict):
            d["fig"] = f["fig"]
        if f.get("figNote"):
            d["figNote"] = _raw(f["figNote"])
        out.append(d)
    return out


def render_html(repo: str) -> str:
    """设计版:吐自包含引擎(templates/report-engine.html)+ 注入从 findings 映射的 DATA。

    注:新设计无「待补」段——未过自校验闸的缺陷(`_needs_fix`)按设计不进 DATA / 不在 HTML 呈现;
    需查待补项请看 MD 版(render() 仍保留待补段)。故 `_valid` / `_needs_fix` 仅占位不用。
    """
    meta, _valid, _needs_fix, quant, nonquant = _partition(repo)
    data = _to_data(quant + nonquant, meta.get("doc", ""))
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")   # 防 </script> 截断
    gentime = datetime.now().isoformat(timespec="minutes").replace("T", " ")
    return (_TEMPLATE.read_text(encoding="utf-8")
            .replace("__CODEROOT__", _esc(meta.get("code_root", "?")))
            .replace("__GENTIME__", gentime)
            .replace("__DATA_JSON__", data_json))


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染 tech-docs-guard 体检报告(MD + HTML,含自校验闸/去重/未评标注)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", help="REPORT.md 输出路径(默认 CWD/cann-ops-report/tech-docs-guard/<repo>/REPORT.md)")
    ap.add_argument("--format", choices=["md", "html", "both"], default="both")
    ap.add_argument("--with-minor", action="store_true", help="保留 minor(瑕疵)条目;默认舍弃")
    args = ap.parse_args()

    global WITH_MINOR
    if args.with_minor:
        WITH_MINOR = True

    md_out = Path(args.out) if args.out else _state.repo_dir(args.repo) / "REPORT.md"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    html_out = md_out.with_suffix(".html")

    if args.format in ("md", "both"):
        md_out.write_text(render(args.repo), encoding="utf-8")
        print(f"[report] {args.repo} → {md_out}")
    if args.format in ("html", "both"):
        html_out.write_text(render_html(args.repo), encoding="utf-8")
        print(f"[report] {args.repo} → {html_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
