"""tech-docs-guard 产物读写 + 数据模型(三态 / 两类 / 自校验闸)。

产物落 CWD/cann-ops-report/tech-docs-guard/<repo>/。findings.json 是机读缺陷台账,
每条遵守「干净交接」格式;render 前过自校验闸:可量化条必带代码位置,不可量化条必带判例+steelman。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPORT_ROOT = Path.cwd() / "cann-ops-report"
TUTEVAL_ROOT = REPORT_ROOT / "tech-docs-guard"

# 五轴
AXES = {"findable", "trustworthy", "learnable", "operable", "readable"}
AXIS_ZH = {"findable": "找得到", "trustworthy": "信得过", "learnable": "学得会",
           "operable": "可操作", "readable": "读得懂"}

# 指标第一分类
CLASSES = {"quantifiable", "non_quantifiable"}

# 缺陷形态 + 「错」的来源
FORMS = {"缺", "糊", "错", "冗"}
ERROR_SOURCES = {"code_mismatch", "stale", "concept_wrong", "self_contradiction"}

# 三态(可量化对照代码的结论)。CONSISTENT/NO_STATIC 不算缺陷,SUSPECTED/CONFIRMED 算。
QUANT_VERDICTS = {"CONSISTENT", "CONFIRMED_MISMATCH", "SUSPECTED", "NO_STATIC_EVIDENCE"}
# 不可量化(教学判断)的结论。CONFIRMED_CONCEPT_WRONG 必须有外部反证。
NONQUANT_VERDICTS = {"TEACHING_JUDGMENT", "SUSPECTED_CONCEPT_RISK", "CONFIRMED_CONCEPT_WRONG"}

# 进了报告才算「缺陷」的 verdict(CONSISTENT / NO_STATIC_EVIDENCE 只是记录,不算缺陷)
DEFECT_VERDICTS = {"CONFIRMED_MISMATCH", "SUSPECTED",
                   "TEACHING_JUDGMENT", "SUSPECTED_CONCEPT_RISK", "CONFIRMED_CONCEPT_WRONG"}

# v4:开发者影响(正交维度,可选)。按「后果+方向」判,不按 category 机械映射。
IMPACT = {"blocker", "misleading", "minor"}
IMPACT_ZH = {"blocker": "阻断", "misleading": "误导", "minor": "瑕疵"}
IMPACT_ORDER = {"blocker": 0, "misleading": 1, "minor": 2}   # 报告排序:阻断在前
# v4:成因标签(正交维度,可选),首批高频根因
ROOT_CAUSES = {"copy_paste_not_updated", "version_or_contract_drift",
               "template_placeholder_left", "fabricated_template_tree"}


def repo_dir(repo: str) -> Path:
    return TUTEVAL_ROOT / repo


def meta_path(repo: str) -> Path:
    return repo_dir(repo) / "doc_meta.json"


def findings_path(repo: str) -> Path:
    return repo_dir(repo) / "findings.json"


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_meta(repo: str) -> dict:
    p = meta_path(repo)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_meta(repo: str, meta: dict) -> None:
    _atomic_write(meta_path(repo), meta)


def load_findings(repo: str) -> list:
    p = findings_path(repo)
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_findings(repo: str, findings: list) -> None:
    _atomic_write(findings_path(repo), findings)


def add_finding(repo: str, finding: dict) -> dict:
    """追加一条缺陷(干净交接格式)。不校验——校验留给 self_check_finding(render 时强制)。"""
    findings = load_findings(repo)
    finding.setdefault("idx", len(findings) + 1)
    finding.setdefault("next_owner", "文档作者")
    findings.append(finding)
    save_findings(repo, findings)
    return finding


def self_check_finding(f: dict) -> list[str]:
    """自校验闸:返回该条缺陷的违规清单(空=通过)。render 时强制,不过的不准进报告正文。

    - 通用:必须有 cls/axis/quote/verdict/improvement。
    - 可量化:verdict ∈ QUANT_VERDICTS;若是缺陷(CONFIRMED/SUSPECTED)必带 code_location。
    - 不可量化:verdict ∈ NONQUANT_VERDICTS;必带 precedent(判例)+ steelman;
      CONFIRMED_CONCEPT_WRONG 还必须带 external_evidence(否则只能 SUSPECTED_CONCEPT_RISK)。
    """
    problems = []
    cls = f.get("cls")
    if cls not in CLASSES:
        problems.append(f"cls 非法: {cls!r}")
    if f.get("axis") not in AXES:
        problems.append(f"axis 非法: {f.get('axis')!r}")
    if not f.get("quote"):
        problems.append("缺 quote(文档原文)")
    if not f.get("improvement"):
        problems.append("缺 improvement(改进建议)")
    v = f.get("verdict")
    if cls == "quantifiable":
        if v not in QUANT_VERDICTS:
            problems.append(f"可量化 verdict 非法: {v!r}")
        if v in {"CONFIRMED_MISMATCH", "SUSPECTED"} and not f.get("code_location"):
            problems.append("可量化缺陷必带 code_location(grep 不到≠编造)")
    elif cls == "non_quantifiable":
        if v not in NONQUANT_VERDICTS:
            problems.append(f"不可量化 verdict 非法: {v!r}")
        if not f.get("precedent"):
            problems.append("不可量化必带 precedent(判例:读者会卡在哪)")
        if not f.get("steelman"):
            problems.append("不可量化必带 steelman(已打过最强反论)")
        if v == "CONFIRMED_CONCEPT_WRONG" and not f.get("external_evidence"):
            problems.append("「确认讲错」必带 external_evidence,否则只能 SUSPECTED_CONCEPT_RISK")
    return problems


def slugify(text: str, maxlen: int = 30) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    return s[:maxlen].strip("_") or "x"
