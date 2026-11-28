# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""确定性巡检：链接改写与参数表越界（只校验告警，不做文本替换）。

被 generation_validation.main 与 ui.cli 的生成-验证重试环共用；
命中即注入 issues 触发重试再生成。
另提供共享辅助函数 build_sig_corpus / resolve_orig_doc_full，供两个入口
统一构建签名语料与原文档全文（修复：changed_only 证据过滤导致签名形参
提取失效、doc_path 为相对目录导致原文档恒读不到的问题）。
"""

import json
import re
from pathlib import Path

_MD_LINK_RE = re.compile(r"\]\(([^)#\s]+\.md[^)\s]*)\)")
_TR_CELL_RE = re.compile(r"<tr>\s*<td>(.*?)</td>", re.S)
_MD_ROW_RE = re.compile(r"(?m)^\|([^|\n]+)\|")
_HEADING_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
_LEAD_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# 非参数名词忽略集（小写）：dtype/排布格式/维度词会被反引号包裹写在约束文本里，
# 但它们不是参数名，天然不在任何允许集 → 比对必中（全量误报），直接跳过。
# 仅用于 param_ref_out_of_scope 的 constraint 分支（param_row_out_of_scope 的
# 输入是参数表首列，语义不同，不受此忽略集影响）。
_NON_PARAM_TOKENS = frozenset(
    {
        # dtype 族
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "bf16",
        "fp16",
        "fp32",
        "bool",
        "double",
        "char",
        "string",
        # format / 排布局部词
        "nd",
        "nz",
        "fractal_nz",
        "format_fractal_nz",
        "nchw",
        "nhwc",
        "ncdhw",
        "ndhwc",
        # 维度/杂项
        "dtype",
        "format",
        "shape",
    }
)
# 维度词形（dim0/dim1/... 不枚举，按词形跳过）
_DIM_TOKEN_RE = re.compile(r"^dim[0-9]+$")

# FRACTAL_NZ 方向识别：证据校验宏 == FORMAT_FRACTAL_NZ 表示报错条件=禁止 NZ，
# != FORMAT_FRACTAL_NZ 表示要求非 NZ；生成文本里用否定/肯定措辞表达方向，
# 两侧方向一致才放行（守卫本意是抓"方向读反"，不是抓"新引入该词"）。
# 允许 ge:: 等命名空间前缀（实际代码常为 == ge::FORMAT_FRACTAL_NZ）。
_NS_PREFIX = r"(?:[A-Za-z_][A-Za-z0-9_]*::)*"
_NZ_EVIDENCE_FORBID_RE = re.compile(r"==\s*" + _NS_PREFIX + r"FORMAT_FRACTAL_NZ")
_NZ_EVIDENCE_REQUIRE_RE = re.compile(r"!=\s*" + _NS_PREFIX + r"FORMAT_FRACTAL_NZ")
_NZ_NEGATION_RE = re.compile(r"不能|不得|不可|不支持|禁止|不为|不使用|避免|无法")
_NZ_AFFIRM_RE = re.compile(r"必须|要求|只支持|只能|支持")


def _lost_md_links(orig_section: str, generated_section: str) -> list[str]:
    """原章节有、生成章节丢失或被改写的 .md 相对链接。

    针对 docs/zh/context → docs/context 这类改写（链接仍在但路径变了），
    集合比对后原链接不再出现即命中。
    """
    orig_links = set(_MD_LINK_RE.findall(orig_section))
    gen_links = set(_MD_LINK_RE.findall(generated_section))
    return sorted(orig_links - gen_links)


def _table_params(section: str) -> list[str]:
    """提取参数表首列参数名（HTML <tr><td> 与 markdown 竖线表两种形式）。

    错误码行（首列 ACLNN_ERR_*，属独立错误码表而非参数表）不算参数。
    """
    cells = [c.strip() for c in _TR_CELL_RE.findall(section)]
    cells += [r.strip() for r in _MD_ROW_RE.findall(section)]
    params = []
    for c in cells:
        c = re.sub(r"<[^>]+>", "", c).replace("`", "").strip()
        if c and _LEAD_IDENT_RE.match(c.replace(" ", "")):
            c = c.replace(" ", "")
            if c.startswith("ACLNN_ERR_"):
                continue
            params.append(c)
    return params


def _signature_params(func: str, corpus: str) -> set[str]:
    """从语料中提取所有 `func(...)` 平衡括号内的形参名。

    用于放行真实存在的参数：新 PR 新增的参数会出现在新代码签名的 diff 证据里。
    """
    params: set[str] = set()
    for m in re.finditer(re.escape(func) + r"\s*\(", corpus):
        depth = 1
        end = -1
        for j in range(m.end(), min(m.end() + 4000, len(corpus))):
            if corpus[j] == "(":
                depth += 1
            elif corpus[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            continue
        body_start = m.end()
        for arg in corpus[body_start:end].split(","):
            idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", arg.split("=")[0])
            if idents:
                params.add(idents[-1])
    return params


def _aclnn_signature_params(sig_corpus: str) -> set[str]:
    """从语料中提取所有 aclnn 接口（aclnnXxx / aclnnXxxGetWorkspaceSize）签名形参名的并集。"""
    params: set[str] = set()
    for m in re.finditer(r"\baclnn[A-Za-z0-9_]+\s*\(", sig_corpus):
        func = m.group(0)[:-1].strip()
        params |= _signature_params(func, sig_corpus)
    return params


def _norm_ident(s: str) -> str:
    """参数名规范化：去 Optional 体例后缀、下划线与大小写差异（tpSendCountsOptional ≡ tpSendCounts ≡ oriX ≡ ori_x）。"""
    if s.endswith("Optional"):
        s = s[: -len("Optional")]
    return re.sub(r"_", "", s).lower()


def build_sig_corpus(doc_original_md, raw_input, extracted, evidence_path=None) -> str:
    """构建签名语料：本 doc_part 全部原章节 + 证据链原文 + 编排后证据（供参数放行判断）。

    批次启用证据过滤时（evidences_changed_only/ 目录），过滤后的 constraint.json
    可能不含任何 aclnn 签名 → 签名形参提取失效（真参数被误报）；此时追加读兄弟
    全量证据目录下的同名文件，补回被过滤掉的 aclnn 签名。任何异常静默返回基础语料。
    """
    corpus = (
        "\n".join(doc_original_md)
        + json.dumps(raw_input, ensure_ascii=False, default=str)
        + json.dumps(extracted, ensure_ascii=False, default=str)
    )
    try:
        if evidence_path:
            _ev = Path(evidence_path)
            if _ev.parent.name == "evidences_changed_only":
                _full = _ev.parent.parent / "evidences" / _ev.name
                if _full.is_file():
                    with open(_full, "r", encoding="utf-8") as f:
                        corpus += json.dumps(json.load(f), ensure_ascii=False, default=str)
    except Exception:
        pass
    return corpus


def resolve_orig_doc_full(doc_path, doc_name, evidence_path=None) -> str:
    """按候选路径依次解析原文档全文，读到即返回；全程容错，异常返回空串。

    候选顺序：
    a. doc_path 本身（兼容绝对文件路径调用方）；
    b. doc_path 为目录时拼 doc_name（anchor 的 doc_path 常为目录）；
    c. evidence_path 给出且 doc_name 非空时，对证据路径前 6 级祖先尝试
       祖先/"doc"/doc_name —— 覆盖 <版本目录>/re_detection/evidences*/x.json
       布局回溯到 <算子输出目录>/doc/<doc_name>（doc_path 为相对目录 "." 时
       (a)/(b) 在流水线 CWD 下均非文件，靠此通道才能读到原文档）。
    """
    try:
        candidates: list[Path] = []
        if doc_path:
            _p = Path(doc_path)
            if _p.is_dir():
                if doc_name:
                    candidates.append(_p / doc_name)
            else:
                candidates.append(_p)
        if evidence_path and doc_name:
            for ancestor in list(Path(evidence_path).resolve().parents)[:6]:
                candidates.append(ancestor / "doc" / doc_name)
        for cand in candidates:
            if cand.is_file():
                return cand.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def _evidence_nz_direction(corpus: str) -> str | None:
    """证据语料中 NZ 校验的方向：仅 FORBID（== FORMAT_FRACTAL_NZ，报错条件=禁止 NZ）
    → "forbid"；仅 REQUIRE（!= FORMAT_FRACTAL_NZ）→ "require"；
    都命中或都不命中 → None（无法定向）。"""
    forbid = bool(_NZ_EVIDENCE_FORBID_RE.search(corpus))
    require = bool(_NZ_EVIDENCE_REQUIRE_RE.search(corpus))
    if forbid and not require:
        return "forbid"
    if require and not forbid:
        return "require"
    return None


def _nz_mention_direction(line: str) -> str | None:
    """一行 FRACTAL_NZ 表述的方向：先查否定措辞 → "forbid"，否则查肯定措辞 →
    "require"，都无 → None（方向未知）。"""
    if _NZ_NEGATION_RE.search(line):
        return "forbid"
    if _NZ_AFFIRM_RE.search(line):
        return "require"
    return None


def collect_issues(
    orig_section: str,
    generated_text: str,
    doc_part: str,
    sig_corpus: str,
    orig_doc_full: str = "",
) -> list[dict]:
    """确定性巡检入口：返回需注入 result_dict['issues'] 的问题列表（空列表=通过）。"""
    issues: list[dict] = []

    lost_links = _lost_md_links(orig_section, generated_text)
    if lost_links:
        issues += [
            {
                "id": "md_link_lost",
                "reason": f"原章节中的链接 {link} 在生成结果中丢失或被改写，必须原样保留（相对路径不得增删目录段），"
                f"除非链接指向的内容已被删除",
            }
            for link in lost_links
        ]

    if "FRACTAL_NZ" in generated_text and "FRACTAL_NZ" not in orig_section:
        # 方向感知：证据含 OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ)（报错条件=
        # 禁止 NZ）而 writer 写"不能为 FRACTAL_NZ"时方向一致，放行；方向读反、
        # 方向未知或证据无 NZ 信息（纯新颖性守卫）才拦截
        evidence_direction = _evidence_nz_direction(sig_corpus)
        direction_verified = evidence_direction is not None and all(
            _nz_mention_direction(line) == evidence_direction
            for line in generated_text.splitlines()
            if "FRACTAL_NZ" in line
        )
        if not direction_verified:
            issues.append(
                {
                    "id": "format_assert_introduced",
                    "reason": "生成结果新引入了原文中不存在的 FRACTAL_NZ 表述。昇腾校验宏 "
                    "OP_TILING_CHECK(format == FORMAT_FRACTAL_NZ, ...) 的第一参数是报错条件，"
                    "意为 FRACTAL_NZ 被禁止（支持 ND 等）；请核对证据中格式校验的方向，"
                    "修正为正确的支持格式或删除该条约束",
                }
            )

    if doc_part == "sub_function_hash":
        heading_m = _HEADING_RE.search(generated_text) or _HEADING_RE.search(orig_section)
        func = heading_m.group(1).strip() if heading_m else ""
        if func:
            allowed = set(_table_params(orig_section)) | _signature_params(func, sig_corpus)
            stray = [p for p in dict.fromkeys(_table_params(generated_text)) if p not in allowed]
            if stray:
                issues += [
                    {
                        "id": "param_row_out_of_scope",
                        "reason": f"参数 `{p}` 不属于 {func} 小节的参数表：既不在原章节参数表中，也不在 {func} 的函数签名里；"
                        f"执行接口小节只能保留执行参数（workspace/workspaceSize/executor/stream），"
                        f"算子输入/输出参数属于 GetWorkspaceSize 小节，必须删除该行",
                    }
                    for p in stray
                ]

    elif doc_part == "constraint":
        # 允许集：原约束节标识符 ∪ 原文档全文标识符（参数名在其他小节/示例代码中）∪ aclnn 签名参数；
        # 比对时做规范化（Optional 后缀/下划线/大小写），兼容文档体例名与 kernel 变量名
        allowed = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", orig_section))
        allowed |= _aclnn_signature_params(sig_corpus)
        if orig_doc_full:
            allowed |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", orig_doc_full))
        allowed_norm = {_norm_ident(a) for a in allowed}
        stray = [
            p
            for p in dict.fromkeys(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", generated_text))
            if _norm_ident(p) not in allowed_norm and not p.startswith("ACLNN_ERR_")
            # 非参数名词（dtype/format/排布/dim0 等反引号词）不是参数名，直接跳过
            and p.lower() not in _NON_PARAM_TOKENS and not _DIM_TOKEN_RE.match(p.lower())
        ]
        if stray:
            issues += [
                {
                    "id": "param_ref_out_of_scope",
                    "reason": f"约束说明中新引入了参数 `{p}` 的约束条目：该名称既不在原文档中，也不在本算子 aclnn 接口"
                    f"的函数签名里（多为共享代码文件中其他算子变体如 RMSNorm 融合系的参数，或 kernel 内部"
                    f"变量名），必须删除相关条目或改用本文档参数表中的参数名。"
                    f"若 `{p}` 出现在证据给出的计算公式中（代码内部中间变量）：修正方法是保留原文公式的"
                    f"记号与结构，仅把代码变更引入的增量（如新增项、变化的常量）施加到原文公式上，"
                    f"禁止用代码表达式整体替换原文公式；如需解释该中间量，按其定义内联展开为文档符号"
                    f"（例如 tokenNeedSizeDispatch 类中间量应展开为其 Align 系列定义），不得在文档中"
                    f"新增该变量名作为公式符号",
                }
                for p in stray
            ]

    return issues
