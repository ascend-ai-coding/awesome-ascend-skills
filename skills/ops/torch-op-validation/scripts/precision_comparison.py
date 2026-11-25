#!/usr/bin/env python3
"""Validate and display supplied per-leaf 2.1 results without judging accuracy."""

from decimal import Decimal
import html
import json
import math

API_TYPES = {
    "non_computational": "非计算类", "integer": "整数计算类",
    "quantized": "量化计算类", "floating": "浮点计算类", "random": "随机数生成类",
}
STATUSES = {
    "PASS", "FAIL", "ERROR", "TIMEOUT", "UNSUPPORTED", "NOT_RUN",
    "NOT_APPLICABLE", "BLOCKED", "MEASURED",
}
ISSUE_STATUSES = {"FAIL", "ERROR", "TIMEOUT", "BLOCKED"}
IDENTITY_FIELDS = (
    "case_id", "operator", "torch_api", "backend_operator", "phase", "api_type",
    "api_family", "api_layer", "output_name", "shape", "dtype", "candidate_name",
    "peer_name", "golden_name",
)
OPTIONAL_STRINGS = {"torch_api", "backend_operator"}
ROW_KEY = ("case_id", "phase", "output_name", "candidate_name", "peer_name", "golden_name")
STANDARD_FIELDS = (
    "level", "numerical_status", "formal_status", "detail", "mare_ratio", "mere_ratio",
    "rmse_ratio", "ratio_denominator_err", "small_value_threshold", "small_value_error",
    "candidate_error_count", "peer_error_count", "small_value_ratio", "nonfinite_status",
    "recheck_status",
)
STANDARD_STATUSES = {"numerical_status", "formal_status", "nonfinite_status", "recheck_status"}
STANDARD_NUMBERS = {
    "mare_ratio", "mere_ratio", "rmse_ratio", "ratio_denominator_err",
    "small_value_threshold", "small_value_error", "small_value_ratio",
}
STANDARD_COUNTS = {"candidate_error_count", "peer_error_count"}
PAIRS = ("candidate_golden", "peer_golden", "candidate_peer")
PAIR_FIELDS = (
    # Count elements whose original-dtype bit patterns differ, not differing bits.
    "bitwise_equal", "bitwise_mismatch_count", "max_ulp", "max_abs", "mean_abs",
    "max_rel", "mean_rel", "rmse", "excluded_count",
)
PAIR_COUNTS = {"bitwise_mismatch_count", "excluded_count"}
CSV_FIELDS = (
    "schema_version", "run_id", "standard_id", "evidence_class",
    *("shape_json" if field == "shape" else field for field in IDENTITY_FIELDS),
    *("standard21_" + field for field in STANDARD_FIELDS), "standard21_issue_fields_json",
    *(pair + "_" + field for pair in PAIRS for field in PAIR_FIELDS),
    "candidate_peer_row_max_mean_rel", "diagnostic_definition_json", "context_json",
)
LABELS = {
    "case_id": "用例", "operator": "记录算子", "torch_api": "Torch API",
    "backend_operator": "后端算子", "phase": "阶段", "api_type": "API 精度类别",
    "api_family": "API 功能族", "api_layer": "API 层级", "output_name": "输出／梯度槽",
    "shape": "Shape", "dtype": "Dtype", "candidate_name": "候选端",
    "peer_name": "同精度对照端", "golden_name": "参考 Golden",
    "level": "等级", "numerical_status": "数值状态", "formal_status": "正式验收状态",
    "detail": "判定说明", "mare_ratio": "MARE 比值", "mere_ratio": "MERE 比值",
    "rmse_ratio": "RMSE 比值", "ratio_denominator_err": "正常误差比分母下限 err",
    "small_value_threshold": "小值阈值 threshold", "small_value_error": "小值计错界 error",
    "candidate_error_count": "候选小值错误数", "peer_error_count": "对照小值错误数",
    "small_value_ratio": "小值计错比", "nonfinite_status": "非有限值状态",
    "recheck_status": "复检状态", "bitwise_equal": "逐位一致",
    "bitwise_mismatch_count": "位模式不一致元素数", "max_ulp": "最大 ULP",
    "max_abs": "最大绝对误差", "mean_abs": "平均绝对误差",
    "max_rel": "最大相对误差", "mean_rel": "平均相对误差", "rmse": "RMSE",
    "excluded_count": "剔除数", "row_max_mean_rel": "行最大位置平均相对差",
}
PAIR_LABELS = {
    "candidate_golden": "被测端 A vs Golden G（直接比较）",
    "peer_golden": "同精度对照 P vs Golden G（直接比较）",
    "candidate_peer": "被测端 A vs 同精度对照 P（直接比较）",
}
PAIR_ROLES = {
    "candidate_golden": ("candidate_name", "golden_name"),
    "peer_golden": ("peer_name", "golden_name"),
    "candidate_peer": ("candidate_name", "peer_name"),
}
GOLDEN_DTYPE_RULES = (
    ("float16", "float32", "计算前升精"),
    ("bfloat16", "float32", "计算前升精"),
    ("float32", "float32", "不升精"),
    ("complex32", "complex64", "计算前升精"),
    ("complex64", "complex64", "不升精"),
)


def _json(value, indent=None):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=indent)


def _string(value, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or "\0" in value:
        raise ValueError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} string without null characters")


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")


def _finite_json(value, label):
    """Do not allow nonfinite numbers to hide in supplied method/source metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            _string(key, label + " key", allow_empty=True)
            _finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{label}[{index}]")
    elif isinstance(value, str):
        _string(value, label, allow_empty=True)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} must contain only finite numbers or null")


def _number(value, label, integer=False):
    if value is None:
        return
    valid = type(value) is int or (not integer and type(value) is float and math.isfinite(value))
    if not valid or value < 0:
        kind = "integer" if integer else "number"
        raise ValueError(f"{label} must be a nonnegative finite {kind} or null")


def validate_rows(value):
    """Validate an optional schema-1.0 extension before any report is written."""
    if not isinstance(value, list):
        raise ValueError("precision_comparisons must be a JSON array")
    seen = set()
    for index, row in enumerate(value):
        label = f"precision_comparisons[{index}]"
        _object(row, label)
        _finite_json(row, label)
        for field in IDENTITY_FIELDS:
            if field == "shape":
                continue
            item = row.get(field)
            if field in OPTIONAL_STRINGS and item is None:
                continue
            _string(item, f"{label}.{field}", allow_empty=field in OPTIONAL_STRINGS)
        if row["api_type"] not in API_TYPES:
            raise ValueError(f"{label}.api_type must be one of {', '.join(API_TYPES)}")
        shape = row.get("shape")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"{label}.shape must be an array of nonnegative integer dimensions")
        key = tuple(row[field] for field in ROW_KEY)
        if key in seen:
            raise ValueError(f"{label} duplicates precision comparison key {key!r}")
        seen.add(key)

        standard = row.get("standard21")
        _object(standard, label + ".standard21")
        unknown = standard.keys() - set(STANDARD_FIELDS) - {"issue_fields"}
        if unknown:
            raise ValueError(f"{label}.standard21 has unknown fields: {', '.join(sorted(unknown))}")
        for field in STANDARD_FIELDS:
            item = standard.get(field)
            location = f"{label}.standard21.{field}"
            if field in STANDARD_NUMBERS or field in STANDARD_COUNTS:
                _number(item, location, integer=field in STANDARD_COUNTS)
            elif item is not None:
                _string(item, location, allow_empty=field == "detail")
                if field == "level" and item not in {"L0", "L1", "L2"}:
                    raise ValueError(f"{location} must be L0, L1, L2 or null")
                if field in STANDARD_STATUSES and item not in STATUSES:
                    raise ValueError(f"{location} has an unknown status")
        issues = standard.get("issue_fields", {})
        _object(issues, label + ".standard21.issue_fields")
        for field, explanation in issues.items():
            if field not in STANDARD_FIELDS:
                raise ValueError(f"{label}.standard21.issue_fields may reference only standard21 fields: {field}")
            _string(explanation, f"{label}.standard21.issue_fields.{field}")

        diagnostics = row.get("diagnostics")
        _object(diagnostics, label + ".diagnostics")
        if diagnostics.keys() - set(PAIRS):
            raise ValueError(f"{label}.diagnostics has an unknown pair")
        for pair, metrics in diagnostics.items():
            location = f"{label}.diagnostics.{pair}"
            _object(metrics, location)
            allowed = set(PAIR_FIELDS) | ({"row_max_mean_rel"} if pair == "candidate_peer" else set())
            if metrics.keys() - allowed:
                raise ValueError(f"{location} has unknown diagnostic fields")
            for field, item in metrics.items():
                if field == "bitwise_equal":
                    if item is not None and type(item) is not bool:
                        raise ValueError(f"{location}.{field} must be a boolean or null")
                else:
                    _number(item, f"{location}.{field}", integer=field in PAIR_COUNTS)
        for field in ("context", "diagnostic_definition"):
            item = row.get(field)
            if item is not None and not isinstance(item, (dict, str)):
                raise ValueError(f"{label}.{field} must be a JSON object, string or null")
    return value


def csv_rows(run, rows):
    """Flatten supplied values, leaving unknown measurements as blank CSV cells."""
    for row in rows:
        result = {
            "schema_version": "1.0", "run_id": run["id"], "standard_id": run["standard_id"],
            "evidence_class": run["evidence_class"],
            **{field: row.get(field) for field in IDENTITY_FIELDS if field != "shape"},
            "shape_json": _json(row["shape"]),
            **{"standard21_" + field: row["standard21"].get(field) for field in STANDARD_FIELDS},
            "standard21_issue_fields_json": _json(row["standard21"].get("issue_fields", {})),
        }
        for pair in PAIRS:
            metrics = row["diagnostics"].get(pair, {})
            fields = (*PAIR_FIELDS, "row_max_mean_rel") if pair == "candidate_peer" else PAIR_FIELDS
            for field in fields:
                result[pair + "_" + field] = metrics.get(field)
        for field in ("context", "diagnostic_definition"):
            result[field + "_json"] = "" if row.get(field) is None else _json(row[field])
        yield result


def _text(value):
    return html.escape(str(value), quote=True)


def _display(value, count=False):
    if value is None:
        return '<span class="muted">未提供</span>'
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float) and not count:
        return _text(format(Decimal(str(value)), ".6e"))
    if isinstance(value, list):
        return _text(_json(value))
    return _text(value)


def _cell(value, field, issue=None, count=False):
    attrs = f' data-field="{_text(field)}"'
    if issue:
        attrs += f' class="standard21-issue" title="{_text(issue)}"'
    return f"<td{attrs}>{_display(value, count=count)}</td>"


def _pair_definition(row, pair):
    definitions = row.get("diagnostic_definition")
    if not isinstance(definitions, dict):
        return {}
    value = definitions.get(pair)
    return value if isinstance(value, dict) else {}


def _binary_not_applicable(row, pair):
    status = _pair_definition(row, pair).get("original_binary_status")
    return status if isinstance(status, str) and status.startswith("NOT_APPLICABLE") else None


def _actual_dtypes(row):
    """Read recorded types; do not infer Golden precision from its device name."""
    context = row.get("context")
    supplied = context.get("actual_output_dtypes", {}) if isinstance(context, dict) else {}
    supplied = supplied if isinstance(supplied, dict) else {}
    result = {key: value for key, value in supplied.items()
              if key in ("candidate", "peer", "golden") and isinstance(value, str) and value}
    for pair, roles in (("candidate_golden", ("candidate", "golden")),
                        ("peer_golden", ("peer", "golden"))):
        recorded = _pair_definition(row, pair).get("original_dtypes")
        if isinstance(recorded, list) and len(recorded) == 2:
            for role, dtype in zip(roles, recorded):
                if isinstance(dtype, str) and dtype:
                    result.setdefault(role, dtype)
    result.setdefault("candidate", row["dtype"])
    return result


def _phase(value):
    return {"forward": "前向", "backward": "反向"}.get(value, value)


def render_method_section(rows):
    """Explain recorded roles and the floating double-reference method."""
    if not rows:
        return ""
    names = {field: " / ".join(sorted({row[field] for row in rows}))
             for field in ("candidate_name", "peer_name", "golden_name")}
    roles = (
        ("被测实现 A（候选）", names["candidate_name"], "取得待评估的实际输出与梯度。"),
        ("同精度对照 P", names["peer_name"], "提供同精度实现，用于比较两端各自相对 Golden 的误差。"),
        ("参考实现 G（Golden）", names["golden_name"], "参考计算按 Golden dtype 映射升精或保持原类型；A 与 P 共用这一份 Golden。"),
    )
    role_rows = "".join("<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>"
                        for row in roles)
    mapping_rows = "".join("<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>"
                           for row in GOLDEN_DTYPE_RULES)
    dtype_rows = []
    for row in rows:
        dtypes = _actual_dtypes(row)
        values = (row["case_id"], row["operator"], _phase(row["phase"]), row["output_name"],
                  dtypes.get("candidate"), dtypes.get("peer"), dtypes.get("golden"))
        dtype_rows.append("<tr>" + "".join(f"<td>{_display(value)}</td>" for value in values) + "</tr>")
    pair_rows = []
    for pair, purpose in (
        ("candidate_golden", "A 相对 G 的误差；浮点 2.1 的被测端误差来源。"),
        ("peer_golden", "P 相对同一个 G 的误差；浮点 2.1 的对照端误差来源。"),
        ("candidate_peer", "A 与 P 的直接差异；浮点分支中用于定位两端差别。"),
    ):
        left, right = PAIR_ROLES[pair]
        actual = " / ".join(sorted({row[left] + " vs " + row[right] for row in rows}))
        reference = names[right]
        pair_rows.append("<tr>" + "".join(f"<td>{_text(value)}</td>" for value in
            (actual, reference, purpose)) + "</tr>")
    return (
        '<section class="precision-method" aria-labelledby="precision-method-title">'
        '<h2 id="precision-method-title">三方角色、执行精度与计算方法</h2>'
        '<p>下表先说明执行端的角色，再说明三组比较。设备名称来自本报告记录；'
        '“候选／对照／Golden”是角色名称，不是三组误差的名称。</p>'
        '<div class="scroll"><table><caption>本报告的执行端</caption><thead><tr>'
        '<th scope="col">角色</th><th scope="col">记录中的执行端</th><th scope="col">作用</th>'
        f'</tr></thead><tbody>{role_rows}</tbody></table></div>'
        '<h3>三组直接比较</h3><div class="scroll"><table><thead><tr>'
        '<th scope="col">实际比较双方</th><th scope="col">相对误差的参考端</th><th scope="col">用途</th>'
        f'</tr></thead><tbody>{"".join(pair_rows)}</tbody></table></div>'
        '<p>直接比较中，实际端为 a，参考端为 b：<code>AE=abs(a-b)</code>，'
        '<code>RE=AE/(abs(b)+1e-7)</code>。A–G 与 P–G 均以 G 为参考；A–P 以 P 为参考。</p>'
        '<h3>浮点双标杆的 2.1 判定</h3>'
        '<p>对同一前向输出或反向梯度，先在标准指定的统计范围内分别计算 '
        '<code>MARE=max(RE)</code>、<code>MERE=mean(RE)</code>、'
        '<code>RMSE=sqrt(mean((actual-G)^2))</code>，再比较两端的误差：</p>'
        '<pre>E_A = metric(A, G)\nE_P = metric(P, G)\nRatio = E_A / max(E_P, err)</pre>'
        '<p><code>metric</code> 分别取 MARE、MERE、RMSE；<code>err</code> 为相应 dtype 的 '
        'Small Value Threshold，不是 RE 的 <code>1e-7</code>，也不是小值域计错界 '
        '<code>error</code>。三项 Ratio 使用记录中适用的 L0／L1／L2 上限，保留实际分母。'
        '含分母下限的 Ratio 小于 1，不能单凭此认定被测实现比对照实现更精确。</p>'
        '<p>正常值域与小值域分别检查。小值域在 <code>abs(G)&lt;threshold</code> 的位置统计 '
        '<code>abs(A-G)&gt;error</code> 和 <code>abs(P-G)&gt;error</code> 的错误数，要求 '
        '<code>ErrorCount_A/max(ErrorCount_P,1)&lt;=2</code>。非有限值按适用标准分支单列。'
        '直接诊断的全有限元素汇总，不自动等于正常值域的指标。</p>'
        '<p>以上误差比用于浮点双标杆及量化浮点输出的适用分支。非计算类、整数类、'
        '量化整型输出和随机数类按各自 2.1 条款验收；其中适用的直接位比较或绝对误差检查仍参与判定。'
        '前向按每个输出、反向按每个梯度分别记录；反向使用各端自身的前向缓存和同一冻结上游梯度。</p>'
        '<h3>Golden dtype 映射</h3>'
        '<p>左列为待测 Torch/TorchNPU dtype，右列为参考计算采用的 dtype。'
        'FP32 和 complex64 保持原 dtype 是有效指定方案，不能自动改成 FP64 或 complex128。'
        '未覆盖的 dtype 须取得适用规则，不能由精度名称自行推断。</p>'
        '<div class="scroll"><table id="golden-dtype-mapping"><thead><tr>'
        '<th scope="col">待测 dtype</th><th scope="col">Golden dtype</th><th scope="col">处理</th>'
        f'</tr></thead><tbody>{mapping_rows}</tbody></table></div>'
        '<p>complex32 的实部、虚部各为 FP16；complex64 的实部、虚部各为 FP32。'
        '映射不代表当前算子／平台支持或已经测试这些 dtype；实际执行范围见本轮记录。</p>'
        '<h3>本轮执行精度（来自记录）</h3>'
        '<p>下面列出各输出／梯度叶子已记录的实际 dtype。Golden 方案须逐算子核验：'
        '需要升精时，应按映射提升参与计算的冻结输入、权重和上游梯度，再重新执行前向与反向；'
        '不升精分支保持原 dtype，不能只因参考端与待测端 dtype 相同而判缺少 Golden。'
        '仅将低精度结果转换成高精度，不能作为升精计算。输出 dtype 也不能单独证明内部累加精度；'
        '具体输入、实现、计算模式与执行证据应在原始记录中保留。缺少 Golden 方案或执行证据时不得推定已满足要求。'
        '旧方案不符映射时保留原结果溯源，受影响条目须补算参考计算及比较，不能只改标签。</p>'
        '<div class="scroll"><table><thead><tr><th scope="col">用例</th><th scope="col">算子</th>'
        '<th scope="col">阶段</th><th scope="col">输出／梯度</th><th scope="col">A 实际输出 dtype</th>'
        '<th scope="col">P 实际输出 dtype</th><th scope="col">G 实际输出 dtype</th>'
        f'</tr></thead><tbody>{"".join(dtype_rows)}</tbody></table></div></section>'
    )


def render_diagnostics_summary(rows):
    """Keep all three supplied pair measurements visible without the wide table."""
    if not rows:
        return ""
    body = []
    for row in rows:
        for pair in PAIRS:
            metrics = row["diagnostics"].get(pair, {})
            left, right = PAIR_ROLES[pair]
            values = (row["operator"], _phase(row["phase"]) + " / " + row["output_name"],
                      row[left] + " vs " + row[right])
            cells = [f"<td>{_text(value)}</td>" for value in values]
            for field in ("max_abs", "max_rel", "mean_rel", "max_ulp"):
                cells.append(f'<td>{_display(metrics.get(field), count=field == "max_ulp")}</td>')
            binary_na = _binary_not_applicable(row, pair)
            binary = _text(binary_na) if binary_na else _display(metrics.get("bitwise_mismatch_count"), count=True)
            cells.append(f"<td>{binary}</td>")
            mask = _pair_definition(row, pair).get("full_finite_pair_mask")
            if isinstance(mask, dict) and type(mask.get("count")) is int:
                excluded = metrics.get("excluded_count")
                scope = "有效元素 " + str(mask["count"]) + "；剔除数 " + ("未提供" if excluded is None else str(excluded))
            else:
                scope = "统计掩码未提供；见计算源与缺项"
            cells.append(f"<td>{_text(scope)}</td>")
            body.append(f'<tr data-pair="{pair}">{"".join(cells)}</tr>')
    return (
        '<section aria-labelledby="precision-direct-title"><h2 id="precision-direct-title">三组直接差异</h2>'
        '<p>每个输出／梯度均展示 A–G、P–G、A–P 三组已记录结果。下表为直接诊断；'
        '浮点 2.1 的正常值域误差比和小值域判定另列于宽表。'
        '不同原 dtype 的逐位比较按记录标为不适用。ULP 的转换／舍入、相对误差分母、掩码与聚合方式见计算源，'
        '未经降精度的原 Golden 继续用于 2.1 检查。</p>'
        '<div class="scroll"><table id="precision-direct"><thead><tr><th scope="col">算子</th>'
        '<th scope="col">阶段／输出</th><th scope="col">实际比较双方</th><th scope="col">最大绝对误差</th>'
        '<th scope="col">最大相对误差</th><th scope="col">平均相对误差</th><th scope="col">最大 ULP</th>'
        '<th scope="col">原 dtype 位模式不一致元素数／适用性</th><th scope="col">统计范围</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>'
    )


def _details(row):
    missing = ["standard21." + field for field in STANDARD_FIELDS if row["standard21"].get(field) is None]
    for pair in PAIRS:
        metrics = row["diagnostics"].get(pair, {})
        fields = (*PAIR_FIELDS, "row_max_mean_rel") if pair == "candidate_peer" else PAIR_FIELDS
        missing.extend(f"diagnostics.{pair}.{field}" for field in fields
                       if metrics.get(field) is None
                       and not (field in ("bitwise_equal", "bitwise_mismatch_count")
                                and _binary_not_applicable(row, pair)))
    sections = ['<p>计算源：输入 bundle.precision_comparisons；生成器展示已有记录，不重新判定 2.1。</p>']
    for field, label in (("context", "来源与上下文"), ("diagnostic_definition", "诊断计算方法")):
        value = row.get(field)
        content = "未提供" if value is None else _json(value, indent=2) if isinstance(value, dict) else value
        sections.append(f"<p>{label}</p><pre>{_text(content)}</pre>")
    if row.get("diagnostic_definition") is None:
        sections.append('<p>未提供 ULP 计算 dtype、相对误差分母／掩码、行最大位置选择或剔除定义；不能据此推测。</p>')
    for pair in PAIRS:
        status = _binary_not_applicable(row, pair)
        if status:
            sections.append(f'<p>{_text(PAIR_LABELS[pair])}：原 dtype 二进制比较 {_text(status)}</p>')
    issues = row["standard21"].get("issue_fields", {})
    if issues:
        sections.append(f'<p>2.1 问题说明</p><pre>{_text(_json(issues, indent=2))}</pre>')
    sections.append('<p>缺测字段保留空值，不按零计。</p>')
    sections.append(f'<pre>{_text(chr(10).join(missing) if missing else "无缺测数值字段")}</pre>')
    return f'<details><summary>计算源与缺项</summary>{"".join(sections)}</details>'


def render_section(rows, *, include_method=True, include_summary=True):
    """Build an accessible, horizontally scrollable table with per-field issues."""
    if not rows:
        return ""
    groups = [("用例、API 与输出槽", len(IDENTITY_FIELDS)), ("昇腾算子精度标准 2.1（输入结论）", len(STANDARD_FIELDS))]
    for pair in PAIRS:
        left, right = PAIR_ROLES[pair]
        titles = {row[left] + " vs " + row[right] for row in rows}
        label = next(iter(titles)) + "（直接比较）" if len(titles) == 1 else PAIR_LABELS[pair]
        groups.append((label, len(PAIR_FIELDS) + (pair == "candidate_peer")))
    group_header = "".join(f'<th scope="colgroup" colspan="{count}">{_text(label)}</th>' for label, count in groups)
    labels = [LABELS[field] for field in IDENTITY_FIELDS + STANDARD_FIELDS]
    for pair in PAIRS:
        labels.extend(LABELS[field] for field in PAIR_FIELDS)
        if pair == "candidate_peer":
            labels.append(LABELS["row_max_mean_rel"])
    columns = "".join(f'<th scope="col">{_text(label)}</th>' for label in labels)
    body = []
    for row in rows:
        cells = []
        for field in IDENTITY_FIELDS:
            value = row.get(field)
            if field == "phase":
                value = {"forward": "前向", "backward": "反向"}.get(value, value)
            elif field == "api_type":
                value = API_TYPES[value]
            cells.append(_cell(value, field))
        standard = row["standard21"]
        for field in STANDARD_FIELDS:
            value = standard.get(field)
            issue = standard.get("issue_fields", {}).get(field)
            if field in STANDARD_STATUSES and value in ISSUE_STATUSES:
                issue = issue or f"输入记录的 2.1 状态：{value}"
            cells.append(_cell(value, "standard21." + field, issue, field in STANDARD_COUNTS))
        for pair in PAIRS:
            fields = (*PAIR_FIELDS, "row_max_mean_rel") if pair == "candidate_peer" else PAIR_FIELDS
            for field in fields:
                cells.append(_cell(row["diagnostics"].get(pair, {}).get(field), pair + "." + field,
                                   count=field in PAIR_COUNTS or field == "max_ulp"))
        cells.append(f'<td>{_details(row)}</td>')
        body.append(f'<tr>{"".join(cells)}</tr>')
    return (
        (render_method_section(rows) if include_method else "") +
        (render_diagnostics_summary(rows) if include_summary else "") +
        '<section aria-labelledby="precision-comparison-title"><h2 id="precision-comparison-title">逐输出三方精度宽表</h2>'
        '<p class="muted">每行对应一个前向输出或反向梯度叶子。2.1 是精度验收依据；右侧三组为诊断数据，'
        '大 ULP、两端直接相对差和逐位不一致不会触发浮点 2.1 失败。'
        '红色仅表示输入明确列出的 2.1 问题或 FAIL／ERROR／TIMEOUT／BLOCKED 状态，缺测显示“未提供”。</p>'
        '<div class="scroll precision-scroll" tabindex="0" role="region" aria-label="逐输出三方精度宽表，可横向滚动">'
        '<table id="precision-comparison"><caption>2.1 判定与诊断指标分列；所有结论及方法来自输入记录</caption>'
        f'<thead><tr>{group_header}<th scope="col" rowspan="2">计算源与缺项</th></tr><tr>{columns}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></section>'
    )


STYLE = """\
.precision-method h3 { margin-top:20px; }
.precision-method caption { text-align:left; padding:10px; }
.precision-method code { overflow-wrap:anywhere; }
#precision-direct { font-variant-numeric:tabular-nums; }
#precision-direct th { white-space:normal; }
.precision-scroll { margin-top:12px; max-height:78vh; }
#precision-comparison { width:max-content; min-width:100%; font-variant-numeric:tabular-nums; }
#precision-comparison caption { text-align:left; padding:10px; color:var(--muted); }
#precision-comparison th, #precision-comparison td { min-width:110px; max-width:330px; }
#precision-comparison thead { position:sticky; top:0; z-index:1; }
#precision-comparison thead tr:first-child th { background:#dce8f2; border-right:1px solid #becfdd; }
#precision-comparison .standard21-issue { color:#8b1d1d; background:#fce3e3; font-weight:600; }
#precision-comparison details { min-width:230px; max-width:480px; }
#precision-comparison pre { max-height:360px; overflow:auto; }
"""
