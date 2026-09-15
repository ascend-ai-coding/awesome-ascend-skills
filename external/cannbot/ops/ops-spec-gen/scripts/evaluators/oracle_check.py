# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stage 9 — oracle reachability.

Real-import based check: try to actually import the framework declared in
spec.math_semantics.reference_oracle.framework, then walk the attribute chain
to confirm `api` is a callable. Also validates that every `${attr.X}` placeholder
in `kwargs` references a real attribute name.

Skips when:
  * reference_oracle.absent == true (spec 显式声明无 oracle)
  * the requested framework is not installed   (info-level SKIP, not FAIL)

Emits errors when:
  * absent: false but framework is installed and api can't be resolved (typo)
  * placeholder ${attr.X} references non-existent attribute
  * ${kind} is unknown (only attr/input/output supported)

Cost note: importing torch can take 1-3 s. We import lazily here, so users without
torch installed simply get a SKIP. Repeated calls within one Python process reuse
the import cache.
"""

from __future__ import annotations

import functools
import importlib
import re
from pathlib import Path
from typing import Any

import yaml


# 占位符语法：${ns.path}
# - ns ∈ {attr, input, output}
# - path = name[.subfield[.subfield]...]，每段必须是 [a-zA-Z_][a-zA-Z0-9_]*
# - **不**支持下标 / 切片（例如 ${attr.foo[0]} 非法）—— 真要支持下标需要在 spec
#   schema 与 stage 9 占位符求值层同时扩展，不是 regex 改一下就够
_PLACEHOLDER_RE = re.compile(
    r"\$\{(?P<ns>attr|input|output)\.(?P<path>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"
)


@functools.lru_cache(maxsize=1)
def _known_frameworks() -> tuple[str, ...]:
    """Load known framework list from registries/framework_oracle_registry.yaml.

    Cached per-process. Adding a framework should only need a yaml line, no code change.
    """
    registry_path = Path(__file__).resolve().parent.parent.parent / "registries" / "framework_oracle_registry.yaml"
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        return tuple(data.get("frameworks") or [])
    except FileNotFoundError:
        return ("torch", "numpy", "scipy", "tensorflow", "jax")


def _resolve_attribute_chain(root: Any, dotted: str) -> Any:
    """Walk `a.b.c` against root via getattr; raise AttributeError on miss."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _check_placeholders(spec: dict, findings: list[dict]) -> None:
    """Scan kwargs values for ${ns.path} and confirm path exists in spec."""
    oracle = (spec.get("math_semantics") or {}).get("reference_oracle") or {}
    kwargs = oracle.get("kwargs") or {}

    attr_names = {a.get("name") for a in (spec.get("attributes") or [])}
    input_names = {i.get("name") for i in (spec.get("inputs") or [])}
    output_names = {o.get("name") for o in (spec.get("outputs") or [])}

    for kw_name, kw_val in kwargs.items():
        if not isinstance(kw_val, str):
            continue
        for m in _PLACEHOLDER_RE.finditer(kw_val):
            ns = m.group("ns")
            path = m.group("path")
            head = path.split(".")[0]
            valid = {
                "attr": attr_names,
                "input": input_names,
                "output": output_names,
            }[ns]
            if head not in valid:
                findings.append({
                    "severity": "error",
                    "rule_id": "oracle_reachable.placeholder_unresolved",
                    "field_path": f"math_semantics.reference_oracle.kwargs.{kw_name}",
                    "message": (
                        f"占位符 ${{{ns}.{path}}} 引用不存在的 {ns}={head!r}；"
                        f"已知 {ns}s={sorted(valid)}"
                    ),
                    "suggested_fix": (
                        f"修正占位符为已声明的 {ns} 名，或在 spec 中添加 {head!r}"
                    ),
                })


def _resolve_api_callable(framework: str, api_path: str, field_path: str,
                          findings: list[dict]):
    """真 import framework 并 getattr 到 api。返回 (ok, callable_or_none)。

    framework_not_installed 时 ok=False；调用方据此决定走 SKIP。
    """
    api_head, *api_rest = api_path.split(".", 1)
    if api_head != framework:
        findings.append({
            "severity": "error",
            "rule_id": "oracle_reachable.api_framework_mismatch",
            "field_path": field_path,
            "message": (
                f"api={api_path!r} 与 framework={framework!r} 不匹配（首段须等于 framework）"
            ),
            "suggested_fix": f"把 api 改为 {framework}.<...> 形式",
        })
        return False, None

    try:
        mod = importlib.import_module(framework)
    except ImportError:
        findings.append({
            "severity": "info",
            "rule_id": "oracle_reachable.framework_not_installed",
            "field_path": "math_semantics.reference_oracle.framework",
            "message": f"framework={framework!r} 未安装；stage 9 framework 部分跳过",
            "suggested_fix": f"pip install {framework} 后重跑 stage 9（占位符仍已校验）",
        })
        return False, None

    if api_rest:
        try:
            target = _resolve_attribute_chain(mod, api_rest[0])
        except AttributeError as e:
            findings.append({
                "severity": "error",
                "rule_id": "oracle_reachable.api_not_found",
                "field_path": field_path,
                "message": f"无法解析 {api_path!r}：{e}",
                "suggested_fix": (
                    f"确认 {framework} 中是否真有 {api_rest[0]!r}（拼写 / 版本差异）"
                ),
            })
            return False, None
    else:
        target = mod

    if not callable(target):
        findings.append({
            "severity": "error",
            "rule_id": "oracle_reachable.api_not_callable",
            "field_path": field_path,
            "message": f"{api_path!r} 已找到但不是 callable (是 {type(target).__name__})",
            "suggested_fix": "选择实际可调用的同义 API",
        })
        return False, None

    return True, target


def _check_arg_reference(arg, attr_names: set, input_names: set,
                         seen_node_ids: set, node_id: str,
                         arg_index: int, findings: list[dict]) -> None:
    """校验 composition 节点的单个 arg 引用合法。

    合法 arg 形式：
      * 字符串裸标识符（input 名 或 已出现的前序节点 id）
      * `${attr.X}` / `${input.X}` 占位符
      * 字面量数字 / bool / null

    禁止：`${output.X}`（output 是 spec 输出名空间，与 oracle 节点拓扑无关）
    """
    if not isinstance(arg, str):
        return  # 字面量直接放行

    placeholders = list(_PLACEHOLDER_RE.finditer(arg))
    if placeholders:
        for m in placeholders:
            ns = m.group("ns")
            path = m.group("path")
            head = path.split(".")[0]
            if ns == "output":
                findings.append({
                    "severity": "error",
                    "rule_id": "oracle_reachable.placeholder_unresolved",
                    "field_path": f"math_semantics.reference_oracle.composition[{node_id}].args[{arg_index}]",
                    "message": (
                        "composition.args 不允许 ${output.X}（output 是 spec 输出名空间，"
                        "与 oracle 节点拓扑无关）；如需引用前序节点请用裸 id"
                    ),
                    "suggested_fix": "把 ${output.X} 替换为前序节点 id 或 ${input.X}",
                })
                continue
            valid = {"attr": attr_names, "input": input_names}[ns]
            if head not in valid:
                findings.append({
                    "severity": "error",
                    "rule_id": "oracle_reachable.placeholder_unresolved",
                    "field_path": f"math_semantics.reference_oracle.composition[{node_id}].args[{arg_index}]",
                    "message": (
                        f"占位符 ${{{ns}.{path}}} 引用不存在的 {ns}={head!r}；"
                        f"已知 {ns}s={sorted(valid)}"
                    ),
                    "suggested_fix": f"修正占位符或在 spec 中添加 {head!r}",
                })
        return

    # 裸字符串 → 必须是 input 名 或前序节点 id
    if arg in input_names or arg in seen_node_ids:
        return

    findings.append({
        "severity": "error",
        "rule_id": "oracle_reachable.composition_arg_unresolved",
        "field_path": f"math_semantics.reference_oracle.composition[{node_id}].args[{arg_index}]",
        "message": (
            f"composition 节点 {node_id!r} 的 args[{arg_index}]={arg!r} 既不是 input 名也"
            f"不是前序节点 id；已知 inputs={sorted(input_names)}，前序节点={sorted(seen_node_ids)}"
        ),
        "suggested_fix": (
            "args 元素必须是：input 名 / 前序节点 id / ${attr.X} / ${input.X} / 字面量"
        ),
    })


def _validate_composition_topology(composition: list, output_id: str,
                                   spec: dict, findings: list[dict]) -> bool:
    """节点 id 唯一 / output 存在 / args 引用合法（隐含禁循环）。"""
    attr_names = {a.get("name") for a in (spec.get("attributes") or [])}
    input_names = {i.get("name") for i in (spec.get("inputs") or [])}

    seen_ids: set = set()
    ok = True

    for idx, node in enumerate(composition):
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            findings.append({
                "severity": "error",
                "rule_id": "oracle_reachable.composition_node_invalid",
                "field_path": f"math_semantics.reference_oracle.composition[{idx}].id",
                "message": "composition 节点缺 id 或 id 非字符串",
                "suggested_fix": "每个节点必须有非空字符串 id",
            })
            ok = False
            continue

        if node_id in seen_ids:
            findings.append({
                "severity": "error",
                "rule_id": "oracle_reachable.composition_id_collision",
                "field_path": f"math_semantics.reference_oracle.composition[{idx}].id",
                "message": f"composition 节点 id={node_id!r} 重复",
                "suggested_fix": "节点 id 必须唯一",
            })
            ok = False
            continue

        if node_id in input_names:
            findings.append({
                "severity": "error",
                "rule_id": "oracle_reachable.composition_id_shadows_input",
                "field_path": f"math_semantics.reference_oracle.composition[{idx}].id",
                "message": (
                    f"composition 节点 id={node_id!r} 与 input 同名，会让 args 引用歧义"
                ),
                "suggested_fix": "重命名节点 id（如加 _node 后缀）",
            })
            ok = False

        for ai, arg in enumerate(node.get("args") or []):
            _check_arg_reference(arg, attr_names, input_names, seen_ids,
                                 node_id, ai, findings)

        # kwargs 占位符校验（与 args 对称）
        for kw_name, kw_val in (node.get("kwargs") or {}).items():
            if not isinstance(kw_val, str):
                continue
            for m in _PLACEHOLDER_RE.finditer(kw_val):
                ns = m.group("ns")
                path = m.group("path")
                head = path.split(".")[0]
                if ns == "output":
                    findings.append({
                        "severity": "error",
                        "rule_id": "oracle_reachable.placeholder_unresolved",
                        "field_path": f"math_semantics.reference_oracle.composition[{node_id}].kwargs.{kw_name}",
                        "message": (
                            "composition.kwargs 不允许 ${output.X}（output 是 spec 输出名空间，"
                            "与 oracle 节点拓扑无关）"
                        ),
                        "suggested_fix": "把 ${output.X} 替换为 ${attr.X} 或字面量",
                    })
                    continue
                valid = {"attr": attr_names, "input": input_names}[ns]
                if head not in valid:
                    findings.append({
                        "severity": "error",
                        "rule_id": "oracle_reachable.placeholder_unresolved",
                        "field_path": f"math_semantics.reference_oracle.composition[{node_id}].kwargs.{kw_name}",
                        "message": (
                            f"占位符 ${{{ns}.{path}}} 引用不存在的 {ns}={head!r}；"
                            f"已知 {ns}s={sorted(valid)}"
                        ),
                        "suggested_fix": f"修正占位符或在 spec 中添加 {head!r}",
                    })

        seen_ids.add(node_id)

    if output_id not in seen_ids:
        findings.append({
            "severity": "error",
            "rule_id": "oracle_reachable.composition_output_unresolved",
            "field_path": "math_semantics.reference_oracle.output",
            "message": (
                f"output={output_id!r} 不在 composition 节点 id 集合 {sorted(seen_ids)}"
            ),
            "suggested_fix": "把 output 改为某个已声明节点的 id",
        })
        ok = False

    if any(f["severity"] == "error" for f in findings):
        ok = False
    return ok


def _check_available_dtype(spec: dict, oracle: dict, findings: list[dict]) -> None:
    """available_for_dtype 与 supported_combinations 输入 dtype 子集校验。"""
    avail = set(oracle.get("available_for_dtype") or [])
    declared_dtypes: set = set()
    for combo in (spec.get("dtype_policy") or {}).get("supported_combinations") or []:
        for d in (combo.get("inputs") or {}).values():
            declared_dtypes.add(d)
    not_in_oracle = declared_dtypes - avail if avail else set()
    if not_in_oracle:
        findings.append({
            "severity": "warning",
            "rule_id": "oracle_reachable.dtype_unsupported",
            "field_path": "math_semantics.reference_oracle.available_for_dtype",
            "message": (
                f"输入 dtype {sorted(not_in_oracle)} 不在 oracle "
                f"available_for_dtype 中；这些 case 将走降级路径"
            ),
            "suggested_fix": "扩 available_for_dtype 或在 supported_combinations 移除这些 dtype",
        })


def stage_9(spec: dict) -> tuple[str, list[dict]]:
    """Resolve the oracle by real import. Skip if framework not installed.

    两种 oracle 模式：单 callable（api 字段）/ DAG composition（composition + output 字段）。
    absent=true 时直接 SKIP（spec 作者显式声明无 oracle，由 invariants + boundary 覆盖）。
    """
    findings: list[dict] = []

    oracle = (spec.get("math_semantics") or {}).get("reference_oracle") or {}
    absent = bool(oracle.get("absent", False))

    if absent:
        return "SKIP", [{
            "severity": "info",
            "rule_id": "oracle_reachable.absent",
            "field_path": "math_semantics.reference_oracle.absent",
            "message": "oracle 显式声明缺失，stage 9 跳过（语义由 invariants + boundary_conditions 覆盖）",
            "suggested_fix": None,
        }]

    # Always check single-api kwargs placeholders even if framework missing
    _check_placeholders(spec, findings)

    framework = oracle.get("framework")
    api_path = oracle.get("api")
    composition = oracle.get("composition")
    output_id = oracle.get("output")

    if not framework:
        findings.append({
            "severity": "error",
            "rule_id": "oracle_reachable.incomplete",
            "field_path": "math_semantics.reference_oracle",
            "message": "framework 字段缺失（且 absent=false）",
            "suggested_fix": "补 framework，或把 absent 设为 true",
        })
        return "FAIL", findings

    if not api_path and not composition:
        findings.append({
            "severity": "error",
            "rule_id": "oracle_reachable.incomplete",
            "field_path": "math_semantics.reference_oracle",
            "message": "api 或 composition 必须二选一（且 absent=false）",
            "suggested_fix": "选择单 callable 模式（api）或 DAG 模式（composition + output）",
        })
        return "FAIL", findings

    if framework not in _known_frameworks():
        findings.append({
            "severity": "warning",
            "rule_id": "oracle_reachable.unknown_framework",
            "field_path": "math_semantics.reference_oracle.framework",
            "message": f"framework={framework!r} 不在已知列表 {_known_frameworks()}",
            "suggested_fix": "确认拼写；若为新框架请在 registries/framework_oracle_registry.yaml 中加一行",
        })

    # ---------- DAG composition 模式 ----------
    if composition:
        if not output_id:
            findings.append({
                "severity": "error",
                "rule_id": "oracle_reachable.incomplete",
                "field_path": "math_semantics.reference_oracle.output",
                "message": "composition 模式必须声明 output 字段（指向某节点 id）",
                "suggested_fix": "补 output: <node_id>",
            })
            return "FAIL", findings

        topo_ok = _validate_composition_topology(composition, output_id, spec, findings)
        if not topo_ok:
            return "FAIL", findings

        any_skip = False
        for node in composition:
            nid = node["id"]
            n_api = node["api"]
            field = f"math_semantics.reference_oracle.composition[{nid}].api"
            n_ok, _ = _resolve_api_callable(framework, n_api, field, findings)
            if not n_ok:
                if any(f["rule_id"] == "oracle_reachable.framework_not_installed"
                       for f in findings):
                    any_skip = True
                    break  # framework_not_installed 整个 stage SKIP
        if any_skip:
            status = "FAIL" if any(f["severity"] == "error" for f in findings) else "SKIP"
        else:
            status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"

        _check_available_dtype(spec, oracle, findings)
        if status == "PASS" and any(f["severity"] == "error" for f in findings):
            status = "FAIL"
        return status, findings

    # ---------- 单 api 模式（向后兼容）----------
    api_ok, _ = _resolve_api_callable(
        framework, api_path,
        "math_semantics.reference_oracle.api",
        findings,
    )

    if any(f["rule_id"] == "oracle_reachable.framework_not_installed" for f in findings):
        status = "FAIL" if any(f["severity"] == "error" for f in findings) else "SKIP"
        return status, findings

    _check_available_dtype(spec, oracle, findings)

    status = "FAIL" if any(f["severity"] == "error" for f in findings) else "PASS"
    return status, findings
