#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Matmul Tiling Solver — TTK 标准化执行模块（算子档案驱动）

将 CSV 校验、baseline 执行、候选扫参、tiling data 提取等流程标准化为可复用的 Python 函数。
AI agent 每次执行 skill 时，直接调用本模块的函数，不再需要从文字描述中自行构造命令。

当前算子档案机制：
    case 分类、tiling 布局、模板集合、模式识别、输入校验、候选硬约束、
    硬件常量、可调宏和 vendor 包由当次针对目标仓生成的
    references/profiles/<op_name>/ 提供。
    执行层仍有 vendor/JIT 模型；新算子必须
    先通过 references/operator-guide.md 的兼容性审计，不能仅凭存在算子档案就
    视为已支持。

    from ttk_runner import load_profile
    load_profile("<op_name>")   # 显式指定算子（可选，见下）

    validate_case_csv / extract_all_params / extract_shape_params 会按 CSV 行的
    op_name 自动 load_profile（全部行必须同一算子），solver.md 工作流无需显式调用；
    其余 per-op 入口（allowed_templates / check_candidate / generate_ranking_block /
    扫参脚本生成等）要求已有活跃档案，未加载时报错指路 load_profile。

使用方式：
    from ttk_runner import *

    # 0. （可选）显式加载算子档案——validate/extract 会按 CSV 自动加载
    load_profile("<op_name>")

    # 1. 校验用户 CSV + 提取参数
    validate_case_csv("user_case.csv")
    params = extract_shape_params(row)

    # 2. 跑 baseline
    baseline = run_baseline("user_case.csv", M, N, K)

    # 3. 跑候选扫参
    results = run_ttk_scan(candidates, "user_case.csv", M, N, K)

    # 4. 输出推荐榜
    ranking = generate_ranking(baseline, results)
"""

import ast
import csv
import glob
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# 常量
# ============================================================================

# 部署路径：新环境/新算子首次使用时调用 configure_paths() 一次（写入 scripts/
# paths.json，跨进程持久生效）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR = os.path.dirname(_SCRIPT_DIR)
_PATHS_SIDECAR = os.path.join(_SCRIPT_DIR, "paths.json")

# 算子仓默认根目录：未在 paths.json 显式配置的算子，op_base = _OP_BASE_ROOT +
# 档案 identity.repo_rel（如 matmul/quant_batch_matmul_v3）。
_OP_BASE_ROOT = "/workspace/ops-nn"

_PATH_DEFAULTS = {
    # per-op 算子目录映射（op_host/op_kernel/tests 所在目录）{op_name: path}。
    "op_base": {},
    "ttk_dir": "/workspace/ops-test-kit",  # ttk 工具目录（ttk 命令的 cwd）
    "run_dir": "/workspace/solver_run",  # 产出目录（缺省落盘处）
    "case_dir": "/workspace/cases",  # case CSV 目录（_result.csv 兜底查找）
}


def _load_paths() -> dict:
    cfg = dict(_PATH_DEFAULTS)
    try:
        with open(_PATHS_SIDECAR) as f:
            loaded = json.load(f)
        cfg.update({k: v for k, v in loaded.items() if k in cfg})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if not isinstance(cfg["op_base"], dict):
        raise RuntimeError("scripts/paths.json 的 op_base 必须是 {op_name: path} 字典")
    return cfg


def _resolve_op_base(op_name: str, repo_rel: str) -> str:
    """
    per-op 算子仓 base 解析：paths.json "op_base" 字典的显式条目 >
    默认 _OP_BASE_ROOT/<repo_rel>。
    """
    raw = _load_paths()["op_base"]
    if raw.get(op_name):
        return raw[op_name]
    return f"{_OP_BASE_ROOT}/{repo_rel}"


def _apply_op_paths(op_name: str, profile: dict) -> None:
    """按活跃档案解析 per-op 部署路径（OP_BASE/PLUGIN_DIR/KERNEL_META_DIR）。"""
    global OP_BASE, PLUGIN_DIR, KERNEL_META_DIR
    OP_BASE = _resolve_op_base(op_name, profile["identity"]["repo_rel"])
    PLUGIN_DIR = f"{OP_BASE}/tests/assets"
    KERNEL_META_DIR = f"{TTK_DIR}/kernel_meta"


def configure_paths(
    op=None, op_base=None, ttk_dir=None, run_dir=None, case_dir=None
) -> dict:
    """
    设置部署路径并持久化到 sidecar（本模块同目录 paths.json），跨进程生效。

    新环境/新算子首次使用时调用一次；参数为 None 的项保持原值。返回完整配置。

    op_base 为 per-op 配置：sidecar 内 "op_base" 是 {op_name: path} 字典——
    新算子接入时传 op=<op_name> + op_base=<算子仓路径>（op 缺省时用当前
    活跃档案的算子名）。未显式配置的算子走默认
    /workspace/ops-nn/<identity.repo_rel>（load_profile 时解析）。
    ttk_dir/run_dir/case_dir 为跨算子共享的全局配置。
    """
    cfg = _load_paths()
    for key, val in (
        ("ttk_dir", ttk_dir),
        ("run_dir", run_dir),
        ("case_dir", case_dir),
    ):
        if val:
            cfg[key] = val
    raw_base = cfg["op_base"]
    if op_base:
        target = op or _ACTIVE_OP_NAME
        if target:
            base_map = dict(raw_base)
            base_map[target] = op_base
            cfg["op_base"] = base_map
        else:
            raise ValueError(
                "configure_paths(op_base=...) 需要知道配置给哪个算子：传 op=<op_name>，"
                "或先 load_profile(<op_name>) 设定活跃算子（op_base 是 per-op 配置）。"
            )
    with open(_PATHS_SIDECAR, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    global TTK_DIR, RUN_DIR, CASE_DIR
    TTK_DIR = cfg["ttk_dir"]
    RUN_DIR = cfg["run_dir"]
    CASE_DIR = cfg["case_dir"]
    if _ACTIVE_OP_NAME:
        # 活跃算子的 op_base 可能被显式更新（或 ttk_dir 变了）→ 重解析 per-op 路径
        _apply_op_paths(_ACTIVE_OP_NAME, _require_profile())
    os.makedirs(RUN_DIR, exist_ok=True)
    return cfg


_cfg = _load_paths()
TTK_DIR = _cfg["ttk_dir"]
RUN_DIR = _cfg["run_dir"]
CASE_DIR = _cfg["case_dir"]
# per-op 路径（OP_BASE/PLUGIN_DIR/KERNEL_META_DIR）由 load_profile 按活跃档案解析，
# 加载前为 None（引擎所有 per-op 数据都要求先 load_profile）。
OP_BASE: Optional[str] = None
PLUGIN_DIR: Optional[str] = None
KERNEL_META_DIR: Optional[str] = None
VENDOR_CONFIG = "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini"

# ============================================================================
# per-op 档案层：load_profile(op_name) 加载
# references/profiles/<op_name>/profile.json 并动态导入同目录
# recognition.py / constraints.py 适配器。算子差异尽量由活跃档案声明；执行层固定
# 契约见 references/operator-guide.md。档案由当次目标算子仓推导，
# evidence 记录每条规则的源码出处。
# ============================================================================

_PROFILE_ROOT = os.path.join(_SKILL_DIR, "references", "profiles")
_PROFILE_CACHE: dict = {}  # op_name → profile dict（JSON 只读一次）
_RECOGNITION_CACHE: dict = {}  # op_name → recognition 适配器模块
_CONSTRAINTS_CACHE: dict = {}  # op_name → constraints 适配器模块
_ACTIVE_OP_NAME: Optional[str] = None

# ---- 以下模块级缓存全部由 load_profile 从活跃档案派生（加载前为空值）----

# vendor 包名映射（模板名/normal → 安装侧 vendor 包目录名，derive 见 _derive_vendor_packages）
VENDOR_PACKAGES: dict = {}

# TUNE_ 宏名映射（候选字段名 → #define 宏名；derive 见 _derive_macro_map）
MACRO_MAP: dict = {}

# skill 规定的每模板最少候选数（档案 constraints.required_candidates_per_template）
REQUIRED_CANDIDATES_PER_TEMPLATE: Optional[int] = None


def _available_profiles() -> list:
    """扫描 profiles 目录，返回含 profile.json 的算子档案包名列表。"""
    out = []
    if os.path.isdir(_PROFILE_ROOT):
        for name in sorted(os.listdir(_PROFILE_ROOT)):
            if os.path.isfile(os.path.join(_PROFILE_ROOT, name, "profile.json")):
                out.append(name)
    return out


def _import_profile_module(op_name: str, filename: str, kind: str):
    """经 importlib 从算子档案目录动态加载 recognition/constraints 适配器模块。"""
    path = os.path.join(_PROFILE_ROOT, op_name, filename)
    mod_name = f"ttk_profile_{op_name}_{kind}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载算子 '{op_name}' 的 {kind} 适配器模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


TEMPLATE_ALIAS = {
    "l0cpingpong": "l0cpp",
    "streamK": "streamk",
    "streamk": "streamk",
    "swat": "swat",
    "afull": "afull",
}


def normalize_template(name: str) -> str:
    return TEMPLATE_ALIAS.get(name, name)


def _derive_vendor_packages(profile: dict) -> dict:
    """
    vendor 包名映射（模板名/normal → 安装侧 vendor 包目录名，档案声明驱动）：
    - worktree 模板（templates.<t>）：vendor_install_dir（swat/afull/l0cpp/streamk）
    - normal 包：stub 模式（声明 tune_macro_define_block 的量化模式）kernel_files
      安装路径的 vendors/<pkg>/op_impl 前缀——stub 模式扫参全程钉 normal 包，
      sed 目标与运行时包保持一致
    """
    pkgs: dict = {
        normalize_template(name): cfg["vendor_install_dir"]
        for name, cfg in (profile.get("templates") or {}).items()
        if cfg.get("vendor_install_dir")
    }
    # 现有构建契约的 normal 包名；operator-guide.md 将其列为固定假设。
    normal = "custom_normal_nn"
    for mode_cfg in profile["quant_modes"].values():
        if not mode_cfg.get("tune_macro_define_block"):
            continue  # worktree 模式（如 mx）：kernel_files 指向各自 forced 包
        for cc_files in (mode_cfg.get("kernel_files") or {}).values():
            for entry in (cc_files or {}).values():
                install = entry["install"] if isinstance(entry, dict) else entry
                m = re.search(r"/vendors/([^/]+)/op_impl/", install)
                if m:
                    if normal != m.group(1):
                        raise RuntimeError(
                            f"档案 stub 模式 kernel_files 指向多个 normal 包"
                            f"（{normal} / {m.group(1)}）——声明不自洽"
                        )
    pkgs["normal"] = normal
    return pkgs


# TUNE_ 宏声明/赋值解析（档案声明块文本 → 可调宏集合/字段映射）
_TUNE_DEFINE_RE = re.compile(r"#define\s+(TUNE_\w+)")
_TUNE_ASSIGN_RE = re.compile(r"(?:\.|->)(\w+)\s*=\s*(TUNE_\w+)\s*;")


def _derive_macro_map(profile: dict) -> dict:
    """
    从档案声明块派生可调宏映射 {候选字段名: TUNE_ 宏名}。

    - 宏集合：worktree 模板 templates.<t>.tune_macro_define_block 与 stub 模式
      quant_modes.<mode>.tune_macro_define_block 内 `#define TUNE_*` 的并集
      （哪些宏可调由档案声明）。
    - 字段对应：同档案 tuning_blocks.<variant>.block_text /
      tuning_block_template 内 `<field> = TUNE_<MACRO>;` 赋值对——左值即候选
      字段名（唯一可靠源，如 streamk 包的 TUNE_KL1_STREAMK 对应字段 kL1）。
    - 声明了宏但无赋值对应、或同一宏映射到多个字段 → 档案不自洽，报错。
    """
    macro_to_field: dict = {}
    assign_texts = []
    for t_cfg in (profile.get("templates") or {}).values():
        for v_block in (t_cfg.get("tuning_blocks") or {}).values():
            assign_texts.append(v_block.get("block_text") or "")
    for m_cfg in profile["quant_modes"].values():
        assign_texts.append(m_cfg.get("tuning_block_template") or "")
    for text in assign_texts:
        for field_name, macro in _TUNE_ASSIGN_RE.findall(text):
            prev = macro_to_field.setdefault(macro, field_name)
            if prev != field_name:
                raise RuntimeError(
                    f"档案 TUNE 宏 {macro} 映射到多个字段（{prev} / {field_name}）——tuning 块声明不自洽"
                )
    declared: set = set()
    for t_cfg in (profile.get("templates") or {}).values():
        declared |= set(
            _TUNE_DEFINE_RE.findall(t_cfg.get("tune_macro_define_block") or "")
        )
    for m_cfg in profile["quant_modes"].values():
        declared |= set(
            _TUNE_DEFINE_RE.findall(m_cfg.get("tune_macro_define_block") or "")
        )
    unmatched = declared - set(macro_to_field)
    if unmatched:
        raise RuntimeError(
            f"档案声明了无字段赋值对应的 TUNE 宏: {sorted(unmatched)}——define 块与 tuning 块不自洽"
        )
    return {field: macro for macro, field in macro_to_field.items()}


def _validate_profile(profile: dict, op_name: str) -> None:
    """校验执行层实际依赖的算子档案结构，避免缺字段在长任务中途才暴露。"""
    if profile.get("_scaffold") is True:
        raise RuntimeError(
            f"算子 '{op_name}' 的算子档案仍是草稿。请完成所有必填项、"
            "补齐源码 evidence，并删除顶层 _scaffold 后再加载。"
        )
    required = {
        "schema",
        "identity",
        "support",
        "version_anchor",
        "soc",
        "case_classes",
        "fold_rule",
        "struct_layouts",
        "input_rules",
        "recognition",
        "quant_modes",
        "templates",
        "constraints",
        "package_gate",
    }
    missing = sorted(required - profile.keys())
    if missing:
        raise RuntimeError(f"算子 '{op_name}' 的算子档案缺少必需字段: {missing}")
    if profile["identity"].get("op_name") != op_name:
        raise RuntimeError(
            f"算子档案目录名 '{op_name}' 与 identity.op_name="
            f"{profile['identity'].get('op_name')!r} 不一致"
        )

    for section in ("case_classes", "struct_layouts", "quant_modes"):
        if not isinstance(profile[section], dict) or not profile[section]:
            raise RuntimeError(f"算子档案 {section} 必须是非空对象")

    runtime_soc_keys = {
        "l1_size",
        "l0a_size",
        "l0b_size",
        "l0c_size",
        "data_size_l0c",
    }
    if "mx_group_size" in profile["soc"]:
        runtime_soc_keys.add("mx_group_size")
    bad_soc = sorted(
        key
        for key in runtime_soc_keys
        if isinstance(profile["soc"].get(key), bool)
        or not isinstance(profile["soc"].get(key), (int, float))
        or profile["soc"][key] <= 0
    )
    if bad_soc:
        raise RuntimeError(
            f"算子档案字段 soc 缺少执行器所需的正数常量或值非法: {bad_soc}"
        )

    core_query = profile["soc"].get("core_query")
    if core_query is not None and (
        not isinstance(core_query, dict)
        or not all(
            isinstance(core_query.get(key), str) and core_query[key].strip()
            for key in ("core_type", "spec_key", "evidence")
        )
    ):
        raise RuntimeError("soc.core_query 必须声明 core_type/spec_key/evidence")

    for dtype, size in profile.get("dtype_sizeof", {}).items():
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or size <= 0
            or not isinstance(profile.get("dtype_evidence", {}).get(dtype), str)
            or not profile["dtype_evidence"][dtype].strip()
        ):
            raise RuntimeError(
                f"dtype_sizeof.{dtype} 必须为正数字节数并附 dtype_evidence"
            )

    required_candidates = profile["constraints"].get("required_candidates_per_template")
    if (
        isinstance(required_candidates, bool)
        or not isinstance(required_candidates, int)
        or required_candidates <= 0
    ):
        raise RuntimeError(
            "算子档案字段 constraints.required_candidates_per_template 必须是正整数"
        )

    key_decl = profile.get("tiling_key")
    if key_decl is None:
        raise RuntimeError(f"算子 '{op_name}' 的算子档案缺少 tiling_key 位段声明")
    if key_decl is not None:
        shift = key_decl.get("kernel_type_shift")
        bits = key_decl.get("kernel_type_bits")
        if (
            not isinstance(shift, int)
            or shift < 0
            or not isinstance(bits, int)
            or not 1 <= bits <= 32
        ):
            raise RuntimeError(
                f"算子 '{op_name}' 的 tiling_key.kernel_type_shift/bits 非法: "
                f"{shift}/{bits}"
            )

    for name, spec in profile["struct_layouts"].items():
        n_slots = spec.get("u32_slots")
        if not isinstance(n_slots, int) or n_slots <= 0:
            raise RuntimeError(f"struct_layouts.{name}.u32_slots 必须是正整数")
        when = spec.get("when", {})
        if not isinstance(when, dict):
            raise RuntimeError(f"struct_layouts.{name}.when 必须是上下文匹配条件对象")
        byte_size = spec.get("byte_size")
        if byte_size is not None and byte_size != n_slots * 4:
            raise RuntimeError(
                f"struct_layouts.{name}: byte_size={byte_size}，但 u32_slots={n_slots} "
                f"应对应 {n_slots * 4} 字节"
            )

    for section, exports in (
        ("recognition", ("derive_facts", "validate_case")),
        ("constraints", ("check_candidate", "validate_rules_against_anchors")),
    ):
        declared = set(profile[section].get("exports", exports))
        absent = sorted(set(exports) - declared)
        if absent:
            raise RuntimeError(f"算子档案字段 {section}.exports 缺少 {absent}")

    scheduling = profile.get("scheduling")
    if scheduling is not None:
        if not isinstance(scheduling, dict) or not all(
            isinstance(scheduling.get(key), str) and scheduling[key].strip()
            for key in ("model", "evidence")
        ):
            raise RuntimeError("scheduling 必须声明调度模型与源码 evidence，或省略该节")
        if "derive_schedule" not in profile["recognition"].get("exports", []):
            raise RuntimeError("调度模型需要 recognition.exports 声明 derive_schedule")

    package_files = profile["package_gate"].get("required_files")
    if not isinstance(package_files, list) or not package_files:
        raise RuntimeError(
            "算子档案字段 package_gate.required_files 必须列出至少一个会影响算子发现/注册的安装产物"
        )
    for idx, item in enumerate(package_files):
        path = item.get("path") if isinstance(item, dict) else None
        scope = item.get("package_scope", "all") if isinstance(item, dict) else None
        if (
            not isinstance(path, str)
            or not path
            or os.path.isabs(path)
            or ".." in path.split(os.sep)
        ):
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].path 必须是 vendor 包内安全相对路径"
            )
        unexpanded = path.replace("{package}", "").replace("{op_name}", "")
        if "{" in unexpanded or "}" in unexpanded:
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].path 只能使用 "
                "{package}/{op_name} 占位符"
            )
        try:
            path.format(package="vendor", op_name=op_name)
        except (KeyError, ValueError) as e:
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].path 只能使用 "
                "{package}/{op_name} 占位符"
            ) from e
        if scope not in ("all", "normal", "forced"):
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].package_scope 非法: {scope!r}"
            )
        contains = item.get("contains", [])
        if (
            not isinstance(contains, list)
            or not contains
            or not all(isinstance(s, str) and s for s in contains)
        ):
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].contains 必须是非空字符串列表"
            )
        if item.get("contains_match", "whitespace") not in ("whitespace", "literal"):
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].contains_match 仅支持 whitespace/literal"
            )
        if item.get("syntax") not in (None, "python"):
            raise RuntimeError(
                f"算子档案 package_gate.required_files[{idx}].syntax 仅支持 python"
            )

    for mode, mode_cfg in profile["quant_modes"].items():
        kernel_files = mode_cfg.get("kernel_files", {})
        for case_class, templates in mode_cfg.get(
            "templates_by_case_class", {}
        ).items():
            table = kernel_files.get(case_class, {})
            missing_entries = sorted(
                normalize_template(t)
                for t in templates
                if normalize_template(t) not in table
            )
            if missing_entries:
                raise RuntimeError(
                    f"quant_modes.{mode}.kernel_files.{case_class} 缺少模板: "
                    f"{missing_entries}"
                )
        tables = {"default": mode_cfg.get("kernel_type_map", {})}
        tables.update(mode_cfg.get("kernel_type_map_by_case_class", {}))
        for case_class, table in tables.items():
            kt_seen = {}
            for template, kernel_type in table.items():
                tmpl = normalize_template(template)
                try:
                    kt = int(kernel_type)
                except (TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"{mode}/{case_class}/{template}: kernel type 不是整数"
                    ) from e
                previous = kt_seen.setdefault(kt, tmpl)
                if previous != tmpl:
                    raise RuntimeError(
                        f"{mode}/{case_class}: kernel type {kt} 同时映射到 {previous}/{tmpl}"
                    )


def load_profile(op_name: str) -> dict:
    """
    加载算子档案并设为引擎的当前档案（多算子引擎的入口，幂等且带缓存）。

    - 读取 references/profiles/<op_name>/profile.json（相对本文件目录）
    - 经 importlib 自同目录动态加载 recognition/constraints 适配器（文件名由档案
      recognition/constraints 节声明）
    - 派生引擎侧缓存：VENDOR_PACKAGES / MACRO_MAP /
      REQUIRED_CANDIDATES_PER_TEMPLATE / 布局槽位表
    - 按 paths.json 的 per-op op_base（或默认 /workspace/ops-nn/<repo_rel>）解析
      OP_BASE/PLUGIN_DIR/KERNEL_META_DIR

    返回算子档案字典（case_classes/quant_modes/struct_layouts/fold_rule/soc 等，
    供调用方内省）。未知算子抛 RuntimeError（列出可用档案并指向
    operator-guide.md 接入程序）。
    """
    global _ACTIVE_OP_NAME, VENDOR_PACKAGES, MACRO_MAP
    global REQUIRED_CANDIDATES_PER_TEMPLATE

    if not op_name or not os.path.isfile(
        os.path.join(_PROFILE_ROOT, str(op_name), "profile.json")
    ):
        raise RuntimeError(
            f"未找到算子 '{op_name}' 的算子档案"
            f"（references/profiles/{op_name}/profile.json 不存在）。"
            f"可用档案: {_available_profiles()}。"
            f"新算子接入请按 references/operator-guide.md 的分析程序"
            f"并通过全部接入门禁后生成算子档案，"
            f"生成后 load_profile(<op_name>) 即可加载。"
        )

    if op_name not in _PROFILE_CACHE:
        profile_path = os.path.join(_PROFILE_ROOT, op_name, "profile.json")
        try:
            with open(profile_path) as f:
                profile = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"算子 '{op_name}' 的 profile.json 加载失败({e})。"
                f"档案损坏时按 references/operator-guide.md 分析程序重新生成。"
            ) from e
        _validate_profile(profile, op_name)
        _PROFILE_CACHE[op_name] = profile
    profile = _PROFILE_CACHE[op_name]

    if op_name not in _RECOGNITION_CACHE:
        _RECOGNITION_CACHE[op_name] = _import_profile_module(
            op_name, profile["recognition"]["module"], "recognition"
        )
    if op_name not in _CONSTRAINTS_CACHE:
        _CONSTRAINTS_CACHE[op_name] = _import_profile_module(
            op_name, profile["constraints"]["module"], "constraints"
        )

    # ---- 设为当前档案并派生算子档案驱动的缓存 ----
    _ACTIVE_OP_NAME = op_name
    VENDOR_PACKAGES = _derive_vendor_packages(profile)
    MACRO_MAP = _derive_macro_map(profile)
    REQUIRED_CANDIDATES_PER_TEMPLATE = int(
        profile["constraints"]["required_candidates_per_template"]
    )
    _apply_op_paths(op_name, profile)
    return profile


def _require_profile() -> dict:
    """取当前算子档案；未加载时报错并指路 load_profile。"""
    if _ACTIVE_OP_NAME is None:
        raise RuntimeError(
            "引擎尚未加载当前算子档案。所有按算子区分的数据（case 分类/"
            "结构体布局/模板集合/量化模式/硬件常量）都来自档案包：先调用 "
            f"load_profile(op_name)（可用档案: {_available_profiles()}）。"
            "validate_case_csv / extract_all_params 会按 CSV 的 op_name 列自动加载。"
        )
    return _PROFILE_CACHE[_ACTIVE_OP_NAME]


def _recognition_module():
    """活跃档案的识别适配器（derive_facts / validate_case）。"""
    _require_profile()
    return _RECOGNITION_CACHE[_ACTIVE_OP_NAME]


def _constraints_module():
    """活跃档案的约束适配器（check_candidate / validate_rules_against_anchors）。"""
    _require_profile()
    return _CONSTRAINTS_CACHE[_ACTIVE_OP_NAME]


def _constraints_ctx() -> dict:
    """constraints 适配器的 ctx（契约见档案包 constraints.py 模块 docstring）。"""
    profile = _require_profile()
    return {
        "soc": profile["soc"],  # 容量/对齐/分组常量（替代引擎旧硬编码）
        "dtype_sizeof": _dtype_sizes(),  # 标准类型与档案声明的扩展尺寸
        "quant_modes": profile["quant_modes"],
        "case_classes": profile["case_classes"],
    }


def _required_candidates_per_template() -> int:
    """每模板最少候选数（活跃档案 constraints.required_candidates_per_template）。"""
    return int(_require_profile()["constraints"]["required_candidates_per_template"])


def allowed_templates(case_class: str, quant_mode: str = "mx") -> list[str]:
    """
    case_class × quant_mode 二维模板白名单（档案 quant_modes.<mode>.templates_by_case_class）。

    模板名保持语义名（swat/afull/...），各模式的 KTYPE 等价值由
    quant_modes.<mode>.kernel_type_map 声明（如 pertoken: swat→2, afull→3）。
    """
    quant_modes = _require_profile()["quant_modes"]
    mode_cfg = quant_modes.get(quant_mode)
    if mode_cfg is None:
        raise KeyError(
            f"量化模式 '{quant_mode}' 未在档案 quant_modes 声明（已接入: {sorted(quant_modes)}）。"
            f"未接入模式的 case 不在本 skill 扫参范围内。"
        )
    return mode_cfg["templates_by_case_class"].get(case_class, [])


def injectable_templates(
    case_class: str, quant_mode: str = "mx", natural_template: str = ""
) -> list[str]:
    """返回当前 case 可实际改写 TUNE_ 宏的模板。

    templates_by_case_class 描述 host 可达性；kernel_files 中
    insert_mode="none" 表示只能运行、无可验证的注入点，不进入扫参。
    install-stub 不能改变 host 选模板结果；传入 natural_template 时只保留
    baseline 实际命中的模板。
    """
    mode_cfg = _require_profile()["quant_modes"][quant_mode]
    table = mode_cfg.get("kernel_files", {}).get(case_class, {})
    result = []
    for template in allowed_templates(case_class, quant_mode):
        tmpl = normalize_template(template)
        entry = table.get(tmpl)
        if entry is None:
            continue
        if isinstance(entry, dict) and entry.get("insert_mode") == "none":
            continue
        result.append(tmpl)
    if mode_cfg.get("tune_macro_define_block") and natural_template:
        natural = normalize_template(natural_template)
        return [t for t in result if t == natural]
    return result


def _all_templates() -> set:
    """模板名全集（活跃档案全部模式声明模板的并集）。"""
    return {
        normalize_template(t)
        for mode_cfg in _require_profile()["quant_modes"].values()
        for ts in (mode_cfg.get("templates_by_case_class") or {}).values()
        for t in ts
    }


def _vendor_package_for_template(template: str, quant_mode: str) -> str:
    """选择扫参运行与 kernel 安装路径共同使用的 vendor 包。"""
    if _require_profile()["quant_modes"][quant_mode].get("tune_macro_define_block"):
        return VENDOR_PACKAGES["normal"]
    tmpl = normalize_template(template)
    if tmpl in VENDOR_PACKAGES:
        return VENDOR_PACKAGES[tmpl]
    if "swat" in VENDOR_PACKAGES:
        return VENDOR_PACKAGES["swat"]
    raise KeyError(
        f"模板 '{tmpl}' 无 vendor 包声明（VENDOR_PACKAGES: {sorted(VENDOR_PACKAGES)}）"
    )


def _kernel_entry_to_path(entry, package: str) -> str:
    """绝对路径直接使用；包内相对路径从 vendor 包安装目录解析。"""
    path = entry["install"] if isinstance(entry, dict) else entry
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(VENDOR_CONFIG), package, path)


# ============================================================================
# NPU 信息确定性探测：核数等估算必须用实测值
# ============================================================================


def _tbe_site_packages(cann_path: str = "") -> str:
    """使用显式 CANN 路径、当前环境或 Python 已选包，不按目录名猜版本。"""

    def resolve(root):
        root = os.path.realpath(root)
        site = (
            root
            if os.path.isdir(os.path.join(root, "tbe"))
            else os.path.join(root, "python", "site-packages")
        )
        if not os.path.isdir(os.path.join(site, "tbe")):
            raise RuntimeError(
                f"当前 CANN 路径没有 tbe 包: {root}，请检查环境或 cann_path"
            )
        return os.path.realpath(site)

    if cann_path:
        return resolve(cann_path)
    roots = {
        os.path.realpath(os.environ[key])
        for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
        if os.environ.get(key)
    }
    if len(roots) > 1:
        raise RuntimeError(
            f"当前 CANN 环境变量指向不同安装: {sorted(roots)}，请先对齐环境"
        )
    if roots:
        return resolve(next(iter(roots)))
    spec = importlib.util.find_spec("tbe")
    locations = list(spec.submodule_search_locations or []) if spec else []
    sites = {os.path.realpath(os.path.dirname(path)) for path in locations}
    if len(sites) == 1:
        return resolve(next(iter(sites)))
    raise RuntimeError(
        "无法唯一确定当前 CANN：请加载目标版本 set_env.sh 或传 cann_path"
    )


def get_soc_version(result_csv_path: str = "") -> str:
    """
    从 ttk result CSV 的 soc 列读取 SoC 版本（如 Ascend950PR_9589）。

    result CSV 由 Step 1 baseline 生成，Step 2 候选生成前必然存在。
    读不到（文件缺失/列为空）直接抛异常，不提供默认值——
    SoC 版本是后续核数探测的输入，错误假设会导致全链路估算失真。
    """
    path = result_csv_path or _latest_result_csv()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"result CSV 不存在（{path}）。请先执行 Step 1 baseline（生成 *_result.csv）"
            f"再调用 get_soc_version()。"
        )
    with open(path) as f:
        for row in csv.DictReader(f):
            soc = (row.get("soc") or "").strip()
            if soc:
                return soc
    raise ValueError(f"result CSV '{path}' 的 soc 列为空，无法确定 SoC 版本")


def _latest_result_csv() -> str:
    """找最近更新的 *_result.csv（ttk 在当前 case CSV 旁生成）"""
    import glob as _glob

    paths = _glob.glob(os.path.join(CASE_DIR, "*_result.csv"))
    if not paths:
        return ""
    return max(paths, key=os.path.getmtime)


def get_aic_num(soc_version: str, cann_path: str = "") -> int:
    """按档案声明的核类型与规格键查询目标 SoC；不将通用 CORE_NUM 当作固定口径。

    soc.core_query 的 core_type/spec_key/evidence 由 agent 从当前 host 调度与
    平台接口确认。查询的是平台编译规格，不代表实时空闲核数。
    """
    query = _require_profile()["soc"].get("core_query", {})
    if not all(
        isinstance(query.get(key), str) and query[key].strip()
        for key in ("core_type", "spec_key", "evidence")
    ):
        raise RuntimeError(
            "核数查询需要 soc.core_query 的 core_type/spec_key/evidence，不能猜核数口径"
        )
    site = _tbe_site_packages(cann_path)
    if site not in sys.path:
        sys.path.insert(0, site)
    try:
        tbe_platform = importlib.import_module("tbe.common.platform")
    except ImportError as e:
        raise RuntimeError(f"无法导入当前 CANN 的 tbe 平台接口（{site}）: {e}") from e
    loaded_path = os.path.realpath(tbe_platform.__file__)
    if os.path.commonpath([site, loaded_path]) != site:
        raise RuntimeError(
            f"tbe 已从其他 CANN 加载: {loaded_path}；请在目标版本环境的新进程重试"
        )
    tbe_platform.set_current_compile_soc_info(soc_version, query["core_type"])
    core_num = int(tbe_platform.get_soc_spec(query["spec_key"]))
    if core_num <= 0:
        raise RuntimeError(
            f"SoC {soc_version} 的 {query['spec_key']} 返回异常核数 {core_num}"
        )
    print(
        f"[core_probe] CANN={site}, SoC={soc_version}, "
        f"core_type={query['core_type']}, spec_key={query['spec_key']}, count={core_num}"
    )
    return core_num


# 安装目录 kernel .h 文件路径按 模板 x case_class x quant_mode 查活跃档案
# quant_modes（get_kernel_file_path）；可调宏映射 MACRO_MAP 由 load_profile 从
# 档案声明块派生（见 _derive_macro_map）。


# ============================================================================
# 数据结构
# ============================================================================


class TilingData:
    """保留档案实际解出的字段；不补不存在的通用参数。"""

    def __init__(self, **values):
        self._values = dict(values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as e:
            raise AttributeError(f"tiling data 没有字段 {name!r}") from e

    def to_dict(self) -> dict:
        return dict(self._values)


@dataclass
class BaselineResult:
    """baseline 执行结果；tiling_data 字典在构造时转换为 TilingData。"""

    template: str = ""  # op_host 选中的模板: swat/afull/l0cpp/streamk
    perf_us: float = 0.0  # DYN_PERF (微秒)
    tiling_data: TilingData | dict = field(default_factory=TilingData)
    precision_status: str = "UNKNOWN"  # PASS / FAIL / COMPARE_FAILURE / UNKNOWN
    # tiling 解析失败原因（空串=解析成功）。perf 与模板名不受解析失败影响：
    # 模板按 dyn_tiling_key 独立反查（布局解析失败不连坐模板检测），
    # perf 恒保留——tiling 不可解析只降级 tiling_data 本身。
    tiling_parse_error: str = ""

    def __post_init__(self):
        if isinstance(self.tiling_data, dict):
            self.tiling_data = TilingData(**self.tiling_data)


@dataclass
class CandidateResult:
    """候选执行结果；tiling_data 字典在构造时转换为 TilingData。"""

    template: str = ""
    tiling_data: TilingData | dict = field(default_factory=TilingData)
    perf_us: float = 0.0
    status: str = (
        "ok"  # ok / timeout / precision_fail / template_mismatch / template_unverified
    )
    precision: Optional[str] = (
        None  # 原始 PRECISION_STATUS（PASS/FAIL/COMPARE_FAILURE）；
    )
    # None=未测得。timeout 恒 None（精度未完成测量，不冒充 FAIL）
    candidate: dict = field(
        default_factory=dict
    )  # 原始可注入字段，供非 QBMM 算子档案的榜单完整渲染

    def __post_init__(self):
        if isinstance(self.tiling_data, dict):
            self.tiling_data = TilingData(**self.tiling_data)


# ============================================================================
# 1. CSV 校验与参数提取
# ============================================================================

# ttk 新版 CSV 列名（顺序敏感，不含老格式的 network_name，新增 m_feat/n_feat/k_feat）
REQUIRED_CSV_COLUMNS = [
    "testcase_name",
    "op_name",
    "input_shapes",
    "input_ori_shapes",
    "output_shapes",
    "output_ori_shapes",
    "input_dtypes",
    "output_dtypes",
    "input_formats",
    "input_ori_formats",
    "output_formats",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "is_enabled",
]

# 新版 ttk 新增的可选列（m_feat/n_feat/k_feat），存在则使用，不存在不影响
OPTIONAL_CSV_COLUMNS = ["m_feat", "n_feat", "k_feat"]

# dtype 字符串 → sizeof(bytes)
DTYPE_SIZEOF = {
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float4_e2m1": 0.5,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int32": 4,
    "float8_e8m0": 1,
    # 标量 scale 族补全（U64/I64 占 8B——缺省值 1 会让 L1 记账少算 8 倍；int4 半字节）
    "uint64": 8,
    "int64": 8,
    "int8": 1,
    "int4": 0.5,
    "hifloat8": 1,
}


def _dtype_sizes() -> dict:
    return {**DTYPE_SIZEOF, **_require_profile().get("dtype_sizeof", {})}


def _dtype_size(dtype: str) -> float:
    sizes = _dtype_sizes()
    if dtype not in sizes:
        raise ValueError(
            f"未知 dtype {dtype!r} 的字节数；请从当前类型定义确认 dtype_sizeof/dtype_evidence，"
            "不能默认按 1 字节计算。"
        )
    size = sizes[dtype]
    if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
        raise ValueError(f"dtype {dtype!r} 的字节数非法: {size!r}")
    return size


def _parse_output_dtype(raw: str) -> str:
    """解析 TTK 的 output_dtypes；拒绝会被其 field_parser 当成变量的裸字符串。"""
    text = (raw or "").strip()
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"output_dtypes={text!r} 不是合法 Python 字面量；标量 dtype 也必须加引号，"
            "例如 'float16'"
        ) from e
    if isinstance(value, tuple):
        if len(value) != 1:
            raise ValueError(f"output_dtypes tuple 必须只有 1 项，实际为 {value!r}")
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError(f"output_dtypes 必须是 dtype 字符串，实际为 {value!r}")
    return value


def validate_case_csv(csv_path: str) -> bool:
    """
    检查用户提供的 CSV 文件是否符合 ttk 新版格式。

    校验项：
    1. 文件存在且可读
    2. 包含必需的列名
    3. op_name：全部行同一算子，且已接入（按首行 op_name load_profile，
       逐行对照档案 identity.op_name）
    4. is_enabled 为启用状态（"1"、"1.0" 或不区分大小写的 "true"）
    5. 每行能解析出适配器声明的形状、类型、转置和模式事实
    6. 输入槽位、属性和可选输入规则由当前算子的 validate_case 验证

    返回 True 表示格式合格。
    """
    if not os.path.exists(csv_path):
        print(f"[validate] CSV 文件不存在: {csv_path}")
        return False

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[validate] CSV 无列名")
            return False

        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
        if missing:
            print(f"[validate] 缺少列: {missing}")
            return False

        # 老格式有 network_name 列但新格式没有——如果检测到 network_name 提示用户转换
        if "network_name" in reader.fieldnames:
            print("[validate] 检测到老格式列 'network_name'，请转换为新版 ttk CSV 格式")
            return False

        rows = list(reader)

    if not rows:
        print("[validate] CSV 无数据行")
        return False

    # ---- op_name 自动推导：按首行加载档案，全部行对照档案 identity.op_name ----
    try:
        profile = load_profile(rows[0].get("op_name"))
    except RuntimeError as e:
        print(f"[validate] {e}")
        return False
    expected_op = profile["identity"]["op_name"]

    for i, row in enumerate(rows):
        if row.get("op_name") != expected_op:
            print(
                f"[validate] 行 {i + 1}: op_name={row.get('op_name')}，期望 {expected_op}"
            )
            return False

        enabled_val = row.get("is_enabled", "").strip().casefold()
        if enabled_val not in ("1", "1.0", "true"):
            print(f"[validate] 行 {i + 1}: is_enabled={enabled_val}，期望 1 或 True")
            return False

        try:
            params = extract_shape_params(row)
        except Exception as e:
            print(f"[validate] 行 {i + 1}: 解析 shape 失败: {e}")
            return False

        try:
            _parse_output_dtype(row["output_dtypes"])
        except ValueError as e:
            print(f"[validate] 行 {i + 1}: {e}")
            return False

        try:
            ctx = _case_context(row)
            ctx.update(params)
            err = _recognition_module().validate_case(ctx, profile["input_rules"])
        except Exception as e:
            print(f"[validate] 行 {i + 1}: 输入校验失败: {e}")
            return False
        if err:
            print(f"[validate] 行 {i + 1}: {err}")
            return False

    print(f"[validate] {csv_path}: {len(rows)} 行，格式合格")
    return True


def _case_context(row: dict) -> dict:
    """解析 CSV 原始列，不解释输入位置或算子属性语义。"""
    profile = _require_profile()
    return {
        "row": dict(row),
        "input_shapes": ast.literal_eval(row["input_shapes"]),
        "input_dtypes": ast.literal_eval(row["input_dtypes"]),
        "input_formats": ast.literal_eval(row["input_formats"]),
        "attributes": ast.literal_eval(row["attributes"]),
        "y_dtype": _parse_output_dtype(row["output_dtypes"]),
        "fold_rule": profile["fold_rule"],
        "quant_modes": profile["quant_modes"],
        "input_rules": profile["input_rules"],
        "case_classes": profile["case_classes"],
    }


def extract_shape_params(row: dict) -> dict:
    """解析 CSV；所有输入、维度、转置、格式和模式事实由识别适配器推导。

    适配器返回现有执行接口需要的事实，并可添加算子特有事实；引擎不从固定
    输入槽位或固定属性名补值。返回键契约见 references/operator-guide.md。
    """
    row_op = row.get("op_name")
    if row_op and row_op != _ACTIVE_OP_NAME:
        load_profile(row_op)
    ctx = _case_context(row)
    facts = _recognition_module().derive_facts(ctx)
    required = {
        "M",
        "N",
        "K",
        "x1_dtype",
        "x2_dtype",
        "scale_dtype",
        "bias",
        "trans_a",
        "trans_b",
        "x1_format",
        "x2_format",
        "case_class",
        "batch_count",
        "quant_mode",
    }
    if not isinstance(facts, dict):
        raise ValueError("recognition.derive_facts 必须返回事实字典")
    missing = sorted(required - facts.keys())
    if missing:
        raise ValueError(f"recognition.derive_facts 缺少执行接口事实: {missing}")

    for dtype in (facts["x1_dtype"], facts["x2_dtype"], ctx["y_dtype"]):
        _dtype_size(dtype)
    for dtype in (facts["scale_dtype"], facts["bias"]):
        if dtype is not None and dtype != "none":
            _dtype_size(dtype)

    try:
        tol = (
            ast.literal_eval(row["precision_tolerances"])
            if row.get("precision_tolerances")
            else None
        )
        ptol = float(tol[0][1]) if tol and tol[0] and len(tol[0]) > 1 else 0.001
    except (ValueError, IndexError, TypeError):
        ptol = 0.001
    return {
        **facts,
        "testcase_name": row.get("testcase_name", ""),
        "y_dtype": ctx["y_dtype"],
        "ptol": ptol,
    }


def extract_all_params(csv_path: str) -> list[dict]:
    """
    从多行 CSV 提取所有 case 的参数。

    op_name 自动推导：全部行必须同一算子（混合报错），按首行 op_name
    load_profile——solver.md 工作流无需显式 load_profile。

    返回 params dict 列表，每个 dict 含 testcase_name + M/N/K/dtype/transpose/bias 等。
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    op_names = {r.get("op_name") for r in rows}
    if len(op_names) > 1:
        raise ValueError(
            f"CSV 含多个 op_name（{sorted(str(name) for name in op_names)}）：一个 CSV 只能含"
            f"一个算子的 case，请先 split_csv 拆分或分文件组织。"
        )
    load_profile(rows[0].get("op_name"))
    return [extract_shape_params(row) for row in rows]


def split_csv(csv_path: str, output_dir: str = "") -> list[str]:
    """
    将多行 CSV 拆分为单行 CSV 文件。

    返回单行 CSV 文件路径列表，按行顺序排列。
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not output_dir:
        output_dir = os.path.dirname(csv_path)

    paths = []
    for i, row in enumerate(rows, 1):
        case_name = row.get("testcase_name", f"case_{i}")
        out_path = os.path.join(output_dir, f"{case_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        paths.append(out_path)

    print(f"[split_csv] {len(paths)} cases split into {output_dir}")
    return paths


# ============================================================================
# 2. timeout 计算
# ============================================================================


def calc_timeout(m: int, n: int, k: int, batch_count: int = 1) -> tuple[int, int]:
    """
    根据 shape 规模计算 npu-timeout（毫秒）和 proc-timeout（秒）。

    batch_count: with_batch case 的调度 batch 数（M 为 per-batch 口径）；
    without_batch（含折叠）M 已含 batch、batch_count=1。

    返回 (npu_timeout_ms, proc_timeout_seconds)。
    """
    flops = m * n * k * batch_count
    if k > 20000:
        npu_timeout, proc_timeout = 90, 1800
    elif k > 12000 or flops > 1e9:
        npu_timeout, proc_timeout = 60, 1200
    else:
        npu_timeout, proc_timeout = 30, 600
    return npu_timeout * 1000, proc_timeout


def estimate_baseline_eta(
    m: int, n: int, k: int, n_cases: int = 1, batch_count: int = 1
) -> int:
    """估算 baseline 总耗时（秒）"""
    eta_per_case = 60
    if m * n * k * batch_count > 1e9:
        eta_per_case += 30
    if k > 20000:
        eta_per_case += 30
    return eta_per_case * n_cases


def estimate_scan_eta(
    m: int, n: int, k: int, n_candidates: int, batch_count: int = 1
) -> int:
    """估算扫参总耗时（秒）"""
    if m * n * k * batch_count > 1e9 and k > 20000:
        eta_per = 95
    elif m * n * k * batch_count > 1e9:
        eta_per = 35
    else:
        eta_per = 30
    return eta_per * n_candidates


# ============================================================================
# 3. vendor 包切换
# ============================================================================


def switch_vendor(template: str) -> None:
    """切换到指定模板的 vendor 包（包名映射由活跃档案派生，见 _derive_vendor_packages）"""
    _require_profile()
    pkg = VENDOR_PACKAGES[normalize_template(template)]
    with open(VENDOR_CONFIG, "w") as f:
        f.write(f"load_priority={pkg}\n")


def _require_runtime_paths() -> None:
    """在生成长任务前一次性检查部署路径，避免把 None 或失效路径写进脚本。"""
    profile = _require_profile()
    checks = {
        "TTK_DIR": TTK_DIR,
        "OP_BASE": OP_BASE,
        "PLUGIN_DIR": PLUGIN_DIR,
    }
    invalid = [
        f"{name}={path!r}"
        for name, path in checks.items()
        if not path or not os.path.isdir(path)
    ]
    if invalid:
        raise RuntimeError(
            "部署路径未配置或不存在: "
            + ", ".join(invalid)
            + "。请先调用 configure_paths(...)。"
        )
    expected_commit = profile.get("version_anchor", {}).get("op_repo_commit")
    if not expected_commit:
        raise RuntimeError("算子档案字段 version_anchor.op_repo_commit 缺失")
    try:
        actual_commit = subprocess.run(
            ["git", "-C", OP_BASE, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"无法读取算子仓版本: {OP_BASE}: {e.stderr.strip()}") from e
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"算子档案的版本锚点与算子仓不一致: "
            f"档案={expected_commit}, 仓库={actual_commit}。请先对账并重新生成算子档案。"
        )
    os.makedirs(KERNEL_META_DIR, exist_ok=True)
    if not os.path.isdir(os.path.dirname(VENDOR_CONFIG)):
        raise RuntimeError(f"vendor 目录不存在: {os.path.dirname(VENDOR_CONFIG)}")


def validate_vendor_package_gate(
    templates: Optional[list[str]] = None, output_path: str = None, strict: bool = True
) -> dict:
    """校验 normal/forced vendor 中影响算子发现与注册的关键安装产物。

    检查项来自算子档案的 `package_gate.required_files`，避免只看构建返回码或 kernel
    头文件就误判“构建成功”。每项可声明：

    - path: vendor 包根目录内的相对路径，可含 {package}/{op_name} 占位符；
    - package_scope: all / normal / forced（缺省 all）；
    - syntax: python 时用 ast.parse 做无副作用语法检查；
    - contains: 必须出现的字符串列表，如 REG_OP 或算子注册名；
    - contains_match: 缺省 whitespace，合并换行/缩进并去除续行；literal 为逐字符匹配。

    templates 为空时校验算子档案中的全部强制模板包；有值时只校验对应模板。
    output_path 非空时先写临时文件并原子替换 report，再按 strict 决定是否抛错，
    因此失败证据也不会丢。返回的 report 可直接作为 build gate 证据。
    """
    _require_runtime_paths()
    profile = _require_profile()
    requested = (
        [normalize_template(t) for t in templates]
        if templates is not None
        else sorted(_all_templates())
    )
    roles = {"normal": VENDOR_PACKAGES["normal"]}
    for tmpl in requested:
        if tmpl in VENDOR_PACKAGES:
            roles[tmpl] = VENDOR_PACKAGES[tmpl]
    unsafe = sorted(
        package
        for package in roles.values()
        if not package or os.path.basename(package) != package
    )
    if unsafe:
        raise RuntimeError(f"算子档案中的 vendor_install_dir 不是安全目录名: {unsafe}")

    vendors_root = os.path.dirname(VENDOR_CONFIG)
    checks = []
    errors = []
    for role, package in roles.items():
        is_normal = role == "normal"
        for rule in profile["package_gate"]["required_files"]:
            scope = rule.get("package_scope", "all")
            if scope == "normal" and not is_normal:
                continue
            if scope == "forced" and is_normal:
                continue
            relative = rule["path"].format(package=package, op_name=_ACTIVE_OP_NAME)
            path = os.path.join(vendors_root, package, relative)
            rec = {"role": role, "package": package, "path": path, "ok": True}
            if not os.path.isfile(path):
                rec["ok"] = False
                rec["error"] = "missing"
            else:
                try:
                    with open(path, errors="replace") as f:
                        content = f.read()
                    if rule.get("syntax") == "python":
                        ast.parse(content, filename=path)
                    if rule.get("contains_match", "whitespace") == "literal":
                        contains = rule.get("contains", [])
                        missing_text = [s for s in contains if s not in content]
                    else:

                        def normalize(text):
                            return " ".join(re.sub(r"\\\r?\n", "", text).split())

                        normalized = normalize(content)
                        missing_text = [
                            s
                            for s in rule.get("contains", [])
                            if normalize(s) not in normalized
                        ]
                    if missing_text:
                        rec["ok"] = False
                        rec["error"] = f"missing content: {missing_text}"
                except (OSError, SyntaxError) as e:
                    rec["ok"] = False
                    rec["error"] = str(e)
            checks.append(rec)
            if not rec["ok"]:
                errors.append(f"{role}: {rule['path']}: {rec['error']}")

    report = {
        "op_name": _ACTIVE_OP_NAME,
        "packages": roles,
        "checks": checks,
        "ok": not errors,
    }
    if output_path:
        _ensure_parent(output_path)
        tmp_path = f"{output_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    if errors and strict:
        raise RuntimeError("vendor package gate 失败:\n- " + "\n- ".join(errors))
    return report


# ============================================================================
# 4. kernel 文件宏更新
# ============================================================================


def get_kernel_file_path(
    template: str, case_class: str = "without_batch", quant_mode: str = "mx"
) -> str:
    """
    获取安装目录 kernel .h 文件路径（按 模板 x case_class x quant_mode，来自
    活跃档案 quant_modes）。pertoken 打 qbmm_mix_* 文件（BLAZE MIX 变体），
    mx 打 qbmm_mx_*——swat/afull 复用同一 vendor 包，运行时 host 按模式路由到
    对应 tiling 类的强制值。
    """
    tmpl = normalize_template(template)
    quant_modes = _require_profile()["quant_modes"]
    mode_cfg = quant_modes.get(quant_mode)
    if mode_cfg is None:
        raise KeyError(
            f"量化模式 '{quant_mode}' 未在档案 quant_modes 声明（已接入: {sorted(quant_modes)}）。"
        )
    table = mode_cfg["kernel_files"].get(case_class)
    entry = table.get(tmpl) if table else None
    if entry is None:
        avail = allowed_templates(case_class, quant_mode)
        raise KeyError(
            f"模板 '{tmpl}' 在 case_class '{case_class}' × quant_mode '{quant_mode}' 下不可用"
            f"（档案声明可用: {avail}）。"
        )
    if isinstance(entry, dict) and entry.get("insert_mode") == "none":
        raise ValueError(
            f"模板 '{tmpl}' 在 case_class '{case_class}' × quant_mode '{quant_mode}' 下"
            "没有可验证的 TUNE_ 宏注入点，不可扫参。"
        )
    return _kernel_entry_to_path(entry, _vendor_package_for_template(tmpl, quant_mode))


def update_kernel_macros(
    template: str,
    tiling_data: TilingData | dict,
    case_class: str = "without_batch",
    quant_mode: str = "mx",
) -> None:
    """
    更新安装目录 kernel .h 文件顶部的 #define TUNE_* 宏值。

    直接修改安装目录副本（不是 worktree 源码），ttk -d JIT 编译会读取新值。
    死字段（档案 quant_modes.<mode>.dead_fields，如 pertoken 的 scaleKL1——host
    不写/kernel 不读）不注入 sed，宏保持文件内默认值；候选携带死字段值时警告
    并忽略（每字段每模式仅警告一次，避免刷屏）。
    """
    kernel_file = get_kernel_file_path(template, case_class, quant_mode)
    td = tiling_data.to_dict() if isinstance(tiling_data, TilingData) else tiling_data
    td = dict(td)  # 剔除死字段时不改调用方对象
    _validate_candidate_fields(td, "update_kernel_macros")

    dead_fields = set(
        _require_profile()["quant_modes"].get(quant_mode, {}).get("dead_fields", {})
        or {}
    )
    for df in dead_fields:
        if td.get(df) is not None:
            _warn_dead_field_once(quant_mode, df, td[df])
        td[df] = None  # 从 sed 集合剔除（None 不进 MACRO_MAP 循环）

    required_fields = _kernel_macro_fields(kernel_file) - dead_fields
    provided_fields = {k for k in MACRO_MAP if td.get(k) is not None}
    if required_fields != provided_fields:
        raise ValueError(
            f"候选与目标 kernel 宏集不一致："
            f"缺少={sorted(required_fields - provided_fields)}，"
            f"不属于该模板={sorted(provided_fields - required_fields)}"
        )

    with open(kernel_file) as f:
        content = f.read()

    changed = []
    for key, macro in MACRO_MAP.items():
        if key in td and td[key] is not None:
            value = td[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"候选字段 {key} 必须是整数，实际为 {value!r}")
            pattern = rf"(?m)^\s*#define\s+{re.escape(macro)}\s+[-+]?\d+\s*$"
            replacement = f"#define {macro} {td[key]}"
            content, count = re.subn(pattern, replacement, content)
            if count != 1:
                raise RuntimeError(
                    f"{kernel_file}: 宏 {macro} 命中 {count} 次（应为 1）；"
                    "安装侧文件与算子档案的注入声明不一致"
                )
            changed.append((macro, value))

    if not changed:
        raise ValueError("候选没有任何可注入字段")

    with open(kernel_file, "w") as f:
        f.write(content)

    with open(kernel_file) as f:
        written = f.read()
    for macro, value in changed:
        if not re.search(rf"(?m)^#define\s+{re.escape(macro)}\s+{value}\s*$", written):
            raise RuntimeError(f"{kernel_file}: 宏 {macro}={value} 写入后校验失败")


_DEAD_FIELD_WARNED: set = set()


def _warn_dead_field_once(quant_mode: str, field: str, value) -> None:
    key = (quant_mode, field)
    if key in _DEAD_FIELD_WARNED:
        return
    _DEAD_FIELD_WARNED.add(key)
    print(
        f"[warn] 字段 '{field}' 在 quant_mode '{quant_mode}' 下是死字段（host 不写/kernel 不读），"
        f"候选值 {value} 已忽略——生成候选时请勿携带该字段"
    )


def clear_kernel_meta() -> None:
    """清理 kernel_meta 缓存（每次跑 ttk 前必须执行）"""
    _require_runtime_paths()
    for root, names in (
        (
            KERNEL_META_DIR,
            os.listdir(KERNEL_META_DIR) if os.path.isdir(KERNEL_META_DIR) else [],
        ),
    ):
        for name in names:
            path = os.path.join(root, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    for name in os.listdir(TTK_DIR):
        if name.startswith("dyn_op_") or name.endswith(".ttk"):
            path = os.path.join(TTK_DIR, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


# ============================================================================
# 5. ttk 命令执行
# ============================================================================


def _clean_ttk_cache() -> None:
    """
    清理 TTK_DIR 下的 dyn_op 缓存与 ttk 产物（目录与文件通配删除）。
    """
    for pattern in ("kernel_meta/dyn_op_*", "*.ttk"):
        for path in glob.glob(os.path.join(TTK_DIR, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)


def _cache_cleanup_command() -> str:
    """
    生成后台脚本内嵌的缓存清理命令（python glob 删除，不依赖 shell 递归强删）。
    """
    patterns = ", ".join(
        json.dumps(os.path.join(base, tail))
        for base, tail in (
            (KERNEL_META_DIR, "*"),
            (TTK_DIR, "dyn_op_*"),
            (TTK_DIR, "*.ttk"),
        )
        if base
    )
    return (
        "python3 -c 'import glob, os, shutil\n"
        f"for pat in ({patterns}):\n"
        "    for p in glob.glob(pat):\n"
        "        try:\n"
        "            if os.path.islink(p) or os.path.isfile(p):\n"
        "                os.remove(p)\n"
        "            else:\n"
        "                shutil.rmtree(p)\n"
        "        except OSError:\n"
        "            pass'"
    )


def build_ttk_command(
    case_csv: str,
    npu_timeout: int,
    proc_timeout: int,
) -> list[str]:
    """
    构造 ttk 标准命令参数列表（--compare=close 开启精度校验，golden 由 --plugin 提供）。

    外层 timeout 命令负责进程级超时（秒）；npu_timeout 单位为毫秒。
    工作目录由调用方以 cwd=TTK_DIR 指定。
    """
    _require_runtime_paths()
    return [
        "timeout",
        str(proc_timeout),
        "python3",
        "-m",
        "ttk",
        "kernel",
        "-i",
        case_csv,
        "-d",
        "--compare=close",
        "--pc=1",
        "--run=20",
        "--npu-timeout",
        str(npu_timeout),
        "--plugin",
        PLUGIN_DIR,
    ]


def parse_ttk_output(output: str) -> Optional[float]:
    """
    从 ttk 终端输出解析 DYN_PERF（微秒）。

    返回 None 表示未找到或执行失败。
    """
    for line in output.split("\n"):
        if "DYN_PERF:" in line:
            parts = line.split("DYN_PERF:")
            if len(parts) > 1:
                val = parts[1].split()[0]
                try:
                    return float(val)
                except ValueError:
                    pass
    return None


def parse_precision_status(output: str) -> Optional[str]:
    """
    从 ttk 终端输出解析 PRECISION_STATUS（PASS / FAIL / COMPARE_FAILURE 等）。

    返回 None 表示输出中没有该行（执行异常中断，精度无法判定）。
    """
    for line in output.split("\n"):
        if "PRECISION_STATUS:" in line:
            parts = line.split("PRECISION_STATUS:")
            if len(parts) > 1:
                return parts[1].split()[0]
    return None


def parse_observed_template(output: str, context: Optional[dict] = None) -> str:
    """从 TTK 日志的 Tiling Key 反查实际 kernel 模板。

    返回 unknown 表示日志缺少可解码 key，不允许据此产生性能推荐。
    """
    match = re.search(r"(?m)^Tiling Key:\s+\d+\s+\((0x[0-9a-fA-F]+)\)", output or "")
    return (
        _detect_template_from_key_str(match.group(1), context) if match else "unknown"
    )


def run_ttk_single(
    case_csv: str,
    m: int,
    n: int,
    k: int,
    batch_count: int = 1,
) -> Optional[float]:
    """
    执行单次 ttk 命令，返回 DYN_PERF（微秒）。

    失败或超时返回 None。
    """
    npu_timeout, proc_timeout = calc_timeout(m, n, k, batch_count)
    _require_runtime_paths()
    _clean_ttk_cache()
    cmd = build_ttk_command(case_csv, npu_timeout, proc_timeout)
    # subprocess timeout = proc_timeout + 30s margin
    try:
        result = subprocess.run(
            cmd,
            cwd=TTK_DIR,
            capture_output=True,
            text=True,
            timeout=proc_timeout + 30,
        )
        return parse_ttk_output(result.stdout)
    except subprocess.TimeoutExpired:
        return None


# ============================================================================
# 6. tiling data 提取
# ============================================================================

# ttk 写入 result CSV 的 dyn_tiling_data 有两种格式（不同版本 ttk）：
#   纯字面量:  (1, 8, 0, ...)
#   numpy repr: (np.int32(1), np.int32(8), ...)   ← numpy 2.x 环境下出现
# 用 ast.literal_eval（非 eval，不执行任意代码），numpy 前缀先做文本规范化剥掉。
_NUMPY_SCALAR_RE = re.compile(
    r"np\.(?:u?int(?:8|16|32|64)|float(?:16|32|64)|bool)\(([^()]*)\)"
)


def _parse_tiling_data(td_str: str) -> list:
    """
    解析 dyn_tiling_data 字段为 int 列表。

    - np.int32(1) 等 numpy 标量包装 → 文本层面剥壳为 1
    - ast.literal_eval 解析（不依赖 numpy、不执行任意代码）
    - 所有元素统一转 Python int，下游 json.dump / 算术运算天然安全
    """
    normalized = _NUMPY_SCALAR_RE.sub(r"\1", td_str.strip())
    values = ast.literal_eval(normalized)
    return [int(v) for v in values]


# ---- 布局驱动解析（活跃档案 struct_layouts，三类 case 共用）----


def _layout_fields(layout_name: str) -> dict:
    """展开布局声明（含 extends 继承合并）→ {字段名: (字节偏移, fmt)}。"""
    spec = _require_profile()["struct_layouts"][layout_name]
    fields = {}
    parent = spec.get("extends")
    if parent:
        fields.update(_layout_fields(parent))
    for name, meta in spec.get("fields", {}).items():
        fields[name] = (meta["offset"], meta["fmt"])
    for name, meta in spec.get("extra_fields", {}).items():
        fields[name] = (meta["offset"], meta["fmt"])
    return fields


def _tiling_context(key_str: str, context: Optional[dict] = None) -> dict:
    ctx = dict(context or {})
    ctx["key_str"] = key_str
    ctx["template"] = _detect_template_from_key_str(key_str, ctx)
    try:
        ctx["kernel_type"] = _decode_kernel_type(key_str)
    except (KeyError, TypeError, ValueError):
        ctx["kernel_type"] = None
    return ctx


def _result_context(row: dict, context: Optional[dict] = None) -> dict:
    if context is not None:
        return {**row, **context}
    columns = {
        "input_shapes",
        "input_dtypes",
        "input_formats",
        "attributes",
        "output_dtypes",
    }
    return {**row, **extract_shape_params(row)} if columns <= row.keys() else dict(row)


def _select_layout(
    n: int, context: Optional[dict] = None, include_documented: bool = False
) -> str:
    """按槽位数及 agent 从源码声明的上下文条件选择唯一布局。"""
    ctx = context or {}
    matches = []
    for name, spec in _require_profile()["struct_layouts"].items():
        if spec["u32_slots"] != n or (name.startswith("_") and not include_documented):
            continue
        if all(
            key in ctx and ctx[key] in (value if isinstance(value, list) else [value])
            for key, value in spec.get("when", {}).items()
        ):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"tiling 布局无法唯一确定: {n} 槽位，匹配={matches}。"
            "请根据当前结构体与分发源码补齐 struct_layouts.*.when 或调用上下文。"
        )
    return matches[0]


def _parse_struct_fields(values: list, layout_name: str) -> dict:
    """u32 数组 → 按布局声明的字节偏移+fmt 解出字段字典。"""
    raw = struct.pack(f"<{len(values)}I", *values)
    return {
        name: struct.unpack_from(f"<{fmt}", raw, off)[0]
        for name, (off, fmt) in _layout_fields(layout_name).items()
    }


def _td_from_values(
    values: list, key_str: str = "", context: Optional[dict] = None
) -> tuple[TilingData, str]:
    """u32 值列表与 key/用例上下文 → (TilingData, template_name)。"""
    layout_name = _select_layout(len(values), _tiling_context(key_str, context))
    f = _parse_struct_fields(values, layout_name)
    td = TilingData(**f)
    template = _detect_template_from_key_str(key_str, context)
    return td, template


def extract_baseline_tiling(
    result_csv_path: str, testcase_name: str = "", context: Optional[dict] = None
) -> tuple[TilingData, str]:
    """
    从 result CSV 提取指定 case 的 baseline tiling data（布局档案驱动）。

    如果指定 testcase_name，从多行 result CSV 中匹配对应行；否则取第一行。
    返回 (TilingData, template_name)。
    """
    with open(result_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if testcase_name and row.get("testcase_name", "") != testcase_name:
                continue
            td = _parse_tiling_data(row["dyn_tiling_data"])
            key_str = row.get("dyn_tiling_key", "")
            break
        else:
            raise ValueError(
                f"testcase_name '{testcase_name}' not found in {result_csv_path}"
            )

    return _td_from_values(td, key_str, _result_context(row, context))


def _decode_kernel_type(key_str: str) -> int:
    """按算子档案声明的位段从 TTK tiling key 文本中提取 kernel type。"""
    match = re.search(r"0x([0-9a-fA-F]+)", key_str or "")
    if match is None:
        raise ValueError(f"无法解析 tiling key: {key_str!r}")
    decl = _require_profile()["tiling_key"]
    shift = decl["kernel_type_shift"]
    bits = decl["kernel_type_bits"]
    return (int(match.group(1), 16) >> shift) & ((1 << bits) - 1)


def _detect_template_from_key_str(key_str: str, context: Optional[dict] = None) -> str:
    """按源码声明的 key 位段、模式及用例类别反查；缺证据或有歧义返回 unknown。"""
    profile = _require_profile()
    try:
        kt = _decode_kernel_type(key_str)
    except (KeyError, TypeError, ValueError):
        return "unknown"
    ctx = context or {}
    modes = profile["quant_modes"]
    if ctx.get("quant_mode") is not None:
        modes = {ctx["quant_mode"]: modes.get(ctx["quant_mode"], {})}
    matches = set()
    case_class = ctx.get("case_class")
    for mode_cfg in modes.values():
        table = mode_cfg.get("kernel_type_map", {})
        by_case = mode_cfg.get("kernel_type_map_by_case_class", {})
        tables = (
            [by_case.get(case_class, table)]
            if case_class
            else [table, *by_case.values()]
        )
        allowed = mode_cfg.get("templates_by_case_class", {}).get(case_class)
        for mapping in tables:
            for template, kernel_type in mapping.items():
                tmpl = normalize_template(template)
                if int(kernel_type) == kt and (
                    allowed is None or tmpl in map(normalize_template, allowed)
                ):
                    matches.add(tmpl)
    return next(iter(matches)) if len(matches) == 1 else "unknown"


def extract_anchor(
    scan_log_path: str,
    templates: Optional[list[str]] = None,
    context: Optional[dict] = None,
) -> dict:
    """
    从扫参日志（或任意含 ttk 输出的日志）提取各 forced 包的锚点 tiling（Step 1.5）。

    锚点 = forced 包 host 为该 shape 自算的 tiling（TILING TUNING BLOCK 覆盖前的
    dyn_tiling_data 值）+ blockDim。候选参数与锚点的偏离分析是约束体系的输入。

    按当前档案布局解析 SCAN_START 块的全部字段，再补日志中的 blockDim 和
    布局名。每个模板取第一个探测块；未知布局或路由不一致不产生锚点。
    """
    if templates is None:
        templates = sorted(
            _all_templates()
        )  # 档案模板并集（含非 MX 的 mix/mix_afull/perblock）
    templates = list(templates)

    with open(scan_log_path, errors="replace") as f:
        full_log = f.read()

    anchors: dict[str, dict] = {}
    for tmpl in templates:
        m = re.search(
            rf"===== SCAN_START {tmpl} 0 =====(.*?)===== SCAN_END {tmpl} 0 =====",
            full_log,
            re.S,
        )
        if m is None and len(templates) == 1:
            # 单模板调用时兼容无 SCAN 标记的探测日志。多模板时绝不能把同一个
            # baseline 块复制成多个模板锚点，否则会制造看似完整的错误 anchors。
            m = re.search(
                r"(Tiling Data Parsed Dict:.*?)(?=Tiling Data Parsed Dict|$)",
                full_log,
                re.S,
            )
        if m is None:
            print(f"[anchor] {tmpl}: 未找到带模板标记的锚点块，跳过")
            continue
        block = m.group(1)

        observed = parse_observed_template(block, context)
        if observed != normalize_template(tmpl):
            print(f"[anchor] {tmpl}: 实际模板为 {observed}，不记录该锚点")
            continue

        pm = re.search(r"Tiling Data Parsed Dict: (.*)", block)
        if pm is None:
            continue
        # np.int32(x) 包装优先（numpy 2.x 环境）；剥壳失败再退回裸数字
        wrapped = re.findall(r"np\.int(?:8|16|32|64)\((-?\d+)\)", pm.group(1))
        vals = [int(x) for x in wrapped]
        if not vals:
            vals = [int(x) for x in re.findall(r"-?\d+", pm.group(1))]

        bd_m = re.search(r"^BlockDim: (\d+)", block, re.M)
        block_dim = int(bd_m.group(1)) if bd_m else None

        anchor: dict = {"blockDim": block_dim}
        try:
            key_match = re.search(
                r"(?m)^Tiling Key:\s+\d+\s+\((0x[0-9a-fA-F]+)\)", block
            )
            key_str = key_match.group(1) if key_match else ""
            layout_name = _select_layout(len(vals), _tiling_context(key_str, context))
        except ValueError as e:
            print(f"[anchor] {tmpl}: {e}，跳过")
            continue
        f = _parse_struct_fields(vals, layout_name)
        anchor.update(f)
        anchor["blockDim"] = block_dim
        anchor["layout"] = layout_name
        anchors[tmpl] = anchor

    return anchors


def _ensure_parent(path: str) -> None:
    """落盘前确保父目录存在（RUN_DIR 按类别分子目录时防 FileNotFoundError；裸文件名安全跳过）。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_anchors(anchors: dict, output_path: str) -> str:
    """锚点 JSON 落盘（Step 1.5 产物，Step 2/3 的输入）"""
    import json

    _ensure_parent(output_path)
    with open(output_path, "w") as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)
    print(f"[anchor] {len(anchors)} 个模板锚点已输出到 {output_path}")
    return output_path


# ============================================================================
# 7. baseline 完整执行流程
# ============================================================================


def generate_baseline_script(
    case_csv: str,
    m: int,
    n: int,
    k: int,
    output_log: str = None,
    progress_log: str = None,
    done_flag: str = None,
    log_tag: str = "",
    batch_count: int = 1,
    all_params: Optional[list[dict]] = None,
) -> str:
    """
    生成 baseline 执行的 bash 脚本。支持单行或多行 CSV。

    单用例由 M/N/K 计算 timeout；多用例传 all_params，NPU 超时取最大值，
    整个 TTK 进程的超时为各用例进程预算之和。
    batch_count: with_batch case 的调度 batch 数（M 为 per-batch 口径）。
    路径解析：显式传入的 output_log/progress_log/done_flag 优先；
    缺省时用 baseline_paths(log_tag) 派生（log_tag 为空即无 tag 的默认路径）。
    """
    _require_runtime_paths()
    if all_params is None:
        npu_timeout, proc_timeout = calc_timeout(m, n, k, batch_count)
    else:
        limits = [
            calc_timeout(p["M"], p["N"], p["K"], p.get("batch_count", 1))
            for p in all_params
        ]
        npu_timeout = max(npu for npu, _ in limits)
        proc_timeout = sum(proc for _, proc in limits)
    paths = baseline_paths(log_tag)
    script_path = paths["script"]
    output_log = output_log or paths["output"]
    progress_log = progress_log or paths["progress"]
    done_flag = done_flag or paths["done"]

    q = shlex.quote
    script = f"""#!/bin/bash
# Matmul tiling baseline runner — auto-generated by ttk_runner.py
set -e

# 切换到 normal 包
printf '%s' {q("load_priority=" + VENDOR_PACKAGES["normal"])} > {q(VENDOR_CONFIG)}

# 清缓存
{_cache_cleanup_command()}

# 跑 ttk baseline
cd {q(TTK_DIR)}
timeout {proc_timeout} python3 -m ttk kernel -i {q(case_csv)} -d \
  --compare=close --pc=1 --run=20 --npu-timeout {npu_timeout} \
  --plugin {q(PLUGIN_DIR)}

echo "DONE baseline" >> {q(progress_log)}
touch {q(done_flag)}
"""
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(
        script_path,
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
    )
    return script_path


def _runner_script_alive(script_path: str) -> bool:
    """
    精确检测执行脚本进程是否存活。

    扫描 /proc/*/cmdline，要求 argv[0] 以 bash 结尾且 argv[1] 精确等于
    script_path（即 "bash /tmp/run_scan.sh" 进程本体）。

    不用 pgrep -f 的原因：-f 对完整命令行做子串正则匹配，调用方自己的
    包装进程（如 bash -c 'python3 -c "...run_scan.sh..."'）命令行里只要
    含有该字符串就会误匹配，造成"脚本明明死了却报存活"的假阴性。
    精确 argv 匹配天然排除这类自匹配。
    """
    try:
        pids = os.listdir("/proc")
    except OSError:
        return True  # /proc 不可读：无法判定，保守视为存活
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if (
            len(argv) >= 2
            and argv[0].endswith(b"bash")
            and argv[1].decode(errors="replace") == script_path
        ):
            return True
    return False


def check_task_done(done_flag: str = "/tmp/baseline_done.flag") -> bool:
    """
    检查后台任务是否完成（done.flag 是否存在）。

    兜底：flag 不存在时，额外检测执行脚本进程是否存活。脚本异常死亡
    （kill -9 / OOM 等不可捕获信号）后 flag 永远不会出现，纯 flag 轮询会
    无限空转——此处检测到"脚本已死且无 flag"时返回 True，让调用方进入
    collect 阶段，从残日志中回收已完成部分（候选级 SCAN_START/END 标记
    的数据不丢，未跑到的候选按 timeout 处理）。

    返回 True 的语义：任务可以收尾了（正常完成 或 异常中断可回收）。

    AI 每隔预估时长的 1/4 调用此函数轮询。
    """
    if os.path.exists(done_flag):
        return True

    # flag 不在 → 检查对应执行脚本进程是否存活（正常场景：正在跑）。
    # 脚本路径候选有两处，任一存活即视为在跑（保守方向——宁可多等不可误判死）：
    #   1) 从 done_flag 名字推导：/tmp/scan_done[_tag].flag ↔ /tmp/run_scan[_tag].sh
    #      （/tmp/baseline_done[_tag].flag ↔ /tmp/run_baseline[_tag].sh，无 tag 即默认路径）
    #   2) 从指针文件读最近 tag 推导（调用方轮询默认 flag 而 start 已自动 tag 的场景）
    name = os.path.basename(done_flag)
    if name.endswith(".flag"):
        name = name[: -len(".flag")]
    if "baseline" in name:
        scripts = [
            f"/tmp/run_baseline{name[len('baseline_done') :]}.sh",
            _script_from_tag_pointer(BASELINE_LAST_TAG_FILE, "baseline"),
        ]
    elif "anchor" in name:
        scripts = [
            f"/tmp/run_anchor{name[len('anchor_done') :]}.sh",
            _script_from_tag_pointer(ANCHOR_LAST_TAG_FILE, "anchor"),
        ]
    else:
        scripts = [
            f"/tmp/run_scan{name[len('scan_done') :]}.sh",
            _script_from_tag_pointer(SCAN_LAST_TAG_FILE, "scan"),
        ]
    alive = any(_runner_script_alive(s) for s in scripts if s)
    if not alive:
        print(
            f"[check_task_done] {scripts} 进程均已终止但 {done_flag} 未产生——"
            f"任务异常中断（可能被 kill -9/OOM），进入收集阶段回收已完成部分"
        )
    return not alive


def start_baseline(
    case_csv: str,
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    batch_count: int = 1,
    all_params: Optional[list[dict]] = None,
) -> int:
    """
    生成 baseline 脚本并在后台启动。

    batch_count: with_batch case 传 batchC（M 为 per-batch 口径）；without_batch
    （含折叠）M 已含 batch、保持默认 1。
    all_params: 多行 CSV 的各用例事实，用于计算整批超时与预计时长。

    返回预估耗时（秒），AI 据此决定轮询间隔（建议预估时长的 1/4）。
    AI 用 check_task_done("/tmp/baseline_done.flag") 轮询
    （自动 tag 时轮询 /tmp/baseline_done_{tag}.flag，见启动日志提示），
    完成后用 collect_baseline(case_csv) 收集结果。
    log_tag 为空时自动从 case_csv 文件名派生——不同 CSV 的 baseline 日志
    互不覆盖，并写 /tmp/baseline_last_tag 指针供 collect/check 兜底定位。
    """
    log_tag = log_tag or _tag_from_case_csv(case_csv)
    paths = baseline_paths(log_tag)
    progress_log = paths["progress"]
    done_flag = paths["done"]
    output_log = paths["output"]

    for f in [progress_log, done_flag, output_log, "/tmp/baseline_done.flag"]:
        # 末项：清理无 tag 的旧 done flag，防其残留导致误判已完成
        if os.path.exists(f):
            os.remove(f)
    with open(BASELINE_LAST_TAG_FILE, "w") as f:
        f.write(log_tag)

    script_path = generate_baseline_script(
        case_csv,
        m,
        n,
        k,
        output_log,
        progress_log,
        done_flag,
        log_tag=log_tag,
        batch_count=batch_count,
        all_params=all_params,
    )
    with open(output_log, "w") as log_fd:
        subprocess.Popen(
            ["bash", script_path],
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    eta = (
        estimate_baseline_eta(m, n, k, batch_count=batch_count)
        if all_params is None
        else sum(
            estimate_baseline_eta(
                p["M"], p["N"], p["K"], batch_count=p.get("batch_count", 1)
            )
            for p in all_params
        )
    )
    print(
        f"[baseline] 已后台启动，预计 {eta}s，建议每 {max(eta // 4, 30)}s 检查一次 done flag: {done_flag}"
    )
    return eta


def collect_baseline(case_csv: str, log_tag: str = "") -> BaselineResult:
    """
    baseline 完成后收集结果：解析 DYN_PERF + 提取 tiling data。

    AI 在 check_task_done() 返回 True 后调用此函数。
    log_tag 应与 start_baseline 一致；忘传时从 case_csv 文件名派生 >
    /tmp/baseline_last_tag 指针兜底定位输出日志。
    """
    log_tag = _resolve_log_tag(log_tag, case_csv, BASELINE_LAST_TAG_FILE)
    output_log = baseline_paths(log_tag)["output"]
    perf = None
    precision = None
    if os.path.exists(output_log):
        with open(output_log) as f:
            output = f.read()
        perf = parse_ttk_output(output)
        precision = parse_precision_status(output)

    result_csv = case_csv.replace(".csv", "_result.csv")
    tiling_data = TilingData()
    template = "unknown"
    tiling_parse_error = ""
    if os.path.exists(result_csv):
        try:
            tiling_data, template = extract_baseline_tiling(result_csv)
        except Exception as e:
            # 与 collect_baseline_batch 同策略：解析失败只降级 tiling_data，perf 保留；
            # 模板按 CSV 首行 key 独立反查（不被布局解析失败连坐）。
            # 捕获面从 (ValueError,KeyError,IndexError) 放宽到 Exception——
            # struct.error 等布局解包异常原先会直接炸出本函数。
            print(f"[baseline] tiling data 提取失败: {e}")
            tiling_parse_error = f"{type(e).__name__}: {e}"
            try:
                with open(result_csv) as f:
                    row = next(csv.DictReader(f), {})
                template = _detect_template_from_key_str(
                    row.get("dyn_tiling_key", ""), _result_context(row)
                )
            except Exception:
                template = "unknown"
    else:
        print(f"[baseline] result CSV 不存在: {result_csv}")
        tiling_parse_error = f"result CSV 不存在: {result_csv}"

    return BaselineResult(
        template=template,
        perf_us=perf or 0.0,
        tiling_data=tiling_data,
        precision_status=precision or "UNKNOWN",
        tiling_parse_error=tiling_parse_error,
    )


def run_baseline(
    case_csv: str,
    m: int,
    n: int,
    k: int,
    batch_count: int = 1,
) -> BaselineResult:
    """
    baseline 一站式执行：启动 → 阻塞等待 → 收集结果。

    适用于 AI 可以长时间阻塞的场景。如果需要 AI 自己控制轮询节奏，
    用 start_baseline() + check_task_done() + collect_baseline() 三步。
    """
    eta = start_baseline(case_csv, m, n, k, batch_count=batch_count)
    done_flag = baseline_paths(_tag_from_case_csv(case_csv))["done"]
    time.sleep(eta)

    for retry in range(3):
        if check_task_done(done_flag):
            break
        time.sleep(max(eta // 4, 30))
    else:
        print("[baseline] 超过最大重试次数，尝试解析已有输出")

    return collect_baseline(case_csv)


def start_baseline_batch(
    csv_path: str,
    all_params: list[dict],
) -> int:
    """
    批量 baseline：一次跑整个 CSV 的所有 case。

    all_params 是每个 case 的 extract_shape_params 结果列表。
    NPU timeout 取各用例预算最大值，proc timeout 和 ETA 按各用例预算求和。
    """
    if not all_params:
        raise ValueError("批量 baseline 的 all_params 不能为空")
    first = all_params[0]
    eta = start_baseline(
        csv_path,
        first["M"],
        first["N"],
        first["K"],
        batch_count=first.get("batch_count", 1),
        all_params=all_params,
    )
    print(f"[baseline_batch] {len(all_params)} cases, eta={eta}s")
    return eta


def collect_baseline_batch(
    csv_path: str,
    all_params: list[dict],
) -> list[BaselineResult]:
    """
    批量 baseline 完成后收集所有 case 的结果。

    从 result CSV 逐行提取每个 case 的 tiling data。
    """
    result_csv = csv_path.replace(".csv", "_result.csv")
    results = []

    if not os.path.exists(result_csv):
        print(f"[baseline_batch] result CSV 不存在: {result_csv}")
        return [BaselineResult() for _ in all_params]

    with open(result_csv) as f:
        reader = csv.DictReader(f)
        result_rows = {row.get("testcase_name", ""): row for row in reader}

    for i, params in enumerate(all_params):
        case_name = params.get("testcase_name", "")
        row = result_rows.get(case_name)
        if row is None:
            print(
                f"[baseline_batch] case {i + 1} '{case_name}' not found in result CSV"
            )
            results.append(BaselineResult())
            continue

        td_str = row.get("dyn_tiling_data", "")
        perf_str = row.get("dyn_perf_us", "")
        key_str = row.get("dyn_tiling_key", "")
        prec_str = (row.get("precision_status") or "").strip()

        try:
            perf = (
                float(perf_str)
                if perf_str and perf_str.replace(".", "").isdigit()
                else 0.0
            )
        except ValueError:
            perf = 0.0

        tiling_parse_error = ""
        try:
            td = _parse_tiling_data(td_str)
            tiling_data, template = _td_from_values(
                td, key_str, _result_context(row, params)
            )
        except Exception as e:
            # 解析失败只降级 tiling_data，perf 保留；模板按 key 独立反查不被连坐
            # （例：legacy 96 值结构布局解析失败，但 KT=2 仍可反查出 mix——
            # 模板名与 tiling 解析是两个独立失败域，见 BaselineResult.tiling_parse_error）
            print(
                f"[baseline_batch] case {i + 1} '{case_name}' tiling 解析失败: {e}，原始片段: {td_str[:80]}"
            )
            tiling_data = TilingData()
            try:
                template = _detect_template_from_key_str(key_str, params)
            except Exception:
                template = "unknown"
            tiling_parse_error = f"{type(e).__name__}: {e}"

        results.append(
            BaselineResult(
                template=template,
                perf_us=perf,
                tiling_data=tiling_data,
                precision_status=prec_str or "UNKNOWN",
                tiling_parse_error=tiling_parse_error,
            )
        )
        print(
            f"[baseline_batch] case {i + 1} '{case_name}': {template}, {perf} us, "
            f"precision={prec_str or 'UNKNOWN'}"
        )

    return results


def verify_layout_coverage(
    result_csv_path: str, verbose: bool = True, all_params: Optional[list[dict]] = None
) -> dict:
    """
    档案推导闭环校验（Step A 档案完成后必跑）：逐行核对 result CSV 实测
    dyn_tiling_data 槽数 vs 活跃档案 struct_layouts 声明，堵"推导漏登记
    结构体变体"的口子（实测教训：bmm BASE 模板 90 槽结构未登记，baseline
    tiling 静默降级 null，榜单渲染全 0 行造成误读）。

    三类结果：
    - parseable: 槽位数在声明表内（运行时可解析）
    - declared_unparseable: 槽位数仅由 "_" 前缀条目登记（已知不可解析路径，
      如 BASE/legacy 结构体——仅文档化槽位，不进运行时布局查找表）
    - undeclared: 槽位数完全未登记 —— 档案推导未完成，补声明后重跑本校验
    另查 KT 维度（kt_undeclared）：dyn_tiling_key 解码出的引擎号不在
    kernel_type_map 登记表内（baseline 模板会落 unknown）。

    返回 {"parseable": {slots: [case名...]}, "declared_unparseable": {...},
    "undeclared": {...}, "kt_undeclared": {kt: [case名...]}}（undeclared 以
    "parse_error" 键归入字符串级解析失败；kt_undeclared 以 "key_parse_error"
    键归入 key 解析失败）。
    """
    _require_profile()
    contexts = {p["testcase_name"]: p for p in (all_params or [])}
    result: dict = {
        "parseable": {},
        "declared_unparseable": {},
        "undeclared": {},
        "kt_undeclared": {},
    }
    with open(result_csv_path) as f:
        for row in csv.DictReader(f):
            case = row.get("testcase_name", "")
            try:
                vals = _parse_tiling_data(row.get("dyn_tiling_data", ""))
            except Exception:
                result["undeclared"].setdefault("parse_error", []).append(case)
                vals = None
            if vals is not None:
                n = len(vals)
                try:
                    ctx = _tiling_context(
                        row.get("dyn_tiling_key", ""),
                        _result_context(row, contexts.get(case)),
                    )
                    layout = _select_layout(n, ctx, include_documented=True)
                    if layout.startswith("_"):
                        result["declared_unparseable"].setdefault(n, []).append(case)
                    else:
                        _parse_struct_fields(vals, layout)
                        result["parseable"].setdefault(n, []).append(case)
                except (ValueError, KeyError, struct.error):
                    result["undeclared"].setdefault(n, []).append(case)
            try:
                kt = _decode_kernel_type(row.get("dyn_tiling_key", ""))
                ctx = _result_context(row, contexts.get(case))
                if (
                    _detect_template_from_key_str(row.get("dyn_tiling_key", ""), ctx)
                    == "unknown"
                ):
                    result["kt_undeclared"].setdefault(kt, []).append(case)
            except Exception:
                result["kt_undeclared"].setdefault("key_parse_error", []).append(case)
    if verbose:
        for n, cases in sorted(result["undeclared"].items(), key=lambda x: str(x[0])):
            print(
                f"[verify_layout_coverage] 警告: 槽位 {n} 未在档案 struct_layouts 声明"
                f"（{len(cases)} case: {', '.join(cases[:3])}{'…' if len(cases) > 3 else ''}）"
                f"——档案推导未完成，补声明后重跑本校验"
            )
        for n, cases in sorted(result["declared_unparseable"].items()):
            print(
                f"[verify_layout_coverage] 提示: 槽位 {n} 为已登记不可解析路径"
                f"（{len(cases)} case）"
            )
        for kt, cases in sorted(
            result["kt_undeclared"].items(), key=lambda x: str(x[0])
        ):
            print(
                f"[verify_layout_coverage] 警告: KT={kt} 未在 kernel_type_map 登记"
                f"（{len(cases)} case: {', '.join(cases[:3])}{'…' if len(cases) > 3 else ''}）"
                f"——baseline 模板将落 unknown，回 Step A 补登记"
            )
    return result


def _extract_perf_status(result_csv_path: str) -> str:
    """从 result CSV 提取 perf_status 字段，用于失败诊断"""
    try:
        with open(result_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get("perf_status", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def output_baseline_info(
    baseline: BaselineResult,
    params: dict,
    case_name: str,
    output_path: str,
    aic_num: int = 0,
) -> str:
    """
    跑完 baseline 后，将 baseline 结果 + shape 参数输出为 JSON 供 AI 阅读生成候选。

    AI 结合此 JSON 与 references/matmul-tiling-theory.md 中的通用切分原理，
    生成 candidates JSON（格式见 load_candidates 的输入）。

    aic_num: AI Core 实测核数（get_aic_num() 的返回值）。多核利用率/尾块负载均衡
    由当前档案的 derive_schedule 按实测核数计算。传 0 时抛异常。

    返回 output_path。
    """
    import json

    if aic_num <= 0:
        raise ValueError(
            "aic_num 未设置。多核利用率等健康度指标必须用实测核数计算："
            "先调用 get_soc_version(result_csv) + get_aic_num(soc) 获取。"
        )

    td = baseline.tiling_data
    batch_count = params.get("batch_count", 1)
    info = {
        "case_name": case_name,
        "shape": {
            "M": params["M"],
            "N": params["N"],
            "K": params["K"],
            "x1_dtype": params["x1_dtype"],
            "x2_dtype": params["x2_dtype"],
            "scale_dtype": params["scale_dtype"],
            "y_dtype": params["y_dtype"],
            "bias": params["bias"],
            "trans_a": params["trans_a"],
            "trans_b": params["trans_b"],
            "x2_format": params["x2_format"],
            # 多 batch 泛化：case 分类与 batch 记账（folded 的 M 已含 batch、batch_count=1）
            "case_class": params.get("case_class", "without_batch"),
            "folded": params.get("folded", False),
            "batch_count": batch_count,
            "x1_batch": params.get("x1_batch", 1),
            "x2_batch": params.get("x2_batch", 1),
            "b_broadcast": params.get("b_broadcast", False),
        },
        "baseline": {
            "template": baseline.template,
            "aic_num": aic_num,
            "core_query": _require_profile().get("soc", {}).get("core_query"),
            "perf_us": baseline.perf_us,
            "precision_status": baseline.precision_status,
            "tiling_data": td.to_dict() or None,
            # tiling_data 为 null 时给出可追溯原因（布局未收录/CSV 缺失等），
            # 避免下游把"没解析到"误读成"值为 0"
            "tiling_parse_error": baseline.tiling_parse_error or None,
        },
    }

    values = td.to_dict()
    capacity_fields = {
        "baseM",
        "baseN",
        "baseK",
        "kAL1",
        "kBL1",
        "scaleKL1",
        "nBufferNum",
        "dbL0C",
    }
    missing_capacity_fields = sorted(capacity_fields - values.keys())
    if not missing_capacity_fields and all(
        values[key] > 0 for key in ("baseM", "baseN", "baseK")
    ):
        sizeof_a = _dtype_size(params["x1_dtype"])
        sizeof_b = _dtype_size(params["x2_dtype"])
        soc = _require_profile()["soc"]  # 容量/分组常量（档案声明，替代引擎旧硬编码）

        l0c_util = (
            td.baseM * td.baseN * soc["data_size_l0c"] * td.dbL0C / soc["l0c_size"]
        )
        l1_k_reuse = td.kAL1 / td.baseK if td.baseK > 0 else 0
        l0a_util = td.baseM * td.baseK * sizeof_a * 2 / soc["l0a_size"]
        l0b_util = td.baseN * td.baseK * sizeof_b * 2 / soc["l0b_size"]

        a_usage = td.baseM * td.kAL1 * sizeof_a * td.nBufferNum
        b_usage = td.baseN * td.kBL1 * sizeof_b * td.nBufferNum
        a_scale = b_scale = 0
        if td.scaleKL1 > 0:
            group_size = soc.get("mx_group_size")
            if (
                isinstance(group_size, bool)
                or not isinstance(group_size, (int, float))
                or group_size <= 0
            ):
                raise ValueError(
                    "当前容量模型含 scale 深度，需要有依据的 soc.mx_group_size"
                )
            a_scale = td.baseM * math.ceil(td.scaleKL1 / group_size) * td.nBufferNum
            b_scale = td.baseN * math.ceil(td.scaleKL1 / group_size) * td.nBufferNum
        sizeof_bias = _dtype_size(params["bias"]) if params["bias"] != "none" else 0
        bias_usage = td.baseN * sizeof_bias * 2 if params["bias"] != "none" else 0
        l1_total = a_usage + b_usage + a_scale + b_scale + bias_usage

        health_dict = {
            "l0c_util": f"{l0c_util:.0%}",
            "l1_k_reuse": f"{l1_k_reuse:.1f}x",
            "l0a_util": f"{l0a_util:.0%}",
            "l0b_util": f"{l0b_util:.0%}",
            "l1_util": f"{l1_total / soc['l1_size']:.0%}",
            "l1_remaining_bytes": soc["l1_size"] - l1_total,
        }
        info["baseline"]["health"] = health_dict
    elif values:
        info["baseline"]["health_unavailable"] = (
            f"现有容量模型缺少字段 {missing_capacity_fields} 或基础块非正数；"
            "原始字段已完整保留，按当前算子机制分析。"
        )

    scheduling = _require_profile().get("scheduling")
    if scheduling is not None and values:
        derive_schedule = getattr(_recognition_module(), "derive_schedule", None)
        if not callable(derive_schedule):
            raise ValueError("档案声明了 scheduling，但识别适配器没有 derive_schedule")
        metrics = derive_schedule(
            {
                "params": params,
                "tiling_data": values,
                "template": baseline.template,
                "aic_num": aic_num,
                "core_query": _require_profile().get("soc", {}).get("core_query"),
                "soc": _require_profile()["soc"],
            }
        )
        if not isinstance(metrics, dict):
            raise ValueError("recognition.derive_schedule 必须返回调度指标字典")
        health = info["baseline"].setdefault("health", {})
        overlap = health.keys() & metrics.keys()
        if overlap:
            raise ValueError(f"调度指标覆盖了容量指标: {sorted(overlap)}")
        health.update(metrics)
        info["baseline"]["scheduling_model"] = dict(scheduling)
    else:
        info["baseline"]["scheduling_unavailable"] = (
            "未声明有源码依据的调度模型或未解析到切分数据，不估算任务数、尾轮和 K 循环。"
        )

    _ensure_parent(output_path)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[baseline_info] 已输出到 {output_path}")
    return output_path


def output_baseline_info_batch(
    baselines: list[BaselineResult],
    all_params: list[dict],
    output_path: str,
    aic_num: int = 0,
) -> str:
    """
    批量输出所有 case 的 baseline info 到一个 JSON 文件。

    AI 读取此 JSON 后，逐个 case 分析并生成 candidates JSON。

    aic_num: AI Core 实测核数（get_aic_num()），透传给单 case 输出。
    """
    import json

    if aic_num <= 0:
        raise ValueError(
            "aic_num 未设置。请先调用 get_soc_version(result_csv) + get_aic_num(soc)，"
            "将实测核数传入 output_baseline_info_batch(..., aic_num=N)。"
        )

    all_info = []
    for i, (baseline, params) in enumerate(zip(baselines, all_params)):
        case_name = params.get("testcase_name", f"case_{i + 1}")
        single_path = output_path.replace(".json", f"_case{i + 1}.json")
        output_baseline_info(baseline, params, case_name, single_path, aic_num=aic_num)
        with open(single_path) as f:
            all_info.append(json.load(f))

    _ensure_parent(output_path)
    with open(output_path, "w") as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)

    print(f"[baseline_info_batch] {len(all_info)} cases 已输出到 {output_path}")
    return output_path


_CANDIDATE_METADATA = {"template", "reason", "expected", "risky"}


def _validate_candidate_fields(candidate: dict, where: str) -> list[str]:
    """校验候选字段能被当前算子档案注入，返回实际可注入字段名。"""
    unknown = sorted(set(candidate) - set(MACRO_MAP) - _CANDIDATE_METADATA)
    if unknown:
        raise ValueError(
            f"候选 {where} 含不可注入字段 {unknown}；可用字段为 "
            f"{sorted(MACRO_MAP)}。字段别名不会自动生效。"
        )
    tune_fields = [k for k in MACRO_MAP if k in candidate and candidate[k] is not None]
    if not tune_fields:
        raise ValueError(f"候选 {where} 没有任何可注入字段")
    for field_name in tune_fields:
        value = candidate[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"候选 {where}.{field_name} 必须是整数，实际为 {value!r}")
    return tune_fields


def _kernel_macro_fields(kernel_file: str) -> set[str]:
    """返回目标 kernel 中可唯一注入的字段集；重复宏直接报错。"""
    try:
        with open(kernel_file) as f:
            content = f.read()
    except OSError as e:
        raise RuntimeError(f"kernel 文件不可读: {kernel_file}: {e}") from e
    available = set()
    duplicated = []
    for field_name, macro in MACRO_MAP.items():
        count = len(
            re.findall(rf"(?m)^\s*#define\s+{re.escape(macro)}\s+[-+]?\d+\s*$", content)
        )
        if count == 1:
            available.add(field_name)
        elif count > 1:
            duplicated.append(f"{field_name}->{macro}({count}次)")
    if duplicated:
        raise RuntimeError(
            f"kernel 文件宏定义重复，无法安全注入 {kernel_file}: "
            + ", ".join(duplicated)
        )
    return available


def load_candidates(json_path: str) -> tuple[list[dict], dict]:
    """
    读取 AI 生成的 candidates JSON 文件。

    每个候选必须显式包含 template/reason/expected。按逐候选模板统计，拒绝
    重复配置，避免列表首项误代表整组或重复点浪费扫描预算。

    返回 (candidates 列表, 各模板候选数统计)。
    """
    _require_profile()
    with open(json_path) as f:
        data = json.load(f)

    candidates = []
    gen_by_template = {}
    seen = {}

    def add_group(items, location: str) -> None:
        if not isinstance(items, list):
            return
        for idx, raw in enumerate(items):
            where = f"{location}[{idx}]"
            if not isinstance(raw, dict):
                raise ValueError(
                    f"候选 {where} 必须是 JSON object，实际为 {type(raw).__name__}"
                )
            candidate = dict(raw)
            if not candidate.get("template"):
                raise ValueError(f"候选 {where} 缺少显式 template；不再默认归入 swat")
            for field_name in ("reason", "expected"):
                if (
                    not isinstance(candidate.get(field_name), str)
                    or not candidate[field_name].strip()
                ):
                    raise ValueError(
                        f"候选 {where} 缺少非空 {field_name}，无法追溯生成意图"
                    )
            candidate["template"] = normalize_template(candidate["template"])

            _validate_candidate_fields(candidate, where)

            # reason/expected/risky 是解释元数据，不构成新的实际配置。
            signature_obj = {
                k: v
                for k, v in candidate.items()
                if k not in ("reason", "expected", "risky")
            }
            signature = json.dumps(signature_obj, sort_keys=True, ensure_ascii=False)
            if signature in seen:
                raise ValueError(
                    f"重复候选配置: {where} 与 {seen[signature]} 仅说明文字不同，"
                    f"会浪费扫描预算"
                )
            seen[signature] = where
            candidates.append(candidate)
            tmpl = candidate["template"]
            gen_by_template[tmpl] = gen_by_template.get(tmpl, 0) + 1

    if isinstance(data, list):
        add_group(data, "$")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                add_group(value, f"$.{key}")
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    add_group(sub_val, f"$.{key}.{sub_key}")
    else:
        raise ValueError(
            f"候选文件根节点必须是 object 或 list，实际为 {type(data).__name__}"
        )

    print(
        f"[load_candidates] 从 {json_path} 加载 {len(candidates)} 个候选: {gen_by_template}"
    )
    return candidates, gen_by_template


# ============================================================================
# 8. 候选扫参完整执行流程
# ============================================================================


def scan_paths(log_tag: str = "") -> dict[str, str]:
    """
    扫参临时文件路径。log_tag 非空时所有文件带 tag 后缀，不同 case 的
    日志互不覆盖；无 tag 时与默认路径一致（向后兼容）。

    注意：tag 只隔离日志与状态文件。kernel .h 宏、kernel_meta、config.ini
    仍是全局唯一资源——扫参依然必须串行，禁止用不同 tag 并行跑多个 scan。
    """
    s = f"_{log_tag}" if log_tag else ""
    return {
        "script": f"/tmp/run_scan{s}.sh",
        "output": f"/tmp/scan_output{s}.log",
        "progress": f"/tmp/scan_progress{s}.log",
        "done": f"/tmp/scan_done{s}.flag",
    }


def baseline_paths(log_tag: str = "") -> dict[str, str]:
    """
    baseline 临时文件路径（与 scan_paths 同构）。log_tag 非空时带后缀，
    不同 CSV 的 baseline 日志互不覆盖；无 tag 时与默认路径一致。
    """
    s = f"_{log_tag}" if log_tag else ""
    return {
        "script": f"/tmp/run_baseline{s}.sh",
        "output": f"/tmp/baseline_output{s}.log",
        "progress": f"/tmp/baseline_progress{s}.log",
        "done": f"/tmp/baseline_done{s}.flag",
    }


def anchor_paths(log_tag: str = "") -> dict[str, str]:
    """锚点探针临时文件路径；与 baseline/scan 使用相同 tag 隔离规则。"""
    s = f"_{log_tag}" if log_tag else ""
    return {
        "script": f"/tmp/run_anchor{s}.sh",
        "output": f"/tmp/anchor_output{s}.log",
        "progress": f"/tmp/anchor_progress{s}.log",
        "done": f"/tmp/anchor_done{s}.flag",
    }


# 最近一次任务的 log_tag 指针：start_* 写入，collect_*/check_task_done 在
# 调用方忘传 tag 时据此定位（顺序：显式 tag > case_csv 派生 > 指针）
SCAN_LAST_TAG_FILE = "/tmp/scan_last_tag"
BASELINE_LAST_TAG_FILE = "/tmp/baseline_last_tag"
ANCHOR_LAST_TAG_FILE = "/tmp/anchor_last_tag"


def _tag_from_case_csv(case_csv: str) -> str:
    """
    从 case CSV 文件名自动派生 log_tag（单 case CSV 即用例名）。

    start_ttk_scan / start_baseline 在 log_tag 为空时调用——即使 AI 忘传
    tag，每个 case 的日志/结果也天然隔离，不会出现后一个 case 覆盖前一个。
    """
    base = os.path.splitext(os.path.basename(case_csv))[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return safe[:80] or "default"


def _resolve_log_tag(log_tag: str, case_csv: Optional[str], pointer_file: str) -> str:
    """log_tag 解析链：显式传参 > case_csv 文件名派生 > 指针文件 > 空串（默认路径）。"""
    if log_tag:
        return log_tag
    if case_csv:
        return _tag_from_case_csv(case_csv)
    try:
        with open(pointer_file) as f:
            return f.read().strip()
    except OSError:
        return ""


def _script_from_tag_pointer(pointer_file: str, kind: str) -> str:
    """从指针文件读最近 tag，拼出对应执行脚本路径；读不到返回空串。"""
    try:
        with open(pointer_file) as f:
            tag = f.read().strip()
    except OSError:
        return ""
    return f"/tmp/run_{kind}_{tag}.sh" if tag else ""


def generate_anchor_probe_script(
    case_csv: str,
    templates: list[str],
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    case_class: str = "without_batch",
    batch_count: int = 1,
    quant_mode: str = "mx",
    baseline_template: str = "",
) -> str:
    """生成只取 host 原生 tiling 的锚点探针脚本，不改任何 TUNE_ 宏。

    worktree 模式逐模板切换 forced vendor；install-stub 模式固定 normal vendor，
    且只能探测 baseline 实际命中的模板。每个块都由确定的 SCAN_START/END
    标记包围，供 extract_anchor() 直接解析。kernel 运行失败不会吞掉 END 标记，
    但缺 layout/blockDim/tiling key 的块会在 collect_anchor_probe() 被门禁拒绝。
    """
    _require_runtime_paths()
    npu_timeout, proc_timeout = calc_timeout(m, n, k, batch_count)
    paths = anchor_paths(log_tag)
    mode_cfg = _require_profile()["quant_modes"][quant_mode]
    uses_stub = bool(mode_cfg.get("tune_macro_define_block"))
    normalized = list(dict.fromkeys(normalize_template(t) for t in templates))
    if not normalized:
        raise ValueError("锚点探针 templates 不能为空")
    injectable = set(
        injectable_templates(case_class, quant_mode, natural_template=baseline_template)
    )
    invalid = sorted(set(normalized) - injectable)
    if invalid:
        raise ValueError(
            f"锚点模板不在当前 case 可注入集合: {invalid}；可用={sorted(injectable)}"
        )
    if uses_stub and not baseline_template:
        raise ValueError("install-stub 模式必须传 baseline_template 后才能探测锚点")

    q = shlex.quote
    lines = [
        "#!/bin/bash",
        "# Matmul tiling anchor probe — auto-generated by ttk_runner.py",
        "trap '' TERM INT",
        f"> {q(paths['output'])}",
        f"> {q(paths['progress'])}",
        "",
    ]
    for tmpl in normalized:
        if uses_stub:
            package = VENDOR_PACKAGES["normal"]
        elif tmpl in VENDOR_PACKAGES:
            package = VENDOR_PACKAGES[tmpl]
        else:
            raise KeyError(
                f"模板 '{tmpl}' 没有 forced vendor 包声明；"
                f"VENDOR_PACKAGES={sorted(VENDOR_PACKAGES)}"
            )
        lines.extend(
            [
                f"printf '%s' {q('load_priority=' + package)} > {q(VENDOR_CONFIG)}",
                f'echo "===== SCAN_START {tmpl} 0 =====" >> {q(paths["output"])}',
                _cache_cleanup_command(),
                f"cd {q(TTK_DIR)} && timeout {proc_timeout} python3 -m ttk kernel "
                f"-i {q(case_csv)} -d --compare=close --pc=1 --run=20 "
                f"--npu-timeout {npu_timeout} --plugin {q(PLUGIN_DIR)} "
                f">> {q(paths['output'])} 2>&1 || true",
                f'echo "===== SCAN_END {tmpl} 0 =====" >> {q(paths["output"])}',
                f'echo "done {tmpl}" >> {q(paths["progress"])}',
                "",
            ]
        )
    lines.append(f"touch {q(paths['done'])}")
    with open(paths["script"], "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(
        paths["script"],
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
    )
    return paths["script"]


def start_anchor_probe(
    case_csv: str,
    templates: list[str],
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    case_class: str = "without_batch",
    batch_count: int = 1,
    quant_mode: str = "mx",
    baseline_template: str = "",
) -> int:
    """后台启动锚点探针，返回预计秒数。探针与 baseline/scan 一样必须串行。"""
    log_tag = log_tag or _tag_from_case_csv(case_csv)
    paths = anchor_paths(log_tag)
    for path in list(paths.values()) + ["/tmp/anchor_done.flag"]:
        if os.path.exists(path):
            os.remove(path)
    with open(ANCHOR_LAST_TAG_FILE, "w") as f:
        f.write(log_tag)
    script = generate_anchor_probe_script(
        case_csv,
        templates,
        m,
        n,
        k,
        log_tag=log_tag,
        case_class=case_class,
        batch_count=batch_count,
        quant_mode=quant_mode,
        baseline_template=baseline_template,
    )
    subprocess.Popen(
        ["bash", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    eta = estimate_baseline_eta(
        m, n, k, n_cases=len(templates), batch_count=batch_count
    )
    print(
        f"[anchor] {len(templates)} 个模板探针已启动，预计 {eta}s，"
        f"done flag: {paths['done']}"
    )
    return eta


def collect_anchor_probe(
    templates: list[str],
    params: dict,
    log_tag: str = "",
    case_csv: str = None,
    save_path: str = None,
    strict: bool = True,
) -> tuple[dict, list[str]]:
    """收集、落盘并校验锚点；缺模板或约束自检异常默认直接阻断后续扫参。"""
    log_tag = _resolve_log_tag(log_tag, case_csv, ANCHOR_LAST_TAG_FILE)
    expected = list(dict.fromkeys(normalize_template(t) for t in templates))
    anchors = extract_anchor(
        anchor_paths(log_tag)["output"], templates=expected, context=params
    )
    missing = sorted(set(expected) - set(anchors))
    incomplete = sorted(
        tmpl
        for tmpl, anchor in anchors.items()
        if anchor.get("blockDim") is None or not anchor.get("layout")
    )
    diagnostics = validate_rules_against_anchors(anchors, params)
    for warning in diagnostics["warnings"]:
        print(f"[anchor warning] {warning}")
    if save_path is None:
        save_path = f"{RUN_DIR}/anchors/anchors_{log_tag or 'default'}.json"
    save_anchors(anchors, save_path)
    failures = [f"缺少锚点模板: {missing}"] if missing else []
    if incomplete:
        failures.append(f"锚点缺少 blockDim/layout: {incomplete}")
    failures += diagnostics["errors"]
    if strict and failures:
        raise RuntimeError("anchor gate 失败:\n- " + "\n- ".join(failures))
    return anchors, failures


def generate_scan_script(
    candidates: list[dict],
    case_csv: str,
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    case_class: str = "without_batch",
    batch_count: int = 1,
    quant_mode: str = "mx",
    baseline_template: str = "",
) -> str:
    """
    生成候选扫参的 bash 脚本。

    case_class: 决定 TUNE_ 宏 sed 的目标 kernel 文件（with_batch 的 swat 是
    qbmm_mx_tensor_api_blaze.h，与 without_batch 不同文件；l0cpp 同文件双变体）。
    quant_mode: 二维路由第二维——模板白名单与 kernel 文件按 quant_modes 查表；
    dead_fields（如 pertoken 的 scaleKL1）不注入 sed，候选带死字段值时警告并忽略。
    baseline_template: normal baseline 实际命中的模板。install-stub 模式
    无法改变 host 路由，因此必须传入，且只扫描该模板。

    脚本对每个候选：
    1. 切换到对应 vendor 包
    2. 用 sed 改安装目录 kernel .h 的 TUNE_ 宏
    3. 清 kernel_meta 缓存
    4. 跑 ttk，输出统一追加到 scan_output[_tag].log（每个候选前加分隔标记）
    5. 写进度

    返回脚本文件路径。
    """
    _require_runtime_paths()
    npu_timeout, proc_timeout = calc_timeout(m, n, k, batch_count)
    paths = scan_paths(log_tag)
    script_path = paths["script"]
    output_log = paths["output"]
    progress_log = paths["progress"]
    done_flag = paths["done"]

    mode_cfg = _require_profile()["quant_modes"][quant_mode]
    uses_stub = bool(mode_cfg.get("tune_macro_define_block"))
    if uses_stub and not baseline_template:
        raise ValueError(
            f"量化模式 '{quant_mode}' 使用 install-stub，请传入 baseline_template，"
            "避免把候选标签误当成 host 实际路由。"
        )
    allowed = injectable_templates(case_class, quant_mode, baseline_template)
    if uses_stub and not allowed:
        raise ValueError(
            f"baseline 模板 '{baseline_template}' 在 case_class '{case_class}' × "
            f"quant_mode '{quant_mode}' 下没有可验证的宏注入点，该 case 不可扫参。"
        )
    dead_fields = set(mode_cfg.get("dead_fields", {}) or {})

    q = shlex.quote
    lines = [
        "#!/bin/bash",
        "# Matmul tiling scan runner — auto-generated by ttk_runner.py",
        "# 信号免疫：候选在 NPU 上挂死被 timeout 击杀时，SIGTERM 可能经进程组",
        "# 株连本脚本（实测：bash 与 timeout 击杀时刻精确重合而死）。本脚本为无人值守",
        "# 批处理，正常退出条件只有跑完并 touch done flag，因此直接忽略 TERM/INT；",
        "# kill -9 等不可捕获信号场景由 check_task_done() 的进程存活检测兜底。",
        "trap '' TERM INT",
        f"> {q(output_log)}",
        "",
    ]

    # 按模板分组
    by_template: dict[str, list[dict]] = {}
    for candidate_idx, c in enumerate(candidates):
        _validate_candidate_fields(c, f"candidates[{candidate_idx}]")
        tmpl = normalize_template(c.get("template", "swat"))
        c["template"] = tmpl
        by_template.setdefault(tmpl, []).append(c)

    for tmpl, group in by_template.items():
        if tmpl not in allowed:
            raise ValueError(
                f"模板 '{tmpl}' 在 case_class '{case_class}' × quant_mode '{quant_mode}' "
                f"下不可注入（可扫模板: {allowed}）"
            )
        pkg = _vendor_package_for_template(tmpl, quant_mode)
        kernel_file = get_kernel_file_path(tmpl, case_class, quant_mode)
        required_fields = _kernel_macro_fields(kernel_file) - dead_fields
        if not required_fields:
            raise RuntimeError(
                f"模板 '{tmpl}' 的 kernel 文件没有可注入 TUNE_ 宏: {kernel_file}"
            )

        lines.append(f"# === {tmpl} 组 ({len(group)} 个候选) ===")
        lines.append(f"printf '%s' {q('load_priority=' + pkg)} > {q(VENDOR_CONFIG)}")
        lines.append("")

        for i, c in enumerate(group):
            # 生成 sed 命令更新每个宏（死字段按 quant_mode 过滤，不注入 sed）
            sed_cmds = []
            live_fields = [
                key
                for key in MACRO_MAP
                if key in c and c[key] is not None and key not in dead_fields
            ]
            missing_fields = sorted(required_fields - set(live_fields))
            extra_fields = sorted(set(live_fields) - required_fields)
            if missing_fields or extra_fields:
                raise ValueError(
                    f"候选 {tmpl}#{i} 与目标 kernel 宏集不一致："
                    f"缺少={missing_fields}，不属于该模板={extra_fields}。"
                    "每个候选必须完整设置目标 kernel 的可注入宏，"
                    "避免继承前一个候选的值。"
                )
            for key, macro in MACRO_MAP.items():
                if key in c and c[key] is not None:
                    if key in dead_fields:
                        _warn_dead_field_once(quant_mode, key, c[key])
                        continue
                    macro_re = re.escape(macro)
                    value = c[key]
                    sed_cmds.extend(
                        [
                            (
                                f"test \"$(grep -Ec '^[[:space:]]*#define[[:space:]]+{macro_re}"
                                f"[[:space:]]+[-+]?[0-9]+[[:space:]]*$' {q(kernel_file)})\" -eq 1 || "
                                f"{{ echo 'macro {macro} missing or duplicated' >> {q(output_log)}; exit 2; }}"
                            ),
                            (
                                f"sed -Ei 's|^[[:space:]]*#define[[:space:]]+{macro_re}"
                                f"[[:space:]]+[-+]?[0-9]+[[:space:]]*$|#define {macro} {value}|' "
                                f"{q(kernel_file)}"
                            ),
                            (
                                f"grep -Eq '^#define[[:space:]]+{macro_re}"
                                f"[[:space:]]+{value}[[:space:]]*$' {q(kernel_file)} || "
                                f"{{ echo 'macro {macro} write verification failed' >> {q(output_log)}; exit 2; }}"
                            ),
                        ]
                    )

            lines.append(f"# 候选 {i + 1}/{len(group)}: {tmpl}")
            lines.append(f'echo "===== SCAN_START {tmpl} {i} =====" >> {q(output_log)}')
            lines.extend(sed_cmds)
            lines.append(_cache_cleanup_command())
            lines.append(
                f"cd {q(TTK_DIR)} && "
                f"timeout {proc_timeout} python3 -m ttk kernel -i {q(case_csv)} -d "
                f"--compare=close --pc=1 --run=20 --npu-timeout {npu_timeout} "
                f"--plugin {q(PLUGIN_DIR)} >> {q(output_log)} 2>&1"
            )
            lines.append(f'echo "===== SCAN_END {tmpl} {i} =====" >> {q(output_log)}')
            lines.append(
                f'echo "done {tmpl} {i + 1}/{len(group)}" >> {q(progress_log)}'
            )
            lines.append("")

    lines.append(f"touch {q(done_flag)}")

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(
        script_path,
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
    )
    return script_path


def start_ttk_scan(
    candidates: list[dict],
    case_csv: str,
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    case_class: str = "without_batch",
    batch_count: int = 1,
    quant_mode: str = "mx",
    baseline_template: str = "",
) -> int:
    """
    生成扫参脚本并在后台启动。

    case_class/batch_count/quant_mode: extract_shape_params 的返回值直接透传
    （with_batch case 的 kernel 文件/超时与 without_batch 不同；batch_count=batchC；
    quant_mode 决定模板白名单/kernel 文件/死字段，二维路由第二维）。
    install-stub 模式还必须传 baseline_template=baseline.template。

    返回预估耗时（秒），AI 据此决定轮询间隔（建议预估时长的 1/4）。
    log_tag 非空时（如 "case1"）日志/进度/done flag 均带 tag，多个 case
    依次扫参时上一 case 的日志不会被覆盖；AI 轮询对应的
    check_task_done(f"/tmp/scan_done_{log_tag}.flag")，收集时把同一个
    log_tag 传给 collect_ttk_scan()。
    log_tag 为空时自动从 case_csv 文件名派生（忘传也隔离，不留全局路径），
    并写入 /tmp/scan_last_tag 指针供 collect_ttk_scan / check_task_done 兜底定位。
    """
    log_tag = log_tag or _tag_from_case_csv(case_csv)
    paths = scan_paths(log_tag)
    for f in list(paths.values()) + ["/tmp/scan_done.flag"]:
        # 末项：清理无 tag 的 /tmp/scan_done.flag——自动 tag 后它不是本任务的活信号，
        # 残留会让轮询默认路径的调用方误判已完成
        if os.path.exists(f):
            os.remove(f)
    with open(SCAN_LAST_TAG_FILE, "w") as f:
        f.write(log_tag)

    script_path = generate_scan_script(
        candidates,
        case_csv,
        m,
        n,
        k,
        log_tag=log_tag,
        case_class=case_class,
        batch_count=batch_count,
        quant_mode=quant_mode,
        baseline_template=baseline_template,
    )
    subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

    eta = estimate_scan_eta(m, n, k, len(candidates), batch_count=batch_count)
    print(
        f"[scan] {len(candidates)} 个候选，已后台启动，预计 {eta}s，"
        f"建议每 {max(eta // 4, 30)}s 检查一次 done flag: {paths['done']}"
    )
    print(
        f"[scan] 本 case 工件已隔离: 日志={paths['output']} (log_tag={log_tag})，"
        f"收集后逐候选结果默认落盘 {RUN_DIR}/scan_results_{log_tag}.json"
    )
    return eta


def collect_ttk_scan(
    candidates: list[dict],
    log_tag: str = "",
    save_path: str = None,
    case_csv: str = None,
    params: Optional[dict] = None,
) -> list[CandidateResult]:
    """
    扫参完成后收集结果：逐个解析 scan_output[_tag].log 中的性能与精度数据。

    AI 在 check_task_done(带 tag 的 done flag) 返回 True 后调用本函数，
    log_tag 应与 start_ttk_scan 传入的一致；忘传时按
    case_csv 文件名派生 > /tmp/scan_last_tag 指针 的顺序兜底定位。

    save_path 缺省（None）时自动落盘
    {RUN_DIR}/scan_results_{tag}.json——逐候选完整结果（输入参数 +
    perf + 精度 + status）永远留底，多 case 结果互不覆盖；
    传空串 "" 可跳过落盘。

    精度过滤：逐候选解析 PRECISION_STATUS——
    - PASS 且有 DYN_PERF         → status="ok"（进推荐榜）
    - 有 DYN_PERF 但精度非 PASS  → status="precision_fail"（剔除出榜，仅计入
      Step 4 统计。含 FAIL / COMPARE_FAILURE / 状态行缺失的异常块）
    - 无 DYN_PERF                → status="timeout"（含 kernel 挂死：块内可能
      残留 PRECISION_STATUS: FAIL 行，但无性能输出按超时归类）。timeout 时
      precision 字段恒 None——timeout 表示精度未完成测量，不冒充 FAIL，
      避免与 precision_fail 统计重复计数。
    """
    log_tag = _resolve_log_tag(log_tag, case_csv, SCAN_LAST_TAG_FILE)
    if save_path is None:
        save_path = f"{RUN_DIR}/results/scan_results_{log_tag or 'default'}.json"
    if params is None and case_csv:
        with open(case_csv) as f:
            params = extract_shape_params(next(csv.DictReader(f)))
    results = []
    records = []
    by_template: dict[str, list[dict]] = {}
    for c in candidates:
        tmpl = normalize_template(c.get("template", "swat"))
        by_template.setdefault(tmpl, []).append(c)

    paths = scan_paths(log_tag)
    output_log = paths["output"]
    full_log = ""
    if os.path.exists(output_log):
        with open(output_log) as f:
            full_log = f.read()

    precision_failed: list[tuple[str, int]] = []
    for tmpl, group in by_template.items():
        for i, c in enumerate(group):
            candidate_values = {k: c[k] for k in MACRO_MAP if c.get(k) is not None}
            rec = {
                "template": tmpl,
                "group_idx": i,
                **candidate_values,
                "reason": c.get("reason", ""),
                "expected": c.get("expected", ""),
                "observed_template": None,
                "perf_us": None,
                "precision": None,
                "status": "",
            }
            start_marker = f"===== SCAN_START {tmpl} {i} ====="
            end_marker = f"===== SCAN_END {tmpl} {i} ====="
            start_idx = full_log.find(start_marker)
            end_idx = full_log.find(end_marker)
            if start_idx == -1 or end_idx == -1:
                rec["status"] = "timeout"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        status="timeout",
                        candidate=candidate_values,
                    )
                )
                continue
            start = start_idx + len(start_marker)
            chunk = full_log[start:end_idx]
            perf = parse_ttk_output(chunk)
            if perf is None:
                rec["status"] = "timeout"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        status="timeout",
                        candidate=candidate_values,
                    )
                )
                continue
            precision = parse_precision_status(chunk)
            observed = parse_observed_template(chunk, params)
            rec["perf_us"] = perf
            rec["precision"] = precision
            rec["observed_template"] = observed
            td = TilingData(**candidate_values)
            if observed == "unknown":
                rec["status"] = "template_unverified"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        tiling_data=td,
                        perf_us=perf,
                        status="template_unverified",
                        precision=precision,
                        candidate=candidate_values,
                    )
                )
            elif observed != tmpl:
                rec["status"] = "template_mismatch"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        tiling_data=td,
                        perf_us=perf,
                        status="template_mismatch",
                        precision=precision,
                        candidate=candidate_values,
                    )
                )
            elif precision == "PASS":
                rec["status"] = "ok"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        tiling_data=td,
                        perf_us=perf,
                        status="ok",
                        precision=precision,
                        candidate=candidate_values,
                    )
                )
            else:
                precision_failed.append((tmpl, i))
                rec["status"] = "precision_fail"
                records.append(rec)
                results.append(
                    CandidateResult(
                        template=tmpl,
                        tiling_data=td,
                        perf_us=perf,
                        status="precision_fail",
                        precision=precision,
                        candidate=candidate_values,
                    )
                )

    if save_path:
        import json

        _ensure_parent(save_path)
        with open(save_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        print(f"[scan] 逐候选结果已落盘: {save_path}（{len(records)} 条）")

    if precision_failed:
        print(
            f"[scan] 精度未通过 {len(precision_failed)} 个候选（已剔除出推荐榜）: "
            + ", ".join(f"{t}#{i}" for t, i in precision_failed)
        )
    route_failed = [
        r
        for r in records
        if r["status"] in ("template_mismatch", "template_unverified")
    ]
    if route_failed:
        print(f"[scan] 模板路由未验证 {len(route_failed)} 个候选（已剔除出推荐榜）")
    return results


def run_ttk_scan(
    candidates: list[dict],
    case_csv: str,
    m: int,
    n: int,
    k: int,
    log_tag: str = "",
    save_path: str = None,
    case_class: str = "without_batch",
    batch_count: int = 1,
    quant_mode: str = "mx",
    baseline_template: str = "",
) -> list[CandidateResult]:
    """
    扫参一站式执行：启动 → 阻塞等待 → 收集结果。

    适用于 AI 可以长时间阻塞的场景。如果需要 AI 自己控制轮询节奏，
    用 start_ttk_scan() + check_task_done() + collect_ttk_scan() 三步。
    """
    eta = start_ttk_scan(
        candidates,
        case_csv,
        m,
        n,
        k,
        log_tag=log_tag,
        case_class=case_class,
        batch_count=batch_count,
        quant_mode=quant_mode,
        baseline_template=baseline_template,
    )
    done_flag = scan_paths(log_tag or _tag_from_case_csv(case_csv))["done"]
    time.sleep(eta)

    for retry in range(5):
        if check_task_done(done_flag):
            break
        time.sleep(max(eta // 4, 30))
    else:
        print("[scan] 超过最大重试次数，尝试解析已有结果")

    return collect_ttk_scan(
        candidates, log_tag=log_tag, save_path=save_path, case_csv=case_csv
    )


# ============================================================================
# 9. 推荐榜生成
# ============================================================================


def generate_ranking(
    baseline: BaselineResult,
    candidates: list[CandidateResult],
    top_n: int = 10,
) -> str:
    """
    生成单个 case 的完整推荐榜 markdown。

    包含：标题行 + 表格 + 最优推荐 + Step 2-4 统计。

    baseline 未测得性能（perf_us <= 0）时直接抛异常——加速比相对 baseline
    计算，无基准则整个榜单无意义。baseline 跑不通属异常场景（ttk 超时/崩溃），
    调用方应先重跑 baseline 再生成推荐榜。
    """
    if baseline.perf_us <= 0:
        raise ValueError(
            f"baseline 未测得性能 (perf_us={baseline.perf_us})，无法生成推荐榜。"
            f"请重跑 baseline（start_baseline_batch → collect_baseline_batch）后再调用。"
        )
    if baseline.precision_status != "PASS":
        raise ValueError(
            f"baseline 精度未通过 (precision_status={baseline.precision_status})，无法生成推荐榜。"
            f"baseline 跑在 normal 包 + op_host 默认 tiling 上，精度失败属环境/算子异常"
            f"（对比 golden 链路或 NPU 状态），请检查 /tmp/baseline_output.log 与 "
            f"result CSV 的 precision_status 列后重跑 baseline。"
        )
    if not baseline.template or baseline.template == "unknown":
        raise ValueError(
            "baseline 的 tiling key 无法反查模板，不能生成可追溯推荐榜。"
            "请检查算子档案的 tiling_key 与 quant_modes.*.kernel_type_map。"
        )

    all_entries = []
    for c in candidates:
        if c.status == "ok" and c.perf_us > 0:
            all_entries.append(
                {
                    "template": c.template,
                    "tiling": c.candidate or c.tiling_data.to_dict(),
                    "perf": c.perf_us,
                    "is_baseline": False,
                }
            )
    # 未解析与实际字段值为 0 分开表示，保留所有真实字段。
    baseline_td_valid = bool(baseline.tiling_data.to_dict())
    all_entries.append(
        {
            "template": f"baseline ({baseline.template})",
            "tiling": baseline.tiling_data.to_dict() if baseline_td_valid else None,
            "tiling_note": None
            if baseline_td_valid
            else (
                baseline.tiling_parse_error
                or f"baseline tiling 未解析（模板 {baseline.template}，布局未收录）"
            ),
            "perf": baseline.perf_us,
            "is_baseline": True,
        }
    )

    all_entries.sort(key=lambda x: x["perf"])
    top = all_entries[:top_n]
    # baseline 掉出 Top10 时追加为表格末尾一行（排名列显示真实名次），
    # 保证榜单始终有性能基准参照
    baseline_entry = next(e for e in all_entries if e["is_baseline"])
    if baseline_entry not in top:
        top.append(baseline_entry)
    baseline_perf = baseline.perf_us

    lines = [
        "| 排名 | 性能模板 | tiling data | 耗时(us) | vs baseline |",
        "|------|---------|-------------|----------|-------------|",
    ]
    # 排名一律取 all_entries 中的真实位置（含 perf 相同的稳定顺序），
    # 追加的 baseline 行因此显示其真实名次（如 11、13）
    rank_of = {id(e): i + 1 for i, e in enumerate(all_entries)}
    for entry in top:
        rank = rank_of[id(entry)]
        if entry["tiling"] is None:
            td_str = f"N/A（{entry.get('tiling_note') or 'tiling 未解析'}）"
        else:
            td_str = (
                "{" + ", ".join(f"{k}:{v}" for k, v in entry["tiling"].items()) + "}"
            )
        speedup = baseline_perf / entry["perf"] if entry["perf"] > 0 else 0
        arrow = "↑" if speedup > 1.0 else ("↓" if speedup < 1.0 else "")

        if entry["is_baseline"]:
            lines.append(
                f"| **{rank}** | **{entry['template']}** | **{td_str}** | **{entry['perf']:.3f}** | **1.00x** |"
            )
        else:
            lines.append(
                f"| {rank} | {entry['template']} | {td_str} | {entry['perf']:.3f} | {speedup:.2f}x {arrow} |"
            )

    best = all_entries[0] if all_entries else None
    if best:
        if best["tiling"] is None:
            best_td = f"N/A（{best.get('tiling_note') or 'tiling 未解析'}）"
        else:
            best_td = (
                "{" + ", ".join(f"{k}:{v}" for k, v in best["tiling"].items()) + "}"
            )
        best_speedup = baseline_perf / best["perf"] if best["perf"] > 0 else 0
        pct = (best_speedup - 1) * 100
        if best["is_baseline"]:
            lines.append("")
            lines.append("最优推荐（排名第 1）")
            lines.append(f"- 性能模板: {baseline.template} (baseline)")
            lines.append(f"- tiling data: {best_td}")
            lines.append(f"- DYN_PERF: {best['perf']:.3f} us")
            lines.append(
                "- 加速比: 1.00x（未找到优于 baseline 的配置，op_host 默认 tiling 已最优）"
            )
        else:
            lines.append("")
            lines.append("最优推荐（排名第 1）")
            lines.append(f"- 性能模板: {best['template']}")
            lines.append(f"- tiling data: {best_td}")
            lines.append(f"- DYN_PERF: {best['perf']:.3f} us")
            lines.append(
                f"- 加速比: {best_speedup:.2f}x（比 baseline 提升 {pct:.1f}%）"
            )

    ok_count = sum(1 for c in candidates if c.status == "ok")
    precision_fail_count = sum(1 for c in candidates if c.status == "precision_fail")
    route_fail_count = sum(
        1
        for c in candidates
        if c.status in ("template_mismatch", "template_unverified")
    )
    timeout_count = sum(1 for c in candidates if c.status == "timeout")
    total = len(candidates)

    lines.append("")
    lines.append("Step 4 执行结果")
    lines.append(f"- 总执行: {total} 组")
    lines.append(f"- 通过（性能+精度）: {ok_count} 组")
    lines.append(f"- 精度未通过（已剔除）: {precision_fail_count} 组")
    lines.append(f"- 模板路由不一致/无法验证（已剔除）: {route_fail_count} 组")
    lines.append(f"- 超时/无性能数据: {timeout_count} 组")

    return "\n".join(lines)


def generate_ranking_block(
    case_idx: int,
    case_name: str,
    params: dict,
    baseline: BaselineResult,
    candidates: list[CandidateResult],
    gen_by_template: dict,
    filter_by_template: dict,
    top_n: int = 10,
) -> str:
    """
    生成单个 case 的完整推荐榜块（含标题行 + 表格 + 最优推荐 + Step 2-4 统计）。
    """
    m, n, k = params["M"], params["N"], params["K"]
    dtype_str = params["x1_dtype"]
    trans = f"{'t' if params['trans_a'] else 'n'}{'t' if params['trans_b'] else 'n'}"
    bias_info = f"bias={params['bias']}" if params["bias"] != "none" else "bias=none"

    header = (
        f"用例 {case_idx}: {case_name}\n"
        f"Shape: M={m}, N={n}, K={k}, {dtype_str}, {trans}, {bias_info} | "
        f"Baseline: {baseline.template}, {baseline.perf_us:.3f} us"
    )

    ranking = generate_ranking(baseline, candidates, top_n)

    gen_lines = []
    filter_lines = []
    # 统计键一律使用归一后的模板名，并保留档案声明顺序。
    declared = allowed_templates(
        params.get("case_class", "without_batch"), params.get("quant_mode", "mx")
    )
    templates = list(
        dict.fromkeys(
            [normalize_template(t) for t in declared]
            + list(gen_by_template)
            + list(filter_by_template)
        )
    )
    for tmpl in templates:
        gen_n = gen_by_template.get(tmpl, 0)
        pass_n = filter_by_template.get(tmpl, 0)
        if gen_n == 0:
            gen_lines.append(f"- {tmpl}: N/A，不可行")
            filter_lines.append(f"- {tmpl}: N/A")
        else:
            gen_lines.append(f"- {tmpl}: {gen_n} 个候选")
            filter_lines.append(f"- {tmpl}: {pass_n} 个通过")

    gen_section = "Step 2 候选生成\n" + "\n".join(gen_lines)
    filter_section = "Step 3 约束筛选结果\n" + "\n".join(filter_lines)

    return f"{header}\n\n{ranking}\n\n{gen_section}\n\n{filter_section}"


def generate_rankings_md(blocks: list[str], output_path: str) -> str:
    """
    将多个 case 的推荐榜块拼接成一个 markdown 文件，用 --- 分隔。

    返回 output_path。
    """
    content = "\n\n---\n\n".join(blocks)
    _ensure_parent(output_path)
    with open(output_path, "w") as f:
        f.write(content + "\n")
    print(f"[rankings] 已输出到 {output_path}")
    return output_path


# ============================================================================
# 11. 约束筛选（档案约束适配器执行，静态验证不跑 ttk）
# ============================================================================
# 硬约束（约束 1-17）与锚点一致性校验的判定逻辑全部在活跃档案包的
# constraints 适配器（按算子区分，判定与证据由算子档案维护；含
# afull 覆盖检查/合并尾块 stale/MTE2 内轴字节 等内联辅助）。硬件常量
# （L1/L0A/L0B/L0C 容量、MTE2 对齐、MX 分组）经 ctx.soc 从档案注入，
# 引擎不再持有硬编码副本。


def check_candidate(
    candidate: dict, params: dict, anchor: Optional[dict] = None
) -> tuple[bool, str]:
    """
    用硬约束（约束 1-17，判定的 ground truth 见活跃档案包 constraints.md）验证候选。
    不合格返回 (False, reason)。

    判定逻辑在活跃档案包的 constraints 适配器（per-op，档案声明驱动）；引擎侧只
    组装 ctx（soc 常量切片 + dtype 字节表 + 档案 quant_modes/case_classes，契约见
    档案包 constraints.py 模块 docstring）。

    anchor: 该模板的锚点 tiling（extract_anchor 产物，含 blockDim）。
    有锚点时启用 streamK 安全域（约束 7）、afull 覆盖检查（约束 11）与
    合并尾块 stale 检查（约束 14）；无锚点时这三项降级跳过，其余约束不变。
    """
    return _constraints_module().check_candidate(
        candidate, params, anchor, _constraints_ctx()
    )


def filter_candidates(
    candidates: list[dict],
    params: dict,
    anchors: Optional[dict] = None,
) -> tuple[list[dict], dict, int]:
    """
    anchors: {模板名: 锚点 dict}（extract_anchor 产物）。传入时 streamK 候选
    启用安全域检查（tail/blockDim 一致性）；None 时降级为仅通用约束。
    """
    passed = []
    by_template = {}
    failed = 0

    for c in candidates:
        tmpl = normalize_template(c.get("template", "swat"))
        anchor = anchors.get(tmpl) if anchors else None
        ok, reason = check_candidate(c, params, anchor=anchor)
        if ok:
            by_template[tmpl] = by_template.get(tmpl, 0) + 1
            passed.append(c)
        else:
            failed += 1

    return passed, by_template, failed


def validate_rules_against_anchors(
    anchors: Optional[dict], params: dict
) -> dict[str, list[str]]:
    """
    锚点一致性校验（取得模板锚点后调用；判读规则见 skill 工作流）。

    锚点是 forced 包 host 派生链的输出，结构上必然满足全部**派生不变量**
    （对齐/整除/容量/枚举值域——host 自己保证的性质，约束 1-5、7a/7b、
    8-10、12、13 这类）。此类约束拒掉锚点 = 规则读错了派生链，必须修规则。

    两类预期内的例外（不算规则 bug）：
    - afull 且 shape 不满足 A 全载（baseM×K×sizeof_A > L1/2）：锚点不反映
      A-full-load 派生路径，afull 配置类规则不适用，跳过并给出提示
    - 调度器语义类规则（"kernel 在该形态下会坏"，7c/11 型）拒锚点：forced 包
      绕过了 host 的 benefit gate，锚点本身可能是坏配置——锚点探针实测裁决
      （把 TUNE_ 宏设为锚点值跑一次 forced 包）：FAIL → 规则获证；PASS → 证伪

    判定逻辑在活跃档案包的 constraints 适配器；引擎侧组装 ctx（同 check_candidate）。

    返回 {"errors": [...], "warnings": [...]}。只有 errors 阻断严格门禁；
    warnings 用于规则不适用等提示。兼容旧适配器返回 list[str]，列表仅表示错误。
    """
    result = _constraints_module().validate_rules_against_anchors(
        anchors, params, _constraints_ctx()
    )
    if isinstance(result, list):
        return {"errors": result, "warnings": []}
    return {"errors": result.get("errors", []), "warnings": result.get("warnings", [])}


def validate_candidate_count(
    gen_by_template: dict,
    filter_by_template: dict,
    allow_fewer: bool = False,
) -> None:
    """
    校验候选生成数量满足 skill 要求。

    规则：AI 判定可行并生成了候选的模板，生成数必须 ≥ 档案声明的每模板最少候选数
    （constraints.required_candidates_per_template，load_profile 时同步到模块常量
    REQUIRED_CANDIDATES_PER_TEMPLATE）。判 N/A 的模板（gen_by_template 中不存在）
    不受约束——但 N/A 判定必须有定量依据（见 solver.md 主流程第 2 节：afull 全载容量、
    streamK 的 7c 全灭域），l0cpp 不允许判 N/A。

    校验对象是生成数（gen），不是筛选通过数（filter）——约束筛选淘汰是正常机制。

    allow_fewer=True 时跳过校验（显式豁免：AI 主动决策裁剪数量，须在最终报告中注明）。
    不满足且未豁免时抛 ValueError，打印各模板缺口，由 AI 补足候选后重试。
    """
    if allow_fewer:
        print("[validate] allow_fewer=True，跳过候选数量校验（豁免需在最终报告中注明）")
        return

    required = _required_candidates_per_template()
    shortfall = {}
    for tmpl, gen_n in gen_by_template.items():
        if gen_n < required:
            shortfall[tmpl] = gen_n

    if shortfall:
        detail = ", ".join(
            f"{t}: {n}/{required}（缺 {required - n}）"
            for t, n in sorted(shortfall.items())
        )
        raise ValueError(
            f"候选数量不足 skill 要求（每模板 {required} 个）: {detail}。"
            f"请为上述模板补足候选后重新执行 Step 3。"
            f"如需裁剪数量，必须显式传 allow_fewer=True 并在最终报告中注明。"
        )
    print(
        f"[validate] 候选数量校验通过: "
        f"{ {t: n for t, n in sorted(gen_by_template.items())} }"
    )
