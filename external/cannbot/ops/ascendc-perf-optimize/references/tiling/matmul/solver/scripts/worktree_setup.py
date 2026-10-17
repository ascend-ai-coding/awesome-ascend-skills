#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tiling Solver — Git Worktree 创建脚本（算子档案驱动）

声明源 = references/profiles/<op>/profile.json。模板补丁与 kernel 注入点由算子档案
声明；ops-nn 构建命令、normal vendor 命名和安装布局仍属于固定引擎契约，见
references/operator-guide.md：

- templates.<name>.*：强制模板包 worktree 的全部配置——vendor_name /
  vendor_install_dir / worktree.path_suffix/.branch / kernel_targets
  （repo_rel × variants）/ forced_get_kernel_type / forced_is_capable
  （anchor_regex + replacement_template（<RETURN> 占位符）+ marker，通用
  regex 注入器，零 per-template 分支）/ tune_macro_define_block /
  define_block_install（插入规则 + 宏完整性修复）/ tuning_blocks
  （block_text + anchor.mode/regex + marker + post_literal_replacements）
- identity.op_name + soc.name：build.sh 参数 --ops=<op_name> --soc=<soc.name>
- quant_modes.<mode>.*：非 MX 模式安装侧打桩（install 路径 / sed_anchor /
  tune_macro_define_block / tuning_block_template / insert_mode），
  锚点样式与块模板由档案驱动

流程：
1. 从 ops-nn 仓库创建强制模板 worktree（档案 templates 声明，如
   swat/afull/l0cpp/streamk），每个 worktree 包含 op_host
   GetKernelType/IsCapable 强制 early return + kernel .h 的 TUNE_ 宏块
   与 TILING TUNING BLOCK
2. 非 MX 量化模式不走 worktree：安装侧直接打桩（备份 .orig → 断言锚点 →
   插入宏块+调参块 → manifest），扫参脚本 sed 安装侧 kernel .h 的
   #define TUNE_* 行、ttk kernel JIT 生效，跑完 restore-stub 还原
3. normal 包（baseline）不建 worktree：直接 ops-nn 主仓编译，
   vendor_name=custom_normal

使用方式：
    python3 worktree_setup.py --op <op_name> [--ops-nn-dir /path/to/ops-nn]
                              [--jobs 32] [--no-build] [--dry-run]

    --op          算子档案名（候选 = references/profiles/ 下当次生成的档案目录）
    --ops-nn-dir  ops-nn 仓库路径（默认 /workspace/ops-nn）。worktree 创建为
                  其兄弟目录：<仓库目录名><templates.<t>.worktree.path_suffix>
    --jobs        编译并行度（默认 32）
    --no-build    只创建 worktree 与校验 vendor，不启动编译
    --dry-run     只打印将要执行的操作，不实际修改文件

非 MX 打桩子命令（安装侧，无需 worktree/重编包）：
    python3 worktree_setup.py install-stub --mode pertoken [--op <op_name>] [--dry-run]
    python3 worktree_setup.py restore-stub [--mode pertoken] [--op <op_name>] [--dry-run]
    python3 worktree_setup.py stub-status [--mode pertoken] [--op <op_name>]

worktree 创建完成后，脚本自动继续编译安装全部 vendor 包（后台 Popen 启动 +
每 ETA/4 轮询三态判定 + 失败自动重试，与 ttk_runner.py 相同的后台模式）。
编译失败会写 /tmp/build_fail.flag 并在 progress log 留下 FAILED 行与诊断尾巴。
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass

# ============================================================================
# 配置
# ============================================================================

# 机器默认 ops-nn 仓库根。per-op 默认 op 目录 = OPS_NN_DEFAULT + "/" +
# identity.repo_rel（档案声明，如 matmul/quant_batch_matmul_v3）；本脚本的
# 仓内文件路径均直接取档案 forced_* / kernel_targets 声明，不做目录拼接
OPS_NN_DEFAULT = "/workspace/ops-nn"

# API 缺省值仅用于给出统一错误；CLI 必须显式指定 --op。
DEFAULT_OP = ""

# normal baseline 包约定（不建 worktree，直接主仓编译；vendor 名与安装目录）
NORMAL_VENDOR_NAME = "custom_normal"
NORMAL_INSTALL_DIR = "custom_normal_nn"

# vendor 包安装根目录（vendor 存在性校验 / config.ini 写入点）
VENDOR_BASE = "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors"


# ============================================================================
# 档案加载
# ============================================================================


def load_profile(op_name: str) -> dict:
    """
    读取 references/profiles/<op_name>/profile.json（相对 skill 根目录）。

    语义与引擎（ttk_runner.py）的 load_profile 一致：未知算子抛
    FileNotFoundError，报错信息列出当前可用的算子档案。
    """
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(skill_dir, "references", "profiles")
    path = os.path.join(root, op_name, "profile.json")
    if not os.path.isfile(path):
        available = []
        if os.path.isdir(root):
            available = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "profile.json")))
        listing = ", ".join(available) if available else "（无）"
        raise FileNotFoundError(f"未知算子档案 {op_name!r}：{path} 不存在。可用算子档案: {listing}")
    with open(path) as f:
        return json.load(f)


def _available_ops() -> list:
    """references/profiles/ 下存在的算子档案目录名（--op 的候选）。"""
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(skill_dir, "references", "profiles")
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "profile.json")))


# ============================================================================
# Worktree 配置（档案 templates 节驱动）
# ============================================================================


# worktree 配置。所有字段均由 build_worktrees() 从档案 templates.<name> 填充
@dataclass
class WorktreeConfig:
    name: str  # 模板名（= 档案 templates key，如 swat / afull / l0cpp / streamk）
    path_suffix: str  # 兄弟目录后缀，拼在仓库目录名后（worktree.path_suffix 声明）
    branch: str  # 分支名（字符串 → git worktree add -b；空 → detached HEAD）
    ktype_return: str  # GetKernelType early return 语句（forced_get_kernel_type.return_statement）
    is_capable_return: str  # IsCapable return 值（forced_is_capable.return_statement）
    kernel_rel_path: str  # 保留字段（单文件场景）；多变体时以 kernel_targets 为准
    vendor_name: str  # 编译时的 vendor_name（templates.<t>.vendor_name）
    path: str = ""  # 绝对路径，由 resolve_worktree_paths() / create_worktree() 填充
    # (kernel 相对路径, 要插入 TUNING BLOCK 的变体列表)。
    # 变体 = tiling 结构体分支：'without_batch'（紧凑 64B，batchC==1 含 host 折叠）
    # 与 'with_batch'（全量 152B，batchC>1）。swat 两文件各一变体；
    # l0cpp 同文件双变体（qbmm_mx_l0c_pingpong.h 含 WithBatch/WithoutBatch 两个入口）。
    kernel_targets: list = None
    # 档案 templates.<name> 声明原文（forced_* / tune_macro_define_block /
    # define_block_install / tuning_blocks），_patch_worktree_files 消费
    template_decl: dict = None

    def __post_init__(self):
        if self.kernel_targets is None:
            self.kernel_targets = [(self.kernel_rel_path, ["without_batch"])]


def build_worktrees(op: str = DEFAULT_OP) -> list[WorktreeConfig]:
    """
    从档案 templates 节构建 worktree 配置列表。

    - 无 worktree 子声明的模板不建强制包（normal baseline 不在 templates 里，
      它不建 worktree，直接主仓编译）
    - branch=null → detached HEAD；字符串 → git worktree add -b <branch>
    """
    profile = load_profile(op)
    configs = []
    for name, tdecl in profile.get("templates", {}).items():
        wtc = tdecl.get("worktree")
        if not wtc:
            continue
        kts = [(kt["repo_rel"], list(kt["variants"])) for kt in tdecl.get("kernel_targets", [])]
        configs.append(
            WorktreeConfig(
                name=name,
                path_suffix=wtc["path_suffix"],
                branch=wtc.get("branch") or "",
                ktype_return=tdecl["forced_get_kernel_type"]["return_statement"],
                is_capable_return=tdecl["forced_is_capable"]["return_statement"],
                kernel_rel_path=kts[0][0] if kts else "",
                vendor_name=tdecl["vendor_name"],
                kernel_targets=kts,
                template_decl=tdecl,
            )
        )
    return configs


def resolve_worktree_paths(ops_nn_dir: str, op: str = DEFAULT_OP) -> list[WorktreeConfig]:
    """把 build_worktrees(op) 各项的 path 解析为 ops_nn_dir 的兄弟目录
    （<目录名><path_suffix>）。幂等（每次返回新配置列表）。
    """
    root = os.path.abspath(ops_nn_dir).rstrip("/")
    parent, base = os.path.dirname(root), os.path.basename(root)
    configs = build_worktrees(op)
    for wt in configs:
        wt.path = os.path.join(parent, base + wt.path_suffix)
    return configs


# ============================================================================
# 文件修改函数（档案 forced_* / tuning_blocks 声明驱动的通用 regex 注入器）
# ============================================================================


def patch_forced_return(file_path: str, decl: dict, label: str, dry_run: bool = False) -> bool:
    """
    按 forced_get_kernel_type / forced_is_capable 声明做通用 regex 注入：
    在 anchor_regex 命中的函数体 { 后插入 replacement_template
    （<RETURN> 占位符替换为 return_statement）。

    幂等：如果已存在 marker 则跳过。
    label 仅用于日志（GetKernelType / IsCapable）。
    """
    with open(file_path) as f:
        content = f.read()

    if decl["marker"] in content:
        marker_pos = content.index(decl["marker"])
        block = content[marker_pos:marker_pos + 1000]
        if decl["return_statement"] not in block:
            print(f"  [ERROR] {label} marker 存在但 return 声明与算子档案不一致: {file_path}")
            return False
        print(f"  [skip] {label} already patched: {file_path}")
        return True

    replacement = decl["replacement_template"].replace("<RETURN>", decl["return_statement"])
    matches = list(re.finditer(decl["anchor_regex"], content))
    if len(matches) != 1:
        print(f"  [ERROR] {label} anchor matched {len(matches)} times in {file_path}; expected 1")
        return False
    new_content = re.sub(decl["anchor_regex"], replacement, content, count=1)

    if not dry_run:
        with open(file_path, "w") as f:
            f.write(new_content)
    print(f"  [patch] {label} early return: {file_path}")
    return True


def patch_kernel_macros(
    file_path: str, template: str, variants=None, dry_run: bool = False, op: str = DEFAULT_OP
) -> bool:
    """
    在 kernel .h 中插入 TUNE_ 宏定义块 + 各变体的 TILING TUNING BLOCK。

    声明源 = profiles/<op>/profile.json templates.<template>：
    - tune_macro_define_block + define_block_install：define 块文本与插入规则
      （最后一个 #include 组之后；无 #include 退到 #pragma once 后；
      marker 已存在但宏不完整时按声明补齐）
    - tuning_blocks.<variant>：block_text + anchor(mode/regex) + marker +
      post_literal_replacements，按 anchor.mode 路由：
      * replace:       锚点整段替换为块文本
      * insert_before: 块文本 + \\n + 原锚点行
    - TUNE_ 宏 define 块每文件一份（幂等标记 define_block_install.marker）。
    - 每个变体独立幂等插入（tuning_blocks.<variant>.marker）。

    variants: 要插入的变体列表，默认 ['without_batch']。
    op: 算子档案名。
    """
    if variants is None:
        variants = ["without_batch"]
    decl = load_profile(op)["templates"][template]
    install = decl.get("define_block_install", {})
    with open(file_path) as f:
        content = f.read()
    original = content

    # ---- 1. TUNE_ 宏 define 块（每文件一份）----
    if install.get("marker", "TUNE PARAMS") in content:
        print(f"  [skip] Kernel macros already patched: {file_path}")
        # marker 已存在但声明中的宏不完整时按算子档案补齐。
        fixup = install.get("legacy_scale_kl1_fixup")
        if fixup:
            fixup_define = re.search(r"#define (\w+)", fixup["define_replacement"]).group(1)
            if fixup_define not in content:
                print(f"  [fix] Adding missing {fixup_define} to: {file_path}")
                content = re.sub(fixup["define_regex"], fixup["define_replacement"], content)
                content = re.sub(fixup["block_regex"], fixup["block_replacement"], content)
    else:
        tune_defines = decl["tune_macro_define_block"]
        # 找到最后一个 #include 行
        matches = list(re.finditer(install["insert_after_last_include_regex"], content))
        if matches:
            insert_pos = matches[-1].end()
            content = content[:insert_pos] + "\n" + tune_defines + "\n" + content[insert_pos:]
        else:
            # fallback: 在 #pragma once 后插入
            content = re.sub(
                install["fallback_after_pragma_once_regex"], r"\1\n" + tune_defines + "\n", content, count=1
            )

    # ---- 2. 各变体的 TILING TUNING BLOCK（独立幂等）----
    # 锚点与块文本全部档案声明（tuning_blocks.<variant>）；锚点不命中必须
    # 报错（define 块插入成功会掩盖块缺失，旋钮存在但无消费块 = 扫参参数
    # 静默不生效，比无 patch 更危险）。
    block_failed = False
    for variant in variants:
        vdecl = decl.get("tuning_blocks", {}).get(variant)
        if vdecl is None:
            print(f"  [ERROR] template '{template}' 无 {variant} 变体（host 拒绝 batch）")
            block_failed = True
            continue
        marker = vdecl["marker"]
        if marker in content:
            if vdecl["block_text"] not in content:
                print(f"  [ERROR] {variant} tuning block marker 存在但内容与算子档案不一致: {file_path}")
                block_failed = True
                continue
            print(f"  [skip] {variant} tuning block already present: {file_path}")
            continue
        block = vdecl["block_text"]
        anchor = vdecl["anchor"]
        if anchor["mode"] == "replace":
            content = re.sub(anchor["regex"], lambda _m: block, content, count=1)
        elif anchor["mode"] == "insert_before":
            content = re.sub(anchor["regex"], lambda m: block + "\n    " + m.group(0), content, count=1)
        else:
            print(f"  [ERROR] 未知 anchor mode {anchor['mode']!r}: {file_path}")
            block_failed = True
            continue
        for lit_old, lit_new in vdecl.get("post_literal_replacements", {}).items():
            content = content.replace(lit_old, lit_new)
        # 锚点命中校验：re.sub 不命中时 content 不变、marker 缺失——必须暴露
        # （define 块插入成功会掩盖块缺失，旋钮存在但无消费块 = 扫参静默不生效）
        if marker not in content:
            print(
                f"  [ERROR] {variant} tuning block anchor NOT FOUND in: {file_path}"
                f"（文件声明风格与锚点不匹配，TUNE 旋钮不会生效）"
            )
            block_failed = True

    if block_failed:
        print(f"  [ERROR] 部分变体块插入失败（见上），该文件 patch 不完整: {file_path}")
        return False

    expected_macros = re.findall(r"#define\s+(TUNE_\w+)", decl["tune_macro_define_block"])
    bad_macros = [m for m in expected_macros if len(re.findall(rf"(?m)^\s*#define\s+{re.escape(m)}\b", content)) != 1]
    missing_markers = [
        decl["tuning_blocks"][v]["marker"]
        for v in variants
        if decl["tuning_blocks"][v]["marker"] not in content
    ]
    if bad_macros or missing_markers:
        print(f"  [ERROR] patch 后验失败: macros={bad_macros}, markers={missing_markers}")
        return False

    if content != original and not dry_run:
        with open(file_path, "w") as f:
            f.write(content)
    action = "patch" if content != original else "ok"
    print(f"  [{action}] Kernel TUNE_ macros + TUNING BLOCK ({'+'.join(variants)}): {file_path}")
    return True


# ============================================================================
# 安装侧声明驱动打桩（非 MX 模式；声明源：profiles/<op>/profile.json quant_modes）
# ============================================================================
# 生命周期: install-stub → 扫参脚本 sed #define TUNE_* 行 → restore-stub
#   install: 备份 .orig → 锚点断言（命中次数必须==1，漂移即报错）→ 插入
#            define 块（最后一个 #include 后，同强制包规则）+ 调参块（按
#            insert_mode: replace=锚点行整行替换 / after=锚点行后插入）→
#            写 manifest（<file>.stub.json，记录占用模式）
#   restore: cp .orig → 原文件 → 重读比对（必须逐字节一致）→ 清理
#            .orig 与 manifest，现场归零
# 注入前断言锚点唯一，注入后校验宏计数与块标记，还原后逐字节比对。
# 任一检查失败都终止，避免把未生效的候选记为有效结果。
# 同文件跨模式占用: manifest 记录注入模式，他模式注入必须先 restore。
#   当前档案形态: 同物理文件的多 entry 共享同一锚点串（实证），每文件
#   注入一份 define 块 + 一个调参块。
# MX 不走此路径（worktree + vendor 包编译），install-stub --mode mx 报错。

STUB_DEFINE_MARKER = "TUNE PARAMS"
STUB_BLOCK_MARKER = "TILING TUNING BLOCK"
STUB_MANIFEST_SUFFIX = ".stub.json"


def _load_stub_profile(op: str = DEFAULT_OP) -> dict:
    """安装侧打桩的声明源（同引擎 load_profile，读 profiles/<op>/profile.json）。"""
    return load_profile(op)


def _read_stub_manifest(path: str):
    mp = path + STUB_MANIFEST_SUFFIX
    if not os.path.exists(mp):
        return None
    try:
        with open(mp) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def stub_file_status(path: str) -> str:
    """单文件打桩状态:
    clean=未打桩 / stubbed=已注入(.orig+manifest) / inconsistent=现场损坏 /
    missing=文件不存在。
    """
    if not os.path.exists(path):
        return "missing"
    has_orig = os.path.exists(path + ".orig")
    has_manifest = os.path.exists(path + STUB_MANIFEST_SUFFIX)
    if has_orig and has_manifest:
        return "stubbed"
    if has_orig or has_manifest:
        return "inconsistent"
    with open(path, "rb") as f:
        if STUB_DEFINE_MARKER.encode() in f.read():
            return "inconsistent"  # 有注入痕迹但无备份/清单（手工注入残留）
    return "clean"


def install_stub_file(
    path: str, mode: str, sed_anchor: str, define_block: str, tuning_block: str, insert_mode: str, dry_run: bool = False
) -> bool:
    """单文件注入（字节级读写，防非 ASCII 内容损坏）。"""
    stub_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "mode": mode,
                "path": path,
                "sed_anchor": sed_anchor,
                "define_block": define_block,
                "tuning_block": tuning_block,
                "insert_mode": insert_mode,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    status = stub_file_status(path)
    if status == "missing":
        print(f"  [ERROR] 文件不存在: {path}")
        return False
    if status == "stubbed":
        manifest = _read_stub_manifest(path)
        owner = (manifest or {}).get("mode")
        if owner != mode:
            print(f"  [ERROR] {path} 已被模式 {owner!r} 占用——先 restore-stub 再注入")
            return False
        if manifest.get("stub_fingerprint") != stub_fingerprint:
            print(f"  [ERROR] {path} 已注入内容与当前算子档案不一致——先 restore-stub 再重新注入")
            return False
        with open(path, "rb") as f:
            current = f.read()
        if define_block.encode().rstrip(b"\n") not in current or tuning_block.encode().rstrip(b"\n") not in current:
            print(f"  [ERROR] {path} 已注入内容不完整——先 restore-stub 再重新注入")
            return False
        print(f"  [skip] 已注入（mode={mode}）: {path}")
        return True
    if status == "inconsistent":
        print(
            f"  [ERROR] {path} 现场不一致（.orig/manifest 单边缺失或有注入痕迹无备份）——"
            f"人工核查清理 .orig/.stub.json 后再注入"
        )
        return False

    with open(path, "rb") as f:
        original = f.read()
    anchor_b = sed_anchor.encode()

    # ---- 断言 1: 锚点命中次数 == 1（sed 静默空转的第一道防线）----
    n_hits = original.count(anchor_b)
    if n_hits != 1:
        reason = (
            "锚点不命中——CANN 升级/文件版本漂移，重新对账并更新算子档案"
            if n_hits == 0
            else "锚点多义——档案 sed_anchor 不唯一"
        )
        print(f"  [ERROR] {path} 锚点命中 {n_hits} 次（应为 1）: {reason}")
        return False

    # ---- define 块: 最后一个 #include 之后（与强制包 patch_kernel_macros 同规则）----
    content = original
    define_b = define_block.encode()
    include_matches = list(re.finditer(rb'((?:#include "[^"]+"\n)+)', content))
    if include_matches:
        pos = include_matches[-1].end()
        content = content[:pos] + b"\n" + define_b.rstrip(b"\n") + b"\n" + content[pos:]
    else:
        content = re.sub(rb"(#pragma once\n)", rb"\1\n" + define_b.rstrip(b"\n") + b"\n", content, count=1)

    # ---- 调参块: 按 insert_mode ----
    block_b = tuning_block.encode()
    if insert_mode == "replace":
        content = content.replace(anchor_b, block_b.rstrip(b"\n"), 1)
    elif insert_mode == "after":
        content = content.replace(anchor_b, anchor_b + b"\n" + block_b.rstrip(b"\n"), 1)
    else:
        print(f"  [ERROR] 未知 insert_mode={insert_mode!r}: {path}")
        return False

    # ---- 断言 2: 宏计数与块标记后验 ----
    expected = define_b.count(b"#define TUNE_")
    actual = content.count(b"#define TUNE_")
    if actual != expected:
        print(f"  [ERROR] {path} 宏计数后验失败: 期望 {expected} 实得 {actual}")
        return False
    if STUB_DEFINE_MARKER.encode() not in content or STUB_BLOCK_MARKER.encode() not in content:
        print(f"  [ERROR] {path} 块标记后验失败（define/tuning block 未落入文件）")
        return False

    if dry_run:
        print(f"  [dry-run] 将注入 mode={mode}: {path}（macros={expected}, insert={insert_mode}）")
        return True

    # ---- 备份 → 写回 → manifest ----
    with open(path + ".orig", "wb") as f:
        f.write(original)
    with open(path, "wb") as f:
        f.write(content)
    manifest = {
        "mode": mode,
        "stubbed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tune_macros": expected,
        "insert_mode": insert_mode,
        "sed_anchor": sed_anchor,
        "stub_fingerprint": stub_fingerprint,
    }
    with open(path + STUB_MANIFEST_SUFFIX, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"  [stub] mode={mode} 注入 {path}（macros={expected}, insert={insert_mode}）")
    return True


def restore_stub_file(path: str, dry_run: bool = False) -> bool:
    """还原单文件: cp .orig → 原文件，重读逐字节比对，清理 .orig/manifest。"""
    orig = path + ".orig"
    if not os.path.exists(orig):
        print(f"  [skip] 无备份（未打桩）: {path}")
        return True
    if dry_run:
        print(f"  [dry-run] 将还原: {path}")
        return True
    with open(orig, "rb") as f:
        backup = f.read()
    with open(path, "wb") as f:
        f.write(backup)
    with open(path, "rb") as f:
        current = f.read()
    if current != backup:
        print(f"  [ERROR] {path} 还原后与备份不一致（写回失败?）")
        return False
    os.remove(orig)
    mp = path + STUB_MANIFEST_SUFFIX
    if os.path.exists(mp):
        os.remove(mp)
    print(f"  [restore] {path}")
    return True


def _stub_modes(profile: dict) -> list:
    """档案内可安装侧打桩的量化模式（kernel_files entry 携带 sed_anchor 声明，
    即非 MX 模式；MX 走 worktree 方案）。"""

    def stubable(cfg) -> bool:
        return any(
            entry.get("sed_anchor") for tmpls in cfg.get("kernel_files", {}).values() for entry in tmpls.values()
        )

    return sorted(m for m, cfg in profile.get("quant_modes", {}).items() if stubable(cfg))


def install_stub(mode: str, dry_run: bool = False, op: str = DEFAULT_OP) -> bool:
    """按档案声明对 mode 的全部 kernel 文件做安装侧打桩（物理文件去重）。

    op: 算子档案名。
    """
    profile = _load_stub_profile(op)
    qm = profile.get("quant_modes", {})
    if mode == "mx":
        print("[ERROR] MX 走 worktree 方案（本脚本默认流程 + vendor 包编译），不用 install-stub")
        return False
    if mode not in qm:
        print(f"[ERROR] 未知模式 {mode!r}，档案现有: {sorted(qm)}")
        return False
    mode_cfg = qm[mode]
    define_block = mode_cfg.get("tune_macro_define_block")
    tuning_block = mode_cfg.get("tuning_block_template")
    if not define_block or not tuning_block:
        print(f"[ERROR] 模式 {mode} 档案缺 tune_macro_define_block / tuning_block_template 声明")
        return False

    print(f"[stub] install-stub mode={mode} dry_run={dry_run}")
    ok = True
    seen_paths = set()
    for cc, tmpls in sorted(mode_cfg.get("kernel_files", {}).items()):
        for tmpl, entry in sorted(tmpls.items()):
            im = entry.get("insert_mode")
            if im == "none":
                print(f"  [skip] {cc}/{tmpl}: insert_mode=none（声明为不可注入）")
                continue
            if not im:
                print(f"  [ERROR] {cc}/{tmpl} 缺 insert_mode 声明（档案不完整）")
                ok = False
                continue
            path = entry["install"]
            if path in seen_paths:
                continue  # 同物理文件多 entry 共享锚点（实证），一份注入即可
            seen_paths.add(path)
            if not install_stub_file(path, mode, entry["sed_anchor"], define_block, tuning_block, im, dry_run):
                ok = False
    if ok:
        print(f"[stub] 完成——扫参脚本 sed #define TUNE_* 行生效；跑完必须 restore-stub")
    return ok


def restore_stub(mode: str = None, dry_run: bool = False, op: str = DEFAULT_OP) -> bool:
    """还原打桩现场。mode=None 时扫描档案所有模式的全部 kernel 文件。

    op: 算子档案名。
    """
    profile = _load_stub_profile(op)
    qm = profile.get("quant_modes", {})
    if mode == "mx":
        print("[skip] MX 无安装侧打桩")
        return True
    if mode and mode not in qm:
        print(f"[ERROR] 未知模式 {mode!r}，档案现有: {sorted(qm)}")
        return False
    modes = [mode] if mode else sorted(qm)
    print(f"[stub] restore-stub modes={modes} dry_run={dry_run}")
    ok = True
    seen = set()
    for m in modes:
        for cc, tmpls in sorted(qm[m].get("kernel_files", {}).items()):
            for tmpl, entry in sorted(tmpls.items()):
                path = entry["install"]
                if path in seen:
                    continue
                seen.add(path)
                if not restore_stub_file(path, dry_run):
                    ok = False
    if ok:
        print("[stub] 现场已还原（.orig/manifest 已清理）")
    return ok


def stub_status(mode: str = None, op: str = DEFAULT_OP) -> None:
    """打印档案内全部（或指定模式）kernel 文件的打桩状态。

    op: 算子档案名。
    """
    profile = _load_stub_profile(op)
    qm = profile.get("quant_modes", {})
    modes = [mode] if mode and mode in qm else sorted(qm)
    seen = set()
    print(f"{'file':<50} {'stub-mode':<16} status")
    print("-" * 80)
    for m in modes:
        for cc, tmpls in sorted(qm[m].get("kernel_files", {}).items()):
            for tmpl, entry in sorted(tmpls.items()):
                path = entry["install"]
                if path in seen:
                    continue
                seen.add(path)
                owner = (_read_stub_manifest(path) or {}).get("mode", "-")
                print(f"{os.path.basename(path):<50} {owner:<16} {stub_file_status(path)}")


# ============================================================================
# Worktree 创建
# ============================================================================


def create_worktree(ops_nn_dir: str, config: WorktreeConfig, dry_run: bool = False, op: str = DEFAULT_OP) -> bool:
    """创建单个 worktree 并做文件修改。

    config.path 按 path_suffix 幂等解析为 ops_nn_dir 的兄弟目录。
    op: 算子档案名，透传给 _patch_worktree_files / patch_kernel_macros。
    """
    root = os.path.abspath(ops_nn_dir).rstrip("/")
    config.path = os.path.join(os.path.dirname(root), os.path.basename(root) + config.path_suffix)
    print(f"\n{'=' * 60}")
    print(f"创建 worktree: {config.name} → {config.path}")
    print(f"{'=' * 60}")

    # 检查 worktree 是否已存在
    if os.path.exists(config.path):
        print(f"  [skip] Worktree already exists: {config.path}")
        return _patch_worktree_files(config, dry_run, op=op)

    # git worktree add
    if config.branch:
        cmd = ["git", "-C", ops_nn_dir, "worktree", "add", "-b", config.branch, config.path]
    else:
        cmd = ["git", "-C", ops_nn_dir, "worktree", "add", "--detach", config.path]

    print(f"  [git] {shlex.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] git worktree add failed: {result.stderr}")
        return False

    # 修改文件
    return _patch_worktree_files(config, dry_run, op=op)


def _patch_worktree_files(config: WorktreeConfig, dry_run: bool = False, op: str = DEFAULT_OP) -> bool:
    """按档案声明修改 worktree 中的 op_host 强制点与 kernel 文件。

    声明源 = templates.<config.name>：forced_get_kernel_type / forced_is_capable
    （file 字段为仓内相对路径）/ kernel_targets（repo_rel × variants）。"""
    wt = config.path
    decl = config.template_decl or load_profile(op)["templates"][config.name]
    ok = True

    # 1. GetKernelType 强制 early return（forced_get_kernel_type 声明）
    gk_path = os.path.join(wt, decl["forced_get_kernel_type"]["file"])
    if os.path.exists(gk_path):
        ok = patch_forced_return(gk_path, decl["forced_get_kernel_type"], "GetKernelType", dry_run) and ok
    else:
        print(f"  [ERROR] File not found: {gk_path}")
        ok = False

    # 2. IsCapable 强制 early return（forced_is_capable 声明）
    ic_path = os.path.join(wt, decl["forced_is_capable"]["file"])
    if os.path.exists(ic_path):
        ok = patch_forced_return(ic_path, decl["forced_is_capable"], "IsCapable", dry_run) and ok
    else:
        print(f"  [ERROR] File not found: {ic_path}")
        ok = False

    # 3. 修改 kernel .h（按 kernel_targets: 文件 × 变体列表）
    for rel_path, variants in config.kernel_targets or [(config.kernel_rel_path, ["without_batch"])]:
        kernel_path = os.path.join(wt, rel_path)
        if os.path.exists(kernel_path):
            ok = patch_kernel_macros(kernel_path, config.name, variants=variants, dry_run=dry_run, op=op) and ok
        else:
            print(f"  [ERROR] File not found: {kernel_path}")
            ok = False
    return ok


# ============================================================================
# 编译安装命令生成
# ============================================================================


def print_build_instructions(ops_nn_dir: str, op: str = DEFAULT_OP) -> None:
    """打印编译安装命令（仅 --dry-run 模式使用）。

    build.sh 参数档案驱动：--soc=<soc.name> --ops=<identity.op_name>。
    """
    profile = load_profile(op)
    soc = profile["soc"]["name"]
    op_name = profile["identity"]["op_name"]
    worktrees = resolve_worktree_paths(ops_nn_dir, op)
    q = shlex.quote
    print(f"\n{'=' * 60}")
    print("编译安装命令（dry-run 模式，实际运行时由 run_build_all 自动执行）")
    print(f"{'=' * 60}\n")

    # normal 包
    print(f"# normal 包（baseline）")
    print(
        f"cd {q(ops_nn_dir)} && bash build.sh --pkg --jit --no_force --soc={q(soc)} "
        f"--vendor_name={q(NORMAL_VENDOR_NAME)} --ops={q(op_name)} -j<jobs> 2>&1"
    )
    package = os.path.join(ops_nn_dir, "build_out", f"cann-ops-nn-{NORMAL_VENDOR_NAME}_linux-x86_64.run")
    print(f"bash {q(package)} --quiet\n")

    # 强制模板包
    for config in worktrees:
        print(f"# {config.name} 包")
        print(
            f"cd {q(config.path)} && bash build.sh --pkg --jit --no_force --soc={q(soc)} "
            f"--vendor_name={q(config.vendor_name)} --ops={q(op_name)} -j<jobs> 2>&1"
        )
        run_file = os.path.join(config.path, "build_out", f"cann-ops-nn-{config.vendor_name}_linux-x86_64.run")
        print(f"bash {q(run_file)} --quiet\n")

    # 验证
    print(f"# 验证 {len(worktrees) + 1} 个包都安装成功")
    install_dirs = [NORMAL_INSTALL_DIR] + [wt.template_decl["vendor_install_dir"] for wt in worktrees]
    print("ls -d " + " ".join(q(os.path.join(VENDOR_BASE, d)) for d in install_dirs))
    print(f'echo "load_priority={NORMAL_INSTALL_DIR}" > {q(os.path.join(VENDOR_BASE, "config.ini"))}\n')


# ============================================================================
# 编译安装自动执行（后台 Popen + 每 ETA/4 轮询 + 三态判定 + 失败重试）
# ============================================================================

PER_PKG_BUILD_SECONDS = 10 * 60  # JIT 模式每包约 10 分钟（host-only，kernel 不编入包、由 ttk 运行时 JIT）

BUILD_PROGRESS_LOG = "/tmp/build_progress.log"
BUILD_DONE_FLAG = "/tmp/build_done.flag"
BUILD_FAIL_FLAG = "/tmp/build_fail.flag"
BUILD_PID_FILE = "/tmp/build.pid"
BUILD_OUTPUT_LOG = "/tmp/build_output.log"
BUILD_STAMP_FILE = ".matmul-tiling-solver-build.json"


def estimate_build_eta(n_packages: int = 5) -> int:
    """估算编译总耗时（秒），JIT 模式每包约 10 分钟（host-only）"""
    return PER_PKG_BUILD_SECONDS * n_packages


def _source_fingerprint(repo_dir: str, profile: dict, vendor_name: str) -> str:
    """对会影响包内容的版本、源码 diff 和构建声明取指纹。"""
    repo_rel = profile["identity"]["repo_rel"]
    try:
        head = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", repo_dir, "diff", "--binary", "HEAD", "--", "build.sh", repo_rel],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        status = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=all", "--", "build.sh", repo_rel],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"无法计算构建源指纹: {repo_dir}: {e.stderr.strip()}") from e
    untracked = {}
    for line in status.splitlines():
        if line.startswith("?? "):
            rel = line[3:]
            path = os.path.join(repo_dir, rel)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    untracked[rel] = hashlib.sha256(f.read()).hexdigest()
    payload = {
        "head": head.strip(),
        "diff": diff,
        "untracked": untracked,
        "op": profile["identity"]["op_name"],
        "soc": profile["soc"]["name"],
        "vendor": vendor_name,
        "profile": profile,
        "build_flags": ["--pkg", "--jit", "--no_force"],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_specs(ops_nn_dir: str, op: str) -> list[dict]:
    """返回 normal 与 forced 包的路径、安装目录和可复用指纹。"""
    profile = load_profile(op)
    specs = [
        {
            "name": "normal",
            "dir": ops_nn_dir,
            "vendor": NORMAL_VENDOR_NAME,
            "install_dir": NORMAL_INSTALL_DIR,
        }
    ]
    for config in resolve_worktree_paths(ops_nn_dir, op):
        specs.append(
            {
                "name": config.name,
                "dir": config.path,
                "vendor": config.vendor_name,
                "install_dir": config.template_decl["vendor_install_dir"],
            }
        )
    for spec in specs:
        spec["fingerprint"] = _source_fingerprint(spec["dir"], profile, spec["vendor"])
        spec["stamp"] = json.dumps(
            {
                "op": profile["identity"]["op_name"],
                "soc": profile["soc"]["name"],
                "vendor": spec["vendor"],
                "source_fingerprint": spec["fingerprint"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return specs


def _stamp_matches(path: str, expected: str) -> bool:
    try:
        with open(path) as f:
            return f.read().strip() == expected
    except OSError:
        return False


def _verify_repo_version(ops_nn_dir: str, profile: dict) -> None:
    """在创建 worktree 前校验算子仓 HEAD 与算子档案的版本锚点。"""
    expected = profile.get("version_anchor", {}).get("op_repo_commit")
    if not expected:
        raise RuntimeError("算子档案字段 version_anchor.op_repo_commit 缺失")
    try:
        actual = subprocess.run(
            ["git", "-C", ops_nn_dir, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"无法读取算子仓版本: {ops_nn_dir}: {e.stderr.strip()}") from e
    if actual != expected:
        raise RuntimeError(
            f"算子档案的版本锚点与算子仓不一致: 档案={expected}, 仓库={actual}。请先对账并重新生成算子档案。"
        )


def generate_build_script(
    ops_nn_dir: str,
    jobs: int = 32,
    progress_log: str = BUILD_PROGRESS_LOG,
    done_flag: str = BUILD_DONE_FLAG,
    fail_flag: str = BUILD_FAIL_FLAG,
    output_log: str = BUILD_OUTPUT_LOG,
    op: str = DEFAULT_OP,
) -> str:
    """
    生成编译+安装全部 vendor 包（normal + 档案 templates 各强制包）的 bash 脚本。

    build.sh 参数档案驱动：--soc=<soc.name> --ops=<identity.op_name>。
    只复用源码指纹一致的 .run 产物。
    """
    profile = load_profile(op)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError(f"jobs 必须是正整数，实际为 {jobs!r}")
    soc = profile["soc"]["name"]
    op_name = profile["identity"]["op_name"]
    packages = _package_specs(ops_nn_dir, op)
    script_path = "/tmp/run_build.sh"
    vendor_base = VENDOR_BASE
    q = shlex.quote

    lines = [
        "#!/bin/bash",
        "# Matmul tiling vendor build runner — generated by worktree_setup.py",
        f"PROGRESS={q(progress_log)}",
        f"DONE={q(done_flag)}",
        f"FAIL={q(fail_flag)}",
        f"OUTPUT={q(output_log)}",
        f"JOBS={jobs}",
        '> "$PROGRESS"',
        'rm -f "$DONE" "$FAIL"',
        "",
        "# 清理异常中断遗留的编译锁。",
        f"find {q(os.path.join(ops_nn_dir, 'scripts/kernel/binary_script'))} "
        "-maxdepth 1 -type f -name '*.lock' -delete 2>/dev/null || true",
    ]
    for spec in packages[1:]:
        lock_dir = os.path.join(spec["dir"], "scripts/kernel/binary_script")
        lines.append(f"find {q(lock_dir)} -maxdepth 1 -type f -name '*.lock' -delete 2>/dev/null || true")
    lines.extend(
        [
            "",
            "# 单包编译+安装。失败时：写 FAILED 行 + output 尾部诊断 + fail flag，然后退出。",
            "# 跨算子、跨版本或未打完补丁的产物不会被复用。",
            "build_one() {",
            "    local name=$1 dir=$2 vendor=$3 install_dir=$4 expected_stamp=$5 rc=0",
            "    local run_file=$dir/build_out/cann-ops-nn-${vendor}_linux-x86_64.run",
            "    local run_stamp=${run_file}.solver-stamp",
            f"    local install_path={q(vendor_base)}/$install_dir",
            f"    local install_stamp=$install_path/{BUILD_STAMP_FILE}",
            '    echo "start $name" >> "$PROGRESS"',
            '    if [ -f "$run_file" ] && [ -f "$run_stamp" ] && '
            '       [ "$(cat "$run_stamp")" = "$expected_stamp" ]; then',
            '        echo "skip compile (fingerprint matched): $name" >> "$PROGRESS"',
            "    else",
            '        rm -f "$run_file" "$run_stamp"',
            f'        (cd "$dir" && bash build.sh --pkg --jit --no_force --soc={q(soc)} '
            f'--vendor_name="$vendor" --ops={q(op_name)} -j"$JOBS" >> "$OUTPUT" 2>&1) || rc=$?',
            "        if [ $rc -ne 0 ]; then",
            '            echo "FAILED $name compile rc=$rc" >> "$PROGRESS"',
            '            echo "---- last 40 lines of $OUTPUT ----" >> "$PROGRESS"',
            '            tail -n 40 "$OUTPUT" >> "$PROGRESS"',
            '            touch "$FAIL"',
            "            exit 1",
            "        fi",
            '        if [ ! -f "$run_file" ]; then',
            '            echo "FAILED $name compile produced no run file" >> "$PROGRESS"',
            '            touch "$FAIL"',
            "            exit 1",
            "        fi",
            '        printf "%s\\n" "$expected_stamp" > "$run_stamp"',
            "    fi",
            "    rc=0",
            '    bash "$run_file" --quiet >> "$OUTPUT" 2>&1 || rc=$?',
            "    if [ $rc -ne 0 ]; then",
            '        echo "FAILED $name install rc=$rc" >> "$PROGRESS"',
            '        echo "---- last 40 lines of $OUTPUT ----" >> "$PROGRESS"',
            '        tail -n 40 "$OUTPUT" >> "$PROGRESS"',
            '        touch "$FAIL"',
            "        exit 1",
            "    fi",
            '    if [ ! -d "$install_path" ]; then',
            '        echo "FAILED $name install directory missing: $install_path" >> "$PROGRESS"',
            '        touch "$FAIL"',
            "        exit 1",
            "    fi",
            '    printf "%s\\n" "$expected_stamp" > "$install_stamp"',
            '    echo "done $name" >> "$PROGRESS"',
            "}",
            "",
        ]
    )

    for i, spec in enumerate(packages, 1):
        lines.append(f'echo "=== package {i}/{len(packages)}: {spec["name"]} ===" >> "$PROGRESS"')
        lines.append(
            "build_one " + " ".join(q(str(spec[k])) for k in ("name", "dir", "vendor", "install_dir", "stamp"))
        )
        lines.append("")

    lines.append(f"# === 验证 {len(packages)} 个 vendor 目录与构建指纹 ===")
    for spec in packages:
        install_path = os.path.join(vendor_base, spec["install_dir"])
        stamp_path = os.path.join(install_path, BUILD_STAMP_FILE)
        lines.extend(
            [
                f"if [ ! -d {q(install_path)} ] || [ ! -f {q(stamp_path)} ] || "
                f'[ "$(cat {q(stamp_path)})" != {q(spec["stamp"])} ]; then',
                f'    echo "FAILED verify {spec["name"]} fingerprint mismatch" >> "$PROGRESS"',
                '    touch "$FAIL"',
                "    exit 1",
                "fi",
            ]
        )
    lines.extend(
        [
            f'echo "load_priority={NORMAL_INSTALL_DIR}" > {q(os.path.join(vendor_base, "config.ini"))}',
            'echo "ALL_DONE" >> "$PROGRESS"',
            'touch "$DONE"',
        ]
    )

    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def check_vendors_exist(op: str = DEFAULT_OP, ops_nn_dir: str = OPS_NN_DEFAULT) -> bool:
    """
    检测 vendor 包（normal baseline + 档案 templates 各强制包）是否已安装。

    返回 True 表示目录存在且构建指纹与当前源码一致。
    """
    stale = []
    specs = _package_specs(ops_nn_dir, op)
    for spec in specs:
        install_path = os.path.join(VENDOR_BASE, spec["install_dir"])
        stamp_path = os.path.join(install_path, BUILD_STAMP_FILE)
        if not os.path.isdir(install_path) or not _stamp_matches(stamp_path, spec["stamp"]):
            stale.append(spec["name"])
    if stale:
        print(f"[build] 缺失或过期的 vendor 包: {', '.join(stale)}")
        return False
    print(f"[build] {len(specs)} 个 vendor 包指纹匹配，跳过编译")
    # 确保 config.ini 指向 normal
    config_ini = os.path.join(VENDOR_BASE, "config.ini")
    with open(config_ini, "w") as f:
        f.write(f"load_priority={NORMAL_INSTALL_DIR}\n")
    print(f"[build] config.ini → {NORMAL_INSTALL_DIR}")
    return True


def _count_need_build(ops_nn_dir: str, op: str = DEFAULT_OP) -> int:
    """统计缺少可复用 .run 产物的包数。"""
    return sum(
        1
        for spec in _package_specs(ops_nn_dir, op)
        if not _stamp_matches(
            os.path.join(spec["dir"], "build_out", f"cann-ops-nn-{spec['vendor']}_linux-x86_64.run.solver-stamp"),
            spec["stamp"],
        )
    )


def start_build(ops_nn_dir: str, jobs: int = 32, op: str = DEFAULT_OP) -> int:
    """
    生成编译脚本并在后台启动（三步式第 1 步）。
    返回预估总耗时（秒），AI 据此以 ETA/4 间隔轮询 check_build_status()。

    op: 算子档案名，决定 --soc/--ops 参数与包列表。
    """
    for f in [BUILD_PROGRESS_LOG, BUILD_DONE_FLAG, BUILD_FAIL_FLAG, BUILD_PID_FILE, BUILD_OUTPUT_LOG]:
        if os.path.exists(f):
            os.remove(f)

    script_path = generate_build_script(ops_nn_dir, jobs=jobs, op=op)

    need_build = _count_need_build(ops_nn_dir, op=op)
    eta = estimate_build_eta(max(need_build, 1))

    # 子进程退出后由内核直接回收（不留 zombie），避免调用方进程长驻时
    # kill(pid,0)/proc 检测被僵尸进程干扰。本函数后续无 wait 类调用，安全。
    import signal

    try:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    except (ValueError, OSError):
        pass  # 非主线程等场景忽略，zombie 由 /proc state 检测兜底

    with open(BUILD_OUTPUT_LOG, "w") as log_fd:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid：脱离会话，父进程退出不影响
        )
    with open(BUILD_PID_FILE, "w") as f:
        f.write(str(proc.pid))

    interval = max(eta // 4, 60)
    print(f"[build] 已后台启动 (pid={proc.pid}, jobs={jobs}, 需编译 {need_build} 包)")
    print(f"[build] 预计 {eta}s（约 {eta // 60} 分钟），建议每 {interval}s（ETA/4）检查一次")
    print(f"[build] 轮询: check_build_status() → running/done/failed/dead")
    return eta


def _process_alive(pid: int) -> bool:
    """
    判定 PID 对应进程是否真实存活。
    """
    try:
        os.kill(pid, 0)  # 信号 0：仅做存在性检查，不投递信号
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但属他人（PID 复用的极端情况），按存活处理

    try:
        with open(f"/proc/{pid}/stat") as f:
            # 格式: pid (comm) state ...；comm 可能含空格/括号，从最后一个 ')' 后取
            state = f.read().split(")")[-1].split()[0]
        return state != "Z"  # zombie = 已退出待回收，视为死亡
    except (FileNotFoundError, IndexError):
        # /proc 读取失败：进程刚退出（竞态），视为死亡
        return False


def check_build_status(op: str = DEFAULT_OP) -> str:
    """
    三态判定编译任务状态（三步式第 2 步）。

    op: 算子档案名。仅为与 start_build/collect_build_result 的三步式调用
    签名保持一致而接受——状态文件是全局 /tmp 路径，与算子无关。

    返回：
    - "done"            : 全部包编译安装成功（done flag 存在）
    - "failed"          : 脚本显式报告失败（fail flag 存在，含失败包名与诊断）
    - "dead"            : 脚本进程已消失但无任何 flag（被 kill -9/OOM 杀死、
                          僵尸、或 PID 复用等异常场景）
    - "running"         : 正在编译
    """
    if os.path.exists(BUILD_DONE_FLAG):
        return "done"
    if os.path.exists(BUILD_FAIL_FLAG):
        return "failed"

    if not os.path.exists(BUILD_PID_FILE):
        return "dead"  # 从未启动过或 PID 文件被清理

    try:
        with open(BUILD_PID_FILE) as f:
            pid = int(f.read().strip().split()[0])
    except ValueError:
        return "dead"

    if _process_alive(pid):
        return "running"
    return "dead"  # 进程不存在/僵尸且没有 done/fail flag → 异常死亡


def _print_failure_diagnosis() -> None:
    """失败时从 progress log 提取 FAILED 行和诊断尾部。"""
    if not os.path.exists(BUILD_PROGRESS_LOG):
        print("[build] 无 progress log")
        return
    with open(BUILD_PROGRESS_LOG) as f:
        content = f.read()
    failed_lines = [ln for ln in content.split("\n") if ln.startswith("FAILED")]
    done_lines = [ln for ln in content.split("\n") if ln.startswith("done ")]
    print(f"[build] 已完成包: {done_lines if done_lines else '无'}")
    print(f"[build] 失败点: {failed_lines if failed_lines else '未知（进程异常死亡）'}")
    # 打印最后一个 FAILED 行之后的诊断尾巴
    if failed_lines:
        idx = content.rfind(failed_lines[-1])
        tail = content[idx:idx + 2000]
        print(f"[build] 诊断摘要:\n{tail}")


def collect_build_result(op: str = DEFAULT_OP, ops_nn_dir: str = OPS_NN_DEFAULT) -> bool:
    """
    编译结束后验证 vendor 目录并写 config.ini（三步式第 3 步）。

    在 check_build_status() 返回 done/failed/dead 后调用。
    返回 True 表示全部 vendor 包就绪且指纹匹配。
    """
    specs = _package_specs(ops_nn_dir, op)
    all_ok = True
    for spec in specs:
        path = os.path.join(VENDOR_BASE, spec["install_dir"])
        stamp = os.path.join(path, BUILD_STAMP_FILE)
        if os.path.isdir(path) and _stamp_matches(stamp, spec["stamp"]):
            print(f"  [ok] {spec['install_dir']}")
        else:
            print(f"  [MISSING/STALE] {spec['install_dir']}")
            all_ok = False

    config_ini = os.path.join(VENDOR_BASE, "config.ini")
    with open(config_ini, "w") as f:
        f.write(f"load_priority={NORMAL_INSTALL_DIR}\n")
    print(f"[build] config.ini → {NORMAL_INSTALL_DIR}")
    return all_ok


def run_build_with_retry(ops_nn_dir: str, jobs: int = 32, max_retries: int = 2, op: str = DEFAULT_OP) -> bool:
    """
    一站式编译：启动 → 每 ETA/4 轮询三态判定 → 失败自动重试（断点续跑）。

    op: 算子档案名，透传给 start_build / collect_build_result。
    """
    for attempt in range(1 + max_retries):
        eta = start_build(ops_nn_dir, jobs=jobs, op=op)
        interval = max(eta // 4, 60)
        deadline = time.time() + 6 * eta  # 总时长上限

        if attempt > 0:
            print(
                f"[build] 第 {attempt} 次重试（断点续跑：已成功包自动跳过）。"
                f"若失败为 kernel 编译超时（rc=124）可考虑降低 jobs 重试，当前 jobs={jobs}。"
            )

        result = "running"
        while result == "running":
            time.sleep(interval)
            result = check_build_status()
            if result == "running" and time.time() > deadline:
                result = "timeout"
                break

        if result == "done":
            print(f"[build] DONE (attempt={attempt})")
            return collect_build_result(op=op, ops_nn_dir=ops_nn_dir)

        # failed / dead / timeout
        print(f"[build] {result.upper()} (attempt={attempt})")
        _print_failure_diagnosis()
        if attempt < max_retries:
            print(f"[build] 将自动重试（剩 {max_retries - attempt} 次）...")
            time.sleep(30)
        else:
            print(f"[build] 重试次数耗尽，请人工检查 {BUILD_OUTPUT_LOG}")

    return collect_build_result(op=op, ops_nn_dir=ops_nn_dir)


def run_build_all(ops_nn_dir: str, jobs: int = 32, build: bool = True, op: str = DEFAULT_OP) -> bool:
    """
    完整的初始化+编译执行流程：
    0. 创建强制模板 worktree（档案 templates 声明；已存在则跳过）
    1. 检测 vendor 包（normal + 各强制包）是否已安装 → 全部存在则跳过编译
    2. build=True 时启动编译（含失败检测与自动重试）；build=False 时只准备不编译
    3. 验证 vendor 目录并写 config.ini

    jobs 为编译并行度（默认 32）。
    op: 算子档案名，是 worktree、构建参数与 vendor 目录的声明来源。
    """
    _verify_repo_version(ops_nn_dir, load_profile(op))
    prepare_results = [
        create_worktree(ops_nn_dir, config, dry_run=False, op=op)
        for config in resolve_worktree_paths(ops_nn_dir, op)
    ]
    if not all(prepare_results):
        print("[build] worktree 补丁未全部成功，终止编译。")
        return False

    if check_vendors_exist(op=op, ops_nn_dir=ops_nn_dir):
        return True

    if not build:
        print("[build] build=False：worktree 已就绪，vendor 包未编译。")
        print(
            f"[build] 后续可调用 start_build(ops_nn_dir, jobs=N) 启动编译，"
            f"再以 ETA/4 间隔轮询 check_build_status()，结束后 collect_build_result()。"
        )
        return False

    return run_build_with_retry(ops_nn_dir, jobs=jobs, op=op)


# ============================================================================
# 主入口
# ============================================================================


def _run_stub_cli(argv):
    parser = argparse.ArgumentParser(description="安装侧声明驱动打桩（非 MX 模式；声明源 profiles/<op>/profile.json）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("install-stub", help="备份+注入（扫参前）")
    p1.add_argument(
        "--mode", required=True, help="档案内可打桩的量化模式（kernel_files 携带 sed_anchor 声明，即非 MX 模式）"
    )
    p1.add_argument("--op", required=True, choices=_available_ops(), help="算子档案名")
    p1.add_argument("--dry-run", action="store_true")
    p2 = sub.add_parser("restore-stub", help="还原现场（扫参后必做）")
    p2.add_argument("--mode", default=None, help="缺省=还原档案内全部文件")
    p2.add_argument("--op", required=True, choices=_available_ops(), help="算子档案名")
    p2.add_argument("--dry-run", action="store_true")
    p3 = sub.add_parser("stub-status", help="查看打桩状态")
    p3.add_argument("--mode", default=None)
    p3.add_argument("--op", required=True, choices=_available_ops(), help="算子档案名")
    args = parser.parse_args(argv)
    available_modes = _stub_modes(load_profile(args.op))
    if args.mode is not None and args.mode not in available_modes:
        parser.error(f"算子 {args.op!r} 不支持 stub mode {args.mode!r}；可用: {available_modes}")
    if args.cmd == "install-stub":
        sys.exit(0 if install_stub(args.mode, args.dry_run, op=args.op) else 1)
    elif args.cmd == "restore-stub":
        sys.exit(0 if restore_stub(args.mode, args.dry_run, op=args.op) else 1)
    else:
        stub_status(args.mode, op=args.op)


def main():
    argv = sys.argv[1:]
    # 非 MX 打桩子命令分流。
    if argv and argv[0] in ("install-stub", "restore-stub", "stub-status"):
        _run_stub_cli(argv)
        return

    parser = argparse.ArgumentParser(
        description="创建算子 tiling solver 的强制模板 worktree（声明源 profiles/<op>/profile.json）"
    )
    parser.add_argument("--ops-nn-dir", default=OPS_NN_DEFAULT, help="ops-nn 仓库路径")
    parser.add_argument(
        "--op", required=True, choices=_available_ops(), help="算子档案名（读 references/profiles/<op>/profile.json）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印操作，不实际修改文件")
    parser.add_argument("--jobs", type=int, default=32, help="编译并行度（默认 32）")
    parser.add_argument("--no-build", action="store_true", help="只创建 worktree 与校验 vendor，不启动编译")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs 必须是正整数")

    ops_nn_dir = os.path.abspath(args.ops_nn_dir)
    op = args.op

    # 主仓库的 .git 是目录，worktree 中是文件。
    if not os.path.exists(os.path.join(ops_nn_dir, ".git")):
        print(f"[ERROR] {ops_nn_dir} 不是 git 仓库")
        sys.exit(1)
    _verify_repo_version(ops_nn_dir, load_profile(op))

    print(f"ops-nn 仓库: {ops_nn_dir}")
    print(f"模式: {'dry-run' if args.dry_run else 'actual'}")

    worktrees = resolve_worktree_paths(ops_nn_dir, op)
    if args.dry_run:
        success_count = sum(create_worktree(ops_nn_dir, config, True, op=op) for config in worktrees)
        print(f"\n预检: {success_count}/{len(worktrees)} 个 worktree 操作可生成")
        print_build_instructions(ops_nn_dir, op=op)
        return

    print(f"\n开始准备并校验 {len(worktrees) + 1} 个 vendor 包 (jobs={args.jobs})")
    build_ok = run_build_all(ops_nn_dir, jobs=args.jobs, build=not args.no_build, op=op)
    if not args.no_build:
        if build_ok:
            print(f"\n[build] 全部编译安装成功，{len(worktrees) + 1} 个 vendor 包就绪")
        else:
            print(f"\n[build] 部分包缺失或过期，请检查 {BUILD_OUTPUT_LOG}")


if __name__ == "__main__":
    main()
