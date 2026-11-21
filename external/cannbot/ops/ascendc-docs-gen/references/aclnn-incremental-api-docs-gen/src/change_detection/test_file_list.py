#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
测试 file_list 功能的脚本

功能：
1. 测试变更检测流程中新增的 file_list 参数
2. 将额外关联文件（如 op_kernel、op_host 下的文件）添加到 code_map 处理流程
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.change_detection.anchor_build import anchor_rebuild
from src.change_detection.anchor_redirect import run_anchor_redirect_pipeline
from src.settings import apply_settings, load_settings


def setup_test_environment():
    """设置测试环境"""
    # 基础路径配置
    workspace_dir = Path(__file__).parent.parent.parent  # nnac-document-generation 根目录

    # 算子仓路径（本机路径经 CANN_OPS_WORKSPACE 环境变量提供, 不在代码中硬编码）
    _ops_ws = os.environ.get("CANN_OPS_WORKSPACE", "")
    operator_repo_path = Path(_ops_ws) / "data" / "ops-math" / "math" / "abs" if _ops_ws else Path("")

    # 测试输出路径
    test_output_dir = Path("test_output") / "file_list_test"

    # 清理并创建输出目录
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # 创建一个"变更后"的算子目录（复制原算子仓）
    changed_operator_path = test_output_dir / "changed_operator" / "abs"
    if changed_operator_path.exists():
        shutil.rmtree(changed_operator_path)
    shutil.copytree(operator_repo_path, changed_operator_path)

    # 创建原始快照目录（模拟已有快照）
    origin_snapshot_path = test_output_dir / "origin_snapshot"
    origin_snapshot_path.mkdir(parents=True, exist_ok=True)

    # 在变更后的算子中做一个小改动（模拟变更）
    test_file = changed_operator_path / "op_api" / "aclnn_abs.cpp"
    if test_file.exists():
        content = test_file.read_text(encoding="utf-8")
        # 在文件末尾添加一行注释
        modified_content = content + "\n// Test modification for file_list feature\n"
        test_file.write_text(modified_content, encoding="utf-8")
        print(f"[INFO] 已模拟变更: {test_file}")

    return {
        "workspace_dir": workspace_dir,
        "operator_repo_path": operator_repo_path,
        "origin_operator_path": operator_repo_path,  # 原始算子路径（用于重定向）
        "changed_operator_path": changed_operator_path,
        "test_output_dir": test_output_dir,
        "origin_snapshot_path": origin_snapshot_path,
    }


def get_extra_file_list():
    """
    获取额外关联文件列表

    从完全不包含 op_api/op_host/op_kernel 目录名的路径选取文件，
    模拟真实场景：额外文件来自其他非标准目录（如 common、utils、tests根目录等）
    """
    # 方案：从 tests 目录的根层级选取文件（不在 ut/st/op_api/op_host 子目录下）
    _ops_ws = os.environ.get("CANN_OPS_WORKSPACE", "")
    operator_repo_path = Path(_ops_ws) / "data" / "ops-math" / "math" / "abs" if _ops_ws else Path("")

    extra_files = []

    # 选择 tests/assets 目录下的文件（完全不包含 op_api/op_host）
    tests_assets_dir = operator_repo_path / "tests" / "assets"
    if tests_assets_dir.exists():
        # 只选取 cpp/h 文件（single_code_map 只处理这些类型）
        pass  # 该目录下没有 cpp/h 文件

    # 选择 examples 目录下的文件（路径中不包含 op_api/op_host/op_kernel）
    examples_dir = operator_repo_path / "examples"
    if examples_dir.exists():
        for f in examples_dir.glob("*.cpp"):
            extra_files.append(str(f))
        for f in examples_dir.glob("*.h"):
            extra_files.append(str(f))

    # 选择根目录下的配置文件作为额外文件（模拟公共配置引用）
    operator_repo_path / "CMakeLists.txt"
    # 注意：single_code_map 只处理 cpp/h 文件，这里仅作演示

    print(f"[INFO] 找到 {len(extra_files)} 个额外关联文件")
    print("[INFO] 这些文件路径中不包含标准 op_kernel/op_host/op_api 目录结构")
    for f in extra_files:
        rel_path = Path(f).relative_to(operator_repo_path)
        # 验证路径中不包含这些关键字
        path_str = str(rel_path)
        has_keywords = any(kw in path_str for kw in ["op_api", "op_host", "op_kernel", "op_graph"])
        status = "[含关键字]" if has_keywords else "[无关键字]"
        print(f"  {status} {rel_path}")

    return extra_files


def run_test_with_file_list(paths: dict, file_list: list):
    """运行带 file_list 的变更检测测试"""

    print("\n" + "=" * 60)
    print("开始测试 file_list 功能")
    print("=" * 60)

    operator_name = "abs"

    # 构建路径
    snapshot_path = paths["test_output_dir"] / "new_snapshot"
    snapshot_path.mkdir(parents=True, exist_ok=True)

    filter_list = [
        "support",
        "summary",
        "function_proto",
        "sub_function_hash",
        "constraint",
    ]

    # 打印测试参数
    print("\n[测试参数]")
    print(f"  算子名称: {operator_name}")
    print(f"  变更后算子路径: {paths['changed_operator_path']}")
    print(f"  输出路径: {snapshot_path}")
    print(f"  额外文件数量: {len(file_list)}")

    # 执行算子重建（传入 file_list）
    print("\n[执行 anchor_rebuild]")
    try:
        rebuild_result = anchor_rebuild(
            operator_name,
            str(paths["changed_operator_path"]),
            str(snapshot_path),
            filter_list,
            str(paths["workspace_dir"]),
            file_list=file_list,
        )
        print("[SUCCESS] anchor_rebuild 完成")
        print(f"  结果统计: {json.dumps(rebuild_result, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"[FAILED] anchor_rebuild 失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 检查 code_map 输出是否包含额外文件
    print("\n[验证 code_map 输出]")
    code_map_dir = snapshot_path / "code_map"
    if code_map_dir.exists():
        files_dir = code_map_dir / "files"
        if files_dir.exists():
            map_files = list(files_dir.glob("*_map.json"))
            print(f"  生成的 code_map 文件数: {len(map_files)}")

            # 检查是否包含额外文件的 code_map
            extra_file_names = [Path(f).name for f in file_list]
            found_extra = []
            for map_file in map_files:
                map_name = map_file.name.replace("_map.json", "")
                if any(map_name == ef.replace(".cpp", "").replace(".h", "") for ef in extra_file_names):
                    found_extra.append(map_file.name)

            if found_extra:
                print(f"  包含额外文件的 code_map: {found_extra}")
            else:
                print("  [WARNING] 未找到额外文件的 code_map")

    # 检查 anchor 输出是否包含额外文件
    print("\n[验证 anchor 输出]")
    anchor_dir = snapshot_path / "anchor"
    if anchor_dir.exists():
        anchor_files = list(anchor_dir.glob("*.json"))
        print(f"  生成的 anchor 文件数: {len(anchor_files)}")

        # 统计 anchor 中的文件路径
        for anchor_file in anchor_files:
            with open(anchor_file, "r", encoding="utf-8") as f:
                anchor_data = json.load(f)
                anchor_tuple = anchor_data.get("anchor_tuple", [])

                # 统计各个 anchor 的文件来源
                operator_files = 0
                extra_files_count = 0
                for anchor in anchor_tuple:
                    file_path = anchor.get("anchor_information", {}).get("level1", {}).get("file_path", "")
                    # 检查是否来自额外文件
                    is_extra = any(file_path.endswith(Path(f).name) for f in file_list)
                    if is_extra:
                        extra_files_count += 1
                    else:
                        operator_files += 1

                print(f"  {anchor_file.name}: operator={operator_files}, extra={extra_files_count}")

    # 检查额外文件判断结果
    print("\n[验证额外文件判断结果]")
    extra_files_info = rebuild_result.get("extra_files_info", {"internal": [], "external": []})
    internal_files = extra_files_info.get("internal", [])
    external_files = extra_files_info.get("external", [])

    print(f"  内部文件（参与重定向）: {len(internal_files)} 个")
    for i, f in enumerate(internal_files, 1):
        print(f"    [{i}] {f}")

    print(f"  外部文件（跳过重定向）: {len(external_files)} 个")
    for i, f in enumerate(external_files, 1):
        print(f"    [{i}] {f}")

    # 执行算子重定向测试
    print("\n" + "=" * 60)
    print("[执行 anchor_redirect 测试]")
    print("=" * 60)

    try:
        # 构建重定向所需参数
        anchor_path = str(snapshot_path / "anchor")
        origin_code_map = str(snapshot_path / "code_map")
        new_code_map = str(snapshot_path / "code_map")

        print("\n[重定向参数]")
        print(f"  anchor_path: {anchor_path}")
        print(f"  origin_map_base: {origin_code_map}")
        print(f"  change_map_base: {new_code_map}")
        print(f"  change_repo_path: {str(paths['changed_operator_path'])}")
        print(f"  extra_files_info: internal={len(internal_files)}, external={len(external_files)}")

        # 执行重定向（传入额外文件判断结果）
        redirect_result = run_anchor_redirect_pipeline(
            anchor_path=anchor_path,
            origin_map_base=origin_code_map,
            change_map_base=new_code_map,
            origin_repo_path=str(paths["origin_operator_path"]),
            change_repo_path=str(paths["changed_operator_path"]),
            output_dir=str(snapshot_path),
            verbose=True,
            extra_files_info=extra_files_info,  # 传入额外文件判断结果
        )

        print("\n[SUCCESS] anchor_redirect 完成")
        print(f"  重定向结果: {json.dumps(redirect_result, ensure_ascii=False, indent=2)[:500]}...")

    except Exception as e:
        print(f"\n[FAILED] anchor_redirect 失败: {e}")
        import traceback

        traceback.print_exc()

    # 验证重定向输出
    print("\n[验证重定向输出]")
    redirect_dir = snapshot_path / "redirect"
    if redirect_dir.exists():
        redirect_files = list(redirect_dir.glob("*.json"))
        print(f"  重定向输出目录: {redirect_dir}")
        print(f"  生成的重定向文件数: {len(redirect_files)}")
        for f in redirect_files:
            print(f"    {f.name}")
    else:
        print("  [WARNING] 重定向输出目录不存在")

    # 写入完成标记
    done_file = snapshot_path / "TEST_DONE"
    with open(done_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completedAt": datetime.now().isoformat(),
                "testType": "file_list_feature",
                "extraFilesCount": len(file_list),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"输出目录: {snapshot_path}")

    return True


def main():
    """主测试流程"""
    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("file_list 功能测试脚本")
    print("=" * 60)

    # 1. 设置测试环境
    print("\n[Step 1] 设置测试环境")
    paths = setup_test_environment()

    # 2. 获取额外文件列表
    print("\n[Step 2] 获取额外关联文件")
    file_list = get_extra_file_list()

    # 3. 运行测试
    print("\n[Step 3] 运行变更检测测试")
    success = run_test_with_file_list(paths, file_list)

    if success:
        print("\n[RESULT] 测试成功")
    else:
        print("\n[RESULT] 测试失败")

    return success


if __name__ == "__main__":
    main()
