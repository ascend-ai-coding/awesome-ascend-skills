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
构建文件树结构脚本
从 results_by_file_v2 目录中的 JSON 文件提取 file_path 字段，
在新目录 file_tree_structure 中复现相同的目录结构，并将对应的 JSON 文件放置到正确的位置。
"""

import json
import shutil
import traceback
from pathlib import Path

from src.logger import debug, error, info, warning


def build_file_tree(out_path):
    """构建文件树结构"""

    # 源目录: step_2 输出 (data/results_by_file/)
    source_dir = out_path / "results_by_file"
    # 目标目录: data/file_tree_structure/
    target_dir = out_path / "file_tree_structure"

    # 确保源目录存在
    if not source_dir.exists():
        error(f"错误: 源目录 {source_dir} 不存在")
        return

    # 创建目标目录
    target_dir.mkdir(exist_ok=True)
    info(f"目标目录: {target_dir}")

    # 统计信息
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历源目录中的所有 JSON 文件
    for json_file in source_dir.glob("*.json"):
        try:
            # 读取 JSON 文件
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 获取 file_path 字段（现为绝对路径）
            file_path = data.get("file_path")
            if not file_path:
                warning(f"警告: {json_file.name} 中没有 file_path 字段")
                skipped_count += 1
                continue

            # 从绝对路径中提取算子目录内部的相对路径
            # 如 D:/data/ops-math/math/abs/op_api/aclnn_abs.h → op_api/aclnn_abs.h
            abs_path_posix = Path(file_path).as_posix()
            operator_name = data.get("operator", "")
            idx = abs_path_posix.find(f"/{operator_name}/")
            if idx >= 0:
                start = idx + len(f"/{operator_name}/")
                relative_within = abs_path_posix[start:]
            else:
                relative_within = Path(file_path).name

            # 获取目录部分（移除文件名）
            directory_part = str(Path(relative_within).parent)

            # 如果 directory_part 为空，说明 file_path 只是文件名，放到根目录
            if directory_part:
                target_sub_dir = target_dir / directory_part
            else:
                target_sub_dir = target_dir

            # 创建目标子目录
            target_sub_dir.mkdir(parents=True, exist_ok=True)

            # 目标文件路径（用 task_id 前缀缩短文件名，避免 Windows MAX_PATH 限制）
            target_file_path = target_sub_dir / (json_file.name.split("_")[0] + ".json")
            # 复制文件（保留元数据）

            shutil.copy2(json_file, target_file_path)

            processed_count += 1
            info(f"已处理: {json_file.name} -> {target_file_path.relative_to(target_dir)}")

        except json.JSONDecodeError:
            error(f"错误: {json_file.name} 不是有效的 JSON 文件")
            error_count += 1
        except Exception as e:
            error(f"错误: 处理 {json_file.name} 时发生异常: {e}")

            traceback.print_exc()
            error_count += 1
            break

    # 打印统计信息
    info("=" * 50)
    info("处理完成:")
    info(f"  成功处理: {processed_count} 个文件")
    info(f"  跳过: {skipped_count} 个文件")
    info(f"  错误: {error_count} 个文件")
    info(f"  结果目录: {target_dir.resolve()}")
    info("=" * 50)


def main(out_path):
    """入口函数，兼容 run_pipeline.py 调用"""
    build_file_tree(out_path)


if __name__ == "__main__":
    main()
