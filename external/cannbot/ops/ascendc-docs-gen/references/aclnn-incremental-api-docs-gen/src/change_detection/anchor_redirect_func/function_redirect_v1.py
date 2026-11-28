# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Entry script for anchor redirect mapping."""

import json
import os
from pathlib import Path

from src.logger import info, debug, warning, error

from .config import MappingContext
from .three_level_mapper import ThreeLevelMapper


def process(
    anchor_path: str,
    origin_map_base: str,
    change_map_base: str,
    change_repo_path: str,
    output_dir: str,
    verbose: bool = False,
) -> dict:
    """
    从单个 anchor JSON 文件中处理所有 type="function" 和 type="file" 的 anchor。
    遍历每个 file-type anchor，对应的变更后文件路径（来自 file 阶段的更新），
    调用 ThreeLevelMapper 进行函数级重定向。

    参数:
        anchor_path: anchor JSON 文件路径（四个阶段共用）
        origin_map_base: 变更前 code_map 目录
        change_map_base: 变更后 code_map 目录
        change_repo_path: 变更后算子仓库路径
        verbose: 是否输出详细日志

    返回:
        更新后的 anchor JSON 数据（含 processed/skipped/remaining/status 统计字段），
        由调用方负责写入文件
    """
    from .utils import load_json

    # Load the anchor JSON once
    try:
        anchor_data = load_json(anchor_path)
        anchor_path_name = Path(anchor_path).name

    except Exception as e:
        # print(f"  [func 阶段] 加载 anchor JSON 失败: {e}")
        anchor_data = {
            "anchor_tuple": [],
            "processed": 0,
            "skipped": 0,
            "remaining": 0,
            "status": f"load error: {e}",
        }
        return anchor_data

    # Collect all file-type anchors to process

    function_anchors = []
    for item in anchor_data.get("anchor_tuple", []):
        anchor_info = item.get("anchor_information", {})
        if anchor_info.get("type") == "function":
            level1 = anchor_info.get("level1", {})
            file_path = level1.get("file_path", "")
            function_name = level1.get("function_name", "")

            if file_path:
                function_anchors.append(
                    {
                        "item": item,
                        "file_path": file_path,
                        "file_name": file_path,
                        "function_name": function_name,
                    }
                )
    info(f"找到 {len(function_anchors)} 个 function 类型的 anchor")
    processed = 0
    skipped = 0
    new_anchors = []
    for fa in function_anchors:
        ops_path = Path(fa["file_path"])
        file_path = Path(*ops_path.parts[1:])
        file_name = fa["file_name"]

        if not file_name:
            file_name = os.path.basename(file_path)

        # Normalize ops_name from file path
        ops_name = file_name.replace(".cpp", "").replace(".h", "")
        ops_name = ops_name.replace("aclnn_", "").replace("aclnn", "")

        if verbose:
            pass
            # # print(f"  [func] 处理文件: {file_path} (ops={ops_name})")

        # Skip if file doesn't exist on disk
        if not os.path.exists(file_path):
            if verbose:
                pass
                # # print(f"  [func] 文件不存在，跳过: {file_path}")
            skipped += 1
            continue

        # Create a temp copy of the anchor JSON for the mapper
        # (ThreeLevelMapper loads from disk and saves to output_dir)
        # import tempfile
        # output_file = os.path.join(output_dir, 'func_result.json')
        # with open(output_file, "w", encoding='utf-8') as fp:
        #     json.dump(anchor_data, fp, ensure_ascii=False, indent=2)

        try:
            context = MappingContext(
                ops_name=ops_name,
                file_path=file_path,
                code_map_dir=change_map_base,
                new_operator_path=change_repo_path,
                anchor_path=anchor_path,
                output_dir=output_dir,
            )

            mapper = ThreeLevelMapper(context, verbose=verbose)
            result = mapper.map()
            new_anchors.append(result.update_anchor)

            # if result.update_anchor:
            #         # save files
            #         output_func_dir = os.path.join(output_dir, f"func_{anchor_path_name}")
            #         with open(output_func_dir, "w", encoding="utf-8") as f:
            #             json.dump(result, f, ensure_ascii=False, indent=2)

            if result.matched:
                processed += 1
                # # print("ori_file_name:", result.old_file_path)
                # # print("new_file_name:", result.new_file_path)
                # # print("ori_functions:", result.old_function_names)
                # # print("new_functions:", result.new_function_names)
                # # print("update_anchors:", result.update_anchor)

            else:
                if verbose:
                    warning(f"  [func] 映射失败 ({ops_name}): {result.message}")

        except Exception:
            if verbose:
                # print(f"  [func] 处理异常 ({file_path}): {e}")
                import traceback
                import sys

                traceback.print_exc()

                # 单独获取 当前异常的 文件名、行号
                exc_type, exc_value, exc_tb = sys.exc_info()

                # print(f"\n报错文件：{fname}")
                # print(f"报错行号：{line_no}")
                # print(f"错误信息：{e}")
    from copy import deepcopy

    anchor_update_output = deepcopy(anchor_data)
    anchor_update_output["anchor_tuple"] = new_anchors

    # 保存anchor
    output_func_dir = os.path.join(output_dir, f"func_{anchor_path_name}")
    # print("FUNC", output_func_dir)
    with open(output_func_dir, "w", encoding="utf-8") as f:
        json.dump(anchor_update_output, f, ensure_ascii=False, indent=2)

    remaining = len(function_anchors) - processed - skipped
    anchor_data["processed"] = processed
    anchor_data["skipped"] = skipped
    anchor_data["remaining"] = remaining
    anchor_data["status"] = "completed"

    return anchor_data
