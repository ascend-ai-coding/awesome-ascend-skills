# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
import sys
from pathlib import Path
from rich.console import Console

from src.change_detection.anchor_build import anchor_rebuild
from src.change_detection.anchor_redirect import run_anchor_redirect_pipeline
from src.settings import apply_settings, load_settings


def change_deteciton_cli(
    origin_operator_path,
    changed_operator_path,
    data_path,
    md_stem,
    output_path,
    filter_list,
    operator_name,
    file_dir_list=None,
    workspace_dir=None,
    verbose=False,
    add_file_list=None,
    modified_file_list=None,
    changed_operator_path_split=None,
):
    """
    origin_operator_path, 修改之前的算子目录
    changed_operator_path, 修改之后的算子目录
    data_path, 快照构建的结果。
    output_path, 输出结果的保存路径
    filter_list, [constraint, ...]
    operator_name, 算子名称
    file_dir_list, [(abs dir for changed file), ...]
    repo_other_file_list, 仓内其他依赖文件变更列表（绝对路径）
    workspace_dir, 算子工程根目录(ops_base)
    add_file_list, 新增文件列表
    modified_file_list, 已修改文件列表
    changed_operator_path_split, 修改之后算子经版本拆分后的目录（绝对路径）
    """
    console = Console()
    workspace_base_dir = Path(workspace_dir) if workspace_dir else Path("./")

    # 将 add_file_list 和 modified_file_list 组织成 extra_files_info
    # external: add_file_list 中同时出现在 modified_file_list 中的文件（外部有变化的文件）
    # internal: modified_file_list 中除去 add_file_list 后的文件（内部有变化的文件）
    add_set = set(add_file_list or [])
    mod_set = set(modified_file_list or [])
    extra_files_info = {
        "external": list(add_set & mod_set),
        "internal": list(mod_set - add_set),
    }

    # changed_operator_path = workspace_base_dir / changed_operator_path
    snapshot_path = workspace_base_dir / data_path

    output_path = workspace_base_dir / output_path
    origin_snapshot_path = workspace_base_dir / data_path
    origin_operator_path = workspace_base_dir / origin_operator_path
    snapshot_path = origin_snapshot_path
    new_snapshot_path = output_path

    op_name_md_stem_path = (
        snapshot_path / operator_name / md_stem
    )  # 获取到"out/算子名称/md_stem/"这个目录下
    anchor_path = op_name_md_stem_path / "anchor"
    origin_code_map = op_name_md_stem_path / "code_map"
    new_code_map = new_snapshot_path / "code_map" / operator_name

    # 执行算子重建
    console.print("  > 重建变更仓代码关联关系")
    # anchor_rebuild_cli(str(changed_operator_path), str(new_snapshot_path), filter_list)
    old_tree_id_path = str(op_name_md_stem_path / "tree_id") if file_dir_list else None
    changed_operator_path_split = (
        str(changed_operator_path_split)
        if changed_operator_path_split
        else str(changed_operator_path)
    )
    result = anchor_rebuild(
        operator_name,
        changed_operator_path_split,  # 分版本结构化的路径
        str(changed_operator_path),  # 修改后的原始仓（绝对路径）
        str(new_snapshot_path),
        filter_list,
        workspace_base_dir,
        file_dir_list=file_dir_list,
        old_tree_id_path=old_tree_id_path,
        md_file=f"{md_stem}.md",
    )
    # import main_snapshot as snapshot
    # result = snapshot.main(changed_operator_path, data_path, workspace_base_dir)
    print("===change-detection-result===")
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("===/change-detection-result===")

    # 执行重定向
    console.print("  > 同步基线仓原始证据至变更仓")
    run_anchor_redirect_pipeline(
        anchor_path=str(anchor_path),
        origin_map_base=str(origin_code_map),
        change_map_base=str(new_code_map),
        origin_repo_path=str(origin_operator_path),
        change_repo_path=str(changed_operator_path),
        output_dir=str(new_snapshot_path),
        verbose=verbose,
        extra_files_info=extra_files_info,
        filter_list=filter_list,
    )


if __name__ == "__main__":
    # 自测入口: 本机路径通过 CANN_OPS_ROOT 环境变量提供, 不在代码中硬编码
    try:
        apply_settings(load_settings())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    _ops_ws = os.environ.get("CANN_OPS_WORKSPACE", "")
    if not _ops_ws:
        raise SystemExit(
            "请设置 CANN_OPS_WORKSPACE 环境变量指向本机 cann-document-generation 工作目录"
        )
    change_deteciton_cli(
        origin_operator_path=f"{_ops_ws}/out/operator_storage/abs/base",
        changed_operator_path=f"{_ops_ws}/data/abs_v2",
        data_path=f"{_ops_ws}/out/",
        md_stem="aclnnAbs",
        output_path=f"{_ops_ws}/change_detection_ouput",
        filter_list=["summary", "function_proto", "sub_function_hash"],
        operator_name="abs",
        file_dir_list=[f"{_ops_ws}/data/abs_v2/op_api/abs.cpp"],
        verbose=False,
    )
