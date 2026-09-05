#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""
ops仓算子构建+安装封装脚本。

封装 ops-nn/cv/math/transformer 仓及 omni-ops 仓的 build.sh 构建流程，
执行编译并将生成的 .run 文件安装到指定路径。
自动检测仓类型并使用对应的构建命令。

用法:
    python build_ops.py --repo-root /path/to/ops-nn --op-name ada_layer_norm_custom \
        --soc ascend910b --install-path /abs/path/to/install

参数:
    --repo-root: ops仓根目录（如 /home/user/ops-nn）
    --op-name: 算子名（如 ada_layer_norm_custom，带 _custom 后缀）
    --soc: 目标芯片（如 ascend910b, ascend950）
    --install-path: 安装目标路径（必须是绝对路径）

依赖:
    - ops仓根目录下需要有 build.sh
    - ASCEND_HOME_PATH 环境变量需要设置
"""

import argparse
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


def detect_cpu_arch() -> str:
    """检测CPU架构。"""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine in ("aarch64", "arm64"):
        return "aarch64"
    else:
        logging.warning(f"未知CPU架构: {machine}, 默认使用aarch64")
        return "aarch64"


def _repo_type_from_name(name: str):
    """按名称关键字匹配仓类型，未命中返回 None。"""
    for repo_type in ("nn", "cv", "math", "transformer"):
        if repo_type in name:
            return repo_type
    return None


def _repo_type_from_build_sh(build_sh: str, repo_root: str):
    """通过 build.sh 内容检测仓类型，检测不到返回 None。"""
    if not os.path.exists(build_sh):
        return None
    try:
        with open(build_sh, "r") as f:
            content = f.read()
    except Exception as e:
        logging.warning("读取 build.sh 失败: %s", e)
        return None

    # omni-ops 特征检测（优先于标准仓检测）:
    # omni-ops 的 build.sh 使用 -n/--op-name 参数风格，
    # 且存在 src/ 目录（如 src/ops-transformer/）
    if (re.search(r'(-n\b|--op-name)', content)
            and os.path.isdir(os.path.join(repo_root, "src"))):
        return "omni"

    # 尝试匹配 REPOSITORY_NAME 变量
    match = re.search(r'REPOSITORY_NAME\s*=\s*["\']?(\w+)', content)
    if match:
        return _repo_type_from_name(match.group(1).lower())
    return None


def detect_repo_type(repo_root: str) -> str:
    """
    通过 build.sh 中的 REPOSITORY_NAME 或目录名检测仓类型。

    Returns:
        "nn" / "cv" / "math" / "transformer" / "omni"
    """
    repo_type = _repo_type_from_build_sh(
        os.path.join(repo_root, "build.sh"), repo_root)
    if repo_type:
        return repo_type

    # 回退: 通过目录名判断
    basename = os.path.basename(os.path.normpath(repo_root)).lower()
    if "omni" in basename:
        return "omni"
    repo_type = _repo_type_from_name(basename)
    if repo_type:
        return repo_type

    logging.warning("无法检测仓类型, 默认 nn")
    return "nn"


def detect_is_omni(repo_root: str) -> bool:
    """检测是否为 omni-ops 仓。

    omni-ops 仓的特征：build.sh 支持 -n/--op-name 参数，且有 src/ 目录。
    """
    build_sh = os.path.join(repo_root, "build.sh")
    has_n_flag = False
    has_src = os.path.isdir(os.path.join(repo_root, "src"))
    if os.path.exists(build_sh):
        with open(build_sh, "r") as f:
            content = f.read()
        has_n_flag = bool(re.search(r'(-n\b|--op-name)', content))
    return has_n_flag and has_src


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.eval_utils import detect_vendor_subdir  # noqa: E402


def _find_run_in_dir(base_dir: str):
    """在单个目录下递归查找 .run 文件，未找到返回 None。"""
    if not os.path.isdir(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".run"):
                return os.path.join(root, f)
    return None


def find_run_file(repo_root: str, is_omni: bool) -> str:
    """
    查找构建产物中的 .run 安装文件。

    omni-ops 仓的 .run 在 build/_CPack_Packages/ 下，
    标准仓的 .run 在 build_out/ 下。

    Args:
        repo_root: 仓根目录
        is_omni: 是否为 omni-ops 仓

    Returns:
        .run 文件的完整路径
    Raises:
        FileNotFoundError: 未找到 .run 文件
    """
    build_out_dir = os.path.join(repo_root, "build_out")
    cpack_dir = os.path.join(repo_root, "build", "_CPack_Packages")
    output_dir = os.path.join(repo_root, "output")

    if is_omni:
        search_dirs = [cpack_dir, output_dir, build_out_dir]
    else:
        search_dirs = [build_out_dir, output_dir]

    for base_dir in search_dirs:
        run_file = _find_run_in_dir(base_dir)
        if run_file:
            return run_file

    searched = ", ".join(search_dirs)
    raise FileNotFoundError(
        f"未在以下目录中找到 .run 文件: {searched}。"
        f" 请检查构建是否成功完成。"
    )


def _validate_inputs(repo_root: str, install_path: str):
    """校验构建输入参数。"""
    if not os.path.isabs(install_path):
        raise ValueError(
            f"install_path 必须是绝对路径, 当前值: {install_path}"
        )

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"ops仓根目录不存在: {repo_root}")

    build_sh = os.path.join(repo_root, "build.sh")
    if not os.path.isfile(build_sh):
        raise FileNotFoundError(f"build.sh 不存在: {build_sh}")


def _clean_build_dirs(repo_root: str):
    """清理旧的构建产物目录。"""
    for d in [os.path.join(repo_root, "build"),
              os.path.join(repo_root, "build_out")]:
        if os.path.exists(d):
            logging.info("清理目录: %s", d)
            shutil.rmtree(d)


def _run_build(repo_root: str, op_name: str, soc: str, is_omni: bool, nproc: int):
    """执行 build.sh 构建（含 omni-ops output 目录备份/恢复保护）。"""
    # omni-ops 仓使用 -n 参数指定算子名，标准仓使用 --pkg --ops= 参数
    if is_omni:
        # omni-ops build.sh 内部自动计算 JOB_NUM，不接受 -j 参数
        build_cmd = ["bash", "build.sh", "-n", op_name, "-c", soc]
    else:
        build_cmd = [
            "bash", "build.sh", "--pkg", "--vendor_name=custom",
            f"--soc={soc}", f"--ops={op_name}", f"-j{nproc}",
        ]
    logging.info("执行构建: %s", " ".join(build_cmd))

    # omni-ops 保护: 备份 repo_root/output 目录，防止 build.sh 误删
    repo_output = os.path.join(repo_root, "output")
    repo_output_bak = repo_output + f"_bak_{os.getpid()}"
    output_backed_up = False
    if is_omni and os.path.exists(repo_output):
        logging.info("omni-ops 保护: 备份 %s -> %s", repo_output, repo_output_bak)
        shutil.move(repo_output, repo_output_bak)
        output_backed_up = True

    try:
        result = subprocess.run(
            build_cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes (large ops repos like ops-transformer need longer)
        )
    finally:
        # 恢复备份的 output 目录
        if output_backed_up and os.path.exists(repo_output_bak):
            if os.path.exists(repo_output):
                # 构建过程可能生成了新的 output，保留原备份
                logging.info("omni-ops 保护: 构建产生了新的 output 目录，保留原备份")
                shutil.rmtree(repo_output)
            logging.info("omni-ops 保护: 恢复 %s -> %s", repo_output_bak, repo_output)
            shutil.move(repo_output_bak, repo_output)

    if result.returncode != 0:
        error_msg = (
            f"构建失败!\n"
            f"Exit Code: {result.returncode}\n"
            f"Stdout (last 2000 chars):\n{result.stdout[-2000:]}\n"
            f"Stderr (last 2000 chars):\n{result.stderr[-2000:]}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    logging.info("构建成功")


def _run_install(run_file: str, install_path: str):
    """执行 .run 安装文件到指定路径。"""
    os.makedirs(install_path, exist_ok=True)
    install_cmd = ["bash", run_file, f"--install-path={install_path}"]
    logging.info("执行安装: %s", " ".join(install_cmd))

    result = subprocess.run(
        install_cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        error_msg = (
            f"安装失败!\n"
            f"Exit Code: {result.returncode}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    logging.info("安装成功")


def build_and_install(repo_root: str, op_name: str, soc: str,
                      install_path: str) -> dict:
    """
    构建并安装 ops 仓算子。

    Args:
        repo_root: ops仓根目录
        op_name: 算子名（如 ada_layer_norm_custom）
        soc: 目标芯片（如 ascend910b）
        install_path: 安装路径（绝对路径）

    Returns:
        dict: {
            "install_path": str,
            "vendor_subdir": str,
            "repo_type": str,
            "run_file": str,
        }

    Raises:
        RuntimeError: 构建或安装失败
    """
    _validate_inputs(repo_root, install_path)

    repo_type = detect_repo_type(repo_root)
    is_omni = detect_is_omni(repo_root)
    cpu_arch = detect_cpu_arch()
    nproc = os.cpu_count() or 8

    logging.info("构建配置:")
    logging.info("  仓类型: %s", repo_type)
    logging.info("  omni-ops: %s", is_omni)
    logging.info("  CPU架构: %s", cpu_arch)
    logging.info("  算子名: %s", op_name)
    logging.info("  目标芯片: %s", soc)
    logging.info("  安装路径: %s", install_path)

    # Step 1: 清理旧的构建产物
    _clean_build_dirs(repo_root)

    # Step 2: 执行构建
    _run_build(repo_root, op_name, soc, is_omni, nproc)

    # Step 3: 查找 .run 文件
    run_file = find_run_file(repo_root, is_omni)
    logging.info("找到 .run 文件: %s", run_file)

    # Step 4: 执行安装
    _run_install(run_file, install_path)

    # Step 5: 检测 vendor 子目录
    vendor_subdir = detect_vendor_subdir(install_path)
    vendor_path = os.path.join(install_path, "vendors", vendor_subdir)
    if not os.path.isdir(vendor_path):
        logging.warning(
            "vendors子目录未找到: %s, 请手动检查 %s/",
            vendor_path, os.path.join(install_path, 'vendors'),
        )

    return {
        "install_path": install_path,
        "vendor_subdir": vendor_subdir,
        "repo_type": repo_type,
        "run_file": run_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build and install ops repository operator"
    )
    parser.add_argument(
        "--repo-root", required=True,
        help="ops仓根目录路径 (如 /home/user/ops-nn)"
    )
    parser.add_argument(
        "--op-name", required=True,
        help="算子名 (如 ada_layer_norm_custom)"
    )
    parser.add_argument(
        "--soc", required=True,
        help="目标芯片 (如 ascend910b, ascend950)"
    )
    parser.add_argument(
        "--install-path", required=True,
        help="安装目标路径 (必须是绝对路径)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        result = build_and_install(
            repo_root=args.repo_root,
            op_name=args.op_name,
            soc=args.soc,
            install_path=args.install_path,
        )
        logging.info("构建安装完成:")
        logging.info("  安装路径: %s", result['install_path'])
        logging.info("  仓类型: %s", result['repo_type'])
        logging.info("  vendor子目录: %s", result['vendor_subdir'])
        logging.info("  .run文件: %s", result['run_file'])

        # 输出JSON供脚本解析（stdout 数据输出，保留 print）
        import json
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        DATA_LOGGER.info("\n--- BUILD_RESULT_JSON ---")
        DATA_LOGGER.info("%s", result_json)
        DATA_LOGGER.info("--- END_BUILD_RESULT_JSON ---")

    except Exception as e:
        logging.error("构建安装失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
