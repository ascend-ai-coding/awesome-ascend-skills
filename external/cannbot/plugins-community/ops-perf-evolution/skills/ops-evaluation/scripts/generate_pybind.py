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
为 ops 仓算子生成 PyBind11 绑定并编译。

与 ascendc-evaluation 版本的区别:
- 适配 ops 仓安装路径下的 vendors 子目录名（custom_nn/custom_cv 等）
- work-dir 指向 ops 仓的安装路径

用法:
    python generate_pybind.py <op_name> --work-dir /path/to/install_dir

示例:
    python generate_pybind.py ada_layer_norm --work-dir /abs/path/to/baseline

依赖:
    - <work_dir>/<op_name>.cpp 必须存在
    - template/ 目录必须存在（从 ascendc-evaluation 复用）
"""

import sys
import shutil
from pathlib import Path
import subprocess
import zipfile
import logging
import argparse


# 复用 ascendc-evaluation 的模板目录
ASCENDC_EVAL_SCRIPTS = Path(__file__).parent.parent.parent / "ascendc-evaluation" / "scripts"
TEMPLATE_DIR = ASCENDC_EVAL_SCRIPTS / "template"


def generate_pybind_bindings(work_dir: Path, op_cpp: Path) -> None:
    """
    生成 PyBind 绑定并编译。

    Args:
        work_dir: 工作目录路径
        op_cpp: 算子 op.cpp 文件路径

    Raises:
        FileNotFoundError: 模板目录或必要的文件不存在
        subprocess.CalledProcessError: 编译失败时抛出异常
    """
    work_dir = Path(work_dir).resolve()
    template_dir = TEMPLATE_DIR
    target_dir = work_dir / "ascend_op_pybind"

    if not template_dir.exists():
        raise FileNotFoundError(f"模板目录不存在: {template_dir}")

    if not target_dir.exists():
        logging.info(f"拷贝模板目录到: {target_dir}")
        shutil.copytree(template_dir, target_dir)
    else:
        logging.info(f"目标目录已存在: {target_dir}")

    op_cpp_path = Path(op_cpp).resolve()
    cpp_path = target_dir / "CppExtension" / "csrc" / "op.cpp"

    if not op_cpp_path.exists():
        raise FileNotFoundError(f"op.cpp 源文件不存在: {op_cpp_path}")

    if cpp_path.exists():
        logging.info(f"删除已存在的 op.cpp: {cpp_path}")
        cpp_path.unlink()

    logging.info(f"拷贝 op.cpp: {op_cpp_path} -> {cpp_path}")
    cpp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(op_cpp_path, cpp_path)

    try:
        logging.info("开始编译 PyBind 绑定")
        extension_dir = target_dir / "CppExtension"

        result = subprocess.run(
            [sys.executable, 'setup.py', 'build', 'bdist_wheel'],
            cwd=str(extension_dir),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logging.info("编译 wheel 包成功")

            dist_dir = extension_dir / "dist"
            if dist_dir.exists():
                for wheel_file in dist_dir.glob("*.whl"):
                    # 隔离 .so 到 work_dir/pybind_lib/
                    pybind_lib_dir = work_dir / "pybind_lib"
                    pybind_lib_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(str(wheel_file), 'r') as whl:
                        for name in whl.namelist():
                            if name.endswith('.so') or name.endswith('.pyd'):
                                whl.extract(name, str(pybind_lib_dir))
                                extracted = pybind_lib_dir / name
                                target = pybind_lib_dir / Path(name).name
                                if extracted != target:
                                    shutil.move(str(extracted), str(target))
                                logging.info(f"Extracted {name} -> {target}")
                    logging.info(f"pybind .so isolated in {pybind_lib_dir}")

                    # 全局安装（兼容性）
                    result_install = subprocess.run(
                        [sys.executable, '-m', 'pip', 'install',
                         str(wheel_file), '--force-reinstall'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result_install.returncode == 0:
                        logging.info(f"安装 {wheel_file.name} 成功 (global)")
                    else:
                        logging.warning(
                            f"安装 {wheel_file.name} 到全局失败 (non-critical): "
                            f"{result_install.stderr}"
                        )

            logging.info("PyBind 绑定编译成功")
        else:
            error_msg = (
                f"编译失败!\n"
                f"Exit Code: {result.returncode}\n"
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
            logging.error(error_msg)
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )

    except subprocess.TimeoutExpired as e:
        raise Exception("编译超时（超过 300 秒）") from e
    except Exception as e:
        raise Exception(f"编译过程中发生错误: {str(e)}") from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Generate PyBind bindings for ops repository operators"
    )
    parser.add_argument("op_name", type=str, help="Operator name")
    parser.add_argument(
        "--work-dir", type=str, required=True,
        help="Work directory containing op_name.cpp (the install path)"
    )

    args = parser.parse_args()

    try:
        work_dir = Path(args.work_dir).resolve()
        op_cpp = work_dir / f"{args.op_name}.cpp"
        generate_pybind_bindings(work_dir, op_cpp)
        logging.info("PyBind bindings generated successfully")
    except Exception as e:
        logging.error(e)
        sys.exit(1)
