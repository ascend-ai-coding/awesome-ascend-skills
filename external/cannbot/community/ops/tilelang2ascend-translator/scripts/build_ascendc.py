#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import argparse
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = SCRIPT_DIR.parent


def _resolve_task_dir(op: str) -> Path:
    op_path = Path(op)
    if op_path.is_dir():
        return op_path.resolve()

    direct = WORKDIR / op
    if direct.is_dir():
        return direct

    raise FileNotFoundError(f"Cannot find task directory for op '{op}'")


def _detect_ascend_path() -> Path:
    for env_name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser().resolve()

    candidates = [
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    return candidates[-1]


def _find_kernel_sources(kernel_dir: Path) -> list[Path]:
    # 新结构: op_host/*.cpp + op_kernel/*.cpp
    sources = []
    for subdir in ("op_host", "op_kernel"):
        sd = kernel_dir / subdir
        if sd.is_dir():
            sources.extend(sorted(sd.glob("*.cpp")))
    if sources:
        return sources

    # 兼容旧结构: kernel/*.cpp (排除 pybind11.cpp)
    sources = sorted(
        path for path in kernel_dir.glob("*.cpp")
        if path.name != "pybind11.cpp"
    )
    if not sources:
        raise FileNotFoundError(f"No kernel .cpp sources found in {kernel_dir}")
    return sources


def _find_pybind_or_register(kernel_dir: Path) -> Path | None:
    # 新结构: 使用 register.cpp
    reg = kernel_dir / "register.cpp"
    if reg.is_file():
        return reg
    # 旧结构: 使用 pybind11.cpp
    pyb = kernel_dir / "pybind11.cpp"
    if pyb.is_file():
        return pyb
    return None


def _extract_pybind_module_name(pybind_path: Path) -> str:
    content = pybind_path.read_text(encoding="utf-8")
    match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", content)
    if not match:
        raise ValueError(f"Unable to detect PYBIND11_MODULE name from {pybind_path}")
    return match.group(1)


def _format_cmake_list(items: list[str], indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{item}" for item in items)


def _collect_include_dirs(kernel_dir: Path) -> list[Path]:
    """Collect include directories for the kernel build."""
    include_dirs = [kernel_dir]
    catlass_include = kernel_dir / "catlass" / "include"
    if catlass_include.is_dir():
        include_dirs.append(catlass_include)
    task_catlass_include = kernel_dir.parent / "catlass" / "include"
    if task_catlass_include.is_dir() and task_catlass_include not in include_dirs:
        include_dirs.append(task_catlass_include)
    return include_dirs


def _cmake_ascendc_dir_block() -> str:
    """Return the CMake block that finds the ascendc_kernel_cmake directory."""
    return """if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()"""


def _cmake_pybind11_block(entry_path: Path, kernel_dir: Path, module_name: str) -> str:
    """Return the CMake block for pybind11 library configuration."""
    return f"""add_library(pybind11_lib SHARED "{entry_path}")
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
  m
  dl
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${{TORCH_PATH}}")
set(ENV{{ASCEND_HOME_PATH}} ${{ASCEND_CANN_PACKAGE_PATH}})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${{TORCH_NPU_PATH}}")
target_link_directories(pybind11_lib PRIVATE
  ${{TORCH_PATH}}/lib
  ${{TORCH_NPU_PATH}}/lib
)
target_include_directories(pybind11_lib PRIVATE
  "{kernel_dir}"
  ${{TORCH_NPU_PATH}}/include
  ${{TORCH_PATH}}/include
  ${{TORCH_PATH}}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${{PYBIND11_INC}})
target_compile_options(pybind11_lib PRIVATE
  ${{PYBIND11_INC}}
  -D_GLIBCXX_USE_CXX11_ABI=1
)

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME {module_name}
  PREFIX ""
  SUFFIX "${{PYTHON_EXTENSION_SUFFIX}}"
)"""


def _render_cmakelists_template(**kwargs) -> str:
    build_dir = kwargs["build_dir"]
    source_lines = kwargs["source_lines"]
    include_lines = kwargs["include_lines"]
    entry_path = kwargs["entry_path"]
    kernel_dir = kwargs["kernel_dir"]
    module_name = kwargs["module_name"]
    return f"""cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "${{SOC_VERSION}}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "${{ASCEND_CANN_PACKAGE_PATH}}" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "{build_dir}")

{_cmake_ascendc_dir_block()}

include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)

ascendc_library(kernels STATIC
{_format_cmake_list(source_lines)}
)

ascendc_include_directories(kernels PRIVATE
{_format_cmake_list(include_lines)}
    ${{ASCEND_CANN_PACKAGE_PATH}}/include
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/runtime
    ${{ASCEND_CANN_PACKAGE_PATH}}/include/experiment/msprof
)

{_cmake_pybind11_block(entry_path, kernel_dir, module_name)}
"""


@dataclass
class BuildConfig:
    kernel_dir: Path
    build_dir: Path
    module_name: str
    sources: list[Path]
    ascend_path: Path
    entry_path: Path


def _generate_cmakelists(config: BuildConfig) -> str:
    include_dirs = _collect_include_dirs(config.kernel_dir)
    source_lines = [str(path) for path in config.sources]
    include_lines = [str(path) for path in include_dirs]
    return _render_cmakelists_template(
        build_dir=config.build_dir,
        source_lines=source_lines,
        include_lines=include_lines,
        entry_path=config.entry_path,
        kernel_dir=config.kernel_dir,
        module_name=config.module_name,
    )


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    logger.info("[build_ascendc] Running: %s", ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _make_cmake_args(source_dir, build_dir, soc_version, ascend_path, build_type):
    return [
        "cmake", "-S", str(source_dir), "-B", str(build_dir),
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_path}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]


class BuildContext(NamedTuple):
    kernel_dir: Path
    build_dir: Path
    ascend_path: Path
    soc_version: str
    build_type: str
    task_dir: Path
    env: dict
    clean: bool


def _build_native(ctx: BuildContext):
    """Mode A: kernel/CMakeLists.txt already exists, use directly."""
    native_cmake = ctx.kernel_dir / "CMakeLists.txt"
    if not native_cmake.is_file():
        return False
    if ctx.clean and ctx.build_dir.exists():
        shutil.rmtree(ctx.build_dir)
    ctx.build_dir.mkdir(parents=True, exist_ok=True)
    cmake_configure = _make_cmake_args(
        ctx.kernel_dir, ctx.build_dir, ctx.soc_version, ctx.ascend_path, ctx.build_type)
    _run(cmake_configure, cwd=ctx.task_dir, env=ctx.env)
    _run(["cmake", "--build", str(ctx.build_dir), "-j"], cwd=ctx.task_dir, env=ctx.env)
    return True


def _build_autogen(ctx: BuildContext):
    """Mode B: no CMakeLists.txt, auto-generate and build."""
    entry_path = _find_pybind_or_register(ctx.kernel_dir)
    if entry_path is None:
        raise FileNotFoundError(
            f"No entry file (register.cpp or pybind11.cpp) found in {ctx.kernel_dir}"
        )

    sources = _find_kernel_sources(ctx.kernel_dir)
    if entry_path.name == "register.cpp":
        module_name = f"{ctx.task_dir.name}_ext"
    else:
        module_name = _extract_pybind_module_name(entry_path)

    cmake_dir = ctx.build_dir / "_autogen_cmake"
    if ctx.clean and ctx.build_dir.exists():
        shutil.rmtree(ctx.build_dir)

    cmake_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = cmake_dir / "CMakeLists.txt"
    cmakelists_path.write_text(
        _generate_cmakelists(
            BuildConfig(
                kernel_dir=ctx.kernel_dir, build_dir=ctx.build_dir,
                module_name=module_name, sources=sources,
                ascend_path=ctx.ascend_path, entry_path=entry_path,
            ),
        ),
        encoding="utf-8",
    )

    cmake_configure = _make_cmake_args(
        cmake_dir, ctx.build_dir, ctx.soc_version, ctx.ascend_path, ctx.build_type)
    _run(cmake_configure, cwd=ctx.task_dir, env=ctx.env)
    _run(["cmake", "--build", str(ctx.build_dir), "-j"], cwd=ctx.task_dir, env=ctx.env)


def build(task: str, soc_version: str, build_type: str, clean: bool) -> Path:
    task_dir = _resolve_task_dir(task)
    kernel_dir = task_dir / "kernel"
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    build_dir = kernel_dir / "build"
    ascend_path = _detect_ascend_path()
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_path)

    ctx = BuildContext(
        kernel_dir=kernel_dir, build_dir=build_dir,
        ascend_path=ascend_path, soc_version=soc_version,
        build_type=build_type, task_dir=task_dir,
        env=env, clean=clean,
    )
    if _build_native(ctx):
        return build_dir
    _build_autogen(ctx)
    return build_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AscendC kernels for a task without task-local run.sh")
    parser.add_argument("task", help="Task directory name or path")
    parser.add_argument("-v", "--soc-version", default="Ascend910B2", help="Ascend SoC version")
    parser.add_argument("--build-type", default="Debug", help="CMake build type")
    parser.add_argument("--clean", action="store_true", help="Remove kernel/build before configuring")
    args = parser.parse_args()

    build_dir = build(
        task=args.task,
        soc_version=args.soc_version,
        build_type=args.build_type,
        clean=args.clean,
    )
    logger.info("[build_ascendc] Build completed: %s", build_dir)


if __name__ == "__main__":
    main()
