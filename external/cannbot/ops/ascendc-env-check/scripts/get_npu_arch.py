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
"""Query the NPU architecture (dav-*) of the current environment via libascend_hal.so.

Usage:
    python3 get_npu_arch.py          # prints e.g. "dav-3510"
    python3 get_npu_arch.py --raw    # prints raw NpuArch number e.g. "3510"

Requires:
    - Ascend driver installed (libascend_hal.so)
    - CANN toolkit installed (for platform_config/*.ini)
"""

import ctypes
import logging
import os
import platform
import sys

_LOGGER = logging.getLogger(__name__)

MAX_CHIP_NAME = 32


class HalChipInfo(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_char * MAX_CHIP_NAME),
        ("name", ctypes.c_char * MAX_CHIP_NAME),
        ("version", ctypes.c_char * MAX_CHIP_NAME),
    ]


SOC_TO_NPUARCH = {
    "Ascend910": 1001,
    "Ascend310P": 2002,
    "Ascend310P1": 2002,
    "Ascend310P3": 2002,
    "Ascend310P5": 2002,
    "Ascend310P7": 2002,
    "Ascend910B": 2201,
    "Ascend910B1": 2201,
    "Ascend910B2": 2201,
    "Ascend910B2C": 2201,
    "Ascend910B3": 2201,
    "Ascend910B4": 2201,
    "Ascend910_93": 2201,
    "Ascend310B": 3002,
    "Ascend310B1": 3002,
    "Ascend310B2": 3002,
    "Ascend310B3": 3002,
    "Ascend310B4": 3002,
    "Ascend950": 3510,
}


def _derive_from_opp_path():
    """Derive toolkit path from ASCEND_OPP_PATH.

    ASCEND_OPP_PATH is always set by set_env.sh as {toolkit_path}/opp.
    Stripping /opp gives the toolkit root. This is the most reliable signal
    because it points to the opp of the *actually used* toolkit.
    """
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if opp and opp.endswith("/opp"):
        toolkit = opp[:-4]
        if os.path.isdir(toolkit) and os.path.isdir(os.path.join(toolkit, "compiler")):
            return toolkit
    return None


def get_cann_home():
    """Locate CANN toolkit installation directory.

    Resolution priority:
    1. ASCEND_TOOLKIT_HOME — explicit toolkit path (highest priority)
    2. ASCEND_HOME — set by set_env.sh, typically points to toolkit directly
    3. ASCEND_OPP_PATH — derive by stripping /opp (very reliable)
    4. ASCEND_HOME_PATH — may be top-level, needs resolution
    5. ASCEND_CANN_HOME — alternative explicit path

    ASCEND_HOME_PATH may point to the top-level directory (e.g. /usr/local/Ascend)
    or directly to the toolkit. This function resolves to the actual toolkit
    directory containing compiler/, lib64/, etc.
    """
    derived = _derive_from_opp_path()
    for var in (
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_HOME",
    ):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved

    if derived:
        return derived

    for var in (
        "ASCEND_HOME_PATH",
        "ASCEND_CANN_HOME",
    ):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved
    raise RuntimeError(
        "Cannot locate CANN toolkit installation. Set one of: "
        "ASCEND_TOOLKIT_HOME, ASCEND_HOME, ASCEND_HOME_PATH, ASCEND_CANN_HOME"
    )


def _resolve_toolkit_path(base_path):
    """Resolve the actual toolkit directory from a potentially top-level path.

    Strategies (in order):
    1. base_path itself contains compiler/ → it's already the toolkit
    2. base_path/ascend-toolkit/{version}/ directories → pick latest with compiler/
    3. base_path/ascend-toolkit/latest symlink → follow (fallback, may point to non-toolkit)
    4. base_path/cann-{version}/ directories → pick one (last resort)
    """
    if os.path.isdir(os.path.join(base_path, "compiler")):
        return base_path

    toolkit_dir = os.path.join(base_path, "ascend-toolkit")
    if os.path.isdir(toolkit_dir):
        candidates = []
        for d in os.listdir(toolkit_dir):
            if d == "latest":
                continue
            dpath = os.path.join(toolkit_dir, d)
            if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
                candidates.append(d)
        candidates.sort(reverse=True)
        if candidates:
            return os.path.join(toolkit_dir, candidates[0])

        latest = os.path.join(toolkit_dir, "latest")
        if os.path.islink(latest):
            real = os.path.realpath(latest)
            if os.path.isdir(real) and os.path.isdir(os.path.join(real, "compiler")):
                return real

    cann_candidates = []
    for d in os.listdir(base_path):
        if not d.startswith("cann-"):
            continue
        dpath = os.path.join(base_path, d)
        if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
            cann_candidates.append(d)
    cann_candidates.sort(reverse=True)
    if cann_candidates:
        return os.path.join(base_path, cann_candidates[0])

    return None


def get_arch_dir():
    arch = platform.machine()
    return f"{arch}-linux"


def load_hal(cann_home):
    """Load libascend_hal.so from driver or CANN paths."""
    arch_dir = get_arch_dir()
    lib_candidates = [
        "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
        os.path.join(cann_home, arch_dir, "devlib", "libascend_hal.so"),
        "libascend_hal.so",
    ]
    for lib_path in lib_candidates:
        try:
            return ctypes.cdll.LoadLibrary(lib_path)
        except OSError:
            continue
    raise RuntimeError(
        "Cannot load libascend_hal.so. Is the Ascend driver installed?"
    )


def get_chip_name(hal):
    """Query chip type+name via halGetChipInfo to build SocVersion string."""
    info = HalChipInfo()
    ret = hal.halGetChipInfo(0, ctypes.byref(info))
    if ret != 0:
        raise RuntimeError(f"halGetChipInfo failed with error code {ret}")
    chip_type = info.type.decode().strip()
    chip_name = info.name.decode().strip()
    return chip_type + chip_name


def read_npu_arch(cann_home, soc_version):
    """Read NpuArch from the platform_config .ini file.

    Falls back to SOC_TO_NPUARCH mapping table when the ini file
    does not contain an explicit NpuArch field (common in newer
    CANN versions where ini only has Arch_type).
    """
    config_dir = os.path.join(
        cann_home, get_arch_dir(), "data", "platform_config"
    )
    if not os.path.isdir(config_dir):
        raise RuntimeError(f"Platform config directory not found: {config_dir}")

    ini_path = os.path.join(config_dir, f"{soc_version}.ini")
    if not os.path.isfile(ini_path):
        raise RuntimeError(
            f"Platform config not found for SoC {soc_version} at {ini_path}"
        )

    fields = {}
    with open(ini_path, "r") as f:
        in_version_section = False
        for line in f:
            line = line.strip()
            if line == "[version]":
                in_version_section = True
                continue
            if in_version_section and line.startswith("["):
                break
            if in_version_section and "=" in line:
                key, val = line.split("=", 1)
                fields[key] = val

    npu_arch = fields.get("NpuArch", "")
    if npu_arch:
        return npu_arch

    short_soc = fields.get("Short_SoC_version", "")
    mapped = SOC_TO_NPUARCH.get(soc_version) or SOC_TO_NPUARCH.get(short_soc)
    if mapped:
        return str(mapped)

    raise RuntimeError(
        f"NpuArch not found in {ini_path} and no mapping for "
        f"SocVersion '{soc_version}' or Short_SoC_version '{short_soc}'"
    )


def _format_output(npu_arch: str, raw_mode: bool) -> str:
    if raw_mode:
        return npu_arch
    return f"dav-{npu_arch}"


def main() -> int:
    raw_mode = "--raw" in sys.argv

    try:
        cann_home = get_cann_home()
        hal = load_hal(cann_home)
        soc_version = get_chip_name(hal)
        npu_arch = read_npu_arch(cann_home, soc_version)
        _LOGGER.info(_format_output(npu_arch, raw_mode))
    except RuntimeError as e:
        _LOGGER.error("Error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
