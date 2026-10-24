#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The chip envelope, DERIVED from the platform ini rather than transcribed.

The rule:

    Read the cube, HBM, UB, L1 and L0 figures out of the platform ini that ships
    with the runtime you are measuring, so no constant lives in this page and
    detaches from the installed version. Resolve <SKU> from the detected device
    rather than assuming it, and record which ini you read alongside the number.

PANKO transcribed six constants instead, taken from one SKU. Checking them back
against that SKU's ini says only that the transcription was faithful; it says
nothing about any other part.

They are already wrong on another SKU in the same family. Two SKUs can share an
NpuArch and carry the same five buffer sizes while differing in core count, so a
transcribed count understates one of them on every tiling decision the search
makes -- and no key coarser than the full SoC name would have caught it.

Per-SKU is not optional, and per-generation does not help either: two SKUs of one
generation can carry identical buffers and core counts that differ several fold.
Only some of the six figures move between generations at all, which is why
transcription survived this long without anyone noticing.

Resolution is direct: `acl.get_soc_name()` returns exactly the ini's basename,
so `<platform_config>/<soc>.ini` needs no mapping table.
"""

import configparser
import glob
import importlib
import os

# Section-scoped on purpose. On 950 SKUs `ub_size` appears in BOTH [AICoreSpec]
# and a [VectorCoreSpec] that 910B3 does not have, so a flat key scan gets two
# hits -- and today they happen to agree, which is exactly how that kind of bug
# reaches production.
FIELDS = {
    "ub_kb": ("AICoreSpec", "ub_size"),
    "l1_kb": ("AICoreSpec", "l1_size"),
    "l0a_kb": ("AICoreSpec", "l0_a_size"),
    "l0b_kb": ("AICoreSpec", "l0_b_size"),
    "l0c_kb": ("AICoreSpec", "l0_c_size"),
}
COUNTS = {
    "cube_cores": ("SoCInfo", "cube_core_cnt"),
    "vector_cores": ("SoCInfo", "vector_core_cnt"),
}


def _txt(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def optional(module_name):
    """The module, or None when it is not installed.

    `acl` and `torch_npu` ship with a runtime this may not be running on, and
    their absence is an answer rather than an error: the caller falls through to
    the next route. Returning None says that, where a swallowed exception would
    only have hidden it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def quietly(fn, *args):
    """`fn(*args)`, or None if the runtime refused to answer.

    Every caller here is asking a driver a question it may decline -- a pyACL
    build that takes no device argument, a device that is not up. The refusal is
    the information, and the next route is tried; there is nothing to report and
    nothing to re-raise.
    """
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None


def live_soc(device=None):
    """(soc, how) straight off the silicon, or ("", "").

    `device` is the one the run will MEASURE on -- the review's first complaint
    is that PANKO takes TILE_FWK_DEVICE_ID and never asks what is behind it.
    Asking without the id asks "what chip is in this box", which is the same
    question only while the box is homogeneous.

    The device-scoped call is tried first and its failure is not fatal: not
    every pyACL build accepts the argument, and a name for the box is still
    better than no name. `how` says which one answered, so a state file never
    claims a device was identified when the box was.

    Deliberately does NOT read PANKO_SOC_NAME. This is what a name gets checked
    AGAINST; a name checked against itself is not a check.
    """
    acl = optional("acl")
    if acl is not None:
        if device not in (None, ""):
            name = quietly(acl.get_soc_name, int(device))
            if name:
                return _txt(name), f"acl(device={device})"
        name = quietly(acl.get_soc_name)
        if name:
            return _txt(name), "acl"
    torch_npu = optional("torch_npu")
    if torch_npu is not None:
        name = quietly(
            torch_npu.npu.get_device_name,
            int(device) if device not in (None, "") else 0,
        )
        if name:
            return _txt(name), "torch_npu"
    return "", ""


def _core_count(impl, name):
    """One core-count binding's answer, or None when it is absent or refuses."""
    fn = getattr(impl, name, None)
    return quietly(fn) if callable(fn) else None


def live_core_counts():
    """(cube, vector, how) off pypto's platform bindings, or (None, None, "").

    pypto answers for the device that is UP; the ini answers for the SKU whose
    name resolved. That is the difference this module exists for -- two SKUs can
    share an NpuArch and every buffer size and differ in core count -- and the
    live figure is the one the kernel will run on, so it wins where both answer.

    Buffer sizes are NOT taken this way. `GetMemoryLimitForArch` is keyed on the
    compilation arch, and that arch maps 910B and 910C to one value, which is the
    granularity this module was written to stop using.
    """
    pypto = optional("pypto")
    impl = getattr(pypto, "pypto_impl", None) if pypto is not None else None
    if impl is None:
        return None, None, ""
    cube = _core_count(impl, "GetAICCoreNum")
    vector = _core_count(impl, "GetAIVCoreNum")
    if not isinstance(cube, int) or not isinstance(vector, int):
        return None, None, ""
    if cube <= 0 or vector <= 0:
        return None, None, ""
    return cube, vector, "pypto_impl"


def soc_name(device=None):
    """The SoC as the runtime reports it, which is the ini's basename.

    The env var comes first because it is for a caller naming a chip
    deliberately -- reading an ini for a box they are not on. That is a legal
    thing to do and a dangerous one, which is why `confirm()` exists.
    """
    env = os.environ.get("PANKO_SOC_NAME", "").strip()
    if env:
        return env, "PANKO_SOC_NAME"
    return live_soc(device)


def confirm(expected, device=None):
    """Is the chip about to be measured the one the envelope describes?

    ("ok" | "mismatch" | "unknown", live_soc, how). "unknown" is not a failure:
    a box with no pyACL can still be searched, and the envelope came from a
    named SoC either way. "mismatch" is a failure, and the more so because the
    ways to get one -- PANKO_SOC_NAME, --chip-envelope -- are exactly the ways
    a human supplies a name by hand.
    """
    live, how = live_soc(device)
    if not live:
        return "unknown", "", ""
    if not expected:
        return "unknown", live, how
    return ("ok" if live == expected else "mismatch"), live, how


def _cann_roots():
    roots = []
    for var in (
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_HOME",
        "ASCEND_HOME_PATH",
        "ASCEND_CANN_HOME",
    ):
        v = os.environ.get(var)
        if v and v not in roots:
            roots.append(v)
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if opp.endswith("/opp") and opp[:-4] not in roots:
        roots.append(opp[:-4])
    return roots


def _config_dirs_under(root):
    """The `data/platform_config` directories under one CANN root.

    The arch directory (aarch64-linux / x86_64-linux) is globbed rather than
    guessed, so both layouts are covered by the same two patterns.
    """
    found = []
    for pat in (
        os.path.join(root, "*", "data", "platform_config"),
        os.path.join(root, "data", "platform_config"),
    ):
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d):
                found.append(d)
    return found


def platform_config_dirs():
    """Every `data/platform_config` reachable from the CANN environment.

    The arch directory (aarch64-linux / x86_64-linux) is globbed rather than
    guessed. NOTE: this is the CANN location, and it is the one to read.
    a5-roofline-and-levers.md names `<pypto root>/framework/src/platform/parser/
    simulation_platform/platform_config/`, which exists only in a pypto SOURCE
    checkout -- the installed package ships no `framework/` -- and even there it
    holds family-level inis rather than per-SKU ones. A family-level ini reports
    one member's `SoC_version` and that member's core count, so it cannot answer
    what a sibling SKU in the same family has. The CANN directory carries the
    SKU-level inis -- dozens of them on the box this was checked against.
    """
    out = []
    for root in _cann_roots():
        for d in _config_dirs_under(root):
            if d not in out:
                out.append(d)
    return out


def find_ini(soc, dirs=None):
    for d in dirs if dirs is not None else platform_config_dirs():
        p = os.path.join(d, f"{soc}.ini")
        if os.path.isfile(p):
            return p
    return ""


def read_ini(path):
    """{field: value} in KB / counts, or (None, reason).

    Every size in the ini is BYTES. Returning KB keeps the rest of the harness
    unchanged, and the conversion is exact for every value observed.
    """
    cp = configparser.ConfigParser(strict=False)
    cp.optionxform = str
    try:
        if not cp.read(path, encoding="utf-8"):
            return None, f"{path} could not be read"
    except configparser.Error as e:
        return None, f"{path} is not a readable ini: {e}"

    env, missing = {}, []
    for key, (sec, opt) in FIELDS.items():
        if cp.has_option(sec, opt):
            try:
                env[key] = int(cp.get(sec, opt)) // 1024
            except ValueError:
                missing.append(f"[{sec}]{opt} is not an integer")
        else:
            missing.append(f"[{sec}]{opt}")
    for key, (sec, opt) in COUNTS.items():
        if cp.has_option(sec, opt):
            try:
                env[key] = int(cp.get(sec, opt))
            except ValueError:
                missing.append(f"[{sec}]{opt} is not an integer")
        else:
            missing.append(f"[{sec}]{opt}")
    if missing:
        return None, f"{os.path.basename(path)} is missing: {', '.join(missing)}"
    env["source"] = path
    return env, ""


def resolve(dirs=None, device=None):
    """(envelope, reason). envelope is None when it could not be derived.

    The caller must NOT fall back to a transcribed default on a None. The review
    is explicit: running an A5 silently under A3 numbers is the failure, and
    every legal candidate it rejects is rejected before any device sees it.
    """
    soc, how = soc_name(device)
    if not soc:
        return None, (
            "the SoC could not be resolved: neither `acl.get_soc_name()` "
            "nor `torch_npu.npu.get_device_name()` answered for device "
            f"{device!r}, and PANKO_SOC_NAME is unset"
        )
    candidates = dirs if dirs is not None else platform_config_dirs()
    if not candidates:
        return None, (
            "no CANN platform_config directory found; set "
            "ASCEND_TOOLKIT_HOME or source the CANN set_env.sh"
        )
    path = find_ini(soc, candidates)
    if not path:
        return None, (
            f"no {soc}.ini under "
            + ", ".join(candidates)
            + f" (SoC resolved via {how})"
        )
    env, why = read_ini(path)
    if env is None:
        return None, why
    env["soc"] = soc
    env["soc_via"] = how
    cube, vector, counted_by = live_core_counts()
    if cube is not None:
        env["cube_cores"], env["vector_cores"] = cube, vector
    env["cores_via"] = counted_by or "ini"
    return env, ""
