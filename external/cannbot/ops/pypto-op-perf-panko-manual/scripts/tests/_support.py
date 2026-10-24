# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared test support: the harness under test, the paths to it, an envelope for
a run with no device, and a writer for synthetic platform inis.

`init` DERIVES the chip envelope from the platform ini and refuses when it
cannot, rather than falling back to a transcribed default -- so a test running
off-device has to supply one, exactly as a user on a box without CANN would.
`CHIP_ENVELOPE_OFFLINE` is that supplied envelope. It describes no device: it is
a set of round numbers large enough for the toy kernels these tests compile.

`write_platform_ini` writes the ini SHAPE -- the sections and option names
`chip_profile` reads -- with values chosen for the test. No real SKU's figures
are stored in this repository, and no ini ships as a file: a test that needs one
writes it into its own temporary directory.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

HARNESS = SCRIPTS_DIR / "panko_harness.py"
CATALOG = SCRIPTS_DIR.parent / "references" / "action_catalog.json"

# The harness loaded once, by path, so a test can call the internals it exists
# to test. Every test file that needs it imports this one rather than repeating
# the eight lines of spec/module/exec_module boilerplate.
_spec = importlib.util.spec_from_file_location("panko_harness", HARNESS)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

CHIP_ENVELOPE_OFFLINE = {
    "ub_kb": 192,
    "l1_kb": 512,
    "l0a_kb": 64,
    "l0b_kb": 64,
    "l0c_kb": 128,
    "cube_cores": 20,
    "vector_cores": 40,
}

# Append to any `init` argv in a test.
CHIP_ARGS = ["--chip-envelope", json.dumps(CHIP_ENVELOPE_OFFLINE)]


def write_platform_ini(
    directory,
    soc,
    *,
    cube=4,
    vector=8,
    ub=32768,
    l0a=8192,
    l0b=8192,
    l0c=16384,
    l1=65536,
    arch="0000",
    short=None,
    vector_ub=None,
):
    """Write `<soc>.ini` into `directory` and return its path.

    The sections and option names are the ones `chip_profile.FIELDS` and
    `COUNTS` read; the numbers are the test's, not any device's. Sizes are
    written in BYTES, which is what an ini holds and what `read_ini` converts.

    `vector_ub` adds a second `[VectorCoreSpec] ub_size`, which some platforms
    carry and others do not -- the case that makes a flat key scan ambiguous.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = [
        "[SoCInfo]",
        f"ai_core_cnt={cube}",
        f"cube_core_cnt={cube}",
        f"vector_core_cnt={vector}",
        "",
        "[AICoreSpec]",
        f"ub_size={ub}",
        f"l0_a_size={l0a}",
        f"l0_b_size={l0b}",
        f"l0_c_size={l0c}",
        f"l1_size={l1}",
        "",
    ]
    if vector_ub is not None:
        body += ["[VectorCoreSpec]", f"ub_size={vector_ub}", ""]
    body += [
        "[version]",
        f"SoC_version={soc}",
        f"Short_SoC_version={short or soc}",
        f"NpuArch={arch}",
        "",
    ]
    path = directory / f"{soc}.ini"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def init_argv(op_dir, op_file, *extra, catalog=CATALOG):
    """The standard `init` argv of a demo campaign: the reference numbers every
    test shares, `--catalog` included unless catalog=None (the path-resolution
    tests omit it on purpose, because the default is the thing under test).
    Extra flags go last.
    """
    argv = [
        "init",
        "--op-dir",
        str(op_dir),
        "--op",
        "demo",
        "--p-ref",
        "100",
        "--preopt-p",
        "500",
    ]
    if catalog is not None:
        argv += ["--catalog", str(catalog)]
    argv += ["--allow-unversioned", "--op-file", str(op_file)]
    return argv + list(extra)


def record_argv(op_dir, digest, p):
    """The `record` argv of one measured candidate of the demo campaign."""
    return [
        "record",
        "--op-dir",
        str(op_dir),
        "--parent",
        "u1",
        "--code-hash",
        digest,
        "--s",
        "1",
        "--p",
        p,
    ]


def feasible_argv(op_dir, op_file):
    """The `feasible` argv of one candidate of the demo campaign."""
    return [
        "feasible",
        "--op-dir",
        str(op_dir),
        "--parent",
        "u1",
        "--op-file",
        str(op_file),
    ]


def run_cli(*args, cwd=None, env=None):
    """Run the harness CLI and return (payload, exit code).

    Every CLI test needs exactly this, and a copy of it in each file is a copy of
    the contract: one JSON object on stdout, and the exit code carries the
    refusal. `cwd` and `env` set what the run can see; a test that has to invoke
    a DIFFERENT copy of the script -- the path-resolution tests run the installed
    symlink -- builds its own argv, because which script runs is the thing those
    tests are about.
    """
    r = subprocess.run(
        [sys.executable, str(HARNESS), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout), r.returncode


def file_hash(path):
    """The code hash the harness computes for a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_state(op_dir):
    """The search state the harness wrote under `op_dir`."""
    return json.loads(
        (Path(op_dir) / "optimization" / "search_state.json").read_text(
            encoding="utf-8"
        )
    )
