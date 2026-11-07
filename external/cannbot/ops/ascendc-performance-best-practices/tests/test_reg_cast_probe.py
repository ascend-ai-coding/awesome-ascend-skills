# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_reg_cast_probe.py"
SPEC = importlib.util.spec_from_file_location("run_reg_cast_probe", SCRIPT)
assert SPEC and SPEC.loader
REG_CAST_PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REG_CAST_PROBE)


def test_probe_sources_preserve_the_static_storage_distinction(tmp_path: Path) -> None:
    positive = REG_CAST_PROBE.POSITIVE_PROBE.read_text(encoding="utf-8")
    control = REG_CAST_PROBE.CONTROL_PROBE.read_text(encoding="utf-8")
    assert "static constexpr AscendC::Reg::CastTrait" in positive
    assert "constexpr AscendC::Reg::CastTrait" in control
    assert "static constexpr AscendC::Reg::CastTrait" not in control
    assert "AscendC::Reg::Cast<float, half, kWiden>" in positive


def test_rewrites_real_xasc_command_without_reusing_target_or_dependency_outputs(tmp_path: Path) -> None:
    source = tmp_path / "kernel.cpp"
    source.write_text("int value;\n", encoding="utf-8")
    database = tmp_path / "compile_commands.json"
    entry = {
        "directory": str(tmp_path),
        "file": "kernel.cpp",
        "command": ("bisheng -Iinclude -O3 --npu-arch=dav-3510 -xasc -MMD -MF old.d -MT old.o -o old.o -c kernel.cpp"),
    }
    probe = REG_CAST_PROBE.POSITIVE_PROBE
    output = tmp_path / "probe.o"

    command, directory = REG_CAST_PROBE.build_probe_command(entry, probe, output, database)

    assert directory == tmp_path
    assert "-xasc" in command
    assert "--npu-arch=dav-3510" in command
    assert str(source) not in command
    assert "kernel.cpp" not in command
    assert "old.o" not in command
    assert "old.d" not in command
    assert str(probe.resolve()) in command
    assert command[-2:] == ["-o", str(output.resolve())]
