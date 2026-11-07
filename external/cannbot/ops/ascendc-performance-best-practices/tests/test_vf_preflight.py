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
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vf_preflight.py"
SPEC = importlib.util.spec_from_file_location("vf_preflight", SCRIPT)
assert SPEC and SPEC.loader
VF_PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VF_PREFLIGHT)


class VfPreflightTest(unittest.TestCase):
    def make_project(self, command: str, source: str) -> Path:
        root = Path(self.temp_dir.name)
        (root / "kernel.cpp").write_text(source, encoding="utf-8")
        (root / "compile_commands.json").write_text(
            json.dumps([{"directory": str(root), "command": command, "file": "kernel.cpp"}]),
            encoding="utf-8",
        )
        return root

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_xasc_rejects_host_visible_cast_trait(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "constexpr AscendC::Reg::CastTrait kTrait = {};\n",
        )
        report = VF_PREFLIGHT.inspect(project)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("xasc_host_visible_casttrait", {item["code"] for item in report["blockers"]})

    def test_xasc_accepts_arch_guarded_trait(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "#if (__NPU_ARCH__ == 3510)\nconstexpr AscendC::Reg::CastTrait kTrait = {};\n#endif\n",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project)["status"], "pass")

    def test_xasc_rejects_host_visible_reg_surface_as_contract_leakage(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            '#include "radix.h"\n',
        )
        (project / "radix.h").write_text(
            """using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
__aicore__ inline void Compute(__local_mem__ uint32_t* src);
""",
            encoding="utf-8",
        )
        report = VF_PREFLIGHT.inspect(project, expect_vf=True)
        blockers = {item["code"]: item for item in report["blockers"]}
        self.assertEqual(report["status"], "blocked")
        self.assertIn("xasc_host_pass_device_api_leakage", blockers)
        self.assertIn(
            "does not mean that the SDK API is missing",
            blockers["xasc_host_pass_device_api_leakage"]["message"],
        )
        self.assertEqual(blockers["xasc_host_pass_device_api_leakage"]["locations"][0]["path"], "radix.h")

    def test_xasc_accepts_arch_guarded_device_only_reg_surface(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            """#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
__aicore__ inline void Compute(__local_mem__ uint32_t* src);
#endif
""",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project, expect_vf=True)["status"], "pass")

    def test_xasc_accepts_reg_header_reached_only_through_arch_guarded_include(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            """#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "radix.h"
#endif
""",
        )
        (project / "radix.h").write_text(
            """using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
__aicore__ inline void Compute(__local_mem__ uint32_t* src);
""",
            encoding="utf-8",
        )
        report = VF_PREFLIGHT.inspect(project, expect_vf=True)
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["blockers"])

    def test_xasc_rejects_header_with_guarded_and_unguarded_include_paths(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            """#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "wrapper.h"
#endif
#include "radix.h"
""",
        )
        (project / "wrapper.h").write_text('#include "radix.h"\n', encoding="utf-8")
        (project / "radix.h").write_text("using AscendC::Reg::CreateMask;\n", encoding="utf-8")
        report = VF_PREFLIGHT.inspect(project, expect_vf=True)
        self.assertIn(
            "xasc_host_pass_device_api_leakage",
            {item["code"] for item in report["blockers"]},
        )

    def test_device_only_accepts_file_scope_reg_surface(self) -> None:
        project = self.make_project(
            "bisheng --cce-aicore-arch=dav-3510 --cce-aicore-only -c kernel.cpp",
            "using AscendC::Reg::CreateMask;\nusing AscendC::Reg::MaskReg;\n",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project, expect_vf=True)["status"], "pass")

    def test_xasc_accepts_function_scope_static_constexpr_trait(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            """__aicore__ inline void Compute() {
  static constexpr AscendC::Reg::CastTrait kTrait = {};
  __VEC_SCOPE__ { AscendC::Reg::RegTensor<float> value; }
}
""",
        )
        report = VF_PREFLIGHT.inspect(project, expect_vf=True, expect_cast=True)
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["blockers"])

    def test_xasc_accepts_static_trait_on_same_line_as_device_function(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "__aicore__ inline void Compute() { static constexpr AscendC::Reg::CastTrait kTrait = {}; }\n",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project, expect_cast=True)["status"], "pass")

    def test_device_only_accepts_namespace_trait(self) -> None:
        project = self.make_project(
            "bisheng --cce-aicore-arch=dav-3510 --cce-aicore-only -c kernel.cpp",
            "constexpr AscendC::Reg::CastTrait kTrait = {};\n",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project)["status"], "pass")

    def test_manual_namespace_alias_is_blocked(self) -> None:
        project = self.make_project(
            "bisheng --cce-aicore-arch=dav-3510 --cce-aicore-only -c kernel.cpp",
            "namespace AscendC { namespace Reg = MicroAPI; }\nAscendC::Reg::RegTensor<float> value;\n",
        )
        report = VF_PREFLIGHT.inspect(project)
        self.assertIn("manual_namespace_alias", {item["code"] for item in report["blockers"]})

    def test_uncompiled_neighbor_does_not_block_target(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "__aicore__ void kernel() { __VEC_SCOPE__ { AscendC::Reg::RegTensor<float> value; } }\n",
        )
        (project / "unrelated.cpp").write_text("constexpr AscendC::Reg::CastTrait kTrait = {};\n", encoding="utf-8")
        self.assertEqual(VF_PREFLIGHT.inspect(project)["status"], "pass")

    def test_comments_do_not_create_vf_findings(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "// AscendC::Reg::CastTrait is discussed here only\n/* MicroAPI::CastTrait */\nint value;\n",
        )
        self.assertEqual(VF_PREFLIGHT.inspect(project)["status"], "not_applicable")

    def test_expected_vf_cannot_be_removed_to_get_not_applicable(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "int value;\n",
        )
        report = VF_PREFLIGHT.inspect(project, expect_vf=True)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("required_vf_missing", {item["code"] for item in report["blockers"]})

    def test_expected_casttrait_is_a_separate_gate(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "__aicore__ void kernel() { __VEC_SCOPE__ { AscendC::Reg::RegTensor<float> value; } }\n",
        )
        report = VF_PREFLIGHT.inspect(project, expect_vf=True, expect_cast=True)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("required_casttrait_missing", {item["code"] for item in report["blockers"]})

    def test_negative_arch_guard_does_not_hide_risky_trait(self) -> None:
        project = self.make_project(
            "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
            "#if !defined(__NPU_ARCH__)\nconstexpr AscendC::Reg::CastTrait kTrait = {};\n#endif\n",
        )
        report = VF_PREFLIGHT.inspect(project)
        self.assertIn("xasc_host_visible_casttrait", {item["code"] for item in report["blockers"]})

    def test_finds_nested_build_compile_database(self) -> None:
        root = Path(self.temp_dir.name)
        source_dir = root / "nested"
        build_dir = source_dir / "build"
        build_dir.mkdir(parents=True)
        (source_dir / "kernel.cpp").write_text(
            "__aicore__ void kernel() { __VEC_SCOPE__ { AscendC::Reg::RegTensor<float> value; } }\n",
            encoding="utf-8",
        )
        (build_dir / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(source_dir),
                        "command": "bisheng --npu-arch=dav-3510 -xasc -c kernel.cpp",
                        "file": "kernel.cpp",
                    }
                ]
            ),
            encoding="utf-8",
        )
        report = VF_PREFLIGHT.inspect(root)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["contracts"], {"xasc_combined": 1})

    def test_scans_project_header_reached_through_include_root(self) -> None:
        root = Path(self.temp_dir.name)
        (root / "src").mkdir()
        (root / "include" / "shared").mkdir(parents=True)
        (root / "src" / "kernel.cpp").write_text('#include "shared/vf.h"\n', encoding="utf-8")
        (root / "include" / "shared" / "vf.h").write_text(
            "constexpr AscendC::Reg::CastTrait kTrait = {};\n", encoding="utf-8"
        )
        (root / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(root),
                        "command": "bisheng --npu-arch=dav-3510 -xasc -Iinclude -c src/kernel.cpp",
                        "file": "src/kernel.cpp",
                    }
                ]
            ),
            encoding="utf-8",
        )
        report = VF_PREFLIGHT.inspect(root)
        self.assertIn("xasc_host_visible_casttrait", {item["code"] for item in report["blockers"]})

    def test_unrelated_device_only_command_cannot_satisfy_wrapper_only_vf_unit(self) -> None:
        root = Path(self.temp_dir.name)
        (root / "vf.cpp").write_text(
            "__aicore__ void vf() { __VEC_SCOPE__ { AscendC::Reg::RegTensor<float> value; } }\n",
            encoding="utf-8",
        )
        (root / "other.cpp").write_text("int other;\n", encoding="utf-8")
        (root / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(root),
                        "command": "bisheng -x cce -c vf.cpp",
                        "file": "vf.cpp",
                    },
                    {
                        "directory": str(root),
                        "command": "bisheng --cce-aicore-only -c other.cpp",
                        "file": "other.cpp",
                    },
                ]
            ),
            encoding="utf-8",
        )

        report = VF_PREFLIGHT.inspect(root)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("backend_contract_missing", {item["code"] for item in report["blockers"]})
        self.assertEqual(report["vf_translation_units"][0]["path"], "vf.cpp")
        self.assertEqual(report["vf_translation_units"][0]["contracts"], {"cce_wrapper": 1})

    def test_unclassified_compile_entry_is_still_scanned(self) -> None:
        root = Path(self.temp_dir.name)
        (root / "vf.cpp").write_text("AscendC::Reg::RegTensor<float> value;\n", encoding="utf-8")
        (root / "other.cpp").write_text("int other;\n", encoding="utf-8")
        (root / "compile_commands.json").write_text(
            json.dumps(
                [
                    {"directory": str(root), "command": "unknowncc -c vf.cpp", "file": "vf.cpp"},
                    {
                        "directory": str(root),
                        "command": "bisheng --cce-aicore-only -c other.cpp",
                        "file": "other.cpp",
                    },
                ]
            ),
            encoding="utf-8",
        )

        report = VF_PREFLIGHT.inspect(root)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("compile_contract_unresolved", {item["code"] for item in report["blockers"]})

    def test_different_final_contracts_on_different_vf_units_are_mapped_independently(self) -> None:
        root = Path(self.temp_dir.name)
        (root / "device.cpp").write_text("AscendC::Reg::RegTensor<float> value;\n", encoding="utf-8")
        (root / "combined.cpp").write_text(
            "#if (__NPU_ARCH__ == 3510)\nAscendC::Reg::CastTrait trait = {};\n#endif\n",
            encoding="utf-8",
        )
        (root / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(root),
                        "command": "bisheng --cce-aicore-only -c device.cpp",
                        "file": "device.cpp",
                    },
                    {
                        "directory": str(root),
                        "command": "bisheng --npu-arch=dav-3510 -xasc -c combined.cpp",
                        "file": "combined.cpp",
                    },
                ]
            ),
            encoding="utf-8",
        )

        report = VF_PREFLIGHT.inspect(root)

        self.assertEqual(report["status"], "pass")
        self.assertNotIn("multiple_build_contracts", {item["code"] for item in report["warnings"]})
        unit_contracts = {item["path"]: item["contracts"] for item in report["vf_translation_units"]}
        self.assertEqual(unit_contracts["device.cpp"], {"device_only": 1})
        self.assertEqual(unit_contracts["combined.cpp"], {"xasc_combined": 1})


if __name__ == "__main__":
    unittest.main()
