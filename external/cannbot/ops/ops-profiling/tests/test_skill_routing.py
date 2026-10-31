# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProfilingSkillRoutingTest(unittest.TestCase):
    def test_upstream_base_capabilities_are_present(self) -> None:
        required = (
            "evals/evals.json",
            "references/msprof-guide.md",
            "references/msprof-op-guide.md",
            "scripts/msprof_profile_run.sh",
            "scripts/msprof_perf_summary.py",
            "scripts/perf_summary.py",
            "msopprof-visualization/SKILL.md",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_route_precedence_is_explicit(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        ordered_phrases = (
            "先探测工具可用性",
            "外部输入显式指定",
            "只有 msopprof 可用",
            "只有 msprof 可用",
            "两者都可用",
            "两者都不可用",
        )
        positions = [skill.index(phrase) for phrase in ordered_phrases]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("不得凭固定偏好选路", skill)
        self.assertIn("向用户确认，或遵循项目既有约定", skill)
        self.assertIn("CANN / ASCEND_HOME 环境问题", skill)
        self.assertIn("回落到 `msprof op`", skill)
        self.assertIn("本 skill 不解析其业务约束，也不覆盖其选择", skill)
        self.assertNotIn("优先使用 `msprof`", skill)
        self.assertNotIn("默认选中 `msprof`", skill)

    def test_new_protocol_is_scoped_to_msprof_application(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        msprof_guide = (ROOT / "references/msprof-guide.md").read_text(encoding="utf-8")
        msprof_op_guide = (ROOT / "references/msprof-op-guide.md").read_text(encoding="utf-8")
        for text in (skill, msprof_guide):
            self.assertIn("10 组", text)
            self.assertIn("中位数", text)
            self.assertIn("collect_perf_application.py", text)
        self.assertNotIn("collect_perf_application.py", msprof_op_guide)
        self.assertNotIn("median_target_total_us", msprof_op_guide)
        self.assertNotIn("必须用 `msprof` 而非 `msprof op`", msprof_guide)


if __name__ == "__main__":
    unittest.main()
