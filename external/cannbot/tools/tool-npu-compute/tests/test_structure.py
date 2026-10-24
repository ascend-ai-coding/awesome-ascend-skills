# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import re
import shlex
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TEST_ROOT.parents[2]
SKILL_ROOT = REPOSITORY_ROOT / "tools" / "tool-npu-compute"
SKILL_PATH = SKILL_ROOT / "SKILL.md"
REFERENCE_ROOT = SKILL_ROOT / "references"
SCRIPT_ROOT = SKILL_ROOT / "scripts"
CLI_USAGE_PATH = REFERENCE_ROOT / "cli-usage.md"
COLLECTION_WORKFLOW_PATH = REFERENCE_ROOT / "collection-workflow.md"


def parse_frontmatter(text):
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise AssertionError("SKILL.md must start with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise AssertionError("SKILL.md frontmatter is not closed") from error

    frontmatter = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise AssertionError(f"invalid frontmatter line: {line}")
        frontmatter[key.strip()] = value.strip()
    return frontmatter


def markdown_links(text):
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def bash_examples(text):
    return "\n".join(re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL))


class SkillStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill_text = SKILL_PATH.read_text(encoding="utf-8")
        cls.frontmatter = parse_frontmatter(cls.skill_text)
        cls.links = markdown_links(cls.skill_text)

    def test_frontmatter_metadata(self):
        self.assertEqual(self.frontmatter.get("name"), "tool-npu-compute")
        self.assertRegex(self.frontmatter["name"], r"^[a-z0-9-]+$")
        self.assertLess(len(self.frontmatter["name"]), 64)
        self.assertTrue(self.frontmatter.get("description"))
        self.assertEqual(self.frontmatter.get("license"), "CANN-2.0")
        self.assertEqual(set(self.frontmatter), {"name", "description", "license"})

    def test_skill_is_concise(self):
        self.assertLess(len(self.skill_text.splitlines()), 500)

    def test_all_local_links_exist(self):
        local_links = [
            link
            for link in self.links
            if "://" not in link and not link.startswith("#")
        ]
        self.assertTrue(local_links)
        for link in local_links:
            target = (SKILL_ROOT / link.split("#", 1)[0]).resolve()
            self.assertTrue(target.is_file(), link)

    def test_resources_are_linked_directly_from_skill(self):
        expected = {
            "references/metric-foundations.md",
            "references/arithmetic-utilization.md",
            "references/pipe-utilization.md",
            "references/pipeline.md",
            "references/resource-conflict-ratio.md",
            "references/memory.md",
            "references/memory-l0.md",
            "references/memory-ub.md",
            "references/l2-cache.md",
            "references/hardware-info.md",
            "references/cli-usage.md",
            "references/collection-workflow.md",
            "references/summary.md",
            "references/data-availability.md",
            "references/verification-basis.md",
            "assets/analysis-report-template.md",
            "assets/metric-catalog.json",
            "assets/hardware-info-catalog.json",
            "assets/pipe-trace-catalog.json",
            "assets/section-catalog.json",
            "assets/summary-catalog.json",
            "scripts/lookup_metric.py",
            "scripts/lookup_section.py",
            "scripts/inspect_csv.py",
            "scripts/inspect_collection.py",
            "scripts/inspect_hardware_info.py",
            "scripts/inspect_pipe_trace.py",
            "scripts/inspect_summary.py",
            "scripts/validate_catalog.py",
        }
        self.assertTrue(expected.issubset(set(self.links)))
        for reference in REFERENCE_ROOT.glob("*.md"):
            self.assertIn(f"references/{reference.name}", self.links)

    def test_required_scripts_exist_and_are_linked_directly(self):
        expected = {
            "inspect_collection.py",
            "inspect_csv.py",
            "inspect_hardware_info.py",
            "inspect_pipe_trace.py",
            "inspect_summary.py",
            "lookup_metric.py",
            "lookup_section.py",
            "validate_catalog.py",
        }
        self.assertEqual({path.name for path in SCRIPT_ROOT.glob("*.py")}, expected)
        for name in expected:
            self.assertIn(f"scripts/{name}", self.links)

    def test_references_do_not_link_to_each_other(self):
        reference_paths = {path.resolve() for path in REFERENCE_ROOT.glob("*.md")}
        for reference in reference_paths:
            for link in markdown_links(reference.read_text(encoding="utf-8")):
                if "://" in link or link.startswith("#"):
                    continue
                target = (reference.parent / link.split("#", 1)[0]).resolve()
                self.assertNotIn(
                    target, reference_paths, f"{reference.name} links to {target.name}"
                )

    def test_cli_references_cover_collection_and_import_workflows(self):
        cli_text = CLI_USAGE_PATH.read_text(encoding="utf-8")
        workflow_text = COLLECTION_WORKFLOW_PATH.read_text(encoding="utf-8")
        for fragment in (
            "npu-compute [options] [--] [program] [program-arguments]",
            "--list-sections",
            "--section <name>",
            "--import <report>",
            "--export <path>",
            "npu-compute: report=",
            "npu-compute: unpacked=",
        ):
            self.assertIn(fragment, cli_text)
        self.assertIn("用户只询问使用方式时，生成命令但不执行", workflow_text)
        self.assertIn("退出状态为 `0`", workflow_text)

    def test_skill_routes_default_set_section_and_list_requests(self):
        for fragment in (
            "--list-sets",
            "--set basic",
            "--set full",
            "默认使用 `basic`",
            "按用户表达顺序",
            "首次出现",
        ):
            self.assertIn(fragment, self.skill_text)
        self.assertIn("未指定指标时，不添加 `--set` 或 `--section`", self.skill_text)
        self.assertIn("npu-compute 默认使用 `basic`", self.skill_text)
        self.assertIn("按命令行出现顺序展开", self.skill_text)

    def test_cli_reference_documents_set_contract_and_examples(self):
        cli_text = CLI_USAGE_PATH.read_text(encoding="utf-8")
        examples = bash_examples(cli_text)
        for option in ("--set <name>", "--list-sets"):
            self.assertRegex(cli_text, rf"\|\s*`{re.escape(option)}`\s*\|")
        for fragment in (
            "npu-compute --list-sets",
            "npu-compute ./add_custom",
            "npu-compute --set basic ./add_custom",
            "npu-compute --set full ./add_custom",
            "--set basic --section ResourceConflictRatio",
            "--set basic -- ./add_custom --help",
        ):
            self.assertIn(fragment, examples)
        self.assertIn("默认使用 `basic`", cli_text)
        self.assertIn("按命令行出现顺序", cli_text)
        self.assertIn("首次出现", cli_text)
        self.assertIn("npu-compute --section Memory --set basic", cli_text)
        self.assertIn("`--` 后第一项是目标程序", cli_text)
        separator_command = next(
            line
            for line in examples.splitlines()
            if line.startswith("npu-compute --set basic -- ./add_custom")
        )
        separator_tokens = shlex.split(separator_command)
        separator_index = separator_tokens.index("--")
        self.assertEqual(separator_tokens[separator_index + 1], "./add_custom")
        self.assertEqual(
            separator_tokens[separator_index + 2:],
            ["--help", "--section", "app-value"],
        )
        self.assertNotIn("采集至少指定一个 Section", cli_text)
        self.assertNotIn("采集需要 Section 和目标程序", cli_text)

    def test_collection_workflow_covers_set_selection_and_separator(self):
        workflow_text = COLLECTION_WORKFLOW_PATH.read_text(encoding="utf-8")
        examples = bash_examples(workflow_text)
        for fragment in (
            "npu-compute --list-sets",
            "npu-compute ./add_custom",
            "npu-compute --set basic ./add_custom",
            "npu-compute --set full ./add_custom",
            "--set basic --section ResourceConflictRatio",
            "--set basic -- ./add_custom --help",
        ):
            self.assertIn(fragment, examples)
        list_sets_position = examples.index("npu-compute --list-sets")
        collection_position = examples.index("npu-compute ./add_custom")
        self.assertLess(list_sets_position, collection_position)
        self.assertIn("按最终命令行出现顺序展开", workflow_text)


if __name__ == "__main__":
    unittest.main()
