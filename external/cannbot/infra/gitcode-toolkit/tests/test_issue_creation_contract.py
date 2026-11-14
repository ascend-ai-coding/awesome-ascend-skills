# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
from pathlib import Path


TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
ISSUE_GEN_SKILL = TOOLKIT_ROOT.parent / "gitcode-issue-gen" / "SKILL.md"


def _section(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def test_issue_creation_docs_use_owner_level_openapi_contract() -> None:
    api = (TOOLKIT_ROOT / "references" / "gitcode-api.md").read_text(encoding="utf-8")
    workflow = (
        TOOLKIT_ROOT / "references" / "issue-creation-workflow.md"
    ).read_text(encoding="utf-8")
    issue_gen = ISSUE_GEN_SKILL.read_text(encoding="utf-8")

    sections = {
        "gitcode-api.md": _section(api, "### 创建 Issue", "### 获取 Issue 列表"),
        "issue-creation-workflow.md": _section(
            workflow, "## Step 6: 创建 Issue", "## Step 7:"
        ),
        "gitcode-issue-gen/SKILL.md": _section(
            issue_gen, "#### 创建 Issue", "#### 认证方式"
        ),
    }

    for document_name, section in sections.items():
        normalized = section.replace("https://api.gitcode.com/api/v5", "")
        assert "POST /repos/{owner}/issues" in normalized, document_name
        assert "POST /repos/{owner}/{repo}/issues" not in normalized, document_name
        assert (
            '"repo": "{repo}"' in section
            or '"repo": "${repo}"' in section
            or '--arg repo "${repo}"' in section
        ), document_name
        assert re.search(r"\bassignee\b", section), document_name
        assert not re.search(r"\bassignees\b", section), document_name
