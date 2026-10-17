#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""从受控骨架创建一个待推导的矩阵乘切分参数算子档案。"""

import argparse
import json
import os
import re
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = SKILL_DIR / "references" / "profiles"


RECOGNITION_TEMPLATE = '''#!/usr/bin/env python3
"""{op_name} 输入与模式识别适配器骨架。"""


def derive_facts(ctx: dict) -> dict:
    """严格复刻主机侧分支顺序，不根据用例名猜测模式。"""
    raise NotImplementedError("请根据目标算子定义和主机侧源码实现事实推导")


def validate_case(ctx: dict, input_rules: dict):
    """支持时返回空值，否则返回具体拒绝原因。"""
    raise NotImplementedError("请根据目标源码实现输入、形状、数据类型和格式校验")
'''


CONSTRAINTS_TEMPLATE = '''#!/usr/bin/env python3
"""{op_name} 候选硬约束适配器骨架。"""


def check_candidate(candidate: dict, params: dict, anchor: dict, ctx: dict):
    raise NotImplementedError("请只实现有源码机制证据的硬约束")


def validate_rules_against_anchors(anchors: dict, params: dict, ctx: dict) -> list[str]:
    raise NotImplementedError("请用主机侧锚点逐条核对硬约束")
'''


CONSTRAINTS_DOC_TEMPLATE = """# {op_name} 约束与候选方向

算子档案通过源码对账和测试后，才能删除 `profile.json` 的 `_scaffold` 标记。

每条规则必须写清：候选字段、公式或判定、为什么属于硬约束、源码位置、反例和锚点自检结果。性能经验写成候选方向，不得冒充硬约束。

## 硬约束

记录必然导致容量超限、越界、漏块、对齐非法或无法注入的规则。

## 候选方向

记录可能改善复用、并行度、流水或搬运效率的方向。

## 保留实测的探索点

记录证据不足但值得用少量预算验证的组合。

## 安装包发现与注册

记录动态实现文件、注册头、共享库或其他发现链路，并与 `profile.json.package_gate` 一致。
"""


def _profile_template(op_name: str, repo_rel: str, repo_commit: str) -> dict:
    return {
        "_scaffold": True,
        "schema": "v3",
        "identity": {
            "op_name": op_name,
            "family": "matmul",
            "repo_rel": repo_rel,
            "display": op_name,
        },
        "support": {
            "level": "unsupported",
            "scope": "以目标源码和实测覆盖为准",
            "known_limits": [],
        },
        "version_anchor": {
            "op_repo_commit": repo_commit,
            "anchored_at": "",
            "note": "由生成流程记录源码对账范围",
        },
        "soc": {
            "name": "",
            "arch": "",
            "l1_size": 0,
            "l0a_size": 0,
            "l0b_size": 0,
            "l0c_size": 0,
            "mte2_align_bytes": 0,
            "mx_group_size": 0,
            "mxfp_divisor_size": 0,
            "data_size_l0c": 0,
        },
        "tiling_key": {
            "kernel_type_shift": 0,
            "kernel_type_bits": 0,
            "evidence": "",
        },
        "case_classes": {},
        "fold_rule": {"conditions": [], "on_fold": {}, "evidence": ""},
        "struct_layouts": {},
        "input_rules": {"slot_order": [], "shape_semantics": "", "evidence": ""},
        "recognition": {
            "module": "recognition.py",
            "dimension": "",
            "exports": ["derive_facts", "validate_case"],
        },
        "quant_modes": {},
        "templates": {},
        "constraints": {
            "module": "constraints.py",
            "exports": ["check_candidate", "validate_rules_against_anchors"],
            "required_candidates_per_template": 1,
            "budget_evidence": "",
        },
        "package_gate": {"required_files": []},
    }


def create_scaffold(
    op_name: str, repo_rel: str, repo_commit: str, output_root: str | os.PathLike = DEFAULT_OUTPUT_ROOT
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", op_name):
        raise ValueError("op_name 只能包含字母、数字和下划线")
    rel = Path(repo_rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("repo_rel 必须是算子仓内安全相对路径")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", repo_commit):
        raise ValueError("repo_commit 必须是 7-64 位十六进制 Git commit")
    target = Path(output_root).resolve() / op_name
    if target.exists():
        raise FileExistsError(f"算子档案目录已存在，拒绝覆盖: {target}")
    target.mkdir(parents=True)
    (target / "profile.json").write_text(
        json.dumps(_profile_template(op_name, repo_rel, repo_commit), indent=2, ensure_ascii=False) + "\n"
    )
    (target / "recognition.py").write_text(RECOGNITION_TEMPLATE.format(op_name=op_name))
    (target / "constraints.py").write_text(CONSTRAINTS_TEMPLATE.format(op_name=op_name))
    (target / "constraints.md").write_text(CONSTRAINTS_DOC_TEMPLATE.format(op_name=op_name))
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", required=True)
    p.add_argument("--repo-rel", required=True)
    p.add_argument("--repo-commit", required=True)
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = p.parse_args()
    print(create_scaffold(args.op, args.repo_rel, args.repo_commit, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
