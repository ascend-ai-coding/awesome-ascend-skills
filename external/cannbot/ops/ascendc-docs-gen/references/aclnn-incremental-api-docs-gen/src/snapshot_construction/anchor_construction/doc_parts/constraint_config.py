# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Constraint DocPartConfig factory function.

Produces the same DocPartConfig as defined in gene_rule_registry.py lines 58-106.
This factory function will eventually replace the inline definition in GeneRuleRegistry.
"""

import logging
from typing import Any

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)

logger = logging.getLogger(__name__)


def make_constraint_config() -> DocPartConfig:
    """Create constraint DocPartConfig matching current output format.

    Returns:
        DocPartConfig instance with:
        - doc_part="constraint"
        - anchor_types=["file", "section"]
        - gene_rule_mapping for file and section anchors
        - id_config: type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True
        - output_filename="constraint.json"
        - sub_part_template with constrain_clarify

    Note:
        This config matches the inline definition in gene_rule_registry.py lines 58-106.
        Gene_rule targets are 1-element lists:
            [['deter_describe'], ['constrain_model_scene'], ['constrain_model_scene_clarify']]
    """
    config = DocPartConfig(
        doc_part="constraint",
        anchor_types=["file", "section"],
        gene_rule_mapping={
            "file": {
                "op_api_aclnn_h": {
                    "deter": "暂空",
                    "target": [
                        ["deter_describe"],
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
                "op_api_aclnn_cpp": {
                    "deter": "暂空",
                    "target": [
                        ["deter_describe"],
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
                "op_host_def_cpp": {
                    "deter": "暂空",
                    "target": [
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
                "op_host_tiling_cpp": {
                    "deter": "暂空",
                    "target": [
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
                "op_kernel_cpp_h": {
                    "deter": "暂空",
                    "target": [
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
            },
            "section": {
                "op_host_tiling_cpp": {
                    "deter": "暂空",
                    "target": [
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
                "op_kernel_cpp_h": {
                    "deter": "暂空",
                    "target": [
                        ["constrain_model_scene"],
                        ["constrain_model_scene_clarify"],
                    ],
                },
            },
        },
        id_config=IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            category_offsets=None,
            shared_counters=True,
        ),
        output_filename="constraint.json",
        sub_part_template={"constrain_clarify": [{"content": "", "evidence_id": []}]},
    )

    logger.debug(f"Created constraint DocPartConfig: {config.doc_part}")
    return config


if __name__ == "__main__":
    """Self-test: Print constraint config."""
    config = make_constraint_config()

    print("=" * 60)
    print("Constraint DocPartConfig")
    print("=" * 60)
    print(f"doc_part: {config.doc_part}")
    print(f"anchor_types: {config.anchor_types}")
    print(f"output_filename: {config.output_filename}")
    print("id_config:")
    print(f"  type_prefix_in_anchor: {config.id_config.type_prefix_in_anchor}")
    print(f"  type_prefix_in_evidence: {config.id_config.type_prefix_in_evidence}")
    print(f"  shared_counters: {config.id_config.shared_counters}")
    print(f"gene_rule_mapping keys: {list(config.gene_rule_mapping.keys())}")
    print(f"sub_part_template keys: {list(config.sub_part_template.keys())}")

    print("\n" + "=" * 60)
    print("Gene rule targets (file anchors):")
    print("=" * 60)
    for file_type, rule in config.gene_rule_mapping["file"].items():
        print(f"{file_type}: {rule['target']}")

    print("\n" + "=" * 60)
    print("Gene rule targets (section anchors):")
    print("=" * 60)
    for section_type, rule in config.gene_rule_mapping["section"].items():
        print(f"{section_type}: {rule['target']}")

    print("\n" + "=" * 60)
    print("Self-test passed")
    print("=" * 60)
