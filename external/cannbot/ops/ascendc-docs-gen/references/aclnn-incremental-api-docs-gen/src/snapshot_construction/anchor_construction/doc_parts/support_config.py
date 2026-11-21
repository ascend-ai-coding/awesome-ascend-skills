# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Support DocPartConfig factory.

Factory function that creates the support DocPartConfig matching the inline definition
in gene_rule_registry.py lines 110-150.

Support uses TYPE1/TYPE2/TYPE3 section patterns with category offsets:
- TYPE1: AICore().AddConfig 匹配芯片
- TYPE2: curShortSocName == SocVersion 匹配 SOC
- TYPE3: #{arch_macro} == {arch_name}

Output structure uses evidence_ids (plural) in sub_part.support_matrix.
"""

import logging

from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)
from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig

logger = logging.getLogger(__name__)


def make_support_config() -> DocPartConfig:
    """Create support DocPartConfig matching gene_rule_registry.py inline definition.

    Returns:
        DocPartConfig instance with:
        - doc_part="support"
        - anchor_types=["file", "section"]
        - gene_rule_mapping with TYPE1/TYPE2/TYPE3 for section
        - id_config: type_prefix_in_anchor=False, type_prefix_in_evidence=False,
                     category_offsets={TYPE1:1, TYPE2:101, TYPE3:201}, shared_counters=False
        - output_filename="support.json"
        - sub_part_template with support_matrix (uses evidence_ids not evidence_id)

    Note:
        This config matches the inline definition in gene_rule_registry.py lines 110-150.
        The factory function provides a clean interface for importing the config without
        accessing the registry singleton.
    """
    config = DocPartConfig(
        doc_part="support",
        anchor_types=["file", "section"],
        gene_rule_mapping={
            "file": {"default": {"deter": "", "target": [["support_matrix"]]}},
            "section": {
                "TYPE1": {
                    "deter": "AICore().AddConfig 匹配芯片: {chips}",
                    "target": [["support_matrix"]],
                },
                "TYPE2": {
                    "deter": "curShortSocName == SocVersion 匹配 SOC: {socs}",
                    "target": [["support_matrix"]],
                },
                "TYPE3": {
                    "deter": "#{arch_macro} == {arch_name}",
                    "target": [["support_matrix"]],
                },
            },
        },
        id_config=IdConfig(
            type_prefix_in_anchor=False,
            type_prefix_in_evidence=False,
            category_offsets={"TYPE1": 1, "TYPE2": 101, "TYPE3": 201},
            shared_counters=False,
        ),
        output_filename="support.json",
        sub_part_template={"support_matrix": [{"content": "", "evidence_ids": []}]},
    )

    logger.debug("Created support DocPartConfig")
    return config


if __name__ == "__main__":
    """Self-test: Print support config details."""
    config = make_support_config()

    print("=" * 60)
    print("Support DocPartConfig")
    print("=" * 60)
    print(f"doc_part: {config.doc_part}")
    print(f"anchor_types: {config.anchor_types}")
    print(f"output_filename: {config.output_filename}")
    print("id_config:")
    print(f"  type_prefix_in_anchor: {config.id_config.type_prefix_in_anchor}")
    print(f"  type_prefix_in_evidence: {config.id_config.type_prefix_in_evidence}")
    print(f"  shared_counters: {config.id_config.shared_counters}")
    print(f"  category_offsets: {config.id_config.category_offsets}")
    print(f"gene_rule_mapping keys: {list(config.gene_rule_mapping.keys())}")
    print(f"section patterns: {list(config.gene_rule_mapping['section'].keys())}")
    print(f"sub_part_template keys: {list(config.sub_part_template.keys())}")
    print("=" * 60)
    print("Self-test passed")
    print("=" * 60)
