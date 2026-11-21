# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Function_proto DocPartConfig factory function.

Produces the same DocPartConfig as defined in gene_rule_registry.py lines 202-238.
This factory function will eventually replace the inline definition in GeneRuleRegistry.
"""

import logging

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)

logger = logging.getLogger(__name__)


def make_function_proto_config() -> DocPartConfig:
    """Create function_proto DocPartConfig matching current output format.

    Returns:
        DocPartConfig instance with:
        - doc_part="function_proto"
        - anchor_types=["file", "function"]
        - gene_rule_mapping for file and function anchors
        - id_config: type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False
        - output_filename="function_proto.json"
        - sub_part_template with function_proto_clarify and sub_function_sign

    Note:
        This config matches the inline definition in gene_rule_registry.py lines 202-238.
        Gene_rule targets are 1-element lists: [['function_proto_clarify'], ['sub_function_sign']] for file,
        [['sub_function_sign']] for function.
    """
    config = DocPartConfig(
        doc_part="function_proto",
        anchor_types=["file", "function"],
        gene_rule_mapping={
            "file": {
                "op_api_aclnn_h": {
                    "deter": "",
                    "target": [["function_proto_clarify"], ["sub_function_sign"]],
                }
            },
            "function": {"default": {"deter": "", "target": [["sub_function_sign"]]}},
        },
        id_config=IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets=None,
            shared_counters=False,
        ),
        output_filename="function_proto.json",
        sub_part_template={
            "function_proto_clarify": [{"content": ""}],
            "sub_function_sign": [{"content": ""}],
        },
    )

    logger.debug(f"Created function_proto DocPartConfig: {config.doc_part}")
    return config


if __name__ == "__main__":
    """Self-test: Print function_proto config."""
    config = make_function_proto_config()

    print("=" * 60)
    print("Function_proto DocPartConfig")
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
    print("Gene rule targets (function anchors):")
    print("=" * 60)
    for function_type, rule in config.gene_rule_mapping["function"].items():
        print(f"{function_type}: {rule['target']}")

    print("\n" + "=" * 60)
    print("Self-test passed")
    print("=" * 60)
