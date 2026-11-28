# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Summary DocPartConfig factory function.

Produces the same DocPartConfig as defined in gene_rule_registry.py lines 154-198.
This factory function will eventually replace the inline definition in GeneRuleRegistry.
"""

import logging

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)

logger = logging.getLogger(__name__)


def make_summary_config() -> DocPartConfig:
    """Create summary DocPartConfig matching current output format.

    Returns:
        DocPartConfig instance with:
        - doc_part="summary"
        - anchor_types=["file", "note"]
        - gene_rule_mapping for file and note anchors
        - id_config: type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False
        - output_filename="summary.json"
        - sub_part_template with summary_clarify and summary_formula

    Note:
        This config matches the inline definition in gene_rule_registry.py lines 154-198.
        Gene_rule targets are 1-element lists:
        - op_api_aclnn_h → [['summary_clarify'], ['summary_formula']]
        - op_api_aclnn_cpp → [['summary_clarify']]
        - op_kernel_cpp_h → [['summary_formula']]
        - note default → [['summary_clarify'], ['summary_formula']]
    """
    config = DocPartConfig(
        doc_part="summary",
        anchor_types=["file", "note"],
        gene_rule_mapping={
            "file": {
                "op_api_aclnn_h": {
                    "deter": "",
                    "target": [["summary_clarify"], ["summary_formula"]],
                },
                "op_api_aclnn_cpp": {"deter": "", "target": [["summary_clarify"]]},
                "op_kernel_cpp_h": {"deter": "", "target": [["summary_formula"]]},
            },
            "note": {
                "default": {
                    "deter": "",
                    "target": [["summary_clarify"], ["summary_formula"]],
                }
            },
        },
        id_config=IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=False,
            category_offsets=None,
            shared_counters=False,
        ),
        output_filename="summary.json",
        sub_part_template={
            "summary_clarify": [{"content": ""}],
            "summary_formula": [{"content": ""}],
        },
    )

    logger.debug(f"Created summary DocPartConfig: {config.doc_part}")
    return config


if __name__ == "__main__":
    """Self-test: Print summary config."""
    config = make_summary_config()

    print("=" * 60)
    print("Summary DocPartConfig")
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
    print("Gene rule targets (note anchors):")
    print("=" * 60)
    for note_type, rule in config.gene_rule_mapping["note"].items():
        print(f"{note_type}: {rule['target']}")

    print("\n" + "=" * 60)
    print("Self-test passed")
    print("=" * 60)
