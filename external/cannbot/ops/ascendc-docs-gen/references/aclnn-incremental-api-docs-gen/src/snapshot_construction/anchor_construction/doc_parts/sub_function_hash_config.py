# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Factory function for sub_function_hash DocPartConfig.

sub_function_hash is the most complex doc_part:
- ALL 4 anchor types (file, function, note, section)
- 2-element target format [[func_id, field]]
- Dynamic errorcode generation based on sub_function_count
- SECTION_MAPPING for section anchors
"""

import logging
from typing import Any

from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
)

logger = logging.getLogger(__name__)


# SECTION_MAPPING for section anchors
SECTION_MAPPING = {
    "SUPPORT_LIST": "dtype",
    "check_dtype": "dtype",
    "CheckFormat": "dmat",
    "CheckShape": "shape",
    "continuous": "tensor",
    "CheckParams": "errorcode",
}


def make_sub_function_hash_config(sub_function_count: int = 2) -> DocPartConfig:
    """Factory function to create sub_function_hash DocPartConfig.

    Args:
        sub_function_count: Number of sub_functions (default: 2)
            Used to dynamically generate errorcode targets.

    Returns:
        DocPartConfig instance for sub_function_hash doc_part

    Note:
        This config matches gene_rule_registry.py lines 242-318 EXACTLY,
        except for the dynamic errorcode generation fix.

        The hardcoded 2-sub-function errorcode targets in the original config
        are replaced with dynamic generation based on sub_function_count:
        - sub_function_count=2 → ["sub_function_1_errorcode", "errorcode"], ["sub_function_2_errorcode", "errorcode"]
        - sub_function_count=3 → ["sub_function_1_errorcode", "errorcode"],
          ["sub_function_2_errorcode", "errorcode"], ["sub_function_3_errorcode", "errorcode"]
        - sub_function_count=1 → ["sub_function_1_errorcode", "errorcode"]
    """
    logger.debug(f"Creating sub_function_hash config with sub_function_count={sub_function_count}")

    # Dynamically generate errorcode targets based on sub_function_count
    errorcode_targets = [[f"sub_function_{i}_errorcode", "errorcode"] for i in range(1, sub_function_count + 1)]

    config = DocPartConfig(
        doc_part="sub_function_hash",
        anchor_types=["file", "function", "note", "section"],
        gene_rule_mapping={
            "file": {
                "aclnn_cpp": {
                    "deter": "",
                    "target": [
                        ["{sub_function_hash}", "des"],
                        ["{sub_function_hash}", "inout"],
                        ["{sub_function_hash}", "usage"],
                        ["{sub_function_hash}", "dtype"],
                        ["{sub_function_hash}", "dmat"],
                        ["{sub_function_hash}", "shape"],
                        ["{sub_function_hash}", "tensor"],
                        ["{sub_error_code}", "errorcode"],
                    ],
                },
                "aclnn_h": {
                    "deter": "",
                    "target": [
                        ["{sub_function_hash}", "des"],
                        ["{sub_function_hash}", "inout"],
                        ["{sub_function_hash}", "usage"],
                        ["{sub_function_hash}", "dtype"],
                        ["{sub_function_hash}", "dmat"],
                        ["{sub_function_hash}", "shape"],
                    ],
                },
                "proto_h": {
                    "deter": "",
                    "target": [
                        ["{sub_function_hash}", "des"],
                        ["{sub_function_hash}", "inout"],
                        ["{sub_function_hash}", "usage"],
                        ["{sub_function_hash}", "dtype"],
                        ["{sub_function_hash}", "dmat"],
                    ],
                },
                "tiling_h": {
                    "deter": "",
                    "target": [
                        ["{sub_function_hash}", "dtype"],
                        ["{sub_function_hash}", "dmat"],
                        ["{sub_function_hash}", "shape"],
                    ],
                },
                "def_h": {
                    "deter": "",
                    "target": [
                        ["{sub_function_hash}", "dtype"],
                        ["{sub_function_hash}", "dmat"],
                        ["{sub_function_hash}", "tensor"],
                    ],
                },
            },
            "function": {
                "default": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "{source_field}"]],
                }
            },
            "note": {
                "default": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "{source_field}"]],
                }
            },
            "section": {
                "SUPPORT_LIST": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "dtype"]],
                },
                "check_dtype": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "dtype"]],
                },
                "CheckFormat": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "dmat"]],
                },
                "CheckShape": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "shape"]],
                },
                "continuous": {
                    "deter": "",
                    "target": [["{sub_function_hash}", "tensor"]],
                },
                "CheckParams": {
                    "deter": "",
                    "target": errorcode_targets,  # Dynamic generation based on sub_function_count
                },
            },
        },
        id_config=IdConfig(
            type_prefix_in_anchor=True,
            type_prefix_in_evidence=True,
            category_offsets=None,
            shared_counters=True,
        ),
        output_filename="sub_function_hash.json",
        sub_part_template={
            "param_matrix": {},
            "fixconfig_1": {},
            "fixconfig_2": {},
            "changeconfig": {},
            "errorcode_matrix": {},
        },
    )

    logger.info(f"Created sub_function_hash config with {sub_function_count} sub_functions")
    return config


if __name__ == "__main__":
    """Self-test: Print config details."""
    print("=" * 60)
    print("sub_function_hash DocPartConfig Factory")
    print("=" * 60)

    # Test with default sub_function_count=2
    config_default = make_sub_function_hash_config()
    print("\n[Default: sub_function_count=2]")
    print(f"  doc_part: {config_default.doc_part}")
    print(f"  anchor_types: {config_default.anchor_types}")
    print(f"  output_filename: {config_default.output_filename}")
    id_cfg = config_default.id_config
    print(
        f"  id_config: type_prefix_in_anchor={id_cfg.type_prefix_in_anchor}, "
        f"type_prefix_in_evidence={id_cfg.type_prefix_in_evidence}, "
        f"shared_counters={id_cfg.shared_counters}"
    )
    print(f"  gene_rule_mapping keys: {list(config_default.gene_rule_mapping.keys())}")
    print(f"  sub_part_template keys: {list(config_default.sub_part_template.keys())}")
    print(f"  CheckParams errorcode targets: {config_default.gene_rule_mapping['section']['CheckParams']['target']}")

    # Test with sub_function_count=3
    config_3 = make_sub_function_hash_config(sub_function_count=3)
    print("\n[sub_function_count=3]")
    print(f"  CheckParams errorcode targets: {config_3.gene_rule_mapping['section']['CheckParams']['target']}")

    # Test with sub_function_count=1
    config_1 = make_sub_function_hash_config(sub_function_count=1)
    print("\n[sub_function_count=1]")
    print(f"  CheckParams errorcode targets: {config_1.gene_rule_mapping['section']['CheckParams']['target']}")

    print("\n" + "=" * 60)
    print("Self-test passed")
    print("=" * 60)
