# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
GeneRuleRegistry + DocPartConfig schema for per-doc_part configuration.

Defines gene_rule mapping, anchor types, ID format, output filename, and doc.sub_part structure
for each doc_part (constraint, support, summary, function_proto, sub_function_hash).
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.generation_validation.config import format_doc_part
from src.snapshot_construction.anchor_construction.common.id_generator import IdConfig

logger = logging.getLogger(__name__)


@dataclass
class DocPartConfig:
    """Configuration for a single doc_part.

    Attributes:
        doc_part: doc_part name (e.g., "constraint", "support", "summary", "function_proto", "sub_function_hash")
        anchor_types: List of anchor types included in this doc_part (e.g., ["file", "section"])
        gene_rule_mapping: Per-anchor-type gene_rule configuration (deter, target)
        id_config: ID format configuration from IdConfig
        output_filename: JSON output filename (e.g., "constraint.json")
        sub_part_template: doc.sub_part structure template
    """

    doc_part: str
    anchor_types: list[str]
    gene_rule_mapping: dict[str, dict[str, Any]]
    id_config: IdConfig
    output_filename: str
    sub_part_template: dict[str, Any]


class GeneRuleRegistry:
    """Registry for gene_rule configurations per doc_part.

    Manages DocPartConfig instances for all 5 doc_parts:
    - constraint: file + section anchors, 1-element targets
    - support: file + section anchors, 1-element targets
    - summary: file + note anchors, 1-element targets
    - function_proto: file + function anchors, 1-element targets
    - sub_function_hash: ALL 4 anchor types, 2-element targets
    """

    def __init__(self):
        """Initialize registry with 5 default DocPartConfig instances."""
        self._configs: dict[str, DocPartConfig] = {}
        self._initialize_default_configs()
        logger.debug(f"GeneRuleRegistry initialized with {len(self._configs)} configs")

    def _initialize_default_configs(self) -> None:
        """Initialize 5 default DocPartConfig instances matching current output format."""

        # 1. constraint doc_part
        constraint_config = DocPartConfig(
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
        self._configs["constraint"] = constraint_config

        # 2. support doc_part
        support_config = DocPartConfig(
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
        self._configs["support"] = support_config

        # 3. summary doc_part
        summary_config = DocPartConfig(
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
        self._configs["summary"] = summary_config

        # 4. function_proto doc_part
        function_proto_config = DocPartConfig(
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
        self._configs["function_proto"] = function_proto_config

        # 5. sub_function_hash doc_part
        sub_function_config = DocPartConfig(
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
                        "target": [
                            ["sub_function_1_errorcode", "errorcode"],
                            ["sub_function_2_errorcode", "errorcode"],
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
            output_filename="sub_function_hash.json",
            sub_part_template={
                "param_matrix": {},
                "fixconfig_1": {},
                "fixconfig_2": {},
                "changeconfig": {},
                "errorcode_matrix": {},
            },
        )
        self._configs["sub_function_hash"] = sub_function_config

        logger.info("Initialized 5 default DocPartConfig instances")

    def get_config(self, doc_part: str) -> DocPartConfig:
        """Get DocPartConfig for a specific doc_part.

        Args:
            doc_part: doc_part name (e.g., "constraint", "support")

        Returns:
            DocPartConfig instance for the doc_part

        Raises:
            KeyError: If doc_part not found in registry
        """
        if doc_part not in self._configs:
            raise KeyError(f"doc_part '{doc_part}' not found in registry. Available: {list(self._configs.keys())}")

        config = self._configs[doc_part]
        logger.debug(f"Retrieved config for doc_part: {doc_part}")
        return config

    def get_all_configs(self) -> dict[str, DocPartConfig]:
        """Get all DocPartConfig instances.

        Returns:
            Dictionary mapping doc_part names to DocPartConfig instances
        """
        logger.debug(f"Retrieved all {len(self._configs)} configs")
        return self._configs.copy()

    def register_config(self, config: DocPartConfig) -> None:
        """Register a custom DocPartConfig.

        Args:
            config: DocPartConfig instance to register

        Note:
            This will overwrite existing config for the same doc_part
        """
        self._configs[config.doc_part] = config
        logger.info(f"Registered custom config for doc_part: {config.doc_part}")


# Singleton instance for convenience
_registry_instance: GeneRuleRegistry | None = None


def get_registry() -> GeneRuleRegistry:
    """Get singleton GeneRuleRegistry instance.

    Returns:
        GeneRuleRegistry singleton instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = GeneRuleRegistry()
        logger.debug("Created GeneRuleRegistry singleton")
    return _registry_instance


if __name__ == "__main__":
    """Self-test: Print all configs."""
    registry = get_registry()

    print("=" * 60)
    print("GeneRuleRegistry - All DocPartConfigs")
    print("=" * 60)

    for doc_part, config in registry.get_all_configs().items():
        print(f"\n[{format_doc_part(doc_part)}]")
        print(f"  anchor_types: {config.anchor_types}")
        print(f"  output_filename: {config.output_filename}")
        id_cfg = config.id_config
        print(
            f"  id_config: type_prefix_in_anchor={id_cfg.type_prefix_in_anchor}, "
            f"type_prefix_in_evidence={id_cfg.type_prefix_in_evidence}, "
            f"shared_counters={id_cfg.shared_counters}"
        )
        print(f"  gene_rule_mapping keys: {list(config.gene_rule_mapping.keys())}")
        print(f"  sub_part_template keys: {list(config.sub_part_template.keys())}")

    print("\n" + "=" * 60)
    print("Self-test passed")
    print("=" * 60)
