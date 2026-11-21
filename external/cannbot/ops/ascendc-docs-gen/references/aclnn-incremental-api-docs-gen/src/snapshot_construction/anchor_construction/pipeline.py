# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
AnchorPipeline - Unified anchor construction pipeline for all 5 doc_parts.

Orchestrates the full anchor construction flow, tying together:
- Extractors (FileExtractor, FunctionExtractor, SectionExtractor, NoteExtractor)
- GeneRuleRegistry (per-doc_part gene_rule configuration)
- IdGenerator (per-doc_part ID format)
- Common utilities (PathConfig, DataLoader, anchor builders)

Produces output matching golden reference structure for:
- constraint: file + section anchors, 1-element targets
- support: file + section anchors, 1-element targets
- summary: file + note anchors, 1-element targets
- function_proto: file + function anchors, 1-element targets
- sub_function_hash: ALL 4 anchor types, 2-element targets

Special handling:
- summary↔function_proto share extraction but output different gene_rules (deep copy)
- Each doc_part has different ID format, output filename, sub_part template
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from src.generation_validation.config import format_doc_part
from src.snapshot_construction.anchor_construction.common import (
    ANCHOR_TYPE_FILE,
    ANCHOR_TYPE_FUNCTION,
    ANCHOR_TYPE_NOTE,
    ANCHOR_TYPE_SECTION,
    DOC_PARTS,
    DataLoader,
    PathConfig,
)
from src.snapshot_construction.anchor_construction.common.id_generator import (
    IdGenerator,
)
from src.snapshot_construction.anchor_construction.extractors.file_extractor import (
    FileExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.function_extractor import (
    FunctionExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.note_extractor import (
    NoteExtractor,
)
from src.snapshot_construction.anchor_construction.extractors.section_extractor import (
    SectionExtractor,
)
from src.snapshot_construction.anchor_construction.gene_rule_registry import (
    DocPartConfig,
    GeneRuleRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)

# 7 param_matrix fields used in template expansion
_PARAM_FIELDS = ["des", "inout", "usage", "dtype", "dmat", "shape", "tensor"]


def _expand_template_targets(anchor_tuple: list[dict], sub_function_count: int) -> list[dict]:
    """Expand template variables in sub_function_hash gene_rule targets.

    Template variables: {sub_function_hash}, {sub_error_code}
    {source_field} is handled implicitly for function/note anchors via _PARAM_FIELDS rebuild.
    Section anchors: normal → copy N times; CheckParams → single anchor with N errorcode targets.
    """
    result: list[dict] = []
    for anchor in anchor_tuple:
        atype = anchor["anchor_information"]["type"]
        targets = anchor["evidence"][0]["gene_rule"]["target"]
        has_tmpl = any("{" in str(e) for t in targets for e in t)
        is_cp = atype == "section" and any(len(t) >= 2 and t[1] == "errorcode" for t in targets)
        if not has_tmpl and not is_cp:
            result.append(anchor)
            continue
        if atype == "section":
            if is_cp:
                a = deepcopy(anchor)
                # fmt: off
                a["evidence"][0]["gene_rule"]["target"] = [
                    [f"sub_function_{n}_errorcode", "errorcode"]
                    for n in range(1, sub_function_count + 1)
                ]
                # fmt: on
                result.append(a)
            else:
                for n in range(1, sub_function_count + 1):
                    a = deepcopy(anchor)
                    # fmt: off
                    a["evidence"][0]["gene_rule"]["target"] = [
                        [e.replace("{sub_function_hash}", f"sub_function_{n}_param_matrix") for e in t]
                        for t in targets
                    ]
                    # fmt: on
                    result.append(a)
        elif atype in ("function", "note"):
            a = deepcopy(anchor)
            new_t: list[list[str]] = []
            for n in range(1, sub_function_count + 1):
                for f in _PARAM_FIELDS:
                    new_t.append([f"sub_function_{n}_param_matrix", f])
                if atype == "function":
                    new_t.append([f"sub_function_{n}_errorcode", "errorcode"])
            a["evidence"][0]["gene_rule"]["target"] = new_t
            result.append(a)
        else:  # file
            a = deepcopy(anchor)
            new_t = []
            for n in range(1, sub_function_count + 1):
                for t in targets:
                    new_t.append(
                        [
                            e.replace("{sub_function_hash}", f"sub_function_{n}_param_matrix").replace(
                                "{sub_error_code}", f"sub_function_{n}_errorcode"
                            )
                            for e in t
                        ]
                    )
            a["evidence"][0]["gene_rule"]["target"] = new_t
            result.append(a)
    return result


class AnchorPipeline:
    """Unified anchor construction pipeline for all 5 doc_parts.

    Orchestrates the full anchor construction flow:
    1. Initialize PathConfig, DataLoader from operator_path/out_path
    2. For each doc_part in doc_part_control (or all 5 if None):
       - Get DocPartConfig from GeneRuleRegistry
       - Initialize IdGenerator with config's id_config
       - Run relevant extractors per config's anchor_types
       - Assign gene_rule via GeneRuleRegistry mapping
       - Assign anchor_id/evidence_id via IdGenerator
       - Assemble output structure: {doc: {...}, anchor_tuple: [...]}
       - Add envidence_id key to each evidence item
       - Write JSON to anchor_dir with correct filename
    3. Special handling for summary↔function_proto coupling
    4. Return dict mapping doc_part names to their output data

    Attributes:
        operator_path: Path to operator source directory
        out_path: Base output directory for all generated artifacts
        doc_part_control: Optional dict/list controlling which doc_parts to process
        path_config: PathConfig instance for path resolution
        data_loader: DataLoader instance for loading metadata
        registry: GeneRuleRegistry singleton instance
    """

    def __init__(
        self,
        path_config: PathConfig,
        doc_part_control: dict | None = None,
        source_files: Optional[list[str]] = None,
    ):
        """Initialize AnchorPipeline with a PathConfig instance.

        Args:
            path_config: PathConfig instance for path resolution (required).
                Operator-level data (doc_json, code_map, tree_id) is resolved
                via path_config (which may redirect to data_root).
            doc_part_control: Optional dict controlling which doc_parts to process.
                If None, processes all 5 doc_parts.
                If dict, keys should be doc_part names with optional config overrides.
            source_files: Optional list of source file paths/names to filter anchors.
                When provided, extractors only return anchors from matching files.
                When None, all discovered files are processed (backward compatible).

        Raises:
            ValueError: If doc_part_control contains invalid doc_part names
        """
        self.path_config = path_config
        self.operator_path = path_config.operator_path
        self.out_path = path_config.out_path
        self.doc_part_control = doc_part_control
        self._source_files = source_files

        self.data_loader = DataLoader(self.path_config)

        # Get GeneRuleRegistry singleton
        self.registry = get_registry()

        # Validate doc_part_control if provided
        if doc_part_control is not None:
            valid_parts = DOC_PARTS
            for part in (doc_part_control.keys() if isinstance(doc_part_control, dict) else doc_part_control):
                if part not in valid_parts:
                    raise ValueError(f"doc_part 类型错误: '{part}' 不是有效的文档部分")

        logger.info(f"AnchorPipeline initialized for operator: {self.operator_path.name}")
        logger.debug(f"  out_path: {self.out_path}")
        logger.debug(f"  doc_part_control: {doc_part_control}")

    def run(self) -> dict[str, Any]:
        """Execute the full anchor construction pipeline for all doc_parts.

        Returns:
            Dict mapping doc_part names to their output data:
            {
                "constraint": {"doc": {...}, "anchor_tuple": [...]},
                "support": {"doc": {...}, "anchor_tuple": [...]},
                "summary": {"doc": {...}, "anchor_tuple": [...]},
                "function_proto": {"doc": {...}, "anchor_tuple": [...]},
                "sub_function_hash": {"doc": {...}, "anchor_tuple": [...]},
            }

        Note:
            - Creates anchor_dir if it doesn't exist
            - Writes JSON files to anchor_dir with correct filenames
            - summary↔function_proto share extraction but output different gene_rules
        """
        logger.info("=" * 60)
        logger.info("AnchorPipeline.run() - Starting full pipeline")
        logger.info("=" * 60)

        # Determine which doc_parts to process
        if self.doc_part_control is None:
            doc_parts_to_process = DOC_PARTS
        elif isinstance(self.doc_part_control, dict):
            doc_parts_to_process = list(self.doc_part_control.keys())
        else:
            doc_parts_to_process = list(self.doc_part_control)

        logger.info(
            f"Processing {len(doc_parts_to_process)} doc_parts: {[format_doc_part(p) for p in doc_parts_to_process]}"
        )

        # Create anchor_dir if needed
        self.path_config.anchor_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Special handling for summary↔function_proto coupling
        # They share extraction but output different gene_rules
        summary_proto_pair = {"summary", "function_proto"}
        needs_shared_extraction = summary_proto_pair.issubset(set(doc_parts_to_process))

        if needs_shared_extraction:
            logger.info("Detected summary↔function_proto coupling - extracting once, deep-copying")
            # Extract shared anchors once
            shared_results = self._extract_summary_proto_shared()

            # Process summary with its gene_rules
            if "summary" in doc_parts_to_process:
                summary_result = self._process_summary_from_shared(shared_results)
                results["summary"] = summary_result
                self._write_output("summary", summary_result)

            # Process function_proto with its gene_rules
            if "function_proto" in doc_parts_to_process:
                function_proto_result = self._process_function_proto_from_shared(shared_results)
                results["function_proto"] = function_proto_result
                self._write_output("function_proto", function_proto_result)

            # Remove summary and function_proto from remaining processing
            remaining_parts = [p for p in doc_parts_to_process if p not in summary_proto_pair]
        else:
            remaining_parts = doc_parts_to_process

        # Process remaining doc_parts individually
        for doc_part in remaining_parts:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing doc_part: {format_doc_part(doc_part)}")
            logger.info(f"{'=' * 60}")

            result = self._process_doc_part(doc_part)
            results[doc_part] = result
            self._write_output(doc_part, result)

        logger.info("\n" + "=" * 60)
        logger.info("AnchorPipeline.run() - Pipeline complete")
        logger.info("=" * 60)
        logger.info(f"Generated {len(results)} doc_part outputs")

        return results

    def _process_doc_part(self, doc_part: str) -> dict[str, Any]:
        """Process a single doc_part through the full pipeline.

        Args:
            doc_part: doc_part name (e.g., "constraint", "support")

        Returns:
            Output dict with structure: {"doc": {...}, "anchor_tuple": [...]}
        """
        # Get DocPartConfig from registry
        config = self.registry.get_config(doc_part)
        logger.debug(f"  Config: anchor_types={config.anchor_types}, output_filename={config.output_filename}")

        # Initialize IdGenerator with config's id_config
        id_generator = IdGenerator(config.id_config)

        # Initialize extractors with this IdGenerator
        extractors = self._create_extractors(id_generator)

        # Run relevant extractors per config's anchor_types
        anchors_by_type = {}

        for anchor_type in config.anchor_types:
            logger.info(f"  Extracting {anchor_type} anchors...")

            if anchor_type == ANCHOR_TYPE_FILE:
                anchors = extractors["file"].extract(self.operator_path, source_files=self._source_files)
            elif anchor_type == ANCHOR_TYPE_FUNCTION:
                anchors = extractors["function"].extract(self.operator_path, source_files=self._source_files)
            elif anchor_type == ANCHOR_TYPE_SECTION:
                # Determine section patterns based on doc_part
                section_patterns = self._get_section_patterns(doc_part)
                anchors = extractors["section"].extract(
                    self.operator_path,
                    section_patterns,
                    source_files=self._source_files,
                )
            elif anchor_type == ANCHOR_TYPE_NOTE:
                anchors = extractors["note"].extract(self.operator_path, source_files=self._source_files)
            else:
                logger.warning(f"  Unknown anchor_type: {anchor_type}")
                continue

            anchors_by_type[anchor_type] = anchors
            logger.info(f"    Extracted {len(anchors)} {anchor_type} anchors")

        # Assign gene_rule via GeneRuleRegistry mapping
        anchor_tuple = self._assign_gene_rules(anchors_by_type, config)

        # Expand template targets for sub_function_hash
        if doc_part == "sub_function_hash":
            sub_function_count = self._get_sub_function_count()
            anchor_tuple = _expand_template_targets(anchor_tuple, sub_function_count)

        # Build doc structure
        doc = self._build_doc_structure(doc_part, config, anchor_tuple)

        # Assemble output
        result = {
            "doc": doc,
            "anchor_tuple": anchor_tuple,
        }

        # Add summary section for support doc_part
        if doc_part == "support":
            result["summary"] = self._build_support_summary(anchors_by_type)

        logger.info(f"  Total anchors: {len(anchor_tuple)}")

        return result

    def _create_extractors(self, id_generator: IdGenerator) -> dict[str, Any]:
        """Create extractor instances with shared IdGenerator.

        Args:
            id_generator: IdGenerator instance for this doc_part

        Returns:
            Dict mapping extractor names to extractor instances
        """
        return {
            "file": FileExtractor(self.path_config, self.data_loader, id_generator),
            "function": FunctionExtractor(self.path_config, self.data_loader, id_generator),
            "section": SectionExtractor(self.path_config, self.data_loader, id_generator),
            "note": NoteExtractor(self.path_config, self.data_loader, id_generator),
        }

    def _get_section_patterns(self, doc_part: str) -> list[str]:
        """Determine section extraction patterns based on doc_part.

        Args:
            doc_part: doc_part name

        Returns:
            List of section pattern types to extract
        """
        if doc_part == "constraint":
            # constraint uses SocVersion and NPU_ARCH patterns
            return ["soc_version", "npu_arch"]
        elif doc_part == "support":
            # support uses TYPE1 (AddConfig), TYPE2 (SocVersion), TYPE3 (NPU_ARCH)
            return ["add_config", "soc_version", "npu_arch"]
        elif doc_part == "sub_function_hash":
            # sub_function uses keyword-based extraction
            return ["keyword"]
        else:
            # Default: all patterns
            return ["soc_version", "npu_arch", "add_config", "keyword"]

    def _assign_gene_rules(
        self,
        anchors_by_type: dict[str, list[dict]],
        config: DocPartConfig,
    ) -> list[dict[str, Any]]:
        """Assign gene_rule to anchors based on GeneRuleRegistry mapping.

        Args:
            anchors_by_type: Dict mapping anchor_type to list of anchors
            config: DocPartConfig for this doc_part

        Returns:
            Flat list of all anchors with gene_rule assigned
        """
        anchor_tuple = []
        gene_rule_mapping = config.gene_rule_mapping

        for anchor_type, anchors in anchors_by_type.items():
            type_mapping = gene_rule_mapping.get(anchor_type, {})

            for anchor in anchors:
                # Determine which gene_rule to apply based on anchor characteristics
                gene_rule_key = self._determine_gene_rule_key(anchor, anchor_type, config.doc_part)

                # Get gene_rule from mapping
                if gene_rule_key in type_mapping:
                    gene_rule = type_mapping[gene_rule_key]
                elif "default" in type_mapping:
                    gene_rule = type_mapping["default"]
                else:
                    # Fallback: use first available mapping
                    gene_rule = list(type_mapping.values())[0] if type_mapping else {"deter": "暂空", "target": []}

                # Assign gene_rule to evidence
                anchor["evidence"][0]["gene_rule"] = {
                    "deter": gene_rule.get("deter", "暂空"),
                    "target": gene_rule.get("target", []),
                }

                # Add envidence_id (compatibility requirement)
                anchor["evidence"][0]["envidence_id"] = anchor["evidence"][0]["evidence_id"]

                anchor_tuple.append(anchor)

        return anchor_tuple

    def _determine_gene_rule_key(
        self,
        anchor: dict[str, Any],
        anchor_type: str,
        doc_part: str,
    ) -> str:
        """Determine which gene_rule key to use for an anchor.

        Args:
            anchor: Anchor dict
            anchor_type: Anchor type (file, function, section, note)
            doc_part: doc_part name

        Returns:
            Gene rule key string (e.g., "op_api_aclnn_h", "TYPE1", "default")
        """
        if anchor_type == ANCHOR_TYPE_FILE:
            file_path = anchor["anchor_information"]["level1"]["file_path"]
            file_name = anchor["anchor_information"]["level1"]["file_name"]

            # Determine file pattern based on path and name
            if "op_api/aclnn" in file_path and file_name.endswith(".h"):
                key = "op_api_aclnn_h"
            elif "op_api/aclnn" in file_path and file_name.endswith(".cpp"):
                key = "op_api_aclnn_cpp"
            elif "op_tiling" in file_path:
                key = "op_host_tiling_cpp"
            elif "op_host" in file_path and "def" in file_name:
                key = "op_host_def_cpp"
            elif "op_host" in file_path and "tiling" in file_name:
                key = "op_host_tiling_cpp"
            elif "op_kernel" in file_path:
                key = "op_kernel_cpp_h"
            elif "op_graph" in file_path and file_name.endswith(".h"):
                key = "proto_h"
            else:
                key = "default"

            # sub_function_hash 使用注册表兼容的短名 key
            if doc_part == "sub_function_hash":
                _short_key_map = {
                    "op_api_aclnn_h": "aclnn_h",
                    "op_api_aclnn_cpp": "aclnn_cpp",
                    "op_host_def_cpp": "def_h",
                    "op_host_tiling_cpp": "tiling_h",
                    "proto_h": "proto_h",
                }
                return _short_key_map.get(key, key)

            return key

        elif anchor_type == ANCHOR_TYPE_SECTION:
            # Check for pattern-specific metadata in evidence
            evidence = anchor["evidence"][0]

            if "support_chips" in evidence:
                chips_data = evidence["support_chips"]
                source = chips_data.get("source", "")

                if source == "AICore().AddConfig":
                    return "TYPE1"
                elif source == "platform_ascendc::SocVersion":
                    return "TYPE2"
                elif source in ["__NPU_ARCH__", "__CCE_AICORE__"]:
                    return "TYPE3"

            if "keyword_source" in evidence:
                return evidence["keyword_source"]

            # Fallback based on doc_part
            if doc_part == "constraint":
                file_path = anchor["anchor_information"]["level1"]["file_path"]
                if "tiling" in file_path:
                    return "op_host_tiling_cpp"
                else:
                    return "op_kernel_cpp_h"

            return "default"

        elif anchor_type == ANCHOR_TYPE_FUNCTION:
            return "default"

        elif anchor_type == ANCHOR_TYPE_NOTE:
            return "default"

        return "default"

    def _build_doc_structure(
        self,
        doc_part: str,
        config: DocPartConfig,
        anchor_tuple: list[dict],
    ) -> dict[str, Any]:
        """Build doc structure for output.

        Args:
            doc_part: doc_part name
            config: DocPartConfig for this doc_part
            anchor_tuple: List of anchors with gene_rule assigned

        Returns:
            Doc dict with structure matching golden reference
        """
        # Use explicit md_file from path_config, fall back to first aclnn*.md
        if self.path_config.md_file:
            first_md_file = Path(self.path_config.md_file)
        else:
            md_files = sorted(self.path_config.operator_path.glob("**/aclnn*.md"))
            first_md_file = md_files[0] if md_files else None

        doc_path = str(first_md_file.parent) if first_md_file else ""
        doc_name = first_md_file.name if first_md_file else ""

        doc_json_file = None
        if first_md_file:
            doc_json_file = self.path_config.doc_json / f"{first_md_file.stem}.json"

        # Build base doc structure
        doc = {
            "doc_path": doc_path,
            "doc_name": doc_name,
            "doc_part": doc_part,
            "md_format": [],
            "sub_part": deepcopy(config.sub_part_template),
        }

        # Load md_format from doc_json if available
        if doc_json_file and doc_json_file.exists():
            try:
                with open(doc_json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    # Extract relevant section based on doc_part
                    if doc_part == "constraint":
                        doc["md_format"] = [json_data.get("constraint", {}).get("origin_txt", "")]
                        doc["sub_part"]["constrain_clarify"][0]["content"] = json_data.get("constraint", {}).get(
                            "constrain_clarify", ""
                        )
                    elif doc_part == "support":
                        support_map = self.data_loader.load_support_matrix()
                        doc["md_format"] = [support_map.get(doc_name, "")]
                        doc["sub_part"]["support_matrix"][0]["content"] = support_map.get(doc_name, "")
                    elif doc_part == "summary":
                        doc["md_format"] = [json_data.get("summary", {}).get("origin_txt", "")]
                        doc["sub_part"]["summary_clarify"][0]["content"] = json_data.get("summary", {}).get(
                            "summary_clarify", ""
                        )
                        doc["sub_part"]["summary_formula"][0]["content"] = json_data.get("summary", {}).get(
                            "summary_formula", ""
                        )
                    elif doc_part == "function_proto":
                        doc["md_format"] = [json_data.get("function_proto", {}).get("origin_txt", "")]
                        doc["sub_part"]["function_proto_clarify"][0]["content"] = json_data.get(
                            "function_proto", {}
                        ).get("function_proto_clarify", "")
                        doc["sub_part"]["sub_function_sign"][0]["content"] = json_data.get("function_proto", {}).get(
                            "sub_function_sign", ""
                        )
            except Exception as e:
                logger.warning(f"Error loading doc_json: {e}")

        # Populate evidence_id list in sub_part
        if doc_part == "constraint":
            evidence_ids = [a["evidence"][0]["evidence_id"] for a in anchor_tuple]
            doc["sub_part"]["constrain_clarify"][0]["evidence_id"] = evidence_ids

        elif doc_part == "support":
            evidence_ids = [a["evidence"][0]["evidence_id"] for a in anchor_tuple]
            doc["sub_part"]["support_matrix"][0]["evidence_ids"] = evidence_ids

        elif doc_part == "sub_function_hash":
            # Populate sub_function_hash sub_part from doc_json data
            if doc_json_file is not None:
                doc = self._populate_sub_function_hash_sub_part(doc, doc_json_file)
            else:
                logger.warning("No doc_json_file available for sub_function_hash sub_part population")

        return doc

    def _get_sub_function_count(self) -> int:
        """Get sub_function_count from doc_json. Returns 1 with warning if unavailable."""
        md_stem = self.path_config.md_stem
        if md_stem:
            doc_json_file = self.path_config.doc_json / f"{md_stem}.json"
        else:
            md_files = sorted(self.path_config.operator_path.glob("**/aclnn*.md"))
            if not md_files:
                logger.warning("No aclnn*.md found, sub_function_count defaults to 1")
                return 1
            doc_json_file = self.path_config.doc_json / f"{md_files[0].stem}.json"
        if not doc_json_file.exists():
            logger.warning(f"doc_json not found: {doc_json_file}, sub_function_count defaults to 1")
            return 1
        try:
            with open(doc_json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            count = len([k for k in json_data if k.startswith("sub_function_hash_")])
            return count if count > 0 else 1
        except Exception as e:
            logger.warning(f"Error reading doc_json: {e}, sub_function_count defaults to 1")
            return 1

    def _populate_sub_function_hash_sub_part(self, doc: dict[str, Any], doc_json_file: Path) -> dict[str, Any]:
        """Populate sub_function_hash sub_part from doc_json data.

        Args:
            doc: Base doc structure with empty sub_part template
            doc_json_file: Path to doc_json file

        Returns:
            Doc dict with populated sub_part matching golden reference structure
        """
        if not doc_json_file.exists():
            logger.warning(f"doc_json file not found: {doc_json_file}")
            return doc

        try:
            with open(doc_json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Find all sub_function_hash_N keys in doc_json
            sub_function_keys = [k for k in json_data.keys() if k.startswith("sub_function_hash_")]
            sub_function_count = len(sub_function_keys)

            logger.debug(f"Found {sub_function_count} sub_functions in doc_json")

            # Initialize sub_part structure
            sub_part = {
                "param_matrix": {},
                "fixconfig_1": {},
                "fixconfig_2": {},
                "changeconfig": {},
                "errorcode_matrix": {},
            }

            # Process each sub_function
            for i in range(1, sub_function_count + 1):
                sub_func_key = f"sub_function_hash_{i}"
                sub_func_data = json_data.get(sub_func_key, {})

                # Extract function name from changeconfig
                func_name = sub_func_data.get("changeconfig", "")

                # Build param_matrix entry
                param_matrix_key = f"sub_function_{i}_param_matrix"
                param_matrix_data = sub_func_data.get(f"subfunction_{func_name}_param_matrix", [])

                # Convert param_matrix array to nested dict structure
                param_matrix_dict = {}
                if param_matrix_data:
                    # Extract field names from first row
                    fields = [
                        "des",
                        "inout",
                        "usage",
                        "dtype",
                        "dmat",
                        "shape",
                        "tensor",
                    ]

                    for field in fields:
                        field_dict = {}
                        for row in param_matrix_data:
                            param_name = row.get("参数名", "")
                            if field == "des":
                                field_dict[param_name] = row.get("描述", "")
                            elif field == "inout":
                                field_dict[param_name] = row.get("输入/输出", "")
                            elif field == "usage":
                                field_dict[param_name] = row.get("使用说明", "")
                            elif field == "dtype":
                                field_dict[param_name] = row.get("数据类型", "")
                            elif field == "dmat":
                                field_dict[param_name] = row.get("数据格式", "")
                            elif field == "shape":
                                field_dict[param_name] = row.get("维度(shape)", "")
                            elif field == "tensor":
                                field_dict[param_name] = row.get("非连续Tensor", "")

                        param_matrix_dict[field] = [{"content": field_dict}]

                sub_part["param_matrix"][param_matrix_key] = param_matrix_dict

                # Build fixconfig_1 entry
                fixconfig_1_key = f"sub_function_{i}_fixconfig_1"
                fixconfig_1_value = sub_func_data.get(f"subfunction_{func_name}_fixconfig_1", "")
                sub_part["fixconfig_1"][fixconfig_1_key] = fixconfig_1_value

                # Build fixconfig_2 entry
                fixconfig_2_key = f"sub_function_{i}_fixconfig_2"
                fixconfig_2_value = sub_func_data.get(f"subfunction_{func_name}_fixconfig_2", "")
                sub_part["fixconfig_2"][fixconfig_2_key] = fixconfig_2_value

                # Build changeconfig entry
                changeconfig_key = f"sub_function_{i}_changeconfig"
                changeconfig_value = sub_func_data.get(f"subfunction_{func_name}_changeconfig", "")
                sub_part["changeconfig"][changeconfig_key] = changeconfig_value

                # Build errorcode_matrix entry
                errorcode_key = f"sub_function_{i}_errorcode"
                errorcode_value = sub_func_data.get(f"subfunction_{func_name}.errorcode", "")
                sub_part["errorcode_matrix"][errorcode_key] = errorcode_value

            doc["sub_part"] = sub_part

            md_format_list = []
            for i in range(1, sub_function_count + 1):
                sub_func_key = f"sub_function_hash_{i}"
                sub_func_data = json_data.get(sub_func_key, {})
                md_format_list.append(sub_func_data.get("origin_txt", ""))
            doc["md_format"] = md_format_list

            logger.info(f"Populated sub_function_hash sub_part with {sub_function_count} sub_functions")

        except Exception as e:
            logger.error(f"Error populating sub_function_hash sub_part: {e}")

        return doc

    def _build_support_summary(self, anchors_by_type: dict[str, list]) -> dict[str, Any]:
        """Build summary section for support doc_part.

        Args:
            anchors_by_type: Dict mapping anchor_type to list of anchors

        Returns:
            Summary dict with statistics
        """
        section_anchors = anchors_by_type.get(ANCHOR_TYPE_SECTION, [])
        file_anchors = anchors_by_type.get(ANCHOR_TYPE_FILE, [])

        # Count by type
        type1_count = 0
        type2_count = 0
        type3_count = 0
        chips = []
        socs = []
        archs = []

        for anchor in section_anchors:
            evidence = anchor["evidence"][0]
            if "support_chips" in evidence:
                chips_data = evidence["support_chips"]
                source = chips_data.get("source", "")

                if source == "AICore().AddConfig":
                    type1_count += 1
                    chips.extend(chips_data.get("chips", []))
                elif source == "platform_ascendc::SocVersion":
                    type2_count += 1
                    socs.extend(chips_data.get("socs", []))
                elif source in ["__NPU_ARCH__", "__CCE_AICORE__"]:
                    type3_count += 1
                    archs.append(chips_data.get("arch", ""))

        # Deduplicate
        chips = list(set(chips))
        socs = list(set(socs))
        archs = list(set(archs))

        all_anchors = section_anchors + file_anchors
        all_evidence_ids = [a["evidence"][0]["evidence_id"] for a in all_anchors]

        # Load support_matrix content
        support_map = self.data_loader.load_support_matrix()
        if self.path_config.md_file:
            md_name = Path(self.path_config.md_file).name
        else:
            md_files = list(self.path_config.operator_path.glob("**/aclnn*.md"))
            first_md_file = md_files[0] if md_files else None
            md_name = first_md_file.name if first_md_file else ""
        content_len = len(support_map.get(md_name, ""))

        return {
            "type1_section_count": type1_count,
            "type2_section_count": type2_count,
            "type3_section_count": type3_count,
            "file_count": len(file_anchors),
            "total_anchor_count": len(all_anchors),
            "total_evidence_count": len(all_evidence_ids),
            "support_matrix_content_len": content_len,
            "chips": chips,
            "socs": socs,
            "archs": archs,
        }

    def _extract_summary_proto_shared(self) -> dict[str, Any]:
        """Extract shared anchors for summary and function_proto.

        summary and function_proto share the same extraction logic but differ in gene_rules.
        This method extracts once and returns raw anchors for both to use.

        Returns:
            Dict with:
                - file_result: Dict with categorized file anchors
                - function_result: Dict with categorized function anchors
                - note_result: Dict with categorized note anchors
        """
        logger.info("  Extracting shared anchors for summary↔function_proto...")

        # Use summary config for extraction (both use same anchor_types)
        summary_config = self.registry.get_config("summary")
        id_generator = IdGenerator(summary_config.id_config)
        extractors = self._create_extractors(id_generator)

        # Extract file anchors
        file_anchors = extractors["file"].extract(self.operator_path, source_files=self._source_files)

        # Categorize file anchors by pattern
        file_result = {
            "op_api_aclnn_h": [],
            "op_api_aclnn_cpp": [],
            "op_kernel_all": [],
        }

        for anchor in file_anchors:
            file_path = anchor["anchor_information"]["level1"]["file_path"]
            file_name = anchor["anchor_information"]["level1"]["file_name"]

            if "op_api/aclnn" in file_path and file_name.endswith(".h"):
                file_result["op_api_aclnn_h"].append(anchor)
            elif "op_api/aclnn" in file_path and file_name.endswith(".cpp"):
                file_result["op_api_aclnn_cpp"].append(anchor)
            elif "op_kernel" in file_path:
                file_result["op_kernel_all"].append(anchor)

        # Extract function anchors
        function_anchors = extractors["function"].extract(self.operator_path, source_files=self._source_files)
        function_result = {"all": function_anchors}

        # Extract note anchors
        note_anchors = extractors["note"].extract(self.operator_path, source_files=self._source_files)
        note_result = {"all": note_anchors}

        logger.info(
            f"    Files: aclnn_h={len(file_result['op_api_aclnn_h'])}, "
            f"aclnn_cpp={len(file_result['op_api_aclnn_cpp'])}, "
            f"kernel={len(file_result['op_kernel_all'])}"
        )
        logger.info(f"    Functions: {len(function_anchors)}")
        logger.info(f"    Notes: {len(note_anchors)}")

        return {
            "file_result": file_result,
            "function_result": function_result,
            "note_result": note_result,
        }

    def _process_summary_from_shared(self, shared_results: dict) -> dict[str, Any]:
        """Process summary doc_part from shared extraction results.

        Args:
            shared_results: Dict from _extract_summary_proto_shared()

        Returns:
            Summary output dict with gene_rules assigned
        """
        logger.info("  Processing summary from shared results...")

        summary_config = self.registry.get_config("summary")

        # Deep copy shared results to avoid mutation
        file_result_copy = deepcopy(shared_results["file_result"])
        note_result_copy = deepcopy(shared_results["note_result"])

        # Assign summary gene_rules
        for item in file_result_copy["op_api_aclnn_h"]:
            item["evidence"][0]["gene_rule"]["target"] = [
                ["summary_clarify"],
                ["summary_formula"],
            ]

        for item in file_result_copy["op_api_aclnn_cpp"]:
            item["evidence"][0]["gene_rule"]["target"] = [["summary_clarify"]]

        for item in file_result_copy["op_kernel_all"]:
            item["evidence"][0]["gene_rule"]["target"] = [["summary_formula"]]

        for item in note_result_copy["all"]:
            item["evidence"][0]["gene_rule"]["target"] = [
                ["summary_clarify"],
                ["summary_formula"],
            ]

        # Combine anchors
        anchor_tuple = (
            file_result_copy["op_api_aclnn_h"]
            + file_result_copy["op_api_aclnn_cpp"]
            + file_result_copy["op_kernel_all"]
            + note_result_copy["all"]
        )

        # Build doc structure
        doc = self._build_doc_structure("summary", summary_config, anchor_tuple)

        return {
            "doc": doc,
            "anchor_tuple": anchor_tuple,
        }

    def _process_function_proto_from_shared(self, shared_results: dict) -> dict[str, Any]:
        """Process function_proto doc_part from shared extraction results.

        Args:
            shared_results: Dict from _extract_summary_proto_shared()

        Returns:
            function_proto output dict with gene_rules assigned
        """
        logger.info("  Processing function_proto from shared results...")

        function_proto_config = self.registry.get_config("function_proto")

        # Deep copy shared results to avoid mutation
        file_result_copy = deepcopy(shared_results["file_result"])
        function_result_copy = deepcopy(shared_results["function_result"])

        # Assign function_proto gene_rules
        for item in file_result_copy["op_api_aclnn_h"]:
            item["evidence"][0]["gene_rule"]["target"] = [
                ["function_proto_clarify"],
                ["sub_function_sign"],
            ]

        for item in function_result_copy["all"]:
            item["evidence"][0]["gene_rule"]["target"] = [["sub_function_sign"]]

        # Combine anchors
        anchor_tuple = file_result_copy["op_api_aclnn_h"] + function_result_copy["all"]

        # Build doc structure
        doc = self._build_doc_structure("function_proto", function_proto_config, anchor_tuple)

        return {
            "doc": doc,
            "anchor_tuple": anchor_tuple,
        }

    def _write_output(self, doc_part: str, result: dict[str, Any]) -> None:
        """Write output JSON to anchor_dir.

        Args:
            doc_part: doc_part name
            result: Output dict to write
        """
        config = self.registry.get_config(doc_part)
        output_file = self.path_config.anchor_dir / config.output_filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        logger.info(f"  Written to: {output_file}")


if __name__ == "__main__":
    """Self-test for AnchorPipeline."""
    import os
    import sys

    print("Testing AnchorPipeline...")

    # Use abs operator for testing
    operator_path = Path(
        os.environ.get(
            "TEST_OPERATOR_PATH",
            Path(os.environ.get("CANN_OPS_ROOT", "")) / "math" / "abs",
        )
    )
    out_path = Path("/tmp/anchor_pipeline_test")

    if not operator_path.exists():
        print(
            f"Error: operator_path not found: {operator_path}\n"
            "请设置 TEST_OPERATOR_PATH 或 CANN_OPS_ROOT 环境变量指向本机 ops 检出"
        )
        sys.exit(1)

    # Create pipeline
    pipeline = AnchorPipeline(operator_path, out_path)

    # Run pipeline
    results = pipeline.run()

    print(f"\nGenerated {len(results)} doc_part outputs:")
    for doc_part, result in results.items():
        print(f"  {format_doc_part(doc_part)}: {len(result.get('anchor_tuple', []))} anchors")

    # Verify output files exist
    anchor_dir = out_path / "abs" / "anchor"
    print(f"\nOutput directory: {anchor_dir}")

    for doc_part in DOC_PARTS:
        config = get_registry().get_config(doc_part)
        output_file = anchor_dir / config.output_filename
        if output_file.exists():
            print(f"  ✓ {config.output_filename}")
        else:
            print(f"  ✗ {config.output_filename} (missing)")

    print("\nAnchorPipeline self-test passed!")
