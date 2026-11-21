# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Configuration for the anchor redirect mapper."""

from dataclasses import dataclass, field


class Config:
    """Global configuration settings."""

    SIMILARITY_THRESHOLD = 0.5
    OUTPUT_DIR = "output/func_result.json"
    # Mapping level names
    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"


@dataclass
class MappingContext:
    """Context for mapping operations."""

    ops_name: str
    file_path: str
    code_map_dir: str
    new_operator_path: str
    anchor_path: str
    output_dir: str = Config.OUTPUT_DIR
    func_save_path: str = Config.OUTPUT_DIR


@dataclass
class LevelResult:
    """Result from Level 1 file matching."""

    matched: bool = False
    old_file_path: str = ""
    new_file_path: str = ""
    new_function_names: list = field(default_factory=list)
    old_function_names: list = field(default_factory=list)
    function_names_changed: bool = False
    message: str = ""
    update_anchor: dict = field(default_factory=dict)


@dataclass
class Level3Result:
    """Result from Level 3 directory matching."""

    candidates: list = field(default_factory=list)

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


@dataclass
class MappingResult:
    """Final mapping result."""

    success: bool = False
    similarity: float = 0.0
    message: str = ""
    mapped_functions: dict = field(default_factory=dict)
    updated_evidence: list = field(default_factory=list)

    def __post_init__(self):
        if self.mapped_functions is None:
            self.mapped_functions = {}
        if self.updated_evidence is None:
            self.updated_evidence = []

    def __str__(self):
        return f"MappingResult(success={self.success}, similarity={self.similarity:.2f}, message={self.message})"
