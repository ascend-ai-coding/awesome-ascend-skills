# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
ID Generator for anchor and evidence IDs.

Supports per-doc_part ID format configuration to reproduce all existing ID formats:
- constraint/sub_function: type prefix in both anchor and evidence IDs
- support: category offsets, type prefix in anchor only
- summary/function_proto: type prefix in anchor only
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IdConfig:
    """Configuration for ID generation format.

    Attributes:
        type_prefix_in_anchor: Whether anchor IDs include the type prefix
        type_prefix_in_evidence: Whether evidence IDs include the type prefix
        category_offsets: Per-category start offsets
            (e.g., TYPE1→1, TYPE2→101, TYPE3→201)
        shared_counters: Whether anchor and evidence share the same counter
            (True for constraint/sub_function, False for support/summary)
    """

    type_prefix_in_anchor: bool = True
    type_prefix_in_evidence: bool = True
    category_offsets: dict[str, int] | None = None
    shared_counters: bool = True


class IdGenerator:
    """Generator for anchor and evidence IDs with configurable format.

    Supports reproducing all existing ID formats across different doc_parts:
    - constraint/sub_function: type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True
    - support: category_offsets with TYPE1→1, TYPE2→101, TYPE3→201, shared_counters=False
    - summary/function_proto: type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False
    """

    def __init__(self, config: IdConfig):
        """Initialize the ID generator with configuration.

        Args:
            config: IdConfig instance specifying ID format
        """
        self.config: IdConfig = config
        self._counters: dict[str, int] = {}
        self._anchor_counters: dict[str, int] = {}
        self._evidence_counters: dict[str, int] = {}
        logger.debug(f"IdGenerator initialized with config: {config}")

    def _get_counter_key(self, category: str = "") -> str:
        """Get counter key for a category.

        Args:
            category: Category name (empty string for default)

        Returns:
            Counter key string
        """
        return category or "_default_"

    def _init_counter(self, category: str = "") -> int:
        """Initialize counter with category offset if configured.

        Args:
            category: Category name

        Returns:
            Initial counter value (offset-1 so first increment gives offset)
        """
        if self.config.category_offsets and category in self.config.category_offsets:
            return self.config.category_offsets[category] - 1
        else:
            return 0

    def _increment_shared_counter(self, category: str = "") -> int:
        """Increment shared counter for a category.

        Args:
            category: Category name (empty string for default)

        Returns:
            New counter value after increment
        """
        key = self._get_counter_key(category)
        if key not in self._counters:
            self._counters[key] = self._init_counter(category)
        self._counters[key] += 1
        return self._counters[key]

    def _increment_anchor_counter(self, category: str = "") -> int:
        """Increment anchor counter for a category (separate counters mode).

        Args:
            category: Category name (empty string for default)

        Returns:
            New counter value after increment
        """
        key = self._get_counter_key(category)
        if key not in self._anchor_counters:
            self._anchor_counters[key] = self._init_counter(category)
        self._anchor_counters[key] += 1
        return self._anchor_counters[key]

    def _increment_evidence_counter(self, category: str = "") -> int:
        """Increment evidence counter for a category (separate counters mode).

        Args:
            category: Category name (empty string for default)

        Returns:
            New counter value after increment
        """
        key = self._get_counter_key(category)
        if key not in self._evidence_counters:
            self._evidence_counters[key] = self._init_counter(category)
        self._evidence_counters[key] += 1
        return self._evidence_counters[key]

    def next_anchor_id(self, anchor_type: str = "", category: str = "") -> str:
        """Generate next anchor ID.

        Args:
            anchor_type: Type of anchor (e.g., "file", "function", "section", "note")
            category: Category name for offset-based counters (e.g., "TYPE1", "TYPE2")

        Returns:
            Formatted anchor ID string

        Examples:
            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True)
            >>> gen = IdGenerator(config)
            >>> gen.next_anchor_id("file")
            'anchor_file_001'

            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False)
            >>> gen = IdGenerator(config)
            >>> gen.next_anchor_id("section")
            'anchor_section_001'

            >>> config = IdConfig(category_offsets={"TYPE1": 1, "TYPE2": 101}, shared_counters=False)
            >>> gen = IdGenerator(config)
            >>> gen.next_anchor_id("section", "TYPE1")
            'anchor_section_001'
            >>> gen.next_anchor_id("section", "TYPE2")
            'anchor_section_101'
        """
        if self.config.shared_counters:
            seq = self._increment_shared_counter(category)
        else:
            seq = self._increment_anchor_counter(category)

        if self.config.type_prefix_in_anchor and anchor_type:
            anchor_id = f"anchor_{anchor_type}_{seq:03d}"
        else:
            anchor_id = f"anchor_{seq:03d}"

        logger.debug(f"Generated anchor_id: {anchor_id} (type={anchor_type}, category={category}, seq={seq})")
        return anchor_id

    def next_evidence_id(self, anchor_type: str = "", category: str = "") -> str:
        """Generate next evidence ID.

        Args:
            anchor_type: Type of anchor (e.g., "file", "function", "section", "note")
            category: Category name for offset-based counters (e.g., "TYPE1", "TYPE2")

        Returns:
            Formatted evidence ID string

        Examples:
            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True)
            >>> gen = IdGenerator(config)
            >>> gen.next_evidence_id("file")
            'ev_file_002'  # Counter already incremented by next_anchor_id

            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False)
            >>> gen = IdGenerator(config)
            >>> gen.next_evidence_id("section")
            'ev_001'

            >>> config = IdConfig(category_offsets={"TYPE1": 1, "TYPE2": 101}, shared_counters=False)
            >>> gen = IdGenerator(config)
            >>> gen.next_evidence_id("section", "TYPE1")
            'ev_001'
            >>> gen.next_evidence_id("section", "TYPE2")
            'ev_101'
        """
        if self.config.shared_counters:
            # Use the same counter as anchor (already incremented)
            key = self._get_counter_key(category)
            seq = self._counters.get(key, 0)
        else:
            seq = self._increment_evidence_counter(category)

        if self.config.type_prefix_in_evidence and anchor_type:
            evidence_id = f"ev_{anchor_type}_{seq:03d}"
        else:
            evidence_id = f"ev_{seq:03d}"

        logger.debug(f"Generated evidence_id: {evidence_id} (type={anchor_type}, category={category}, seq={seq})")
        return evidence_id

    def next_pair(self, anchor_type: str = "", category: str = "") -> tuple[str, str]:
        """Generate next anchor and evidence ID pair.

        This is the recommended method when shared_counters=True, as it generates
        both IDs with the same counter value in one call.

        Args:
            anchor_type: Type of anchor (e.g., "file", "function", "section", "note")
            category: Category name for offset-based counters (e.g., "TYPE1", "TYPE2")

        Returns:
            Tuple of (anchor_id, evidence_id)

        Examples:
            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=True, shared_counters=True)
            >>> gen = IdGenerator(config)
            >>> gen.next_pair("file")
            ('anchor_file_001', 'ev_file_001')

            >>> config = IdConfig(type_prefix_in_anchor=True, type_prefix_in_evidence=False, shared_counters=False)
            >>> gen = IdGenerator(config)
            >>> gen.next_pair("section")
            ('anchor_section_001', 'ev_001')
        """
        if self.config.shared_counters:
            # Increment once and use same value for both
            seq = self._increment_shared_counter(category)

            if self.config.type_prefix_in_anchor and anchor_type:
                anchor_id = f"anchor_{anchor_type}_{seq:03d}"
            else:
                anchor_id = f"anchor_{seq:03d}"

            if self.config.type_prefix_in_evidence and anchor_type:
                evidence_id = f"ev_{anchor_type}_{seq:03d}"
            else:
                evidence_id = f"ev_{seq:03d}"

            logger.debug(
                f"Generated pair: ({anchor_id}, {evidence_id}) (type={anchor_type}, category={category}, seq={seq})"
            )
            return anchor_id, evidence_id
        else:
            # Use separate counters
            anchor_id = self.next_anchor_id(anchor_type, category)
            evidence_id = self.next_evidence_id(anchor_type, category)
            return anchor_id, evidence_id

    def reset(self, category: str = "") -> None:
        """Reset counter for a category.

        Args:
            category: Category name to reset (empty string for default)
        """
        key = self._get_counter_key(category)

        if self.config.shared_counters:
            self._counters[key] = self._init_counter(category)
        else:
            self._anchor_counters[key] = self._init_counter(category)
            self._evidence_counters[key] = self._init_counter(category)

        logger.debug(f"Reset counter for category: {category}")
