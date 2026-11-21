#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""
Unified SkipInfo class for tracking skip events across extraction modules.

This module provides a parameterized SkipInfo class that can be configured
with different skip types for different extraction contexts.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SkipInfo:
    """Class to track skip information with configurable skip types.

    This class provides a unified interface for tracking skip events across
    different extraction modules. Each module can instantiate SkipInfo with
    its own set of skip types.

    Attributes:
        skip_types: Dictionary mapping skip type keys to their descriptions.
        skip_counts: Dictionary tracking count of each skip type.
        skip_details: Dictionary storing details of each skip event.

    Example:
        >>> skip_types = {
        ...     'FILE_READ_ERROR': 'File read error',
        ...     'NO_MATCHING_FILES': 'No matching files found'
        ... }
        >>> skip_info = SkipInfo(skip_types)
        >>> skip_info.record_skip('FILE_READ_ERROR', '/path/to/file', 'Permission denied')
        >>> skip_info.get_total_skips()
        1
    """

    def __init__(self, skip_types: Dict[str, str]):
        """Initialize SkipInfo with configurable skip types.

        Args:
            skip_types: Dictionary mapping skip type keys to their descriptions.
                       Example: {'FILE_READ_ERROR': 'File read error'}
        """
        self.skip_types = skip_types
        self.skip_counts: Dict[str, int] = {skip_type: 0 for skip_type in skip_types}
        self.skip_details: Dict[str, List[Dict[str, str]]] = {skip_type: [] for skip_type in skip_types}

    def record_skip(self, skip_type: str, item: str, reason: str = "") -> None:
        """Record a skip event.

        Args:
            skip_type: The type of skip (must be a key in skip_types).
            item: The item that was skipped (e.g., file path, function name).
            reason: Optional reason for the skip.

        Note:
            If skip_type is not in the configured skip_types, the event is ignored
            and a warning is logged.
        """
        if skip_type not in self.skip_counts:
            logger.warning(f"Unknown skip type '{skip_type}'. Valid types: {list(self.skip_types.keys())}")
            return

        self.skip_counts[skip_type] += 1
        self.skip_details[skip_type].append({"item": item, "reason": reason})

        logger.debug(f"Recorded skip: type={skip_type}, item={item}, reason={reason}")

    def get_total_skips(self) -> int:
        """Get total number of skips across all types.

        Returns:
            Sum of all skip counts.
        """
        return sum(self.skip_counts.values())

    def merge(self, other: "SkipInfo") -> None:
        """Merge another SkipInfo instance into this one.

        Combines skip counts and details from another SkipInfo instance.
        Only skip types that exist in both instances are merged.

        Args:
            other: Another SkipInfo instance to merge into this one.

        Note:
            Skip types that exist in 'other' but not in 'self' are ignored.
        """
        for skip_type in other.skip_types:
            if skip_type not in self.skip_counts:
                logger.warning(
                    f"Skip type '{skip_type}' from other SkipInfo not in current skip types. "
                    f"Skipping merge for this type."
                )
                continue

            self.skip_counts[skip_type] += other.skip_counts[skip_type]
            self.skip_details[skip_type].extend(other.skip_details[skip_type])

        logger.debug(f"Merged SkipInfo. Total skips: {self.get_total_skips()}")

    def get_skip_count(self, skip_type: str) -> int:
        """Get count for a specific skip type.

        Args:
            skip_type: The skip type to query.

        Returns:
            Count for the specified skip type, or 0 if type doesn't exist.
        """
        return self.skip_counts.get(skip_type, 0)

    def get_skip_details(self, skip_type: str) -> List[Dict[str, str]]:
        """Get details for a specific skip type.

        Args:
            skip_type: The skip type to query.

        Returns:
            List of detail dictionaries for the specified skip type,
            or empty list if type doesn't exist.
        """
        return self.skip_details.get(skip_type, [])

    def to_dict(self) -> Dict[str, Any]:
        """Export skip information as a dictionary.

        Returns:
            Dictionary containing skip_types, skip_counts, and skip_details.
        """
        return {
            "skip_types": self.skip_types,
            "skip_counts": self.skip_counts,
            "skip_details": self.skip_details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkipInfo":
        """Create a SkipInfo instance from a dictionary.

        Args:
            data: Dictionary containing skip_types, skip_counts, and skip_details.

        Returns:
            New SkipInfo instance.
        """
        instance = cls(data["skip_types"])
        instance.skip_counts = data["skip_counts"]
        instance.skip_details = data["skip_details"]
        return instance
