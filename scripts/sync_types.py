#!/usr/bin/env python3
"""Type definitions for external skills synchronization."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class ExternalSource:
    """External skills repository source configuration.

    Attributes:
        name: Unique identifier for the external source.
        url: Git repository URL (HTTPS or SSH).
        branch: Branch name to clone (default: "main").
        enabled: Whether this source is active (default: True).
        skills_path: Subdirectory path where skills are located (default: "" for root).
    """

    name: str
    url: str
    branch: str = "main"
    enabled: bool = True
    skills_path: str = ""


@dataclass
class Skill:
    """Represents a skill directory with SKILL.md.

    Attributes:
        name: Skill directory name.
        path: Absolute path to skill directory.
        source: ExternalSource this skill comes from.
        has_skill_md: Whether SKILL.md file exists.
    """

    name: str
    path: Path
    source: ExternalSource
    has_skill_md: bool


@dataclass
class SyncResult:
    """Result of syncing skills from a source.

    Attributes:
        synced: List of successfully synced skill names.
        skipped: List of (name, reason) tuples for skipped skills.
        errors: List of (name, error) tuples for failed operations.
    """

    synced: List[str]
    skipped: List[Tuple[str, str]]
    errors: List[Tuple[str, str]]


@dataclass
class ConflictInfo:
    """Information about a skill name conflict.

    Attributes:
        skill_name: The conflicting skill name.
        local_path: Path to existing local skill.
        external_source: Name of external source providing the skill.
    """

    skill_name: str
    local_path: str
    external_source: str
