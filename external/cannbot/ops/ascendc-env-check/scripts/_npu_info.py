#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified, robust NPU device information parser.

Design principles (from first principles):
1. npu-smi is a CLI for humans, not machines. Its output format changes.
2. The ONLY reliable npu-smi outputs are:
   a. `npu-smi info -t <type> -i <id>` -> key:value format (stable)
   b. `npu-smi info -m` -> fixed-width mapping (relatively stable)
3. `npu-smi info` (main table) is UNRELIABLE and must NEVER be parsed
   for critical data. It is used ONLY as last-resort fallback.
4. Runtime discovery: parse `npu-smi info --help` to discover available
   subcommands rather than hard-coding them.
5. Layered fallback: for each data need, try multiple subcommands in order
   of preference, gracefully degrading when a subcmd is unavailable.
6. Batch with `common`: `-t common` contains temperature, power, usage
   rates etc. Cache its output to reduce npu-smi calls per device.
7. All parsing results are validated. Format mismatch -> explicit warning,
   never silent wrong data.

Usage:
    python3 _npu_info.py --list              # list all NPU devices
    python3 _npu_info.py --json              # full info as JSON
    python3 _npu_info.py --health            # health status only
    python3 _npu_info.py --chip-name         # chip names only
    python3 _npu_info.py --discover          # show available subcommands

Import:
    from _npu_info import NpuInfoCollector
    collector = NpuInfoCollector()
    for npu_id in collector.get_npu_ids():
        print(collector.get_health(npu_id))
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of a parse operation with confidence and warnings."""
    value: any
    confidence: str  # "high", "medium", "low"
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.value is not None


def _run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[str, int, str]:
    """Run a command and return (stdout, returncode, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.returncode, result.stderr
    except FileNotFoundError:
        return "", -1, f"Command not found: {cmd[0]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return "", -1, str(e)


# ---------------------------------------------------------------------------
# NpuSmiDiscovery — runtime discovery of available subcommands
# ---------------------------------------------------------------------------


def _is_valid_type_name(t: str) -> bool:
    """Check if a string looks like a valid npu-smi info -t type name."""
    return t and (t.replace("-", "").replace("_", "").isalnum())


def _extract_types_from_text(text: str) -> List[str]:
    """Extract valid type names from a comma-separated text fragment."""
    types: List[str] = []
    for t in text.split(","):
        t = t.strip().rstrip(".")
        if _is_valid_type_name(t):
            types.append(t)
    return types


class NpuSmiDiscovery:
    """Discover available npu-smi info -t subcommands from --help output."""

    def __init__(self):
        self._available_types: Optional[set] = None

    @property
    def available_types(self) -> set:
        if self._available_types is None:
            self._available_types = self._discover_types()
        return self._available_types

    def is_available(self, subcmd: str) -> bool:
        return subcmd in self.available_types

    def get_available_list(self) -> List[str]:
        return sorted(self.available_types)

    @staticmethod
    def _discover_types() -> set:
        """Parse `npu-smi info --help` to get supported -t types.

        Help format (excerpt):
            -t type        Show information for type
                          type: board, flash, memory, usages, sensors, temp, power, volt,
                                common, health, product, ecc, ip, sys-time, ...
                                custom-op-secverify-cert.

            Options:
               -i %d      Card ID
        """
        types: set = set()
        stdout, rc, _ = _run_cmd(["npu-smi", "info", "--help"])
        if rc != 0 or not stdout:
            return types

        in_type_list = False
        for line in stdout.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            if "type:" in stripped.lower():
                in_type_list = True
                prefix = stripped.split("type:", 1)[1]
                types.update(_extract_types_from_text(prefix))
                continue

            if in_type_list:
                if stripped == "Options:" or stripped.startswith("-"):
                    break
                types.update(_extract_types_from_text(stripped))

        return types

# ---------------------------------------------------------------------------
# Key-value parser with flexible matching
# ---------------------------------------------------------------------------


def _parse_kv(output: str, key_pattern: str) -> Optional[str]:
    """Parse key:value format output with prefix matching.

    The key in npu-smi output may have units appended, e.g.:
        'NPU Temperature (C)            : 39'
        'Temperature(C)                 : 39'   # from -t common
        'NPU Real-time Power(W)         : 88.3'
        'HBM Capacity(MB)               : 65536'
        'HBM Usage Rate(%)              : 5'

    We match by checking if the output key starts with our pattern.
    """
    for line in output.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key_part, val_part = line.split(":", 1)
        key_part = key_part.strip()
        if key_part.startswith(key_pattern):
            return val_part.strip()
    return None


def _parse_kv_fuzzy(
    output: str, key_patterns: List[str]
) -> Tuple[Optional[str], List[str]]:
    """Try multiple key patterns, return first match + warnings."""
    warnings: List[str] = []
    for i, key in enumerate(key_patterns):
        val = _parse_kv(output, key)
        if val is not None:
            if i > 0:
                warnings.append(
                    f"Primary key '{key_patterns[0]}' not found, "
                    f"used fallback '{key}'"
                )
            return val, warnings
    return None, [f"None of keys {key_patterns} found"]


def _validate_kv_format(output: str) -> Tuple[bool, List[str]]:
    """Check if output looks like key:value format."""
    warnings: List[str] = []
    lines = [l for l in output.strip().split("\n") if l.strip()]
    if not lines:
        return False, ["Empty output"]

    kv_lines = [l for l in lines if ":" in l]
    if len(kv_lines) < 2:
        warnings.append(
            f"Output has only {len(kv_lines)} key:value lines. "
            f"First 100 chars: {output[:100]}"
        )
        return False, warnings

    # Check for table markers (|, +, =) which indicate main table format
    table_markers = sum(1 for l in lines if l.strip().startswith(("|", "+", "=")))
    if table_markers > 2:
        warnings.append(
            "Output contains table markers (|,+,-), may be table format "
            "instead of key:value"
        )
        return False, warnings

    return True, warnings


# ---------------------------------------------------------------------------
# Layered query strategies
# ---------------------------------------------------------------------------

# Ordered list of (subcmd, key_patterns) for each data need.
# Primary (first) is preferred; later entries are fallbacks.
QUERY_STRATEGIES: Dict[str, List[Tuple[str, List[str]]]] = {
    "health": [
        ("health", ["Health"]),
    ],
    "temperature": [
        ("temp", ["NPU Temperature", "Temperature", "HBM Temperature"]),
        ("common", ["Temperature", "NPU Temperature", "HBM Temperature"]),
        ("sensors", ["Temperature", "NPU Temperature"]),
    ],
    "power": [
        ("power", ["NPU Real-time Power", "Power"]),
        ("common", ["NPU Real-time Power", "Power"]),
    ],
    "memory_capacity": [
        ("memory", ["HBM Capacity"]),
        ("usages", ["HBM Capacity"]),
    ],
    "memory_usage": [
        ("usages", ["HBM Usage Rate"]),
        ("common", ["HBM Usage Rate"]),
    ],
    "aicore_usage": [
        ("usages", ["Aicore Usage Rate"]),
        ("common", ["Aicore Usage Rate"]),
    ],
    "aivector_usage": [
        ("usages", ["Aivector Usage Rate"]),
        ("common", ["Aivector Usage Rate"]),
    ],
    "ctrlcpu_usage": [
        ("usages", ["Ctrlcpu Usage Rate"]),
        ("common", ["Ctrlcpu Usage Rate"]),
    ],
    "npu_util": [
        ("usages", ["NPU Utilization"]),
        ("common", ["NPU Utilization"]),
    ],
    "product_name": [
        ("product", ["Product Name"]),
        ("board", ["Product Name"]),
    ],
}


def _query_typed_subcommand(
    npu_id: int,
    subcmd: str,
    key_patterns: List[str],
) -> ParseResult:
    """Query a typed subcommand and extract value by key patterns."""
    warnings: List[str] = []
    stdout, rc, stderr = _run_cmd(
        ["npu-smi", "info", "-t", subcmd, "-i", str(npu_id)]
    )

    if rc != 0:
        err_msg = stderr.strip() or stdout.strip()
        if "does not support" in err_msg.lower():
            return ParseResult(
                None, "low",
                [f"Device NPU {npu_id} does not support -t {subcmd}"]
            )
        return ParseResult(
            None, "low",
            [f"npu-smi info -t {subcmd} -i {npu_id} failed: {err_msg}"]
        )

    if not stdout.strip():
        return ParseResult(
            None, "low",
            [f"npu-smi info -t {subcmd} returned empty output"]
        )

    is_kv, fmt_warnings = _validate_kv_format(stdout)
    warnings.extend(fmt_warnings)
    if not is_kv:
        warnings.append(
            f"Output does not appear to be key:value format. "
            f"Raw (first 200 chars): {stdout[:200]}"
        )

    if not key_patterns:
        return ParseResult(stdout.strip(), "medium" if warnings else "high", warnings)

    val, key_warnings = _parse_kv_fuzzy(stdout, key_patterns)
    warnings.extend(key_warnings)

    if val is None:
        return ParseResult(None, "low", warnings)

    confidence = "medium" if warnings else "high"
    return ParseResult(val, confidence, warnings)


def _query_raw_output(npu_id: int, subcmd: str) -> Optional[str]:
    """Fetch raw output of a typed subcommand for caching."""
    stdout, rc, _ = _run_cmd(
        ["npu-smi", "info", "-t", subcmd, "-i", str(npu_id)]
    )
    if rc == 0 and stdout.strip():
        return stdout.strip()
    return None


def _build_strategy_result(
    result: ParseResult,
    raw_output: bool,
    npu_id: int,
    subcmd: str,
    warnings: List[str],
) -> ParseResult:
    """Build final ParseResult, optionally fetching raw output."""
    if raw_output:
        raw = _query_raw_output(npu_id, subcmd)
        if raw:
            return ParseResult(raw, result.confidence, warnings + result.warnings)
    return ParseResult(
        result.value, result.confidence, warnings + result.warnings
    )


def query_with_strategy(
    discovery: NpuSmiDiscovery,
    npu_id: int,
    data_need: str,
    raw_output: bool = False,
) -> ParseResult:
    """Try multiple subcommands for a data need, with auto-discovery.

    Args:
        discovery: NpuSmiDiscovery instance
        npu_id: NPU card ID
        data_need: key in QUERY_STRATEGIES
        raw_output: if True, return full raw output of first successful subcmd
    """
    warnings: List[str] = []
    strategies = QUERY_STRATEGIES.get(data_need, [])

    if not strategies:
        return ParseResult(None, "low", [f"No query strategy defined for '{data_need}'"])

    for subcmd, key_patterns in strategies:
        if not discovery.is_available(subcmd):
            warnings.append(f"Subcommand '{subcmd}' not available in this environment")
            continue

        result = _query_typed_subcommand(npu_id, subcmd, key_patterns)
        if result.value is not None:
            return _build_strategy_result(
                result, raw_output, npu_id, subcmd, warnings
            )
        warnings.extend(result.warnings)

    return ParseResult(None, "low", warnings)


# ---------------------------------------------------------------------------
# Mapping table queries
# ---------------------------------------------------------------------------


def query_npu_smi_mapping() -> ParseResult:
    """Get NPU IDs from `npu-smi info -m` (mapping table)."""
    warnings: List[str] = []
    stdout, rc, stderr = _run_cmd(["npu-smi", "info", "-m"])
    if rc != 0:
        return ParseResult(
            None, "low", [f"npu-smi info -m failed (rc={rc}): {stderr.strip()}"]
        )

    lines = [l.rstrip() for l in stdout.split("\n") if l.strip()]
    if not lines:
        return ParseResult(None, "low", ["npu-smi info -m returned empty output"])

    header = lines[0]
    if "NPU ID" not in header:
        warnings.append(
            f"Mapping table header missing 'NPU ID'. Header: {header[:80]}"
        )

    npu_ids: set = set()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                npu_id = int(parts[0])
                npu_ids.add(npu_id)
            except ValueError:
                warnings.append(f"Cannot parse NPU ID from: {line[:60]}...")
        else:
            warnings.append(
                f"Mapping line too short ({len(parts)} fields): {line[:60]}..."
            )

    if not npu_ids:
        return ParseResult(
            None, "low", warnings + ["No NPU IDs found in mapping table"]
        )

    confidence = "medium" if warnings else "high"
    return ParseResult(sorted(npu_ids), confidence, warnings)


def _parse_chip_name_from_line(line: str, target_npu_id: int) -> Optional[str]:
    """Parse a single mapping line and return chip name if it matches target."""
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        if int(parts[0]) != target_npu_id:
            return None
        if parts[2] == "-":
            return None
        return " ".join(parts[3:])
    except ValueError:
        return None


def query_chip_name_from_mapping(npu_id: int) -> ParseResult:
    """Get chip name for a specific NPU from mapping table.

    Skips MCU chips (Chip Logic ID == "-").
    """
    warnings: List[str] = []
    stdout, rc, stderr = _run_cmd(["npu-smi", "info", "-m"])
    if rc != 0:
        return ParseResult(
            None, "low", [f"npu-smi info -m failed: {stderr.strip()}"]
        )

    lines = [l.rstrip() for l in stdout.split("\n") if l.strip()]
    for line in lines[1:]:
        chip_name = _parse_chip_name_from_line(line, npu_id)
        if chip_name:
            return ParseResult(chip_name, "high", warnings)

    return ParseResult(
        None, "low", warnings + [f"No non-MCU chip found for NPU {npu_id}"]
    )


# ---------------------------------------------------------------------------
# High-level collector with common-cache
# ---------------------------------------------------------------------------


def _parse_table_npu_ids(stdout: str) -> List[int]:
    """Parse NPU IDs from npu-smi info main table output."""
    ids = set()
    for line in stdout.split("\n"):
        m = re.search(r'^\|\s*(\d+)\s+\S+.*\|', line)
        if not m:
            continue
        try:
            ids.add(int(m.group(1)))
        except ValueError:
            pass
    return sorted(ids)


class NpuInfoCollector:
    """Collect NPU device information using structured subcommands."""

    def __init__(self):
        self.warnings: List[str] = []
        self._npu_ids: Optional[List[int]] = None
        self._discovery = NpuSmiDiscovery()
        # Cache: npu_id -> raw output of -t common
        self._common_cache: Dict[int, str] = {}

    @property
    def discovery(self) -> NpuSmiDiscovery:
        return self._discovery

    # -- Discovery ----------------------------------------------------------

    def get_npu_ids(self) -> List[int]:
        """Return sorted list of NPU device IDs."""
        if self._npu_ids is not None:
            return self._npu_ids

        result = query_npu_smi_mapping()
        self.warnings.extend(result.warnings)

        if result.value:
            self._npu_ids = result.value
            return self._npu_ids

        # FALLBACK: try npu-smi info main table (LAST RESORT, unreliable)
        self.warnings.append(
            "FALLBACK: parsing npu-smi info main table (unreliable)"
        )
        stdout, rc, stderr = _run_cmd(["npu-smi", "info"])
        if rc == 0 and stdout:
            ids = _parse_table_npu_ids(stdout)
            if ids:
                self._npu_ids = ids
                self.warnings.append(
                    "WARNING: NPU IDs from unreliable table format"
                )
                return self._npu_ids

        return []


    def get_chip_name(self, npu_id: int) -> Optional[str]:
        result = query_chip_name_from_mapping(npu_id)
        self.warnings.extend(result.warnings)
        return result.value

    def get_health(self, npu_id: int) -> Optional[str]:
        result = query_with_strategy(self._discovery, npu_id, "health")
        self.warnings.extend(result.warnings)
        return result.value

    def get_temperature(self, npu_id: int) -> Optional[str]:
        val = self._query_from_cache_or_strategy(
            npu_id, "temperature",
            ["NPU Temperature", "Temperature", "HBM Temperature"]
        )
        return val

    def get_power(self, npu_id: int) -> Optional[str]:
        val = self._query_from_cache_or_strategy(
            npu_id, "power",
            ["NPU Real-time Power", "Power"]
        )
        return val

    def get_memory_info(self, npu_id: int) -> Dict[str, Optional[str]]:
        info: Dict[str, Optional[str]] = {}

        # HBM Capacity is NOT in common cache; always query
        result = query_with_strategy(self._discovery, npu_id, "memory_capacity")
        self.warnings.extend(result.warnings)
        info["hbm_capacity_mb"] = result.value

        # HBM Usage Rate may be in common cache
        info["hbm_usage_rate"] = self._query_from_cache_or_strategy(
            npu_id, "memory_usage", ["HBM Usage Rate"]
        )

        return info

    def get_usage_info(self, npu_id: int) -> Dict[str, Optional[str]]:
        """Return usage info: aicore, aivector, hbm_usage, npu_util, ctrlcpu.

        Tries common cache first, then falls back to usages for fields
        not present in common (e.g., Aivector, NPU Utilization, Ctrlcpu).
        """
        cached = self._get_common_cache(npu_id)
        result = {
            "aicore": None,
            "aivector": None,
            "hbm_usage": None,
            "npu_util": None,
            "ctrlcpu": None,
        }

        if cached:
            result["aicore"] = _parse_kv(cached, "Aicore Usage Rate")
            result["hbm_usage"] = _parse_kv(cached, "HBM Usage Rate")
            # aivector, npu_util, ctrlcpu are NOT in common on some devices

        # Fill missing fields from usages subcommand
        needs_usages = any(v is None for v in result.values())
        if needs_usages and self._discovery.is_available("usages"):
            u_result = _query_typed_subcommand(npu_id, "usages", [])
            self.warnings.extend(u_result.warnings)
            raw = u_result.value or ""
            if result["aicore"] is None:
                result["aicore"] = _parse_kv(raw, "Aicore Usage Rate")
            if result["aivector"] is None:
                result["aivector"] = _parse_kv(raw, "Aivector Usage Rate")
            if result["hbm_usage"] is None:
                result["hbm_usage"] = _parse_kv(raw, "HBM Usage Rate")
            if result["npu_util"] is None:
                result["npu_util"] = _parse_kv(raw, "NPU Utilization")
            if result["ctrlcpu"] is None:
                result["ctrlcpu"] = _parse_kv(raw, "Ctrlcpu Usage Rate")

        return result

    def get_all_info(self, npu_id: int) -> Dict[str, any]:
        """Return comprehensive info for one NPU.

        Optimized to minimize npu-smi calls:
        1. One call to common (cached) for temp, power, usage rates
        2. One call to health for health status
        3. One call to memory for HBM capacity
        """
        # Prime the common cache
        self._get_common_cache(npu_id)

        return {
            "npu_id": npu_id,
            "chip_name": self.get_chip_name(npu_id),
            "health": self.get_health(npu_id),
            "temperature": self.get_temperature(npu_id),
            "power": self.get_power(npu_id),
            "memory": self.get_memory_info(npu_id),
            "usage": self.get_usage_info(npu_id),
        }

    def get_all_warnings(self) -> List[str]:
        return self.warnings.copy()

    def _get_common_cache(self, npu_id: int) -> Optional[str]:
        """Return cached -t common output, or fetch and cache it."""
        if npu_id in self._common_cache:
            return self._common_cache[npu_id]

        if not self._discovery.is_available("common"):
            return None

        result = _query_typed_subcommand(npu_id, "common", [])
        if result.value:
            self._common_cache[npu_id] = result.value
            return result.value
        return None

    def _query_from_cache_or_strategy(
        self, npu_id: int, data_need: str, cache_keys: List[str]
    ) -> Optional[str]:
        """Try cache first, then fall back to query_with_strategy."""
        # 1. Try common cache
        cached = self._get_common_cache(npu_id)
        if cached:
            for key in cache_keys:
                val = _parse_kv(cached, key)
                if val is not None:
                    return val

        # 2. Fall back to layered strategy
        result = query_with_strategy(self._discovery, npu_id, data_need)
        self.warnings.extend(result.warnings)
        return result.value

    # -- Per-device queries -------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _output_discover(discovery: NpuSmiDiscovery) -> int:
    types = discovery.get_available_list()
    _LOGGER.info("npu-smi info -t supports %d subcommand(s):", len(types))
    for t in types:
        _LOGGER.info("  - %s", t)
    return 0


def _output_list(npu_ids: List[int]) -> int:
    _LOGGER.info("%s", " ".join(str(i) for i in npu_ids))
    return 0


def _output_chip_names(collector: NpuInfoCollector, npu_ids: List[int]) -> int:
    for npu_id in npu_ids:
        name = collector.get_chip_name(npu_id)
        _LOGGER.info("NPU %d: %s", npu_id, name or "unknown")
    return 0


def _output_health(collector: NpuInfoCollector, npu_ids: List[int]) -> int:
    for npu_id in npu_ids:
        health = collector.get_health(npu_id)
        _LOGGER.info("NPU %d: %s", npu_id, health or "unknown")
    return 0


def _output_json(collector: NpuInfoCollector, npu_ids: List[int]) -> int:
    data = {
        "npu_count": len(npu_ids),
        "devices": [collector.get_all_info(i) for i in npu_ids],
        "warnings": collector.get_all_warnings(),
    }
    _LOGGER.info("%s", json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def _output_warnings(collector: NpuInfoCollector) -> int:
    for w in collector.get_all_warnings():
        _LOGGER.info("WARN: %s", w)
    return 0


def _output_summary(collector: NpuInfoCollector, npu_ids: List[int]) -> int:
    _LOGGER.info("Detected %d NPU device(s)", len(npu_ids))
    for npu_id in npu_ids:
        info = collector.get_all_info(npu_id)
        chip = info.get("chip_name") or "unknown"
        health = info.get("health") or "unknown"
        temp = info.get("temperature") or "?"
        power = info.get("power") or "?"
        _LOGGER.info(
            "  NPU %d: %s | Health=%s | Temp=%sC | Power=%sW",
            npu_id, chip, health, temp, power,
        )
    warnings = collector.get_all_warnings()
    if warnings:
        _LOGGER.info("")
        _LOGGER.info("WARNINGS (%d):", len(warnings))
        for w in warnings:
            _LOGGER.info("  ! %s", w)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust NPU info parser")
    parser.add_argument("--list", action="store_true", help="List NPU IDs")
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    parser.add_argument("--health", action="store_true", help="Health only")
    parser.add_argument("--chip-name", action="store_true", help="Chip names")
    parser.add_argument("--warnings", action="store_true", help="Show warnings")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover and show available npu-smi -t subcommands",
    )
    args = parser.parse_args()

    if args.discover:
        return _output_discover(NpuSmiDiscovery())

    collector = NpuInfoCollector()
    npu_ids = collector.get_npu_ids()

    if not npu_ids:
        _LOGGER.error("ERROR: No NPU devices detected")
        for w in collector.get_all_warnings():
            _LOGGER.error("WARN: %s", w)
        return 1

    if args.list:
        return _output_list(npu_ids)
    if args.chip_name:
        return _output_chip_names(collector, npu_ids)
    if args.health:
        return _output_health(collector, npu_ids)
    if args.json:
        return _output_json(collector, npu_ids)
    if args.warnings:
        return _output_warnings(collector)
    return _output_summary(collector, npu_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
