# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

SUPPORTED_INPUT_SCHEMAS = {"msopprof-collection/v2", "msopprof-collection/v1"}


# Canonical report page order; shared by visualize.py and self_check.py.
PAGE_ORDER = [
    "details",
    "roofline",
    "timeline",
    "cache",
    "onchip-memory",
    "source",
    "warp-stall",
    "instruction-timeline",
    "raw-data",
]


class VisualizationError(RuntimeError):
    pass


@dataclass
class ArtifactSet:
    visualize_data: Optional[Path]
    trace: Optional[Path]
    csv: Dict[str, Path]


@dataclass
class NormalizedCollection:
    root: Path
    schema: str
    blocks: Dict[str, ArtifactSet]
    manifest: Dict[str, Any]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualizationError(f"Failed to read JSON: {path.name}: {exc}") from exc


def safe_relative_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise VisualizationError("Manifest artifact path must stay relative to the collection root.")
    candidate = (root / Path(*pure.parts)).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise VisualizationError("Manifest artifact path escapes the collection root.") from exc
    return candidate


def normalize_artifacts(root: Path, data: Mapping[str, Any]) -> ArtifactSet:
    csv_map: Dict[str, Path] = {}
    for name, rel in (data.get("csv") or {}).items():
        p = safe_relative_path(root, str(rel))
        if p is not None:
            csv_map[str(name)] = p
    return ArtifactSet(
        visualize_data=safe_relative_path(root, data.get("visualize_data")),
        trace=safe_relative_path(root, data.get("trace")),
        csv=csv_map,
    )


def load_collection(input_root: Path) -> NormalizedCollection:
    root = input_root.resolve()
    manifest_path = root / "collection_manifest.json"
    if not manifest_path.is_file():
        raise VisualizationError("collection_manifest.json was not found in the input root.")
    manifest = read_json(manifest_path)
    schema = str(manifest.get("schema", ""))
    if schema not in SUPPORTED_INPUT_SCHEMAS:
        raise VisualizationError(f"Unsupported collection schema: {schema or '<missing>'}")

    blocks: Dict[str, ArtifactSet] = {}
    if schema == "msopprof-collection/v2":
        for block_id, entry in (manifest.get("blocks") or {}).items():
            blocks[str(block_id)] = normalize_artifacts(root, entry.get("artifacts") or {})
    else:
        runs = manifest.get("runs") or {}
        legacy_map = {
            "details": "primary",
            "roofline": "roofline",
            "timeline": "pipe_timeline",
            "source": "source",
            "warp_stall": "pc_sampling",
            "instruction_timeline": "instruction_timeline",
            "memory_detail": "memory_detail",
            "raw_data": "default_metrics",
        }
        for block_id, run_id in legacy_map.items():
            entry = runs.get(run_id)
            if entry:
                blocks[block_id] = normalize_artifacts(root, entry.get("artifacts") or {})
        # Legacy collections often used Roofline to provide the complete CSV suite.
        if "raw_data" not in blocks or not blocks["raw_data"].csv:
            roofline = blocks.get("roofline")
            if roofline and roofline.csv:
                blocks["raw_data"] = ArtifactSet(None, None, dict(roofline.csv))

    # Stable empty block objects keep the rendering code schema-independent.
    for block_id in [
        "details", "roofline", "timeline", "source", "warp_stall",
        "instruction_timeline", "memory_detail", "raw_data", "timeline_detail",
    ]:
        blocks.setdefault(block_id, ArtifactSet(None, None, {}))
    return NormalizedCollection(root=root, schema=schema, blocks=blocks, manifest=manifest)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        text = str(value).strip()
        if text.upper() in {"", "NA", "NAN", "NONE", "NULL", "INF", "-INF"}:
            return None
        x = float(text)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def mean(values: Iterable[Any]) -> Optional[float]:
    xs = [to_float(v) for v in values]
    ys = [x for x in xs if x is not None]
    return sum(ys) / len(ys) if ys else None


def median(values: Iterable[Any]) -> Optional[float]:
    xs = [to_float(v) for v in values]
    ys = [x for x in xs if x is not None]
    return statistics.median(ys) if ys else None


def percentile(values: Iterable[Any], q: float) -> Optional[float]:
    xs: List[float] = []
    for value in values:
        number = to_float(value)
        if number is not None:
            xs.append(number)
    xs.sort()
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def humanize_key(key: str) -> str:
    text = re.sub(r"[_\-.]+", " ", str(key)).strip()
    aliases = {
        "l2cache hit rate": "L2 Hit Rate",
        "l2 cache hit rate": "L2 Hit Rate",
        "instruction per cycle": "IPC",
        "simt vf instructions instructions": "SIMT Instructions",
        "cycles": "Cycles",
        "throughput": "Throughput",
    }
    low = text.lower()
    if low in aliases:
        return aliases[low]
    if low.endswith("instruction per cycle"):
        return "IPC"
    if low.endswith("instructions") and "simt" in low:
        return "SIMT Instructions"
    parts = []
    for token in text.split():
        up = token.upper()
        if up in {"L2", "L1", "L0", "GM", "UB", "IPC", "PC", "SIMT", "AIC", "AIV", "MTE"}:
            parts.append(up)
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


def metric_is_percent(key: str, values: Sequence[Optional[float]]) -> bool:
    low = key.lower()
    # Only explicit rate/ratio/percent tokens (or a (%)-style unit signal) imply
    # a 0..100 percentage scale. Bare count metrics such as cache "hit" event
    # counts must stay numeric even when every value happens to be <= 100.
    if any(token in low for token in ["rate", "ratio", "percent", "(%)"]):
        finite = [v for v in values if v is not None]
        return not finite or max(abs(v) for v in finite) <= 100.0001
    return False


def flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(obj, Mapping):
        return out
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_numeric(value, name))
        elif not isinstance(value, list):
            x = to_float(value)
            if x is not None:
                out[name] = x
    return out


def iter_dicts(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def positive(value: Any, eps: float = 1e-12) -> bool:
    x = to_float(value)
    return x is not None and abs(x) > eps


def ratio_pct(value: Any) -> Optional[float]:
    x = to_float(value)
    if x is None:
        return None
    # CSV ratios are normally 0..1, while some payload values are already 0..100.
    return x * 100.0 if abs(x) <= 1.000001 else x
